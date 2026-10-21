using System;
using System.Numerics;

namespace Undercurrent.CharacterGen
{
    /// <summary>
    /// Foot-planting inverse kinematics applied to authored poses. Authored poses use fixed hip heights, so on
    /// characters with different leg lengths feet would float or sink. Grounding (1) moves the hips so the lower foot
    /// rests on the floor, then (2) solves hip/knee pitch of the other leg so it is planted too, unless that foot is
    /// clearly lifted (kicks, knees, steps). Engine-free: used by the runtime presenter and the offline preview.
    /// </summary>
    public sealed class PoseGrounding
    {
        private readonly HumanSkeleton _sk;
        private readonly PoseSolver _solver;
        private readonly Vector3 _heel;
        private readonly Vector3 _toe;
        // Scratch buffers reused every call: grounding runs per frame and must not allocate.
        private static readonly Joint[] Feet = { Joint.FootL, Joint.FootR };
        private readonly int[] _dof = new int[3];
        private readonly Vector3[] _jac = new Vector3[3];

        public PoseGrounding(HumanSkeleton skeleton)
        {
            _sk = skeleton;
            _solver = new PoseSolver(skeleton);
            float h = skeleton.Height;
            float ankle = skeleton.Rest[(int)Joint.FootL].Y;
            // Sole contact points relative to the ankle joint in rest pose (heel and ball of the foot).
            _heel = new Vector3(0, -ankle, -0.026f * h);
            _toe = new Vector3(0, -ankle + 0.002f * h, 0.095f * h);
        }

        /// <summary>Height of the lowest sole point of a foot in the last solved pose.</summary>
        public float ContactHeight(PoseSolver solver, Joint foot)
        {
            var r = solver.Rotation[(int)foot];
            var p = solver.Position[(int)foot];
            return Math.Min((p + r.Transform(_heel)).Y, (p + r.Transform(_toe)).Y);
        }

        /// <param name="pose">Modified in place.</param>
        /// <param name="plantSecondFoot">False while walking so the stepping foot keeps its lift.</param>
        /// <param name="liftThreshold">A foot higher than this above the floor (metres) is treated as intentionally lifted.</param>
        public void Apply(Pose pose, bool plantSecondFoot = true, float liftThreshold = 0.12f)
        {
            _solver.Solve(pose, Matrix3.Identity, Vector3.Zero);
            float low = Math.Min(ContactHeight(_solver, Joint.FootL), ContactHeight(_solver, Joint.FootR));
            ShiftHips(pose, -low);
            if (!plantSecondFoot) return;

            for (int pass = 0; pass < 6; pass++)
            {
                Plant(pose, Joint.HipL, Joint.KneeL, Joint.FootL, liftThreshold);
                Plant(pose, Joint.HipR, Joint.KneeR, Joint.FootR, liftThreshold);
                // A leg that is already straight cannot reach the floor: sink the hips by the remaining gap and
                // let the other knee bend more on the next pass.
                _solver.Solve(pose, Matrix3.Identity, Vector3.Zero);
                float residual = 0f;
                foreach (var foot in Feet)
                {
                    float c = ContactHeight(_solver, foot);
                    if (c <= liftThreshold) residual = Math.Max(residual, c);
                }
                if (residual < 0.004f) return;
                ShiftHips(pose, -residual);
            }
            // Budget spent: make sure nothing is left below the floor.
            Plant(pose, Joint.HipL, Joint.KneeL, Joint.FootL, liftThreshold);
            Plant(pose, Joint.HipR, Joint.KneeR, Joint.FootR, liftThreshold);
        }

        private static void ShiftHips(Pose pose, float dy) =>
            pose.HipOffset = new Vector3(pose.HipOffset.X, pose.HipOffset.Y + dy, pose.HipOffset.Z);

        /// <summary>Returns true when the foot ends up planted (or is intentionally lifted).</summary>
        private bool Plant(Pose pose, Joint hip, Joint knee, Joint foot, float liftThreshold)
        {
            _solver.Solve(pose, Matrix3.Identity, Vector3.Zero);
            float above = ContactHeight(_solver, foot);
            if (above > liftThreshold) return true;
            const float SoftBias = 0.001f; // aim so the exact min lands on the floor despite the soft-min bias
            if (Math.Abs(above) < 0.003f) return true;
            var ankle = _solver.Position[(int)foot];
            // Objective vector: keep the ankle's horizontal position, bring the sole contact height to zero.
            var target = new Vector3(ankle.X, -SoftBias, ankle.Z);

            // Damped least squares on (hip pitch, knee pitch, ankle pitch) with a numeric Jacobian.
            var dof = _dof;
            dof[0] = (int)hip; dof[1] = (int)knee; dof[2] = (int)foot;
            var jac = _jac;
            for (int iter = 0; iter < 16; iter++)
            {
                var err = target - Probe(pose, dof, -1, 0f, foot);
                if (err.LengthSquared() < 1e-6f) return true;
                const float step = 0.5f;
                for (int k = 0; k < 3; k++)
                    jac[k] = (Probe(pose, dof, k, step, foot) - Probe(pose, dof, k, -step, foot)) / (2 * step);
                // Normal equations (JᵀJ + λI)Δ = Jᵀe, solved by Cramer's rule.
                const float lambda = 1e-8f; // Jacobian entries are ~1e-2 m/deg, so JᵀJ ~1e-4: keep damping well below that
                float a00 = Vector3.Dot(jac[0], jac[0]) + lambda, a01 = Vector3.Dot(jac[0], jac[1]), a02 = Vector3.Dot(jac[0], jac[2]);
                float a11 = Vector3.Dot(jac[1], jac[1]) + lambda, a12 = Vector3.Dot(jac[1], jac[2]), a22 = Vector3.Dot(jac[2], jac[2]) + lambda;
                float b0 = Vector3.Dot(jac[0], err), b1 = Vector3.Dot(jac[1], err), b2 = Vector3.Dot(jac[2], err);
                float det = a00 * (a11 * a22 - a12 * a12) - a01 * (a01 * a22 - a12 * a02) + a02 * (a01 * a12 - a11 * a02);
                if (Math.Abs(det) < 1e-30f) break; // entries are ~1e-4, so a 3x3 determinant is ~1e-12 when well-conditioned
                float d0 = (b0 * (a11 * a22 - a12 * a12) - a01 * (b1 * a22 - a12 * b2) + a02 * (b1 * a12 - a11 * b2)) / det;
                float d1 = (a00 * (b1 * a22 - a12 * b2) - b0 * (a01 * a22 - a12 * a02) + a02 * (a01 * b2 - b1 * a02)) / det;
                float d2 = (a00 * (a11 * b2 - b1 * a12) - a01 * (a01 * b2 - b1 * a02) + b0 * (a01 * a12 - a11 * a02)) / det;
                Nudge(pose, dof[0], d0, -120f, 60f);
                Nudge(pose, dof[1], d1, 0f, 150f);   // knees never hyperextend
                Nudge(pose, dof[2], d2, -70f, 45f);  // anatomical ankle range
            }
            return (target - Probe(pose, dof, -1, 0f, foot)).Length() < 0.01f;
        }

        private static void Nudge(Pose pose, int joint, float delta, float min, float max)
        {
            var e = pose.Euler[joint];
            delta = Math.Max(-15f, Math.Min(15f, delta));
            pose.Euler[joint] = new Vector3(Math.Max(min, Math.Min(max, e.X + delta)), e.Y, e.Z);
        }

        /// <summary>(ankle x, sole contact height, ankle z) with one degree of freedom temporarily offset.</summary>
        private Vector3 Probe(Pose pose, int[] dof, int which, float delta, Joint foot)
        {
            Vector3 saved = default;
            if (which >= 0)
            {
                saved = pose.Euler[dof[which]];
                pose.Euler[dof[which]] = new Vector3(saved.X + delta, saved.Y, saved.Z);
            }
            _solver.Solve(pose, Matrix3.Identity, Vector3.Zero);
            if (which >= 0) pose.Euler[dof[which]] = saved;
            var ankle = _solver.Position[(int)foot];
            return new Vector3(ankle.X, SmoothContactHeight(_solver, foot), ankle.Z);
        }

        /// <summary>Differentiable soft-minimum of heel/ball heights (bias ≤ 2 mm) so the solver sees a smooth objective.</summary>
        private float SmoothContactHeight(PoseSolver solver, Joint foot)
        {
            const float k = 0.003f;
            var r = solver.Rotation[(int)foot];
            var p = solver.Position[(int)foot];
            float a = (p + r.Transform(_heel)).Y, b = (p + r.Transform(_toe)).Y;
            float m = Math.Min(a, b);
            return m - k * (float)Math.Log(Math.Exp(-(a - m) / k) + Math.Exp(-(b - m) / k));
        }
    }
}
