using System;
using System.Numerics;
using NUnit.Framework;
using Undercurrent.CharacterGen;
using Undercurrent.Core.Content;
using Pose = Undercurrent.CharacterGen.Pose;

namespace Undercurrent.Tests
{
    public class CharacterGenTests
    {
        private static readonly string[] Fighters = { FighterRoster.Mara, FighterRoster.Dima, FighterRoster.Haruto, FighterRoster.Ines, FighterRoster.Rook, FighterRoster.Sable };

        [Test]
        public void GeneratedBodies_AreValidSkinnedMeshes()
        {
            foreach (var id in Fighters)
            {
                var spec = FighterAppearance.For(id).Body;
                var m = HumanBodyGenerator.Generate(spec);
                Assert.AreEqual(m.Positions.Count, m.Normals.Count, id);
                Assert.AreEqual(m.Positions.Count, m.Influences.Count, id);
                Assert.Less(m.Positions.Count, 65000, id + " must fit 16-bit indices");
                Assert.Less(m.TriangleCount, 40000, id + " triangle budget");
                float minY = float.MaxValue, maxY = float.MinValue;
                for (int i = 0; i < m.Positions.Count; i++)
                {
                    var p = m.Positions[i];
                    Assert.IsFalse(float.IsNaN(p.X + p.Y + p.Z), id + " NaN position");
                    Assert.AreEqual(1f, m.Normals[i].Length(), 1e-3f, id + " normal length");
                    var w = m.Influences[i];
                    Assert.That(w.B0, Is.InRange(0, HumanSkeleton.JointCount - 1));
                    Assert.That(w.B1, Is.InRange(0, HumanSkeleton.JointCount - 1));
                    Assert.That(w.W1, Is.InRange(0f, 1f));
                    minY = Math.Min(minY, p.Y);
                    maxY = Math.Max(maxY, p.Y);
                }
                foreach (var list in m.Triangles)
                {
                    Assert.AreEqual(0, list.Count % 3);
                    foreach (var idx in list) Assert.That(idx, Is.InRange(0, m.Positions.Count - 1));
                }
                Assert.AreEqual(0f, minY, 0.02f, id + " feet should touch the floor");
                Assert.AreEqual(spec.Height, maxY, spec.Height * 0.05f, id + " overall height");
                Assert.Greater(m.Triangles[(int)MaterialSlot.Skin].Count, 0);
                Assert.Greater(m.Triangles[(int)MaterialSlot.Hair].Count, 0, id + " hair");
                Assert.Greater(m.Triangles[(int)MaterialSlot.Iris].Count, 0, id + " eyes");
            }
        }

        [Test]
        public void SurfaceNormals_PointOutwards()
        {
            // For the body skin, the vast majority of normals should point away from the body's vertical axis.
            var m = HumanBodyGenerator.Generate(FighterAppearance.For(FighterRoster.Dima).Body);
            int outward = 0, total = 0;
            foreach (var idx in m.Triangles[(int)MaterialSlot.Skin])
            {
                var p = m.Positions[idx];
                if (p.Y < 0.5f || p.Y > 1.4f || Math.Abs(p.X) > 0.12f) continue; // torso only
                var radial = new Vector3(p.X, 0, p.Z);
                if (radial.LengthSquared() < 1e-6f) continue;
                total++;
                if (Vector3.Dot(radial, m.Normals[idx]) > 0f) outward++;
            }
            Assert.Greater(total, 100);
            Assert.Greater(outward / (float)total, 0.9f);
        }

        [Test]
        public void BodiesAreLeftRightSymmetric()
        {
            var m = HumanBodyGenerator.Generate(FighterAppearance.For(FighterRoster.Haruto).Body);
            float sumX = 0f;
            foreach (var p in m.Positions) sumX += p.X;
            Assert.AreEqual(0f, sumX / m.Positions.Count, 0.003f);
        }

        [Test]
        public void RestPoseSkinning_IsIdentity()
        {
            var m = HumanBodyGenerator.Generate(FighterAppearance.For(FighterRoster.Mara).Body);
            var solver = new PoseSolver(m.Skeleton);
            solver.Solve(null, Matrix3.Identity, Vector3.Zero);
            var pos = new Vector3[m.Positions.Count];
            var nrm = new Vector3[m.Positions.Count];
            solver.Skin(m, pos, nrm);
            for (int i = 0; i < pos.Length; i++) Assert.Less(Vector3.Distance(pos[i], m.Positions[i]), 1e-4f);
        }

        [Test]
        public void EulerConvention_MatchesUnity()
        {
            // Unity: Quaternion.Euler(90,0,0) * forward = down; Euler(0,90,0) * forward = right; Euler(0,0,90) * right = up.
            AssertClose(new Vector3(0, -1, 0), Matrix3.Euler(new Vector3(90, 0, 0)).Transform(Vector3.UnitZ));
            AssertClose(new Vector3(1, 0, 0), Matrix3.Euler(new Vector3(0, 90, 0)).Transform(Vector3.UnitZ));
            AssertClose(new Vector3(0, 1, 0), Matrix3.Euler(new Vector3(0, 0, 90)).Transform(Vector3.UnitX));
            // Order Z, then X, then Y.
            var v = Matrix3.Euler(new Vector3(90, 90, 0)).Transform(Vector3.UnitZ);
            AssertClose(Matrix3.RotY(90).Transform(Matrix3.RotX(90).Transform(Vector3.UnitZ)), v);
        }

        [Test]
        public void Grounding_PlantsFeetInEveryGroundedPose()
        {
            var poses = new[] { PoseLibrary.Guard, PoseLibrary.Crouch, PoseLibrary.Block, PoseLibrary.CrouchBlock, PoseLibrary.HitBack,
                PoseLibrary.Dodge, PoseLibrary.Staggered, PoseLibrary.GuardBroken, PoseLibrary.Victory, PoseLibrary.ThrowGrab };
            foreach (var id in Fighters)
            {
                var sk = new HumanSkeleton(FighterAppearance.For(id).Body);
                var grounding = new PoseGrounding(sk);
                var solver = new PoseSolver(sk);
                foreach (var authored in poses)
                {
                    var p = authored.Copy();
                    grounding.Apply(p);
                    solver.Solve(p, Matrix3.Identity, Vector3.Zero);
                    float yl = grounding.ContactHeight(solver, Joint.FootL), yr = grounding.ContactHeight(solver, Joint.FootR);
                    string ctx = $"{id} pose#{Array.IndexOf(poses, authored)}";
                    Assert.AreEqual(0f, Math.Min(yl, yr), 0.005f, ctx + ": lower foot not on the floor");
                    float second = Math.Max(yl, yr);
                    Assert.IsTrue(Math.Abs(second) < 0.015f || second > 0.12f, $"{ctx}: second foot neither planted nor clearly lifted ({second:0.000} m)");
                    foreach (var e in p.Euler) Assert.IsFalse(float.IsNaN(e.X + e.Y + e.Z));
                }
            }
        }

        [Test]
        public void Grounding_LeavesKickingLegLifted()
        {
            var sk = new HumanSkeleton(FighterAppearance.For(FighterRoster.Haruto).Body);
            var p = PoseLibrary.For(AttackStyle.FrontKick).strike.Copy();
            new PoseGrounding(sk).Apply(p);
            var solver = new PoseSolver(sk);
            solver.Solve(p, Matrix3.Identity, Vector3.Zero);
            var g = new PoseGrounding(sk);
            float yl = g.ContactHeight(solver, Joint.FootL), yr = g.ContactHeight(solver, Joint.FootR);
            Assert.Greater(Math.Max(yl, yr), 0.5f, "kicking foot should stay high");
            Assert.AreEqual(0f, Math.Min(yl, yr), 0.005f, "support foot planted");
        }

        [Test]
        public void EveryMove_HasAPose()
        {
            foreach (var id in Fighters)
                foreach (var m in FighterRoster.Get(id).Moves)
                {
                    var (windup, strike) = PoseLibrary.For(PoseLibrary.StyleFor(m));
                    Assert.IsNotNull(windup, $"{id}/{m.Id}");
                    Assert.IsNotNull(strike, $"{id}/{m.Id}");
                }
        }

        private static void AssertClose(Vector3 expected, Vector3 actual)
        {
            Assert.Less(Vector3.Distance(expected, actual), 1e-4f, $"expected {expected} got {actual}");
        }
    }
}
