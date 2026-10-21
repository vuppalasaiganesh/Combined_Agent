using System;
using System.Numerics;

namespace Undercurrent.CharacterGen
{
    /// <summary>
    /// Engine-independent forward kinematics and linear-blend skinning, using Unity's rotation conventions
    /// (Quaternion.Euler applies Z, then X, then Y). Used by the offline preview tool and tests; at runtime Unity's
    /// SkinnedMeshRenderer does the same work on the GPU/CPU.
    /// </summary>
    public sealed class PoseSolver
    {
        public readonly HumanSkeleton Skeleton;
        public readonly Matrix3[] Rotation = new Matrix3[HumanSkeleton.JointCount];
        public readonly Vector3[] Position = new Vector3[HumanSkeleton.JointCount];

        public PoseSolver(HumanSkeleton skeleton) => Skeleton = skeleton;

        /// <summary>Solves world transforms for a pose (null = rest pose). Root rotation/offset apply to the whole body.</summary>
        public void Solve(Pose pose, Matrix3 rootRotation, Vector3 rootOffset)
        {
            for (int i = 0; i < HumanSkeleton.JointCount; i++)
            {
                var local = pose != null ? Matrix3.Euler(pose.Euler[i]) : Matrix3.Identity;
                var offset = Skeleton.LocalOffset(i);
                if (i == (int)Joint.Hips && pose != null) offset += pose.HipOffset;
                int parent = HumanSkeleton.Parent[i];
                if (parent < 0)
                {
                    Rotation[i] = rootRotation * local;
                    Position[i] = rootOffset + rootRotation.Transform(offset);
                }
                else
                {
                    Rotation[i] = Rotation[parent] * local;
                    Position[i] = Position[parent] + Rotation[parent].Transform(offset);
                }
            }
        }

        /// <summary>Skins rest-pose vertices into the solved pose.</summary>
        public void Skin(HumanMesh mesh, Vector3[] positions, Vector3[] normals)
        {
            for (int v = 0; v < mesh.Positions.Count; v++)
            {
                var w = mesh.Influences[v];
                var p = mesh.Positions[v];
                var n = mesh.Normals[v];
                var p0 = Position[w.B0] + Rotation[w.B0].Transform(p - Skeleton.Rest[w.B0]);
                var n0 = Rotation[w.B0].Transform(n);
                if (w.W1 > 0f && w.B1 != w.B0)
                {
                    var p1 = Position[w.B1] + Rotation[w.B1].Transform(p - Skeleton.Rest[w.B1]);
                    var n1 = Rotation[w.B1].Transform(n);
                    p0 = Vector3.Lerp(p0, p1, w.W1);
                    n0 = Vector3.Lerp(n0, n1, w.W1);
                }
                positions[v] = p0;
                normals[v] = n0.LengthSquared() > 1e-12f ? Vector3.Normalize(n0) : Vector3.UnitY;
            }
        }
    }

    /// <summary>Minimal 3×3 rotation matrix (column-vector convention: v' = M·v).</summary>
    public struct Matrix3
    {
        public float M00, M01, M02, M10, M11, M12, M20, M21, M22;

        public static readonly Matrix3 Identity = new Matrix3 { M00 = 1, M11 = 1, M22 = 1 };

        public static Matrix3 RotX(float deg)
        {
            float r = deg * (float)Math.PI / 180f, c = (float)Math.Cos(r), s = (float)Math.Sin(r);
            return new Matrix3 { M00 = 1, M11 = c, M12 = -s, M21 = s, M22 = c };
        }

        public static Matrix3 RotY(float deg)
        {
            float r = deg * (float)Math.PI / 180f, c = (float)Math.Cos(r), s = (float)Math.Sin(r);
            return new Matrix3 { M00 = c, M02 = s, M11 = 1, M20 = -s, M22 = c };
        }

        public static Matrix3 RotZ(float deg)
        {
            float r = deg * (float)Math.PI / 180f, c = (float)Math.Cos(r), s = (float)Math.Sin(r);
            return new Matrix3 { M00 = c, M01 = -s, M10 = s, M11 = c, M22 = 1 };
        }

        /// <summary>Same result as UnityEngine.Quaternion.Euler(e): rotate about Z, then X, then Y.</summary>
        public static Matrix3 Euler(Vector3 e) => RotY(e.Y) * RotX(e.X) * RotZ(e.Z);

        public static Matrix3 operator *(Matrix3 a, Matrix3 b) => new Matrix3
        {
            M00 = a.M00 * b.M00 + a.M01 * b.M10 + a.M02 * b.M20,
            M01 = a.M00 * b.M01 + a.M01 * b.M11 + a.M02 * b.M21,
            M02 = a.M00 * b.M02 + a.M01 * b.M12 + a.M02 * b.M22,
            M10 = a.M10 * b.M00 + a.M11 * b.M10 + a.M12 * b.M20,
            M11 = a.M10 * b.M01 + a.M11 * b.M11 + a.M12 * b.M21,
            M12 = a.M10 * b.M02 + a.M11 * b.M12 + a.M12 * b.M22,
            M20 = a.M20 * b.M00 + a.M21 * b.M10 + a.M22 * b.M20,
            M21 = a.M20 * b.M01 + a.M21 * b.M11 + a.M22 * b.M21,
            M22 = a.M20 * b.M02 + a.M21 * b.M12 + a.M22 * b.M22,
        };

        public Vector3 Transform(Vector3 v) => new Vector3(
            M00 * v.X + M01 * v.Y + M02 * v.Z,
            M10 * v.X + M11 * v.Y + M12 * v.Z,
            M20 * v.X + M21 * v.Y + M22 * v.Z);
    }
}
