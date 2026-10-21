using System;
using System.Collections.Generic;
using System.Numerics;

namespace Undercurrent.CharacterGen
{
    /// <summary>Two-bone skin influence (weight of B1 is W1, weight of B0 is 1 - W1).</summary>
    public struct BoneInfluence
    {
        public int B0, B1;
        public float W1;

        public static BoneInfluence Single(Joint j) => new BoneInfluence { B0 = (int)j, B1 = (int)j, W1 = 0f };

        public static BoneInfluence Blend(Joint a, Joint b, float t) =>
            new BoneInfluence { B0 = (int)a, B1 = (int)b, W1 = Smooth(Clamp01(t)) };

        internal static float Clamp01(float v) => v < 0f ? 0f : v > 1f ? 1f : v;
        internal static float Smooth(float t) => t * t * (3f - 2f * t);
    }

    /// <summary>Engine-independent skinned mesh in rest pose (character space, metres).</summary>
    public sealed class HumanMesh
    {
        public readonly HumanSkeleton Skeleton;
        public readonly List<Vector3> Positions = new List<Vector3>();
        public readonly List<Vector3> Normals = new List<Vector3>();
        public readonly List<BoneInfluence> Influences = new List<BoneInfluence>();
        public readonly List<int>[] Triangles;

        public HumanMesh(HumanSkeleton skeleton)
        {
            Skeleton = skeleton;
            int n = Enum.GetValues(typeof(MaterialSlot)).Length;
            Triangles = new List<int>[n];
            for (int i = 0; i < n; i++) Triangles[i] = new List<int>();
        }

        public int TriangleCount
        {
            get
            {
                int c = 0;
                foreach (var t in Triangles) c += t.Count / 3;
                return c;
            }
        }
    }

    /// <summary>
    /// Generates an original, continuous, skinned human body procedurally: anatomically proportioned torso and limbs
    /// with muscle profiles, a sculpted head (eyes, brows, nose, lips, ears), hair styles, beards and clothing layers
    /// that share the body's skin weights. Stylised-realistic placeholder until licensed scanned/sculpted models exist.
    /// </summary>
    public static class HumanBodyGenerator
    {
        private struct Ring
        {
            public Vector3 Center, U, V;
            public float HalfU, FrontV, BackV;
        }

        private sealed class Builder
        {
            public readonly HumanMesh Mesh;
            public Builder(HumanMesh mesh) => Mesh = mesh;

            public int Vertex(Vector3 p, BoneInfluence w)
            {
                Mesh.Positions.Add(p);
                Mesh.Normals.Add(Vector3.Zero);
                Mesh.Influences.Add(w);
                return Mesh.Positions.Count - 1;
            }

            /// <summary>Adds a triangle, flipping it if needed so its normal points along <paramref name="outward"/>.</summary>
            public void Tri(MaterialSlot slot, int a, int b, int c, Vector3 outward)
            {
                var pa = Mesh.Positions[a];
                var n = Vector3.Cross(Mesh.Positions[b] - pa, Mesh.Positions[c] - pa);
                var list = Mesh.Triangles[(int)slot];
                if (Vector3.Dot(n, outward) >= 0f) { list.Add(a); list.Add(b); list.Add(c); }
                else { list.Add(a); list.Add(c); list.Add(b); }
            }

            /// <summary>Generalised cylinder through rings (super-ellipse cross-sections with separate front/back depth).</summary>
            public void Loft(IList<Ring> rings, int around, MaterialSlot slot, Func<Vector3, BoneInfluence> weight,
                bool capStart, bool capEnd, float squareness = 2f)
            {
                int start = Mesh.Positions.Count;
                float e = 2f / squareness;
                for (int i = 0; i < rings.Count; i++)
                {
                    var r = rings[i];
                    for (int j = 0; j < around; j++)
                    {
                        double a = 2.0 * Math.PI * j / around;
                        float cu = (float)Math.Cos(a), sv = (float)Math.Sin(a);
                        float su = Math.Sign(cu) * (float)Math.Pow(Math.Abs(cu), e);
                        float ss = Math.Sign(sv) * (float)Math.Pow(Math.Abs(sv), e);
                        var p = r.Center + r.U * (su * r.HalfU) + r.V * (ss * (ss > 0 ? r.FrontV : r.BackV));
                        Vertex(p, weight(p));
                    }
                }
                for (int i = 0; i + 1 < rings.Count; i++)
                {
                    for (int j = 0; j < around; j++)
                    {
                        int a = start + i * around + j;
                        int b = start + i * around + (j + 1) % around;
                        int c = start + (i + 1) * around + j;
                        int d = start + (i + 1) * around + (j + 1) % around;
                        var mid = (Mesh.Positions[a] + Mesh.Positions[d]) * 0.5f;
                        var axis = (rings[i].Center + rings[i + 1].Center) * 0.5f;
                        var outward = mid - axis;
                        Tri(slot, a, b, c, outward);
                        Tri(slot, b, d, c, outward);
                    }
                }
                if (capStart) Cap(rings[0], rings[1].Center, start, around, slot, weight);
                if (capEnd) Cap(rings[rings.Count - 1], rings[rings.Count - 2].Center, start + (rings.Count - 1) * around, around, slot, weight);
            }

            private void Cap(Ring r, Vector3 inner, int ringStart, int around, MaterialSlot slot, Func<Vector3, BoneInfluence> weight)
            {
                var outward = r.Center - inner;
                // Slightly domed cap so ends look rounded.
                var tip = r.Center + Vector3.Normalize(outward) * Math.Min(r.HalfU, Math.Min(r.FrontV, r.BackV)) * 0.45f;
                int c = Vertex(tip, weight(tip));
                for (int j = 0; j < around; j++)
                    Tri(slot, c, ringStart + j, ringStart + (j + 1) % around, outward);
            }

            /// <summary>Lat/long sphere deformed by a shape function; triangles kept only where <paramref name="mask"/> holds.</summary>
            public void Sphere(Vector3 center, int lat, int lon, Func<Vector3, Vector3> shape, MaterialSlot slot,
                Func<Vector3, BoneInfluence> weight, Func<Vector3, bool> mask = null, Func<Vector3, MaterialSlot> slotFor = null)
            {
                int start = Mesh.Positions.Count;
                var dirs = new List<Vector3>();
                for (int i = 0; i <= lat; i++)
                {
                    double th = Math.PI * i / lat;
                    for (int j = 0; j <= lon; j++)
                    {
                        double ph = Math.PI + 2.0 * Math.PI * j / lon; // seam at the back
                        var u = new Vector3((float)(Math.Sin(th) * Math.Sin(ph)), (float)Math.Cos(th), (float)(Math.Sin(th) * Math.Cos(ph)));
                        dirs.Add(u);
                        var p = center + shape(u);
                        Vertex(p, weight(p));
                    }
                }
                int stride = lon + 1;
                for (int i = 0; i < lat; i++)
                {
                    for (int j = 0; j < lon; j++)
                    {
                        int a = start + i * stride + j, b = a + 1, c = a + stride, d = c + 1;
                        var du = Vector3.Normalize(dirs[i * stride + j] + dirs[i * stride + j + 1] + dirs[(i + 1) * stride + j] + dirs[(i + 1) * stride + j + 1] + new Vector3(1e-5f, 0, 0));
                        if (mask != null && !mask(du)) continue;
                        var s = slotFor != null ? slotFor(du) : slot;
                        var outward = (Mesh.Positions[a] + Mesh.Positions[d]) * 0.5f - center;
                        if (i > 0) Tri(s, a, b, c, outward);
                        if (i < lat - 1) Tri(s, b, d, c, outward);
                    }
                }
            }

            public void Ellipsoid(Vector3 center, Vector3 radii, MaterialSlot slot, BoneInfluence w, int lat = 8, int lon = 12)
            {
                Sphere(center, lat, lon, u => new Vector3(u.X * radii.X, u.Y * radii.Y, u.Z * radii.Z), slot, _ => w);
            }

            /// <summary>Tube along a polyline with per-point radius.</summary>
            public void Tube(IList<Vector3> points, IList<float> radii, int around, MaterialSlot slot, BoneInfluence w)
            {
                var rings = new List<Ring>();
                for (int i = 0; i < points.Count; i++)
                {
                    var dir = Vector3.Normalize(i + 1 < points.Count ? points[i + 1] - points[i] : points[i] - points[i - 1]);
                    var u = Vector3.Normalize(Vector3.Cross(dir, Math.Abs(dir.Y) > 0.9f ? Vector3.UnitZ : Vector3.UnitY));
                    var v = Vector3.Cross(u, dir);
                    if (Vector3.Dot(Vector3.Cross(u, v), dir) < 0) v = -v;
                    rings.Add(new Ring { Center = points[i], U = u, V = v, HalfU = radii[i], FrontV = radii[i], BackV = radii[i] });
                }
                Loft(rings, around, slot, _ => w, true, true);
            }

            public void ComputeNormals()
            {
                var pos = Mesh.Positions;
                var nrm = new Vector3[pos.Count];
                foreach (var list in Mesh.Triangles)
                {
                    for (int t = 0; t < list.Count; t += 3)
                    {
                        int a = list[t], b = list[t + 1], c = list[t + 2];
                        var n = Vector3.Cross(pos[b] - pos[a], pos[c] - pos[a]); // area-weighted
                        nrm[a] += n; nrm[b] += n; nrm[c] += n;
                    }
                }
                for (int i = 0; i < nrm.Length; i++)
                    Mesh.Normals[i] = nrm[i].LengthSquared() > 1e-20f ? Vector3.Normalize(nrm[i]) : Vector3.UnitY;
            }
        }

        // ------------------------------------------------------------------ profile tables

        /// <summary>Interpolates a keyed profile (keys strictly increasing) with smooth easing between keys.</summary>
        private static float Sample(float[] keys, float[] values, float t)
        {
            if (t <= keys[0]) return values[0];
            for (int i = 0; i + 1 < keys.Length; i++)
            {
                if (t <= keys[i + 1])
                {
                    float k = (t - keys[i]) / (keys[i + 1] - keys[i]);
                    k = k * k * (3f - 2f * k);
                    return values[i] + (values[i + 1] - values[i]) * k;
                }
            }
            return values[values.Length - 1];
        }

        private static float Gauss(float x, float y, float cx, float cy, float sx, float sy)
        {
            float dx = (x - cx) / sx, dy = (y - cy) / sy;
            return (float)Math.Exp(-(dx * dx + dy * dy));
        }

        private static float SmoothStep(float a, float b, float x)
        {
            float t = BoneInfluence.Clamp01((x - a) / (b - a));
            return t * t * (3f - 2f * t);
        }

        private static int Steps(float detail, int baseCount, int min = 6) => Math.Max(min, (int)Math.Round(baseCount * detail));

        // Torso profile (fractions of height): y, half width, front depth, back depth, z offset of centre.
        private static readonly float[] TorsoY = { 0.475f, 0.50f, 0.54f, 0.58f, 0.62f, 0.66f, 0.70f, 0.74f, 0.78f, 0.81f, 0.835f, 0.855f, 0.872f, 0.895f, 0.92f };
        private static readonly float[] TorsoWMale = { 0.058f, 0.086f, 0.093f, 0.090f, 0.080f, 0.081f, 0.086f, 0.090f, 0.093f, 0.090f, 0.074f, 0.050f, 0.038f, 0.034f, 0.032f };
        private static readonly float[] TorsoFMale = { 0.040f, 0.050f, 0.054f, 0.055f, 0.052f, 0.055f, 0.062f, 0.068f, 0.067f, 0.058f, 0.045f, 0.036f, 0.032f, 0.031f, 0.030f };
        private static readonly float[] TorsoBMale = { 0.040f, 0.055f, 0.068f, 0.058f, 0.048f, 0.050f, 0.055f, 0.058f, 0.060f, 0.060f, 0.054f, 0.044f, 0.036f, 0.033f, 0.030f };
        private static readonly float[] TorsoWFem = { 0.062f, 0.096f, 0.103f, 0.096f, 0.071f, 0.073f, 0.079f, 0.083f, 0.085f, 0.083f, 0.068f, 0.045f, 0.033f, 0.029f, 0.028f };
        private static readonly float[] TorsoFFem = { 0.040f, 0.052f, 0.056f, 0.055f, 0.050f, 0.052f, 0.058f, 0.062f, 0.060f, 0.054f, 0.042f, 0.033f, 0.029f, 0.027f, 0.026f };
        private static readonly float[] TorsoBFem = { 0.042f, 0.060f, 0.074f, 0.062f, 0.046f, 0.048f, 0.052f, 0.054f, 0.056f, 0.056f, 0.050f, 0.040f, 0.032f, 0.029f, 0.027f };
        private static readonly float[] TorsoZ = { 0f, 0f, -0.004f, 0f, 0.002f, 0.002f, 0.002f, 0.002f, 0f, -0.004f, -0.008f, -0.012f, -0.012f, -0.008f, -0.004f };

        private static readonly float[] UpperArmT = { 0f, 0.12f, 0.35f, 0.6f, 0.85f, 1f };
        private static readonly float[] UpperArmW = { 0.026f, 0.038f, 0.034f, 0.031f, 0.027f, 0.026f };
        private static readonly float[] UpperArmF = { 0.024f, 0.035f, 0.036f, 0.034f, 0.028f, 0.026f };
        private static readonly float[] UpperArmB = { 0.024f, 0.035f, 0.034f, 0.031f, 0.026f, 0.025f };

        private static readonly float[] ForearmT = { 0f, 0.2f, 0.5f, 0.8f, 1f };
        private static readonly float[] ForearmW = { 0.026f, 0.027f, 0.022f, 0.018f, 0.016f };
        private static readonly float[] ForearmF = { 0.027f, 0.030f, 0.026f, 0.021f, 0.019f };
        private static readonly float[] ForearmB = { 0.026f, 0.029f, 0.025f, 0.020f, 0.018f };

        private static readonly float[] FistT = { 0f, 0.25f, 0.55f, 0.85f, 1f };
        private static readonly float[] FistW = { 0.017f, 0.020f, 0.022f, 0.020f, 0.012f };
        private static readonly float[] FistF = { 0.022f, 0.028f, 0.032f, 0.030f, 0.015f };
        private static readonly float[] FistB = { 0.020f, 0.024f, 0.026f, 0.024f, 0.012f };

        private static readonly float[] ThighT = { 0f, 0.12f, 0.3f, 0.55f, 0.8f, 0.95f, 1f };
        private static readonly float[] ThighW = { 0.062f, 0.066f, 0.062f, 0.055f, 0.045f, 0.038f, 0.036f };
        private static readonly float[] ThighF = { 0.060f, 0.064f, 0.064f, 0.058f, 0.048f, 0.040f, 0.038f };
        private static readonly float[] ThighB = { 0.060f, 0.064f, 0.058f, 0.050f, 0.040f, 0.036f, 0.035f };

        private static readonly float[] ShinT = { 0f, 0.1f, 0.3f, 0.55f, 0.8f, 1f };
        private static readonly float[] ShinW = { 0.037f, 0.036f, 0.038f, 0.032f, 0.024f, 0.021f };
        private static readonly float[] ShinF = { 0.037f, 0.036f, 0.034f, 0.030f, 0.024f, 0.021f };
        private static readonly float[] ShinB = { 0.036f, 0.040f, 0.047f, 0.036f, 0.025f, 0.022f };

        private static readonly float[] FootZ = { -0.03f, -0.018f, 0.015f, 0.05f, 0.088f, 0.11f };
        private static readonly float[] FootY = { 0.028f, 0.030f, 0.030f, 0.022f, 0.014f, 0.011f };
        private static readonly float[] FootW = { 0.020f, 0.023f, 0.027f, 0.032f, 0.030f, 0.022f };
        private static readonly float[] FootUp = { 0.026f, 0.03f, 0.03f, 0.021f, 0.012f, 0.009f };
        private static readonly float[] FootDown = { 0.026f, 0.028f, 0.028f, 0.020f, 0.012f, 0.009f };

        // ------------------------------------------------------------------ generation

        public static HumanMesh Generate(BodySpec spec)
        {
            var sk = new HumanSkeleton(spec);
            var mesh = new HumanMesh(sk);
            var b = new Builder(mesh);
            float h = spec.Height;
            float d = Math.Max(0.5f, spec.Detail);

            BuildTorso(b, spec, sk, 0f, 0.475f, 0.92f, MaterialSlot.Skin);
            BuildHead(b, spec, sk);
            for (int side = -1; side <= 1; side += 2)
            {
                BuildArm(b, spec, sk, side);
                BuildLeg(b, spec, sk, side);
            }
            if (spec.Female) BuildChest(b, spec, sk);
            BuildClothing(b, spec, sk);
            b.ComputeNormals();
            return mesh;
        }

        private static BoneInfluence TorsoWeight(Vector3 p, float h)
        {
            float y = p.Y / h;
            if (y < 0.60f) return BoneInfluence.Single(Joint.Hips);
            if (y < 0.67f) return BoneInfluence.Blend(Joint.Hips, Joint.Spine, (y - 0.60f) / 0.07f);
            if (y < 0.70f) return BoneInfluence.Single(Joint.Spine);
            if (y < 0.76f) return BoneInfluence.Blend(Joint.Spine, Joint.Chest, (y - 0.70f) / 0.06f);
            if (y < 0.85f) return BoneInfluence.Single(Joint.Chest);
            if (y < 0.885f) return BoneInfluence.Blend(Joint.Chest, Joint.Neck, (y - 0.85f) / 0.03f);
            return BoneInfluence.Blend(Joint.Neck, Joint.Head, (y - 0.885f) / 0.025f);
        }

        private static void BuildTorso(Builder b, BodySpec s, HumanSkeleton sk, float offset, float yFrom, float yTo, MaterialSlot slot)
        {
            float h = s.Height;
            var W = s.Female ? TorsoWFem : TorsoWMale;
            var F = s.Female ? TorsoFFem : TorsoFMale;
            var B = s.Female ? TorsoBFem : TorsoBMale;
            int along = Steps(s.Detail, 30);
            var rings = new List<Ring>();
            for (int i = 0; i <= along; i++)
            {
                float y = yFrom + (yTo - yFrom) * i / along;
                // Muscle/build: shoulders and chest scale more than the waist.
                float upper = SmoothStep(0.68f, 0.78f, y) * (1f - SmoothStep(0.83f, 0.86f, y));
                float neck = SmoothStep(0.85f, 0.875f, y);
                float neckK = 1f + neck * (0.12f * s.Build - 0.02f);
                float buildK = 0.93f + 0.16f * s.Build * (0.4f + 0.6f * upper);
                float hw = Sample(TorsoY, W, y) * (neck > 0f ? neckK : buildK) * h + offset;
                float fd = Sample(TorsoY, F, y) * (0.95f + 0.12f * s.Build * upper) * (neck > 0f ? neckK : 1f) * h + offset;
                float bd = Sample(TorsoY, B, y) * (0.96f + 0.08f * s.Build) * (neck > 0f ? neckK : 1f) * h + offset;
                rings.Add(new Ring
                {
                    Center = new Vector3(0, y * h, Sample(TorsoY, TorsoZ, y) * h),
                    U = Vector3.UnitX, V = Vector3.UnitZ, HalfU = hw, FrontV = fd, BackV = bd,
                });
            }
            rings.Reverse(); // top → bottom so U×V points along the path
            bool capTop = false, capBottom = yFrom <= 0.48f; // top ends inside the head
            b.Loft(rings, Steps(s.Detail, 32, 12), slot, p => TorsoWeight(p, h), capTop, capBottom, slot == MaterialSlot.Skin ? 2.15f : 2.05f);
        }

        // ------------------------------------------------------------------ head

        private sealed class HeadShape
        {
            public Vector3 Center;
            public float A, B, C; // half width, half height, half depth
            public bool Female;
            public float Build;

            public Vector3 Point(Vector3 u)
            {
                float x = u.X, y = u.Y, z = u.Z;
                float r = 1f / (float)Math.Sqrt((x / A) * (x / A) + (y / B) * (y / B) + (z / C) * (z / C));
                var p = u * r;

                // Fuller cranium at the back, tapered jaw, chin and neck attachment underneath.
                p.Z *= 1f + 0.10f * BoneInfluence.Clamp01(-z) * BoneInfluence.Clamp01((y + 0.2f) / 0.6f);
                float low = BoneInfluence.Clamp01(-y);
                p.X *= 1f - (Female ? 0.28f : 0.18f) * low * low;
                float underBack = BoneInfluence.Clamp01(0.35f - z);
                p.Y += B * 0.14f * low * low * underBack;
                p.Z *= 1f - 0.22f * low * underBack;
                // Rounder (less egg-shaped) cranium: widen the upper head, lower the crown slightly.
                float upperHead = BoneInfluence.Clamp01(y);
                p.X *= 1f + 0.10f * upperHead * (1f - upperHead * 0.6f);
                p.Z *= 1f + 0.05f * upperHead;
                p.Y *= 1f - 0.05f * upperHead * upperHead;

                float front = SmoothStep(0.25f, 0.75f, z);
                if (front > 0f)
                {
                    float fem = Female ? 0.6f : 1f;
                    float ax = Math.Abs(x);
                    float f = 0f;
                    f += Gauss(x, y, 0f, 0.0f, 0.06f, 0.2f) * 0.16f * (0.7f + 0.3f * fem);    // nasal bridge
                    f += Gauss(x, y, 0f, -0.2f, 0.085f, 0.08f) * 0.22f * (0.8f + 0.2f * fem); // nose tip
                    f += Gauss(ax, y, 0.11f, -0.26f, 0.06f, 0.05f) * 0.05f;                  // nostrils
                    f += Gauss(ax, y, 0.28f, 0.25f, 0.22f, 0.065f) * 0.07f * fem;            // brow ridge
                    f -= Gauss(ax, y, 0.30f, 0.12f, 0.12f, 0.075f) * 0.10f;                  // eye sockets
                    f += Gauss(x, y, 0f, -0.47f, 0.2f, 0.1f) * 0.05f;                       // muzzle under the lips
                    f += Gauss(x, y, 0f, -0.77f, 0.16f, 0.11f) * (Female ? 0.06f : 0.09f);   // chin
                    p.Z += front * C * f;
                    // Cheekbones and (male) square jaw push sideways and forward.
                    float side = Gauss(ax, y, 0.48f, -0.08f, 0.14f, 0.12f) * 0.06f
                                 + (Female ? 0f : Gauss(ax, y, 0.45f, -0.62f, 0.18f, 0.15f) * 0.05f * (0.6f + 0.6f * Build));
                    p.X += Math.Sign(x) * side * A * front;
                    p.Z += side * C * 0.5f * front;
                }
                return p;
            }
        }

        private static HeadShape MakeHead(BodySpec s, HumanSkeleton sk)
        {
            float h = s.Height;
            return new HeadShape
            {
                Center = new Vector3(0, 0.935f * h, 0.006f * h),
                A = (s.Female ? 0.039f : 0.041f) * h,
                B = 0.061f * h,
                C = 0.053f * h,
                Female = s.Female,
                Build = s.Build,
            };
        }

        private static Vector3 Dir(float x, float y)
        {
            float z = (float)Math.Sqrt(Math.Max(0.0, 1.0 - x * x - y * y));
            return Vector3.Normalize(new Vector3(x, y, z));
        }

        private static void BuildHead(Builder b, BodySpec s, HumanSkeleton sk)
        {
            float h = s.Height;
            var head = MakeHead(s, sk);
            var w = BoneInfluence.Single(Joint.Head);
            int lat = Steps(s.Detail, 56, 18), lon = Steps(s.Detail, 72, 24);
            b.Sphere(head.Center, lat, lon, head.Point, MaterialSlot.Skin, _ => w);

            // Lips: sculpted pieces on the muzzle (smooth outline regardless of head tessellation).
            float lipK = s.Female ? 1.2f : 1f;
            var upper = head.Center + head.Point(Dir(0f, -0.43f));
            var lower = head.Center + head.Point(Dir(0f, -0.52f));
            b.Ellipsoid(upper + new Vector3(0, 0, -0.0028f * h), new Vector3(0.0135f, 0.0031f * lipK, 0.0045f * lipK) * h, MaterialSlot.Lips, w, 6, 12);
            b.Ellipsoid(lower + new Vector3(0, 0.0006f * h, -0.003f * h), new Vector3(0.0125f, 0.0037f * lipK, 0.0048f * lipK) * h, MaterialSlot.Lips, w, 6, 12);

            // Eyes: eyeball set into the socket, dark iris in front.
            // Eyes at real scale (eyeball ≈ 24 mm), set into the socket, with upper and lower lids.
            float eyeR = 0.0068f * h;
            for (int side = -1; side <= 1; side += 2)
            {
                var surface = head.Center + head.Point(Dir(side * 0.30f, 0.12f));
                var eye = surface + new Vector3(0, 0, -eyeR * 0.35f);
                b.Ellipsoid(eye, new Vector3(eyeR, eyeR, eyeR), MaterialSlot.EyeWhite, w, 8, 12);
                b.Ellipsoid(eye + new Vector3(0, 0, eyeR * 0.86f), new Vector3(eyeR * 0.52f, eyeR * 0.52f, eyeR * 0.22f), MaterialSlot.Iris, w, 6, 12);
                b.Ellipsoid(eye + new Vector3(0, eyeR * 0.62f, eyeR * 0.2f), new Vector3(eyeR * 1.3f, eyeR * 0.55f, eyeR * 0.95f), MaterialSlot.Skin, w, 6, 12);
                b.Ellipsoid(eye + new Vector3(0, -eyeR * 0.78f, eyeR * 0.15f), new Vector3(eyeR * 1.2f, eyeR * 0.4f, eyeR * 0.9f), MaterialSlot.Skin, w, 6, 12);

                // Brows: separate, slightly arched strokes (no unibrow).
                for (int k = 0; k < 3; k++)
                {
                    float bx = side * (0.17f + 0.085f * k);
                    float by = 0.262f + (k == 1 ? 0.012f : 0f) - (k == 2 ? 0.006f : 0f);
                    var bp = head.Center + head.Point(Dir(bx, by)) + new Vector3(0, 0, 0.0012f * h);
                    b.Ellipsoid(bp, new Vector3(0.0062f * h, (s.Female ? 0.0014f : 0.002f) * h, 0.0016f * h), MaterialSlot.Brow, w, 4, 8);
                }

                var ear = head.Center + new Vector3(side * head.A * 0.99f, 0.0f * h, -0.008f * h);
                b.Ellipsoid(ear, new Vector3(0.0045f * h, 0.0175f * h, 0.0105f * h), MaterialSlot.Skin, w, 6, 10);
            }

            BuildHair(b, s, head);
        }

        private static Vector3 Spherical(float theta, float phi) =>
            new Vector3((float)(Math.Sin(theta) * Math.Sin(phi)), (float)Math.Cos(theta), (float)(Math.Sin(theta) * Math.Cos(phi)));

        /// <summary>
        /// Builds a shell patch over the head between two latitude curves, so its border follows a smooth curve
        /// (hairline, beard line) instead of the head's triangle grid. Thickness fades to zero at the borders.
        /// </summary>
        private static void HeadPatch(Builder b, HeadShape head, float phiFrom, float phiTo, bool closedLoop,
            Func<float, float> thetaStart, Func<float, float> thetaEnd, int nPhi, int nTheta, float thickness,
            float fadeStart, float fadeEnd, float fadeSides, MaterialSlot slot)
        {
            var w = BoneInfluence.Single(Joint.Head);
            int start = b.Mesh.Positions.Count;
            int cols = closedLoop ? nPhi : nPhi + 1;
            for (int i = 0; i < cols; i++)
            {
                float fi = (float)i / nPhi;
                float phi = phiFrom + (phiTo - phiFrom) * fi;
                float t0 = thetaStart(phi), t1 = thetaEnd(phi);
                float side = closedLoop ? 1f : SmoothStep(0f, fadeSides, Math.Min(fi, 1f - fi));
                for (int j = 0; j <= nTheta; j++)
                {
                    float fj = (float)j / nTheta;
                    var u = Spherical(t0 + (t1 - t0) * fj, phi);
                    float fade = SmoothStep(0f, Math.Max(1e-3f, fadeStart), fj) * SmoothStep(0f, Math.Max(1e-3f, fadeEnd), 1f - fj);
                    if (fadeStart <= 0f) fade = SmoothStep(0f, Math.Max(1e-3f, fadeEnd), 1f - fj);
                    var p = head.Center + head.Point(u) * (1.002f + thickness * fade * side);
                    b.Vertex(p, w);
                }
            }
            int rows = nTheta + 1;
            for (int i = 0; i < nPhi; i++)
            {
                int i1 = closedLoop ? (i + 1) % cols : i + 1;
                for (int j = 0; j < nTheta; j++)
                {
                    int a = start + i * rows + j, c = start + i1 * rows + j;
                    var outward = b.Mesh.Positions[a] - head.Center;
                    b.Tri(slot, a, c, a + 1, outward);
                    b.Tri(slot, c, c + 1, a + 1, outward);
                }
            }
        }

        private static float SolveTheta(Func<Vector3, float> f, float phi)
        {
            // f(u) crosses zero going from the crown (positive) downwards (negative); bisection on latitude.
            float lo = 0.01f, hi = 3.1f;
            for (int k = 0; k < 24; k++)
            {
                float mid = (lo + hi) * 0.5f;
                if (f(Spherical(mid, phi)) > 0f) lo = mid; else hi = mid;
            }
            return (lo + hi) * 0.5f;
        }

        private static void BuildHair(Builder b, BodySpec s, HeadShape head)
        {
            float h = s.Height;
            var w = BoneInfluence.Single(Joint.Head);
            int nPhi = Steps(s.Detail, 72, 24), nTheta = Steps(s.Detail, 22, 8);
            const float Pi = (float)Math.PI;

            // Hairline: high at the forehead, above the ears at the sides, down to the nape at the back.
            float Hairline(Vector3 u) => u.Z >= 0f ? 0.06f + 0.36f * u.Z * u.Z + 0.06f * u.Z : 0.06f - 0.62f * (-u.Z);
            float thickness = s.Hair == HairStyle.Shaved ? 0.012f : s.Hair == HairStyle.Braids ? 0.04f : 0.065f;
            HeadPatch(b, head, 0f, 2f * Pi, true, _ => 0.02f, phi => SolveTheta(u => u.Y - Hairline(u), phi),
                nPhi, nTheta, thickness, 0f, 0.3f, 0f, MaterialSlot.Hair);

            switch (s.Hair)
            {
                case HairStyle.Bun:
                {
                    var c = head.Center + head.Point(Vector3.Normalize(new Vector3(0, 0.72f, -0.7f))) * 1.08f;
                    b.Ellipsoid(c, new Vector3(0.028f, 0.025f, 0.026f) * h, MaterialSlot.Hair, w, 8, 12);
                    break;
                }
                case HairStyle.Ponytail:
                {
                    var root = head.Center + head.Point(Vector3.Normalize(new Vector3(0, 0.35f, -0.94f))) * 1.06f;
                    var pts = new List<Vector3> { root, root + new Vector3(0, -0.04f, -0.025f) * h, root + new Vector3(0, -0.10f, -0.035f) * h, root + new Vector3(0, -0.16f, -0.03f) * h };
                    b.Tube(pts, new[] { 0.016f * h, 0.018f * h, 0.014f * h, 0.006f * h }, 10, MaterialSlot.Hair, w);
                    break;
                }
                case HairStyle.Braids:
                {
                    for (int i = -2; i <= 2; i++)
                    {
                        var dir = Vector3.Normalize(new Vector3(i * 0.22f, 0.05f, -0.97f));
                        var root = head.Center + head.Point(dir) * 1.04f;
                        var pts = new List<Vector3> { root, root + new Vector3(i * 0.006f, -0.05f, -0.012f) * h, root + new Vector3(i * 0.01f, -0.11f, -0.006f) * h, root + new Vector3(i * 0.012f, -0.15f, 0.0f) * h };
                        b.Tube(pts, new[] { 0.009f * h, 0.008f * h, 0.007f * h, 0.004f * h }, 8, MaterialSlot.Hair, w);
                    }
                    break;
                }
            }

            if (s.Beard)
            {
                // Beard: from the cheek line (sides) / below the lower lip (centre) down under the jaw.
                float TopY(float phi)
                {
                    float a = Math.Abs(phi) * 180f / Pi;
                    return -0.6f + 0.5f * SmoothStep(16f, 46f, a);
                }
                HeadPatch(b, head, -1.95f, 1.95f, false, phi => (float)Math.Acos(TopY(phi)), _ => 2.55f,
                    Steps(s.Detail, 48, 16), Steps(s.Detail, 18, 6), 0.035f, 0.35f, 0.25f, 0.12f, MaterialSlot.Hair);
                // Moustache above the upper lip.
                HeadPatch(b, head, -0.55f, 0.55f, false, _ => (float)Math.Acos(-0.33f), _ => (float)Math.Acos(-0.425f),
                    18, 6, 0.045f, 0.45f, 0.15f, 0.2f, MaterialSlot.Hair);
            }
        }

        // ------------------------------------------------------------------ limbs

        private static void BuildArm(Builder b, BodySpec s, HumanSkeleton sk, int side)
        {
            float h = s.Height;
            bool left = side < 0;
            var shoulder = sk.Rest[(int)(left ? Joint.ShoulderL : Joint.ShoulderR)];
            var elbow = sk.Rest[(int)(left ? Joint.ElbowL : Joint.ElbowR)];
            var wrist = sk.Rest[(int)(left ? Joint.HandL : Joint.HandR)];
            var jS = left ? Joint.ShoulderL : Joint.ShoulderR;
            var jE = left ? Joint.ElbowL : Joint.ElbowR;
            var jH = left ? Joint.HandL : Joint.HandR;
            float k = (0.86f + 0.26f * s.Build) * (s.Female ? 0.86f : 1f);
            int around = Steps(s.Detail, 16, 8);
            int along = Steps(s.Detail, 12, 6);

            // Upper arm: deltoid cap down to the elbow.
            var top = shoulder + new Vector3(side * 0.006f * h, 0.014f * h, 0);
            var bottom = elbow + new Vector3(0, -0.012f * h, 0);
            b.Loft(Rings(top, bottom, along, UpperArmT, UpperArmW, UpperArmF, UpperArmB, h * k, 0f), around, MaterialSlot.Skin, p =>
            {
                if (p.Y > shoulder.Y - 0.01f * h)
                    return new BoneInfluence { B0 = (int)Joint.Chest, B1 = (int)jS, W1 = 0.5f + 0.5f * BoneInfluence.Smooth(BoneInfluence.Clamp01((shoulder.Y + 0.025f * h - p.Y) / (0.035f * h))) };
                return BoneInfluence.Blend(jS, jE, (elbow.Y + 0.035f * h - p.Y) / (0.045f * h));
            }, true, true);

            // Forearm.
            var fTop = elbow + new Vector3(0, 0.016f * h, 0);
            var fBottom = wrist + new Vector3(0, -0.006f * h, 0);
            Func<Vector3, BoneInfluence> forearmW = p => BoneInfluence.Blend(jE, jH, (wrist.Y + 0.03f * h - p.Y) / (0.03f * h));
            b.Loft(Rings(fTop, fBottom, along, ForearmT, ForearmW, ForearmF, ForearmB, h * k, 0f), around, MaterialSlot.Skin, forearmW, false, true);
            // Hand wraps over the lower forearm: same profile as the skin, slightly inflated, so it never clips.
            float wrapT0 = 1f - 0.05f * h / Vector3.Distance(fTop, fBottom);
            b.Loft(Rings(fTop, fBottom, Math.Max(3, along / 3), ForearmT, ForearmW, ForearmF, ForearmB, h * k, 0.0035f * h, wrapT0, 1f),
                around, MaterialSlot.Wraps, forearmW, false, false);

            // Wrapped fist (fighters always wear wraps or open-finger gloves).
            var fistTop = wrist + new Vector3(0, 0, 0.004f * h);
            var fistBottom = wrist + new Vector3(0, -0.085f * h, 0.006f * h);
            float fk = 0.9f + 0.2f * s.Build;
            b.Loft(Rings(fistTop, fistBottom, along, FistT, FistW, FistF, FistB, h * fk, 0f), around, MaterialSlot.Wraps, _ => BoneInfluence.Single(jH), false, true, 2.6f);
            b.Ellipsoid(wrist + new Vector3(-side * 0.008f * h, -0.035f * h, 0.026f * h), new Vector3(0.01f, 0.022f, 0.011f) * h * fk, MaterialSlot.Wraps, BoneInfluence.Single(jH));
        }

        private static void BuildLeg(Builder b, BodySpec s, HumanSkeleton sk, int side)
        {
            float h = s.Height;
            bool left = side < 0;
            var hip = sk.Rest[(int)(left ? Joint.HipL : Joint.HipR)];
            var knee = sk.Rest[(int)(left ? Joint.KneeL : Joint.KneeR)];
            var ankle = sk.Rest[(int)(left ? Joint.FootL : Joint.FootR)];
            var jH = left ? Joint.HipL : Joint.HipR;
            var jK = left ? Joint.KneeL : Joint.KneeR;
            var jF = left ? Joint.FootL : Joint.FootR;
            float k = 0.9f + 0.2f * s.Build;
            int around = Steps(s.Detail, 18, 8);
            int along = Steps(s.Detail, 14, 6);

            foreach (var (offset, slot, tTo, hem) in LegLayers(s))
            {
                var thighTop = new Vector3(hip.X * 1.08f, 0.60f * h, 0);
                var thighBottom = knee + new Vector3(0, -0.01f * h, 0);
                var thighRings = Rings(thighTop, thighBottom, along, ThighT, ThighW, ThighF, ThighB, h * k, offset);
                float thighEnd = Math.Min(1f, tTo);
                if (thighEnd < 1f) thighRings = Trim(thighRings, thighEnd, hem);
                Func<Vector3, BoneInfluence> thighW = p =>
                {
                    if (p.Y > knee.Y + 0.05f * h)
                        return BoneInfluence.Blend(Joint.Hips, jH, (0.585f * h - p.Y) / (0.085f * h));
                    return BoneInfluence.Blend(jH, jK, (knee.Y + 0.04f * h - p.Y) / (0.06f * h));
                };
                b.Loft(thighRings, around, slot, thighW, slot == MaterialSlot.Skin, thighEnd < 1f && slot == MaterialSlot.Skin);
                if (tTo <= 1f) continue;

                var shinTop = knee + new Vector3(0, 0.022f * h, 0);
                var shinBottom = new Vector3(ankle.X, 0.035f * h, ankle.Z);
                var shinRings = Rings(shinTop, shinBottom, along, ShinT, ShinW, ShinF, ShinB, h * (0.92f + 0.16f * s.Build), offset);
                if (tTo < 2f) shinRings = Trim(shinRings, tTo - 1f, hem);
                b.Loft(shinRings, around, slot, p => BoneInfluence.Blend(jK, jF, (0.085f * h - p.Y) / (0.045f * h)), false, slot == MaterialSlot.Skin);
            }

            // Foot (bare) or shoe, built heel → toe.
            var foot = new List<Ring>();
            float fs = s.Shoes ? 1.08f : 1f;
            for (int i = 0; i < FootZ.Length; i++)
            {
                float z = FootZ[i] * h * (s.Shoes ? 1.04f : 1f);
                foot.Add(new Ring
                {
                    Center = new Vector3(ankle.X + side * 0.004f * h * i / FootZ.Length, FootY[i] * h, z),
                    U = Vector3.UnitX, V = Vector3.UnitY,
                    HalfU = FootW[i] * h * fs, FrontV = FootUp[i] * h * fs, BackV = FootDown[i] * h,
                });
            }
            b.Loft(foot, around, s.Shoes ? MaterialSlot.Shoes : MaterialSlot.Skin, _ => BoneInfluence.Single(jF), true, true, 2.4f);
        }

        /// <summary>Leg surface layers: skin, then clothing shells (offset, material, extent in thigh=0..1 / shin=1..2, hem flare).</summary>
        private static IEnumerable<(float offset, MaterialSlot slot, float tTo, float hem)> LegLayers(BodySpec s)
        {
            float h = s.Height;
            yield return (0f, MaterialSlot.Skin, 2f, 0f);
            if (s.LongPants) yield return (0.009f * h, MaterialSlot.Outfit, 1.9f, 0.004f * h);
            else yield return (0.007f * h, MaterialSlot.Outfit, 0.45f, 0.006f * h);
        }

        private static List<Ring> Rings(Vector3 from, Vector3 to, int steps, float[] t, float[] w, float[] f, float[] bk, float scale, float offset,
            float t0 = 0f, float t1 = 1f)
        {
            var list = new List<Ring>();
            var dir = Vector3.Normalize(to - from);
            // Frame: U is the side axis (X); V is forward (Z) re-orthogonalised against the bone direction.
            var u = Vector3.Normalize(Vector3.Cross(Vector3.UnitZ, dir)) * Math.Sign(-dir.Y == 0 ? 1 : -dir.Y);
            var v = Vector3.Normalize(Vector3.Cross(dir, u));
            if (Vector3.Dot(Vector3.Cross(u, v), dir) < 0) v = -v;
            for (int i = 0; i <= steps; i++)
            {
                float tt = t0 + (t1 - t0) * i / steps;
                list.Add(new Ring
                {
                    Center = Vector3.Lerp(from, to, tt),
                    U = u, V = v,
                    HalfU = Sample(t, w, tt) * scale + offset,
                    FrontV = Sample(t, f, tt) * scale + offset,
                    BackV = Sample(t, bk, tt) * scale + offset,
                });
            }
            return list;
        }

        /// <summary>Cuts a ring list at fraction <paramref name="t"/> and flares the hem.</summary>
        private static List<Ring> Trim(List<Ring> rings, float t, float hem)
        {
            int n = Math.Max(2, (int)Math.Round((rings.Count - 1) * t) + 1);
            var cut = rings.GetRange(0, Math.Min(n, rings.Count));
            for (int i = 0; i < cut.Count; i++)
            {
                float k = (float)i / (cut.Count - 1);
                float flare = hem * k * k;
                var r = cut[i];
                r.HalfU += flare; r.FrontV += flare; r.BackV += flare;
                cut[i] = r;
            }
            return cut;
        }

        // ------------------------------------------------------------------ chest & clothing

        private static void BuildChest(Builder b, BodySpec s, HumanSkeleton sk)
        {
            float h = s.Height;
            var slot = s.Top != TopStyle.None ? MaterialSlot.Outfit : MaterialSlot.Skin;
            float k = 0.92f + 0.12f * s.Build;
            for (int side = -1; side <= 1; side += 2)
            {
                var c = new Vector3(side * 0.04f * h, 0.758f * h, 0.04f * h);
                b.Ellipsoid(c, new Vector3(0.034f, 0.03f, 0.022f) * h * k, slot, BoneInfluence.Single(Joint.Chest), 10, 14);
            }
        }

        private static void BuildClothing(Builder b, BodySpec s, HumanSkeleton sk)
        {
            float h = s.Height;
            // Waist garment (shorts or trousers), waistband trim.
            BuildTorso(b, s, sk, 0.008f * h, 0.475f, 0.60f, MaterialSlot.Outfit);
            BuildTorso(b, s, sk, 0.011f * h, 0.582f, 0.607f, MaterialSlot.OutfitTrim);

            switch (s.Top)
            {
                case TopStyle.TankTop:
                    BuildTorso(b, s, sk, 0.006f * h, 0.598f, 0.835f, MaterialSlot.Outfit);
                    BuildTorso(b, s, sk, 0.0075f * h, 0.598f, 0.612f, MaterialSlot.OutfitTrim);
                    break;
                case TopStyle.SportsBra:
                    BuildTorso(b, s, sk, 0.005f * h, 0.705f, 0.80f, MaterialSlot.Outfit);
                    BuildTorso(b, s, sk, 0.0065f * h, 0.705f, 0.716f, MaterialSlot.OutfitTrim);
                    break;
            }
        }
    }
}
