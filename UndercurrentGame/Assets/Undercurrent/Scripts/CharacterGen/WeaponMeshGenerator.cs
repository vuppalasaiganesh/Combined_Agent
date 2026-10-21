using System;
using System.Collections.Generic;
using System.Numerics;
using Undercurrent.Core.Combat;

namespace Undercurrent.CharacterGen
{
    public enum PropSlot { Steel, Grip, Wood, Brass }

    /// <summary>Small static mesh (weapon) in the holding hand's local space.</summary>
    public sealed class PropMesh
    {
        public readonly List<Vector3> Positions = new List<Vector3>();
        public readonly List<Vector3> Normals = new List<Vector3>();
        public readonly List<int>[] Triangles = { new List<int>(), new List<int>(), new List<int>(), new List<int>() };
    }

    /// <summary>
    /// Procedural weapons held in the right fist. Hand space: the fist hangs along -Y from the wrist joint, the grip
    /// axis runs front-to-back through the fist (+Z = blade direction).
    /// </summary>
    public static class WeaponMeshGenerator
    {
        public static PropMesh Generate(WeaponClass weapon, float bodyHeight)
        {
            var m = new PropMesh();
            float h = bodyHeight / 1.78f;
            var grip = new Vector3(0, -0.045f * bodyHeight, 0.006f * bodyHeight);
            switch (weapon)
            {
                case WeaponClass.Knife:
                    Prism(m, grip, -0.06f * h, 0.05f * h, 0.011f * h, 0.013f * h, 8, PropSlot.Grip, 1f);
                    Prism(m, grip, 0.05f * h, 0.06f * h, 0.006f * h, 0.022f * h, 4, PropSlot.Brass, 1f);
                    Blade(m, grip, 0.06f * h, 0.26f * h, 0.012f * h, 0.0025f * h);
                    break;
                case WeaponClass.Sword:
                    Prism(m, grip, -0.13f * h, 0.05f * h, 0.012f * h, 0.014f * h, 8, PropSlot.Grip, 1f);
                    Prism(m, grip, -0.15f * h, -0.13f * h, 0.018f * h, 0.018f * h, 8, PropSlot.Brass, 1f);
                    Prism(m, grip, 0.05f * h, 0.07f * h, 0.012f * h, 0.085f * h, 4, PropSlot.Brass, 1f); // crossguard
                    Blade(m, grip, 0.07f * h, 0.95f * h, 0.02f * h, 0.003f * h);
                    break;
                case WeaponClass.Staff:
                    Prism(m, grip, -0.70f * h, 1.0f * h, 0.016f * h, 0.016f * h, 10, PropSlot.Wood, 1f);
                    Prism(m, grip, -0.72f * h, -0.66f * h, 0.018f * h, 0.018f * h, 10, PropSlot.Brass, 1f);
                    Prism(m, grip, 0.96f * h, 1.02f * h, 0.018f * h, 0.018f * h, 10, PropSlot.Brass, 1f);
                    break;
            }
            // Built along +Z; turn so the weapon continues the forearm line, angled forward of it
            // (+Z maps to normalize(0, -0.8, -0.6) in hand space): in guard the blade points up-forward,
            // in a thrust it extends the arm towards the opponent.
            var r = Matrix3.RotX(126.87f);
            for (int i = 0; i < m.Positions.Count; i++)
            {
                m.Positions[i] = grip + r.Transform(m.Positions[i] - grip);
                m.Normals[i] = r.Transform(m.Normals[i]);
            }
            return m;
        }

        /// <summary>Capped prism along +Z from z0 to z1 with elliptical section (rx = X radius, ry = Y radius).</summary>
        private static void Prism(PropMesh m, Vector3 origin, float z0, float z1, float rx, float ry, int sides, PropSlot slot, float taper)
        {
            int start = m.Positions.Count;
            for (int ring = 0; ring < 2; ring++)
            {
                float z = ring == 0 ? z0 : z1;
                float k = ring == 0 ? 1f : taper;
                for (int i = 0; i < sides; i++)
                {
                    double a = 2 * Math.PI * (i + 0.5) / sides;
                    var n = new Vector3((float)Math.Cos(a), (float)Math.Sin(a), 0);
                    m.Positions.Add(origin + new Vector3(n.X * rx * k, n.Y * ry * k, z));
                    m.Normals.Add(n);
                }
            }
            for (int i = 0; i < sides; i++)
            {
                int a = start + i, b = start + (i + 1) % sides, c = start + sides + i, d = start + sides + (i + 1) % sides;
                Oriented(m, slot, a, b, c, origin);
                Oriented(m, slot, b, d, c, origin);
            }
            Cap(m, origin + new Vector3(0, 0, z0), start, sides, -Vector3.UnitZ, slot);
            Cap(m, origin + new Vector3(0, 0, z1), start + sides, sides, Vector3.UnitZ, slot);
        }

        private static void Cap(PropMesh m, Vector3 centre, int ringStart, int sides, Vector3 normal, PropSlot slot)
        {
            int baseIndex = m.Positions.Count;
            m.Positions.Add(centre);
            m.Normals.Add(normal);
            for (int i = 0; i < sides; i++)
            {
                m.Positions.Add(m.Positions[ringStart + i]);
                m.Normals.Add(normal);
            }
            var tris = m.Triangles[(int)slot];
            for (int i = 0; i < sides; i++)
            {
                int a = baseIndex + 1 + i, b = baseIndex + 1 + (i + 1) % sides;
                var n = Vector3.Cross(m.Positions[a] - centre, m.Positions[b] - centre);
                if (Vector3.Dot(n, normal) >= 0) { tris.Add(baseIndex); tris.Add(a); tris.Add(b); }
                else { tris.Add(baseIndex); tris.Add(b); tris.Add(a); }
            }
        }

        /// <summary>Adds a triangle wound so its face normal points away from the weapon's long axis (through origin along Z).</summary>
        private static void Oriented(PropMesh m, PropSlot slot, int a, int b, int c, Vector3 origin)
        {
            var pa = m.Positions[a];
            var n = Vector3.Cross(m.Positions[b] - pa, m.Positions[c] - pa);
            var centre = (pa + m.Positions[b] + m.Positions[c]) / 3f;
            var outward = centre - new Vector3(origin.X, origin.Y, centre.Z);
            var tris = m.Triangles[(int)slot];
            if (Vector3.Dot(n, outward) >= 0f) { tris.Add(a); tris.Add(b); tris.Add(c); }
            else { tris.Add(a); tris.Add(c); tris.Add(b); }
        }

        /// <summary>Flat double-edged blade in the YZ plane (edge up/down), tapering to a point at z1.</summary>
        private static void Blade(PropMesh m, Vector3 origin, float z0, float z1, float halfWidth, float halfThick)
        {
            var tris = m.Triangles[(int)PropSlot.Steel];
            // Diamond cross-section: (x=±thick, y=0) and (x=0, y=±width); point at the tip.
            Vector3 P(float z, float w, float t, int corner)
            {
                switch (corner)
                {
                    case 0: return origin + new Vector3(0, w, z);
                    case 1: return origin + new Vector3(t, 0, z);
                    case 2: return origin + new Vector3(0, -w, z);
                    default: return origin + new Vector3(-t, 0, z);
                }
            }
            float zTaper = z0 + (z1 - z0) * 0.8f;
            var tip = origin + new Vector3(0, 0, z1);
            for (int face = 0; face < 4; face++)
            {
                int c0 = face, c1 = (face + 1) % 4;
                var a = P(z0, halfWidth, halfThick, c0);
                var b = P(z0, halfWidth, halfThick, c1);
                var c = P(zTaper, halfWidth * 0.9f, halfThick, c0);
                var d = P(zTaper, halfWidth * 0.9f, halfThick, c1);
                AddFlatQuad(m, tris, origin, a, b, d, c);
                AddFlatTri(m, tris, origin, c, d, tip);
            }
        }

        private static void AddFlatQuad(PropMesh m, List<int> tris, Vector3 axis, Vector3 a, Vector3 b, Vector3 c, Vector3 d)
        {
            AddFlatTri(m, tris, axis, a, b, c);
            AddFlatTri(m, tris, axis, a, c, d);
        }

        /// <summary>Flat-shaded triangle (own vertices) facing away from the blade's centre line.</summary>
        private static void AddFlatTri(PropMesh m, List<int> tris, Vector3 axis, Vector3 a, Vector3 b, Vector3 c)
        {
            var n = Vector3.Cross(b - a, c - a);
            var centre = (a + b + c) / 3f;
            var outward = centre - new Vector3(axis.X, axis.Y, centre.Z);
            if (Vector3.Dot(n, outward) < 0f)
            {
                var t = b; b = c; c = t;
                n = -n;
            }
            n = n.LengthSquared() > 1e-20f ? Vector3.Normalize(n) : Vector3.UnitY;
            int i = m.Positions.Count;
            m.Positions.Add(a); m.Positions.Add(b); m.Positions.Add(c);
            m.Normals.Add(n); m.Normals.Add(n); m.Normals.Add(n);
            tris.Add(i); tris.Add(i + 1); tris.Add(i + 2);
        }
    }
}
