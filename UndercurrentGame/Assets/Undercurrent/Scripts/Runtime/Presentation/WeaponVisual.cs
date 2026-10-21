using System.Collections.Generic;
using Undercurrent.CharacterGen;
using Undercurrent.Core.Combat;
using UnityEngine;

namespace Undercurrent.Runtime.Presentation
{
    /// <summary>Builds the in-hand weapon mesh (knife, sword, staff) from <see cref="WeaponMeshGenerator"/>.</summary>
    public static class WeaponVisual
    {
        private static readonly Dictionary<string, Mesh> Cache = new Dictionary<string, Mesh>();

        public static GameObject Create(WeaponClass weapon, float bodyHeight, Transform parent)
        {
            string key = weapon + "|" + bodyHeight.ToString("0.000");
            if (!Cache.TryGetValue(key, out var mesh) || mesh == null)
            {
                var g = WeaponMeshGenerator.Generate(weapon, bodyHeight);
                mesh = new Mesh { name = "UC_" + weapon };
                var v = new List<Vector3>(g.Positions.Count);
                var n = new List<Vector3>(g.Normals.Count);
                foreach (var p in g.Positions) v.Add(new Vector3(p.X, p.Y, p.Z));
                foreach (var p in g.Normals) n.Add(new Vector3(p.X, p.Y, p.Z));
                mesh.SetVertices(v);
                mesh.SetNormals(n);
                mesh.subMeshCount = g.Triangles.Length;
                for (int i = 0; i < g.Triangles.Length; i++) mesh.SetTriangles(g.Triangles[i], i);
                mesh.RecalculateBounds();
                Cache[key] = mesh;
            }

            var go = new GameObject("Weapon_" + weapon);
            go.transform.SetParent(parent, false);
            go.AddComponent<MeshFilter>().sharedMesh = mesh;
            var r = go.AddComponent<MeshRenderer>();
            r.sharedMaterials = new[]
            {
                VisualAssets.Lit(new Color(0.78f, 0.8f, 0.84f), 0.85f, 0.9f), // steel
                VisualAssets.Lit(new Color(0.15f, 0.1f, 0.08f), 0.25f),       // leather grip
                VisualAssets.Lit(new Color(0.5f, 0.33f, 0.18f), 0.35f),       // wood
                VisualAssets.Lit(new Color(0.75f, 0.58f, 0.25f), 0.7f, 0.8f), // brass fittings
            };
            return go;
        }
    }
}
