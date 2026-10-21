using System;
using System.Globalization;
using System.IO;
using System.Numerics;
using System.Text;
using Undercurrent.CharacterGen;
using Undercurrent.Core.Content;

// Usage: dotnet run -- <outDir> [fighterId:pose:yaw ...]
// Writes one .mesh file per entry: "v x y z nx ny nz r g b" lines then "f a b c" lines (0-based).
internal static class Program
{
    private static int Main(string[] args)
    {
        string outDir = args.Length > 0 ? args[0] : "preview_out";
        Directory.CreateDirectory(outDir);
        var jobs = args.Length > 1 ? args[1..] : new[]
        {
            "mara:rest:0", "dima:rest:0", "haruto:rest:0", "ines:rest:0", "rook:rest:0", "sable:rest:0",
            "mara:guard:90", "dima:guard:90", "haruto:guard:90", "rook:guard:90",
        };
        foreach (var job in jobs)
        {
            var parts = job.Split(':');
            string id = parts[0], pose = parts.Length > 1 ? parts[1] : "rest";
            float yaw = parts.Length > 2 ? float.Parse(parts[2], CultureInfo.InvariantCulture) : 0f;
            var look = FighterAppearance.For(id);
            var mesh = HumanBodyGenerator.Generate(look.Body);
            var solver = new PoseSolver(mesh.Skeleton);
            var p = PoseFor(pose)?.Copy();
            if (p != null && pose != "lying") new PoseGrounding(mesh.Skeleton).Apply(p);
            solver.Solve(p, Matrix3.RotY(yaw), Vector3.Zero);
            var pos = new Vector3[mesh.Positions.Count];
            var nrm = new Vector3[mesh.Positions.Count];
            solver.Skin(mesh, pos, nrm);
            Write(Path.Combine(outDir, $"{id}_{pose}_{yaw:0}.mesh"), mesh, pos, nrm, look);
            Console.WriteLine($"{id}:{pose}  verts={mesh.Positions.Count} tris={mesh.TriangleCount}");
        }
        return 0;
    }

    private static Pose PoseFor(string name)
    {
        switch (name)
        {
            case "rest": return null;
            case "guard": return PoseLibrary.Guard;
            case "crouch": return PoseLibrary.Crouch;
            case "block": return PoseLibrary.Block;
            case "lying": return PoseLibrary.Lying;
            case "victory": return PoseLibrary.Victory;
            case "hit": return PoseLibrary.HitBack;
        }
        foreach (AttackStyle s in Enum.GetValues(typeof(AttackStyle)))
            if (string.Equals(s.ToString(), name, StringComparison.OrdinalIgnoreCase)) return PoseLibrary.For(s).strike;
        throw new ArgumentException("Unknown pose " + name);
    }

    private static void Write(string path, HumanMesh mesh, Vector3[] pos, Vector3[] nrm, FighterAppearance look)
    {
        var inv = CultureInfo.InvariantCulture;
        var sb = new StringBuilder();
        // Vertices are duplicated per material so colours can be per-vertex.
        int count = 0;
        var fsb = new StringBuilder();
        for (int slot = 0; slot < mesh.Triangles.Length; slot++)
        {
            var c = Color(look, (MaterialSlot)slot);
            var tris = mesh.Triangles[slot];
            for (int t = 0; t < tris.Count; t += 3)
            {
                for (int k = 0; k < 3; k++)
                {
                    int v = tris[t + k];
                    sb.Append("v ").Append(pos[v].X.ToString(inv)).Append(' ').Append(pos[v].Y.ToString(inv)).Append(' ').Append(pos[v].Z.ToString(inv))
                      .Append(' ').Append(nrm[v].X.ToString(inv)).Append(' ').Append(nrm[v].Y.ToString(inv)).Append(' ').Append(nrm[v].Z.ToString(inv))
                      .Append(' ').Append(c.R.ToString(inv)).Append(' ').Append(c.G.ToString(inv)).Append(' ').Append(c.B.ToString(inv)).Append('\n');
                }
                fsb.Append("f ").Append(count).Append(' ').Append(count + 1).Append(' ').Append(count + 2).Append('\n');
                count += 3;
            }
        }
        File.WriteAllText(path, sb.ToString() + fsb.ToString());
    }

    private static Rgb Color(FighterAppearance a, MaterialSlot slot)
    {
        switch (slot)
        {
            case MaterialSlot.Skin: return a.Skin;
            case MaterialSlot.Lips: return a.Lips;
            case MaterialSlot.EyeWhite: return new Rgb(0.92f, 0.9f, 0.86f);
            case MaterialSlot.Iris: return a.Iris;
            case MaterialSlot.Brow:
            case MaterialSlot.Hair: return a.Hair;
            case MaterialSlot.Outfit: return a.Outfit;
            case MaterialSlot.OutfitTrim: return a.OutfitTrim;
            case MaterialSlot.Wraps: return a.Wraps;
            default: return a.Shoes;
        }
    }
}
