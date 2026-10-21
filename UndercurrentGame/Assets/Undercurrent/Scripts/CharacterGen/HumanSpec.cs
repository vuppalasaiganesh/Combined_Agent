using System;
using System.Collections.Generic;
using System.Numerics;
using Undercurrent.Core.Content;

namespace Undercurrent.CharacterGen
{
    /// <summary>Skeleton joints. Order guarantees parents precede children.</summary>
    public enum Joint
    {
        Hips, Spine, Chest, Neck, Head,
        ShoulderL, ElbowL, HandL,
        ShoulderR, ElbowR, HandR,
        HipL, KneeL, FootL,
        HipR, KneeR, FootR,
    }

    public enum HairStyle { Shaved, Short, Ponytail, Bun, Braids }

    public enum TopStyle { None, TankTop, SportsBra }

    public enum MaterialSlot { Skin, Lips, EyeWhite, Iris, Brow, Hair, Outfit, OutfitTrim, Wraps, Shoes }

    /// <summary>Body shape and wardrobe for the procedural human generator.</summary>
    public sealed class BodySpec
    {
        public float Height = 1.78f;
        /// <summary>0 = lean, 1 = heavy/muscular.</summary>
        public float Build = 0.5f;
        public bool Female;
        public HairStyle Hair = HairStyle.Short;
        public bool Beard;
        public TopStyle Top = TopStyle.None;
        public bool LongPants;
        public bool Shoes;
        /// <summary>Tessellation multiplier (graphics tier): 0.6 low … 1.2 ultra.</summary>
        public float Detail = 1f;

        public BodySpec Clone() => (BodySpec)MemberwiseClone();
    }

    /// <summary>Linear RGB colour (0..1) without engine dependencies.</summary>
    public struct Rgb
    {
        public float R, G, B;
        public Rgb(float r, float g, float b) { R = r; G = g; B = b; }

        public static Rgb Hex(string rrggbb)
        {
            if (string.IsNullOrEmpty(rrggbb) || rrggbb.Length < 6) return new Rgb(1, 0, 1);
            int v = Convert.ToInt32(rrggbb.Substring(0, 6), 16);
            return new Rgb(((v >> 16) & 255) / 255f, ((v >> 8) & 255) / 255f, (v & 255) / 255f);
        }
    }

    /// <summary>Default look of each fighter: body shape and colours.</summary>
    public sealed class FighterAppearance
    {
        public BodySpec Body;
        public Rgb Skin, Hair, Outfit, OutfitTrim, Wraps, Shoes, Lips, Iris;
        /// <summary>Idle bounce amplitude (metres): boxers bounce, karateka stay still.</summary>
        public float Bounce = 0.01f;

        public FighterAppearance Clone()
        {
            var a = (FighterAppearance)MemberwiseClone();
            a.Body = Body.Clone();
            return a;
        }

        public static readonly Rgb[] SkinTones =
        {
            new Rgb(0.93f, 0.76f, 0.65f),
            new Rgb(0.85f, 0.65f, 0.51f),
            new Rgb(0.74f, 0.54f, 0.40f),
            new Rgb(0.57f, 0.39f, 0.27f),
            new Rgb(0.42f, 0.28f, 0.19f),
            new Rgb(0.30f, 0.20f, 0.14f),
        };

        public static readonly string[] SkinToneNames = { "Fair", "Light", "Medium", "Tan", "Brown", "Deep" };

        private static FighterAppearance Make(BodySpec body, int skin, string hair, string outfit, string trim, string wraps, float bounce, string lips = null) =>
            new FighterAppearance
            {
                Body = body,
                Skin = SkinTones[skin],
                Hair = Rgb.Hex(hair),
                Outfit = Rgb.Hex(outfit),
                OutfitTrim = Rgb.Hex(trim),
                Wraps = Rgb.Hex(wraps),
                Shoes = Rgb.Hex("1E1E22"),
                Lips = lips != null ? Rgb.Hex(lips) : Darken(SkinTones[skin], 0.78f, 0.06f),
                Iris = Rgb.Hex("3B2A1E"),
                Bounce = bounce,
            };

        private static Rgb Darken(Rgb c, float k, float redBias) => new Rgb(Math.Min(1f, c.R * k + redBias), c.G * k * 0.92f, c.B * k * 0.92f);

        private static readonly Dictionary<string, FighterAppearance> Defaults = new Dictionary<string, FighterAppearance>
        {
            [FighterRoster.Mara] = Make(new BodySpec { Height = 1.74f, Build = 0.62f, Female = true, Hair = HairStyle.Braids, Top = TopStyle.SportsBra },
                4, "151110", "2B3A4A", "C9CED6", "EDEDED", 0.012f),
            [FighterRoster.Dima] = Make(new BodySpec { Height = 1.80f, Build = 0.5f, Hair = HairStyle.Short, Beard = true, Shoes = true },
                0, "6E5236", "2B3A4A", "C9CED6", "EDEDED", 0.03f),
            [FighterRoster.Haruto] = Make(new BodySpec { Height = 1.76f, Build = 0.38f, Hair = HairStyle.Bun, LongPants = true, Top = TopStyle.TankTop },
                1, "0D0D10", "2B3A4A", "C9CED6", "EDEDED", 0.004f),
            [FighterRoster.Ines] = Make(new BodySpec { Height = 1.70f, Build = 0.42f, Female = true, Hair = HairStyle.Ponytail, Top = TopStyle.TankTop },
                2, "9E9EA3", "33404D", "E6E6E6", "C23333", 0.012f),
            [FighterRoster.Rook] = Make(new BodySpec { Height = 1.90f, Build = 1f, Hair = HairStyle.Shaved, Beard = true, LongPants = true, Shoes = true },
                3, "1A1412", "403830", "9A7333", "262626", 0.006f),
            [FighterRoster.Sable] = Make(new BodySpec { Height = 1.77f, Build = 0.58f, Female = true, Hair = HairStyle.Short, Top = TopStyle.TankTop },
                5, "0A0808", "141417", "9E8CFF", "1A1A1F", 0.015f),
            [FighterRoster.Kofi] = Make(new BodySpec { Height = 1.88f, Build = 0.95f, Hair = HairStyle.Short, Beard = false, Shoes = true },
                5, "0B0B0C", "5A1A14", "E8A33D", "1C1C1C", 0.02f),
            [FighterRoster.Wren] = Make(new BodySpec { Height = 1.66f, Build = 0.35f, Female = true, Hair = HairStyle.Ponytail, Top = TopStyle.TankTop, LongPants = true },
                1, "8C2F1B", "1F2A22", "D9C27A", "3A3A3A", 0.02f),
            [FighterRoster.Lin] = Make(new BodySpec { Height = 1.70f, Build = 0.4f, Female = true, Hair = HairStyle.Bun, Top = TopStyle.TankTop, LongPants = true },
                1, "101012", "6E1D2A", "E6C36A", "EDEDED", 0.006f),
            [FighterRoster.Brandt] = Make(new BodySpec { Height = 1.86f, Build = 0.8f, Hair = HairStyle.Short, Beard = true, LongPants = true, Shoes = true, Top = TopStyle.TankTop },
                0, "A8742E", "3B2E22", "8A8A8A", "4A3A2A", 0.004f),
            [FighterRoster.Crane] = Make(new BodySpec { Height = 1.84f, Build = 0.7f, Hair = HairStyle.Short, LongPants = true, Shoes = true, Top = TopStyle.TankTop },
                2, "C8C8CC", "0E0E10", "B3122E", "0E0E10", 0.006f),
        };

        public static FighterAppearance For(string fighterId) =>
            (Defaults.TryGetValue(fighterId, out var a) ? a : Defaults[FighterRoster.Mara]).Clone();

        /// <summary>True when the fighter's default outfit is fixed (campaign opponents) rather than player cosmetics.</summary>
        public static bool HasFixedOutfit(string fighterId) =>
            fighterId == FighterRoster.Ines || fighterId == FighterRoster.Rook || fighterId == FighterRoster.Sable || fighterId == FighterRoster.Crane;
    }

    /// <summary>Rest-pose skeleton in character space (metres, feet at y=0, facing +Z, character's left is -X).</summary>
    public sealed class HumanSkeleton
    {
        public const int JointCount = 17;

        public static readonly int[] Parent =
        {
            -1, (int)Joint.Hips, (int)Joint.Spine, (int)Joint.Chest, (int)Joint.Neck,
            (int)Joint.Chest, (int)Joint.ShoulderL, (int)Joint.ElbowL,
            (int)Joint.Chest, (int)Joint.ShoulderR, (int)Joint.ElbowR,
            (int)Joint.Hips, (int)Joint.HipL, (int)Joint.KneeL,
            (int)Joint.Hips, (int)Joint.HipR, (int)Joint.KneeR,
        };

        public readonly Vector3[] Rest = new Vector3[JointCount];
        public readonly float Height;
        public readonly float ShoulderX;
        public readonly float HipX;

        public HumanSkeleton(BodySpec spec)
        {
            float h = spec.Height;
            Height = h;
            ShoulderX = (spec.Female ? 0.088f : 0.098f + 0.008f * spec.Build) * h;
            HipX = (spec.Female ? 0.056f : 0.051f) * h;
            float sx = ShoulderX, hx = HipX;
            Set(Joint.Hips, 0, 0.56f, 0);
            Set(Joint.Spine, 0, 0.62f, -0.006f);
            Set(Joint.Chest, 0, 0.72f, -0.008f);
            Set(Joint.Neck, 0, 0.862f, -0.012f);
            Set(Joint.Head, 0, 0.905f, -0.004f);
            for (int side = -1; side <= 1; side += 2)
            {
                bool left = side < 0;
                Set(left ? Joint.ShoulderL : Joint.ShoulderR, side * sx / h, 0.818f, -0.01f);
                Set(left ? Joint.ElbowL : Joint.ElbowR, side * (sx / h + 0.018f), 0.632f, -0.014f);
                Set(left ? Joint.HandL : Joint.HandR, side * (sx / h + 0.024f), 0.486f, -0.004f);
                Set(left ? Joint.HipL : Joint.HipR, side * hx / h, 0.53f, 0);
                Set(left ? Joint.KneeL : Joint.KneeR, side * hx / h, 0.285f, 0.004f);
                Set(left ? Joint.FootL : Joint.FootR, side * hx / h, 0.039f, -0.008f);
            }

            void Set(Joint j, float x, float y, float z) => Rest[(int)j] = new Vector3(x * h, y * h, z * h);
        }

        public Vector3 LocalOffset(int joint) => Parent[joint] < 0 ? Rest[joint] : Rest[joint] - Rest[Parent[joint]];
    }
}
