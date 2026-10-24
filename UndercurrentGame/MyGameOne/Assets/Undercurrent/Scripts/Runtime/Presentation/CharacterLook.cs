using System.Collections.Generic;
using Undercurrent.Core.Content;
using Undercurrent.Core.Progression;
using UnityEngine;

namespace Undercurrent.Runtime.Presentation
{
    public enum HairStyle
    {
        Shaved,
        Short,
        Ponytail,
        Bun,
        Braids,
    }

    /// <summary>Visual description of a fighter used by the placeholder mannequin (and as tint data for real models).</summary>
    public sealed class CharacterLook
    {
        public float Height = 1.78f;
        /// <summary>0 = lean, 1 = heavy/muscular.</summary>
        public float Build = 0.5f;
        public bool FemaleProportions;
        public Color Skin;
        public Color Hair;
        public HairStyle HairStyle;
        public bool Beard;
        public Color Outfit;
        public Color OutfitTrim;
        public Color Wraps;
        public bool LongPants;
        public bool Shirtless;
        /// <summary>Idle bounce amplitude; boxers bounce, karateka are still.</summary>
        public float Bounce = 0.01f;

        public static readonly Color[] SkinTones =
        {
            new Color(0.96f, 0.80f, 0.69f),
            new Color(0.87f, 0.68f, 0.53f),
            new Color(0.76f, 0.57f, 0.42f),
            new Color(0.58f, 0.40f, 0.27f),
            new Color(0.42f, 0.28f, 0.19f),
            new Color(0.29f, 0.19f, 0.13f),
        };

        public static readonly string[] SkinToneNames = { "Fair", "Light", "Medium", "Tan", "Brown", "Deep" };

        private static readonly Dictionary<string, CharacterLook> Defaults = new Dictionary<string, CharacterLook>
        {
            [FighterRoster.Mara] = new CharacterLook { Height = 1.74f, Build = 0.6f, FemaleProportions = true, Skin = SkinTones[4], Hair = new Color(0.08f, 0.06f, 0.05f), HairStyle = HairStyle.Braids, Bounce = 0.012f },
            [FighterRoster.Dima] = new CharacterLook { Height = 1.80f, Build = 0.45f, Skin = SkinTones[0], Hair = new Color(0.55f, 0.42f, 0.28f), HairStyle = HairStyle.Short, Beard = true, Shirtless = true, Bounce = 0.03f },
            [FighterRoster.Haruto] = new CharacterLook { Height = 1.76f, Build = 0.35f, Skin = SkinTones[1], Hair = new Color(0.05f, 0.05f, 0.06f), HairStyle = HairStyle.Bun, LongPants = true, Bounce = 0.004f },
            [FighterRoster.Ines] = new CharacterLook { Height = 1.70f, Build = 0.4f, FemaleProportions = true, Skin = SkinTones[2], Hair = new Color(0.62f, 0.62f, 0.64f), HairStyle = HairStyle.Ponytail, Bounce = 0.012f, Outfit = new Color(0.2f, 0.25f, 0.3f), OutfitTrim = new Color(0.9f, 0.9f, 0.9f), Wraps = new Color(0.8f, 0.2f, 0.2f) },
            [FighterRoster.Rook] = new CharacterLook { Height = 1.90f, Build = 1f, Skin = SkinTones[3], Hair = new Color(0.1f, 0.08f, 0.07f), HairStyle = HairStyle.Shaved, Beard = true, LongPants = true, Bounce = 0.006f, Outfit = new Color(0.25f, 0.22f, 0.18f), OutfitTrim = new Color(0.6f, 0.45f, 0.2f), Wraps = new Color(0.15f, 0.15f, 0.15f) },
            [FighterRoster.Sable] = new CharacterLook { Height = 1.77f, Build = 0.55f, FemaleProportions = true, Skin = SkinTones[5], Hair = new Color(0.04f, 0.03f, 0.03f), HairStyle = HairStyle.Short, Bounce = 0.015f, Outfit = new Color(0.08f, 0.08f, 0.09f), OutfitTrim = new Color(0.62f, 0.55f, 1f), Wraps = new Color(0.1f, 0.1f, 0.12f) },
        };

        /// <summary>Builds the look for a fighter, applying the player's equipped cosmetics and skin tone if provided.</summary>
        public static CharacterLook For(string fighterId, FighterProgress progress = null)
        {
            var src = Defaults.TryGetValue(fighterId, out var d) ? d : new CharacterLook { Skin = SkinTones[2], Hair = Color.black };
            var look = (CharacterLook)src.MemberwiseClone();

            bool hasOutfitDefault = src.Outfit.a > 0f;
            var outfit = Catalog.GetCosmetic(progress?.EquippedOutfit ?? "outfit_default");
            var wraps = Catalog.GetCosmetic(progress?.EquippedWraps ?? "wraps_white");
            if (progress != null || !hasOutfitDefault)
            {
                look.Outfit = VisualAssets.Hex(outfit?.PrimaryColor ?? "2B3A4A");
                look.OutfitTrim = VisualAssets.Hex(outfit?.SecondaryColor ?? "C9CED6");
                look.Wraps = VisualAssets.Hex(wraps?.PrimaryColor ?? "EDEDED");
            }
            if (progress != null && progress.SkinToneIndex >= 0 && progress.SkinToneIndex < SkinTones.Length)
                look.Skin = SkinTones[progress.SkinToneIndex];
            return look;
        }
    }
}
