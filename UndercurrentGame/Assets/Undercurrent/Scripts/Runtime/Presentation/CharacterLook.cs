using Undercurrent.CharacterGen;
using Undercurrent.Core.Progression;
using Undercurrent.Runtime.Services;
using UnityEngine;

namespace Undercurrent.Runtime.Presentation
{
    /// <summary>
    /// Resolved look of a fighter for rendering: body shape from <see cref="FighterAppearance"/> plus the player's
    /// cosmetics (outfit, wraps, skin tone), as Unity colours.
    /// </summary>
    public sealed class CharacterLook
    {
        public FighterAppearance Appearance;
        public Color Skin, Hair, Outfit, OutfitTrim, Wraps, Shoes, Lips, Iris;
        public float Bounce;
        /// <summary>Mesh tessellation for the current graphics tier.</summary>
        public float Detail = 1f;

        public static string[] SkinToneNames => FighterAppearance.SkinToneNames;

        private static Color ToColor(Rgb c) => new Color(c.R, c.G, c.B);

        /// <summary>Builds the look for a fighter, applying the player's equipped cosmetics and skin tone if provided.</summary>
        public static CharacterLook For(string fighterId, FighterProgress progress = null)
        {
            var a = FighterAppearance.For(fighterId);
            var look = new CharacterLook
            {
                Appearance = a,
                Skin = ToColor(a.Skin),
                Hair = ToColor(a.Hair),
                Outfit = ToColor(a.Outfit),
                OutfitTrim = ToColor(a.OutfitTrim),
                Wraps = ToColor(a.Wraps),
                Shoes = ToColor(a.Shoes),
                Lips = ToColor(a.Lips),
                Iris = ToColor(a.Iris),
                Bounce = a.Bounce,
                Detail = QualityService.CharacterDetail,
            };

            if (progress != null || !FighterAppearance.HasFixedOutfit(fighterId))
            {
                var outfit = Catalog.GetCosmetic(progress?.EquippedOutfit ?? "outfit_default");
                var wraps = Catalog.GetCosmetic(progress?.EquippedWraps ?? "wraps_white");
                look.Outfit = VisualAssets.Hex(outfit?.PrimaryColor ?? "2B3A4A");
                look.OutfitTrim = VisualAssets.Hex(outfit?.SecondaryColor ?? "C9CED6");
                look.Wraps = VisualAssets.Hex(wraps?.PrimaryColor ?? "EDEDED");
            }
            if (progress != null && progress.SkinToneIndex >= 0 && progress.SkinToneIndex < FighterAppearance.SkinTones.Length)
            {
                var tone = FighterAppearance.SkinTones[progress.SkinToneIndex];
                look.Skin = ToColor(tone);
                look.Lips = new Color(Mathf.Min(1f, tone.R * 0.78f + 0.06f), tone.G * 0.72f, tone.B * 0.72f);
            }
            return look;
        }
    }
}
