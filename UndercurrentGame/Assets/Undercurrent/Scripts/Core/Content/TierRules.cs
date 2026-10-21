using System;
using Undercurrent.Core.AI;

namespace Undercurrent.Core.Content
{
    /// <summary>How demanding a campaign level is. Tiers are mixed inside every chapter so the curve breathes.</summary>
    public enum LevelTier
    {
        Easy,
        Medium,
        Hard,
        Extreme,
    }

    /// <summary>
    /// Tier rules in one place: opponent stat scaling, AI skill, reward multiplier. Opponents get stronger in two
    /// independent ways — <b>skill</b> (AI difficulty, beaten by the player's own skill) and <b>stats</b>
    /// (health/damage, matched by the player's power from upgrades, armour and weapons).
    /// </summary>
    public static class TierRules
    {
        /// <summary>Opponent stat growth per campaign level (applies to health and damage alike).</summary>
        public const float ScalePerLevel = 0.007f;

        public static float TierStatFactor(LevelTier tier)
        {
            switch (tier)
            {
                case LevelTier.Easy: return 0.92f;
                case LevelTier.Medium: return 1.0f;
                case LevelTier.Hard: return 1.06f;
                default: return 1.2f;
            }
        }

        /// <summary>Multiplies every coin reward of a level (win, first clear, objectives, replays).</summary>
        public static float RewardMultiplier(LevelTier tier)
        {
            switch (tier)
            {
                case LevelTier.Easy: return 1.0f;
                case LevelTier.Medium: return 1.3f;
                case LevelTier.Hard: return 1.7f;
                default: return 2.3f;
            }
        }

        /// <summary>Opponent health and damage multiplier for a level at <paramref name="campaignIndex"/> (0-based).</summary>
        public static float OpponentScale(LevelTier tier, int campaignIndex) =>
            (1f + ScalePerLevel * Math.Max(0, campaignIndex)) * TierStatFactor(tier);

        /// <summary>
        /// AI skill for a non-boss level: a chapter baseline (Easy → Normal → Hard) shifted by the tier.
        /// Bosses always use <see cref="AIDifficulty.Boss"/>.
        /// </summary>
        public static AIDifficulty AIFor(LevelTier tier, int chapter, bool boss)
        {
            if (boss) return AIDifficulty.Boss;
            int baseline = chapter <= 1 ? (int)AIDifficulty.Easy : chapter <= 3 ? (int)AIDifficulty.Normal : (int)AIDifficulty.Hard;
            int offset = tier == LevelTier.Easy ? -1 : tier == LevelTier.Medium ? 0 : tier == LevelTier.Hard ? 1 : 2;
            return (AIDifficulty)Math.Max((int)AIDifficulty.Beginner, Math.Min((int)AIDifficulty.Expert, baseline + offset));
        }

        public static string Name(LevelTier tier) => tier.ToString().ToUpperInvariant();

        /// <summary>UI colour (hex RRGGBB) for a tier tag.</summary>
        public static string ColorHex(LevelTier tier)
        {
            switch (tier)
            {
                case LevelTier.Easy: return "5AC76B";
                case LevelTier.Medium: return "33B3B8";
                case LevelTier.Hard: return "E8A33D";
                default: return "DC4038";
            }
        }
    }

    /// <summary>
    /// A single number for "how strong is this loadout": 100 = a fresh, unarmed fighter. It is the product of the
    /// health, damage and damage-reduction multipliers (time-to-KO scales with each), times a weapon factor.
    /// Both the player's gear and a level's opponent are rated with the same formula, so "your power vs
    /// recommended power" compares like with like. It deliberately ignores AI skill and fighter style.
    /// </summary>
    public static class PowerRating
    {
        /// <summary>Armed vs unarmed at equal stats wins ≈75–80% in AI-vs-AI tests (TEST_RESULTS.md §2).</summary>
        public const float ArmedFactor = 1.25f;
        public const int MaxWeaponLevel = 5;

        /// <summary>Damage multiplier for weapon strikes at an upgrade level (1..5): +6% per level.</summary>
        public static float WeaponLevelScale(int level) => 1f + 0.06f * (Math.Max(1, Math.Min(MaxWeaponLevel, level)) - 1);

        public static int Of(float health, float damage, float damageTaken, string weaponId, int weaponLevel)
        {
            float p = 100f * health * damage / Math.Max(0.05f, damageTaken);
            if (!string.IsNullOrEmpty(weaponId)) p *= ArmedFactor * WeaponLevelScale(weaponLevel);
            return (int)Math.Round(p);
        }
    }
}
