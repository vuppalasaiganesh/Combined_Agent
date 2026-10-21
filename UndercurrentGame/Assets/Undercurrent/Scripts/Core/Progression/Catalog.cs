using System;
using System.Collections.Generic;

namespace Undercurrent.Core.Progression
{
    public enum CosmeticSlot
    {
        Outfit,
        Wraps,
    }

    [Serializable]
    public sealed class CosmeticDefinition
    {
        public string Id;
        public string DisplayName;
        public CosmeticSlot Slot;
        /// <summary>Price in Marks (earned in-game currency). 0 means reward-only.</summary>
        public int Price;
        public bool RewardOnly;
        /// <summary>Colours as hex RRGGBB, interpreted by the presentation layer.</summary>
        public string PrimaryColor;
        public string SecondaryColor;
    }

    public enum SkillEffect
    {
        StaminaRegenPercent,
        GuardDamageReductionPercent,
        FocusGainPercent,
        GetUpFramesReduction,
    }

    [Serializable]
    public sealed class SkillDefinition
    {
        public string Id;
        public string DisplayName;
        public string Description;
        public int RequiredFighterLevel;
        public SkillEffect Effect;
        public float Value;
    }

    [Serializable]
    public sealed class AchievementDefinition
    {
        public string Id;
        public string DisplayName;
        public string Description;
        public int RewardCurrency;
    }

    /// <summary>Static catalog of cosmetics, skills and achievements.</summary>
    public static class Catalog
    {
        public const int MaxFighterLevel = 10;

        public static readonly IReadOnlyList<CosmeticDefinition> Cosmetics = new List<CosmeticDefinition>
        {
            new CosmeticDefinition { Id = "outfit_default", DisplayName = "Circuit Standard", Slot = CosmeticSlot.Outfit, PrimaryColor = "2B3A4A", SecondaryColor = "C9CED6" },
            new CosmeticDefinition { Id = "outfit_harbor", DisplayName = "Harbor Night", Slot = CosmeticSlot.Outfit, Price = 300, PrimaryColor = "123B5C", SecondaryColor = "E8A33D" },
            new CosmeticDefinition { Id = "outfit_ember", DisplayName = "Ember Line", Slot = CosmeticSlot.Outfit, Price = 450, PrimaryColor = "5A1A14", SecondaryColor = "F06A2A" },
            new CosmeticDefinition { Id = "outfit_moss", DisplayName = "Rooftop Moss", Slot = CosmeticSlot.Outfit, Price = 450, PrimaryColor = "2F4A2A", SecondaryColor = "C8D6A0" },
            new CosmeticDefinition { Id = "outfit_sable_tribute", DisplayName = "Gatekeeper's Tribute", Slot = CosmeticSlot.Outfit, RewardOnly = true, PrimaryColor = "151515", SecondaryColor = "9E8CFF" },
            new CosmeticDefinition { Id = "wraps_white", DisplayName = "White Wraps", Slot = CosmeticSlot.Wraps, PrimaryColor = "EDEDED", SecondaryColor = "EDEDED" },
            new CosmeticDefinition { Id = "wraps_black", DisplayName = "Black Wraps", Slot = CosmeticSlot.Wraps, Price = 150, PrimaryColor = "1C1C1C", SecondaryColor = "1C1C1C" },
            new CosmeticDefinition { Id = "wraps_gold", DisplayName = "Gold Wraps", Slot = CosmeticSlot.Wraps, Price = 600, PrimaryColor = "D4A63A", SecondaryColor = "D4A63A" },
            new CosmeticDefinition { Id = "wraps_crimson", DisplayName = "Crimson Wraps", Slot = CosmeticSlot.Wraps, RewardOnly = true, PrimaryColor = "9C1B22", SecondaryColor = "9C1B22" },
        };

        public static readonly IReadOnlyList<SkillDefinition> Skills = new List<SkillDefinition>
        {
            new SkillDefinition { Id = "second_wind", DisplayName = "Second Wind", Description = "+10% stamina regeneration.", RequiredFighterLevel = 3, Effect = SkillEffect.StaminaRegenPercent, Value = 10f },
            new SkillDefinition { Id = "iron_guard", DisplayName = "Iron Guard", Description = "Blocking drains 15% less stamina.", RequiredFighterLevel = 5, Effect = SkillEffect.GuardDamageReductionPercent, Value = 15f },
            new SkillDefinition { Id = "focus_flow", DisplayName = "Focus Flow", Description = "+15% Focus meter gain.", RequiredFighterLevel = 7, Effect = SkillEffect.FocusGainPercent, Value = 15f },
            new SkillDefinition { Id = "quick_rise", DisplayName = "Quick Rise", Description = "Get up 4 frames faster after a knockdown.", RequiredFighterLevel = 9, Effect = SkillEffect.GetUpFramesReduction, Value = 4f },
        };

        public static readonly IReadOnlyList<AchievementDefinition> Achievements = new List<AchievementDefinition>
        {
            new AchievementDefinition { Id = "first_win", DisplayName = "Into the Current", Description = "Win your first match.", RewardCurrency = 50 },
            new AchievementDefinition { Id = "parry_10", DisplayName = "Read the Tide", Description = "Land 10 parries in total.", RewardCurrency = 100 },
            new AchievementDefinition { Id = "combo_5", DisplayName = "Combination Artist", Description = "Land a 5-hit combo.", RewardCurrency = 100 },
            new AchievementDefinition { Id = "perfect_round", DisplayName = "Untouched", Description = "Win a round without taking damage.", RewardCurrency = 150 },
            new AchievementDefinition { Id = "chapter1", DisplayName = "Past the Gatekeeper", Description = "Complete Chapter 1.", RewardCurrency = 200 },
            new AchievementDefinition { Id = "throws_10", DisplayName = "Freight Handler", Description = "Land 10 throws in total.", RewardCurrency = 100 },
            new AchievementDefinition { Id = "collector_5", DisplayName = "Collector", Description = "Own 5 cosmetic items.", RewardCurrency = 100 },
            new AchievementDefinition { Id = "armory", DisplayName = "Full Armory", Description = "Unlock the knife, sword and staff.", RewardCurrency = 200 },
            new AchievementDefinition { Id = "disarm_5", DisplayName = "Empty Hands", Description = "Disarm opponents 5 times.", RewardCurrency = 150 },
            new AchievementDefinition { Id = "gauntlet", DisplayName = "Still Standing", Description = "Clear a gauntlet level.", RewardCurrency = 150 },
            new AchievementDefinition { Id = "champion", DisplayName = "End of the Undercurrent", Description = "Complete the Final Championship.", RewardCurrency = 500 },
        };

        public static CosmeticDefinition GetCosmetic(string id)
        {
            foreach (var c in Cosmetics) if (c.Id == id) return c;
            return null;
        }

        public static AchievementDefinition GetAchievement(string id)
        {
            foreach (var a in Achievements) if (a.Id == id) return a;
            return null;
        }

        /// <summary>Marks needed to raise a fighter from <paramref name="currentLevel"/> to the next level.</summary>
        public static int UpgradeCost(int currentLevel) => 150 * currentLevel;

        public static int XpToNextLevel(int playerLevel) => 100 + 60 * (playerLevel - 1);
    }
}
