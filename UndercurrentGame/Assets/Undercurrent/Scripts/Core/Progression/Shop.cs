using System;
using System.Collections.Generic;
using Undercurrent.Core.AI;
using Undercurrent.Core.Content;

namespace Undercurrent.Core.Progression
{
    /// <summary>One armour tier. Armour is the player's own kit, shared by every fighter they play.</summary>
    [Serializable]
    public sealed class ArmorDefinition
    {
        public int Tier;
        public string DisplayName;
        public string Description;
        public int Price;
        /// <summary>Multiplier on damage taken (0.8 = 20% less damage).</summary>
        public float DamageTaken;
    }

    /// <summary>A real-money coin pack. Prices come from the store at runtime; nothing here is a price.</summary>
    [Serializable]
    public sealed class CoinPack
    {
        public string ProductId;
        public string DisplayName;
        public int Coins;
    }

    /// <summary>
    /// Prices and stat curves for everything bought with Coins. Every item can be earned by playing:
    /// <c>EconomyTests</c> simulates a full campaign without coin packs to keep that true when numbers change.
    /// </summary>
    public static class Shop
    {
        public const string CurrencyName = "Coins";
        public const int MaxArmorTier = 5;
        public const int MaxWeaponLevel = PowerRating.MaxWeaponLevel;

        public static readonly IReadOnlyList<ArmorDefinition> Armor = new List<ArmorDefinition>
        {
            new ArmorDefinition { Tier = 1, DisplayName = "Padded Vest", Price = 1000, DamageTaken = 0.95f, Description = "Quilted canvas under your shirt. Takes the sting out of body shots." },
            new ArmorDefinition { Tier = 2, DisplayName = "Leather Guards", Price = 2400, DamageTaken = 0.90f, Description = "Forearm and shin guards of boiled leather." },
            new ArmorDefinition { Tier = 3, DisplayName = "Reinforced Gambeson", Price = 4200, DamageTaken = 0.85f, Description = "Layered jacket with stitched ribs; turns glancing blades." },
            new ArmorDefinition { Tier = 4, DisplayName = "Riveted Leather", Price = 6500, DamageTaken = 0.80f, Description = "Steel rivets over the ribs, shoulders and kidneys." },
            new ArmorDefinition { Tier = 5, DisplayName = "Forged Plate Inserts", Price = 9000, DamageTaken = 0.75f, Description = "Thin plates from Brandt's forge, sewn into every guard." },
        };

        public static ArmorDefinition GetArmor(int tier)
        {
            foreach (var a in Armor) if (a.Tier == tier) return a;
            return null;
        }

        public static float ArmorDamageTaken(int tier) => tier <= 0 ? 1f : GetArmor(Math.Min(MaxArmorTier, tier)).DamageTaken;

        public static int WeaponPrice(string weaponId)
        {
            switch (weaponId)
            {
                case WeaponCatalog.Knife: return 1200;
                case WeaponCatalog.Staff: return 1800;
                case WeaponCatalog.Sword: return 2500;
                default: return 0;
            }
        }

        /// <summary>Coins to raise a weapon from <paramref name="currentLevel"/> to the next level.</summary>
        public static int WeaponUpgradeCost(string weaponId, int currentLevel) =>
            (int)Math.Round(WeaponPrice(weaponId) * currentLevel / 10f) * 10;

        /// <summary>Fighter level (1..10) stat curves: +2% health and +2% damage per level.</summary>
        public static float FighterHealthScale(int level) => 1f + 0.02f * (Math.Max(1, Math.Min(Catalog.MaxFighterLevel, level)) - 1);
        public static float FighterDamageScale(int level) => 1f + 0.02f * (Math.Max(1, Math.Min(Catalog.MaxFighterLevel, level)) - 1);

        /// <summary>Versus AI win reward per round won: harder opponents pay more, so Versus is a real way to earn.</summary>
        public static int QuickFightCoins(AIDifficulty difficulty)
        {
            switch (difficulty)
            {
                case AIDifficulty.Beginner: return 10;
                case AIDifficulty.Easy: return 20;
                case AIDifficulty.Normal: return 35;
                case AIDifficulty.Hard: return 55;
                case AIDifficulty.Expert: return 80;
                default: return 110;
            }
        }

        public static readonly IReadOnlyList<CoinPack> CoinPacks = new List<CoinPack>
        {
            new CoinPack { ProductId = "coins_pouch", DisplayName = "Pouch of Coins", Coins = 2500 },
            new CoinPack { ProductId = "coins_purse", DisplayName = "Fighter's Purse", Coins = 7000 },
            new CoinPack { ProductId = "coins_chest", DisplayName = "Champion's Chest", Coins = 16000 },
        };

        public static CoinPack GetCoinPack(string productId)
        {
            foreach (var p in CoinPacks) if (p.ProductId == productId) return p;
            return null;
        }
    }
}
