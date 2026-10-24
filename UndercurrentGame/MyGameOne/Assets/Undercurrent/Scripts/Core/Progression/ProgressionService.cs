using System;
using System.Collections.Generic;
using Undercurrent.Core.Combat;
using Undercurrent.Core.Content;

namespace Undercurrent.Core.Progression
{
    public enum MatchResultKind
    {
        Victory,
        Defeat,
        Draw,
    }

    /// <summary>The facts about a finished match that progression needs, extracted from the simulation.</summary>
    public sealed class MatchSummary
    {
        public MatchResultKind Result;
        public int FightSeconds;
        public int RoundsWon;
        public int RoundsLost;
        public int Parries;
        public int Throws;
        public int MaxCombo;
        public int PerfectRounds;
        public int SpecialsLanded;
        public int DamageDealt;
        public int DamageTaken;

        public static MatchSummary FromSimulation(FightSimulation sim, int playerIndex)
        {
            var p = sim.Fighters[playerIndex];
            var o = sim.Fighters[1 - playerIndex];
            MatchResultKind result;
            if (sim.MatchOutcome == RoundOutcome.Draw || sim.MatchOutcome == RoundOutcome.None) result = MatchResultKind.Draw;
            else result = (sim.MatchOutcome == RoundOutcome.Player0) == (playerIndex == 0) ? MatchResultKind.Victory : MatchResultKind.Defeat;
            return new MatchSummary
            {
                Result = result,
                FightSeconds = sim.FightingTicksElapsed / SimClock.TicksPerSecond,
                RoundsWon = p.RoundsWon,
                RoundsLost = o.RoundsWon,
                Parries = p.Stats.Parries,
                Throws = p.Stats.ThrowsLanded,
                MaxCombo = p.Stats.MaxCombo,
                PerfectRounds = p.Stats.PerfectRounds,
                SpecialsLanded = p.Stats.SpecialsLanded,
                DamageDealt = p.Stats.DamageDealt,
                DamageTaken = p.Stats.DamageTaken,
            };
        }
    }

    public sealed class RewardResult
    {
        public int Currency;
        public int Xp;
        public bool FirstClear;
        public int LevelsGained;
        public int NewPlayerLevel;
        public readonly List<string> CompletedObjectives = new List<string>();
        public readonly List<string> UnlockedFighters = new List<string>();
        public readonly List<string> UnlockedCosmetics = new List<string>();
        public readonly List<string> UnlockedLevels = new List<string>();
        public readonly List<string> UnlockedAchievements = new List<string>();
    }

    public enum PurchaseResult
    {
        Success,
        AlreadyOwned,
        InsufficientFunds,
        NotForSale,
        Unknown,
    }

    /// <summary>All rules that change the player's persistent progress. Pure functions over <see cref="SaveData"/>.</summary>
    public static class ProgressionService
    {
        public const float ReplayCurrencyFactor = 0.5f;
        public const float DefeatXpFactor = 0.25f;

        public static readonly string[] StartingFighters = { FighterRoster.Mara, FighterRoster.Dima };

        public static SaveData CreateNew()
        {
            var save = new SaveData { Version = SaveMigrator.CurrentVersion, SelectedFighterId = FighterRoster.Mara };
            foreach (var id in StartingFighters)
            {
                save.UnlockedFighters.Add(id);
                save.GetOrCreateFighter(id);
            }
            save.OwnedCosmetics.Add("outfit_default");
            save.OwnedCosmetics.Add("wraps_white");
            return save;
        }

        public static bool IsFighterUnlocked(SaveData save, string fighterId) => save.UnlockedFighters.Contains(fighterId);

        public static bool IsLevelUnlocked(SaveData save, LevelDefinition level)
        {
            if (level == null) return false;
            if (string.IsNullOrEmpty(level.RequiresLevelId)) return true;
            return save.IsLevelCompleted(level.RequiresLevelId);
        }

        /// <summary>The level "Continue" should start: the first unlocked, uncompleted level, or the last level.</summary>
        public static LevelDefinition NextLevel(SaveData save)
        {
            LevelDefinition last = null;
            foreach (var l in CampaignDatabase.Levels)
            {
                last = l;
                if (IsLevelUnlocked(save, l) && !save.IsLevelCompleted(l.Id)) return l;
            }
            return last;
        }

        public static RewardResult ApplyMatchResult(SaveData save, LevelDefinition level, MatchSummary summary)
        {
            if (save == null) throw new ArgumentNullException(nameof(save));
            if (level == null) throw new ArgumentNullException(nameof(level));
            if (summary == null) throw new ArgumentNullException(nameof(summary));

            var reward = new RewardResult();
            var progress = save.GetOrCreateLevel(level.Id);
            progress.Attempts++;
            save.MatchesPlayed++;
            save.LastPlayedLevelId = level.Id;
            save.TotalParries += summary.Parries;
            save.TotalThrows += summary.Throws;
            save.TotalPerfectRounds += summary.PerfectRounds;
            if (summary.MaxCombo > save.BestCombo) save.BestCombo = summary.MaxCombo;

            if (summary.Result == MatchResultKind.Victory)
            {
                reward.FirstClear = !progress.Completed;
                progress.Wins++;
                save.TotalWins++;
                if (progress.BestTimeSeconds == 0 || summary.FightSeconds < progress.BestTimeSeconds)
                    progress.BestTimeSeconds = summary.FightSeconds;

                reward.Currency = reward.FirstClear
                    ? level.RewardCurrency + level.FirstClearBonusCurrency
                    : (int)(level.RewardCurrency * ReplayCurrencyFactor);
                reward.Xp = level.RewardXp;

                foreach (var obj in level.Objectives)
                {
                    if (progress.CompletedObjectives.Contains(obj.Id)) continue;
                    if (!IsObjectiveMet(obj, summary)) continue;
                    progress.CompletedObjectives.Add(obj.Id);
                    reward.CompletedObjectives.Add(obj.Id);
                    reward.Currency += obj.BonusCurrency;
                }

                if (reward.FirstClear)
                {
                    progress.Completed = true;
                    if (!string.IsNullOrEmpty(level.UnlocksFighterId) && !save.UnlockedFighters.Contains(level.UnlocksFighterId))
                    {
                        save.UnlockedFighters.Add(level.UnlocksFighterId);
                        save.GetOrCreateFighter(level.UnlocksFighterId);
                        reward.UnlockedFighters.Add(level.UnlocksFighterId);
                    }
                    if (!string.IsNullOrEmpty(level.UnlocksCosmeticId) && !save.OwnedCosmetics.Contains(level.UnlocksCosmeticId))
                    {
                        save.OwnedCosmetics.Add(level.UnlocksCosmeticId);
                        reward.UnlockedCosmetics.Add(level.UnlocksCosmeticId);
                    }
                    foreach (var other in CampaignDatabase.Levels)
                        if (other.RequiresLevelId == level.Id) reward.UnlockedLevels.Add(other.Id);
                }
            }
            else
            {
                // Losing still teaches something: a fraction of XP, no currency.
                reward.Xp = (int)(level.RewardXp * DefeatXpFactor);
            }

            save.Currency += reward.Currency;
            reward.LevelsGained = AddXp(save, reward.Xp);
            reward.NewPlayerLevel = save.PlayerLevel;

            foreach (var id in EvaluateAchievements(save))
            {
                reward.UnlockedAchievements.Add(id);
                var a = Catalog.GetAchievement(id);
                if (a != null)
                {
                    save.Currency += a.RewardCurrency;
                    reward.Currency += a.RewardCurrency;
                }
            }
            return reward;
        }

        public const int QuickFightWinCurrency = 25;
        public const int QuickFightWinXp = 30;
        public const int QuickFightLossXp = 10;

        /// <summary>Rewards for a non-campaign (Versus AI) match: small, so the campaign remains the main progression path.</summary>
        public static RewardResult ApplyQuickFightResult(SaveData save, MatchSummary summary)
        {
            if (save == null) throw new ArgumentNullException(nameof(save));
            if (summary == null) throw new ArgumentNullException(nameof(summary));
            var reward = new RewardResult();
            save.MatchesPlayed++;
            save.TotalParries += summary.Parries;
            save.TotalThrows += summary.Throws;
            save.TotalPerfectRounds += summary.PerfectRounds;
            if (summary.MaxCombo > save.BestCombo) save.BestCombo = summary.MaxCombo;
            if (summary.Result == MatchResultKind.Victory)
            {
                save.TotalWins++;
                reward.Currency = QuickFightWinCurrency;
                reward.Xp = QuickFightWinXp;
            }
            else reward.Xp = QuickFightLossXp;

            save.Currency += reward.Currency;
            reward.LevelsGained = AddXp(save, reward.Xp);
            reward.NewPlayerLevel = save.PlayerLevel;
            foreach (var id in EvaluateAchievements(save))
            {
                reward.UnlockedAchievements.Add(id);
                var a = Catalog.GetAchievement(id);
                if (a == null) continue;
                save.Currency += a.RewardCurrency;
                reward.Currency += a.RewardCurrency;
            }
            return reward;
        }

        public static bool IsObjectiveMet(ObjectiveDefinition obj, MatchSummary s)
        {
            if (s.Result != MatchResultKind.Victory) return false;
            switch (obj.Type)
            {
                case ObjectiveType.WinWithoutLosingRound: return s.RoundsLost == 0;
                case ObjectiveType.WinWithinSeconds: return s.FightSeconds <= obj.Value;
                case ObjectiveType.LandParries: return s.Parries >= obj.Value;
                case ObjectiveType.LandCombo: return s.MaxCombo >= obj.Value;
                case ObjectiveType.PerfectRound: return s.PerfectRounds > 0;
                case ObjectiveType.LandThrows: return s.Throws >= obj.Value;
                case ObjectiveType.LandSpecial: return s.SpecialsLanded > 0;
                default: return false;
            }
        }

        /// <summary>Adds XP and returns how many player levels were gained.</summary>
        public static int AddXp(SaveData save, int xp)
        {
            if (xp <= 0) return 0;
            save.PlayerXp += xp;
            int gained = 0;
            while (save.PlayerXp >= Catalog.XpToNextLevel(save.PlayerLevel))
            {
                save.PlayerXp -= Catalog.XpToNextLevel(save.PlayerLevel);
                save.PlayerLevel++;
                gained++;
            }
            return gained;
        }

        /// <summary>Returns ids of achievements newly unlocked by the current save state, and records them.</summary>
        public static List<string> EvaluateAchievements(SaveData save)
        {
            var result = new List<string>();
            void Check(string id, bool condition)
            {
                if (condition && !save.UnlockedAchievements.Contains(id))
                {
                    save.UnlockedAchievements.Add(id);
                    result.Add(id);
                }
            }

            bool chapter1 = true;
            foreach (var l in CampaignDatabase.LevelsInChapter(1)) chapter1 &= save.IsLevelCompleted(l.Id);

            Check("first_win", save.TotalWins >= 1);
            Check("parry_10", save.TotalParries >= 10);
            Check("combo_5", save.BestCombo >= 5);
            Check("perfect_round", save.TotalPerfectRounds >= 1);
            Check("chapter1", chapter1);
            Check("throws_10", save.TotalThrows >= 10);
            Check("collector_5", save.OwnedCosmetics.Count >= 5);
            return result;
        }

        // ------------------------------------------------------------------ upgrades, skills, cosmetics

        public static bool CanUpgradeFighter(SaveData save, string fighterId, out int cost)
        {
            cost = 0;
            if (!IsFighterUnlocked(save, fighterId)) return false;
            var fp = save.GetOrCreateFighter(fighterId);
            if (fp.Level >= Catalog.MaxFighterLevel) return false;
            cost = Catalog.UpgradeCost(fp.Level);
            return save.Currency >= cost;
        }

        public static bool TryUpgradeFighter(SaveData save, string fighterId)
        {
            if (!CanUpgradeFighter(save, fighterId, out int cost)) return false;
            save.Currency -= cost;
            save.GetOrCreateFighter(fighterId).Level++;
            return true;
        }

        public static List<SkillDefinition> UnlockedSkills(int fighterLevel)
        {
            var list = new List<SkillDefinition>();
            foreach (var s in Catalog.Skills) if (fighterLevel >= s.RequiredFighterLevel) list.Add(s);
            return list;
        }

        public static PurchaseResult TryPurchaseCosmetic(SaveData save, string cosmeticId)
        {
            var c = Catalog.GetCosmetic(cosmeticId);
            if (c == null) return PurchaseResult.Unknown;
            if (save.OwnedCosmetics.Contains(cosmeticId)) return PurchaseResult.AlreadyOwned;
            if (c.RewardOnly || c.Price <= 0) return PurchaseResult.NotForSale;
            if (save.Currency < c.Price) return PurchaseResult.InsufficientFunds;
            save.Currency -= c.Price;
            save.OwnedCosmetics.Add(cosmeticId);
            EvaluateAchievements(save);
            return PurchaseResult.Success;
        }

        public static bool TryEquipCosmetic(SaveData save, string fighterId, string cosmeticId)
        {
            var c = Catalog.GetCosmetic(cosmeticId);
            if (c == null || !save.OwnedCosmetics.Contains(cosmeticId) || !IsFighterUnlocked(save, fighterId)) return false;
            var fp = save.GetOrCreateFighter(fighterId);
            if (c.Slot == CosmeticSlot.Outfit) fp.EquippedOutfit = cosmeticId;
            else fp.EquippedWraps = cosmeticId;
            return true;
        }

        /// <summary>
        /// Builds the combat definition for the player's fighter with level and skill bonuses applied.
        /// Bonuses are deliberately modest so progression helps without making skill irrelevant.
        /// </summary>
        public static FighterDefinition BuildLoadout(SaveData save, string fighterId)
        {
            var def = FighterRoster.Get(fighterId);
            if (def == null) return null;
            var fp = save?.GetFighter(fighterId);
            int level = fp?.Level ?? 1;
            int steps = Math.Max(0, Math.Min(Catalog.MaxFighterLevel, level) - 1);

            def.MaxHealth = (int)Math.Round(def.MaxHealth * (1f + 0.02f * steps));
            def.MaxStamina *= 1f + 0.015f * steps;

            foreach (var skill in UnlockedSkills(level))
            {
                switch (skill.Effect)
                {
                    case SkillEffect.StaminaRegenPercent: def.StaminaRegen *= 1f + skill.Value / 100f; break;
                    case SkillEffect.GuardDamageReductionPercent: def.GuardDamageTakenMultiplier *= 1f - skill.Value / 100f; break;
                    case SkillEffect.FocusGainPercent:
                        def.FocusGainOnHit *= 1f + skill.Value / 100f;
                        def.FocusGainOnDamaged *= 1f + skill.Value / 100f;
                        break;
                    case SkillEffect.GetUpFramesReduction: def.GetUpFrames = Math.Max(8, def.GetUpFrames - (int)skill.Value); break;
                }
            }
            return def;
        }
    }
}
