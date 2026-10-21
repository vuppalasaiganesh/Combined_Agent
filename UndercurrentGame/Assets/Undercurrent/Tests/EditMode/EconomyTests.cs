using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using NUnit.Framework;
using Undercurrent.Core.AI;
using Undercurrent.Core.Combat;
using Undercurrent.Core.Content;
using Undercurrent.Core.Progression;

namespace Undercurrent.Tests
{
    public class TierTests
    {
        [Test]
        public void Tiers_AreMixed_InEveryChapter()
        {
            foreach (var ch in CampaignDatabase.Chapters)
            {
                var tiers = CampaignDatabase.LevelsInChapter(ch.Number).Select(l => l.Tier).Distinct().Count();
                Assert.GreaterOrEqual(tiers, 2, "chapter " + ch.Number);
            }
            foreach (LevelTier tier in Enum.GetValues(typeof(LevelTier)))
                Assert.GreaterOrEqual(CampaignDatabase.Levels.Count(l => l.Tier == tier), 4, tier.ToString());
            Assert.AreEqual(LevelTier.Easy, CampaignDatabase.Levels[0].Tier, "the first fight is easy");
            Assert.AreEqual(LevelTier.Extreme, CampaignDatabase.Levels.Last().Tier, "the final boss is extreme");
        }

        [Test]
        public void HarderTiers_PayMore_AndScaleOpponentsMore()
        {
            Assert.Less(TierRules.RewardMultiplier(LevelTier.Easy), TierRules.RewardMultiplier(LevelTier.Medium));
            Assert.Less(TierRules.RewardMultiplier(LevelTier.Hard), TierRules.RewardMultiplier(LevelTier.Extreme));
            Assert.Less(TierRules.OpponentScale(LevelTier.Easy, 10), TierRules.OpponentScale(LevelTier.Medium, 10));
            Assert.Less(TierRules.OpponentScale(LevelTier.Hard, 10), TierRules.OpponentScale(LevelTier.Extreme, 10));
            Assert.Less(TierRules.OpponentScale(LevelTier.Medium, 0), TierRules.OpponentScale(LevelTier.Medium, 20), "opponents grow across the campaign");

            // Neighbouring levels: the extreme one pays more than the easy one before it.
            var levels = CampaignDatabase.Levels;
            for (int i = 1; i < levels.Count; i++)
                if (levels[i].Tier == LevelTier.Extreme && levels[i - 1].Tier == LevelTier.Easy)
                    Assert.Greater(levels[i].RewardCurrency, levels[i - 1].RewardCurrency);
            var survival = levels.First(l => l.Tier == LevelTier.Extreme && !l.IsBoss);
            var easyBefore = levels.Last(l => l.Tier == LevelTier.Easy && levels.ToList().IndexOf(l) < levels.ToList().IndexOf(survival));
            Assert.Greater(survival.RewardCurrency, easyBefore.RewardCurrency);
            Assert.Greater(survival.RecommendedPower, easyBefore.RecommendedPower);
        }

        [Test]
        public void AISkill_FollowsTier_AndBossesUseBossAI()
        {
            foreach (var l in CampaignDatabase.Levels)
            {
                Assert.AreEqual(TierRules.AIFor(l.Tier, l.Chapter, l.IsBoss), l.Difficulty, l.Id);
                if (l.IsBoss) Assert.AreEqual(AIDifficulty.Boss, l.Difficulty, l.Id);
                if (l.Tier == LevelTier.Easy) Assert.LessOrEqual((int)l.Difficulty, (int)AIDifficulty.Normal, l.Id);
            }
            Assert.AreEqual(AIDifficulty.Beginner, CampaignDatabase.Levels[0].Difficulty, "the tutorial level stays gentle");
        }

        [Test]
        public void BuildOpponent_AppliesStatScale_AndWeapon()
        {
            var level = CampaignDatabase.Levels.Last();
            var bout = level.Bouts()[0];
            var opp = level.BuildOpponent(bout);
            var basis = FighterRoster.Get(bout.OpponentId);
            Assert.Greater(level.OpponentScale, 1.2f);
            Assert.AreEqual((int)Math.Round(basis.MaxHealth * level.OpponentScale), opp.MaxHealth);
            Assert.AreEqual(basis.DamageDealtMultiplier * level.OpponentScale, opp.DamageDealtMultiplier, 1e-4);
            Assert.AreEqual(bout.WeaponId, opp.Weapon.Id);
            Assert.AreEqual(opp.MaxHealth, opp.Unarmed.MaxHealth, "a disarmed opponent keeps its scaled stats");
        }

        [Test]
        public void RecommendedPower_UsesTheSameFormulaAsPlayerPower()
        {
            var s = ProgressionService.CreateNew();
            Assert.AreEqual(100, ProgressionService.PlayerPower(s, FighterRoster.Mara, false), "a fresh unarmed fighter is 100");
            var first = CampaignDatabase.Levels[0];
            Assert.AreEqual((int)Math.Round(100 * first.OpponentScale * first.OpponentScale), first.RecommendedPower);
            Assert.Less(first.RecommendedPower, 100, "the first level is below a fresh fighter");
            var armed = CampaignDatabase.Levels.First(l => l.Mode == LevelMode.Weapons && l.OpponentWeaponId != null && l.PlayerDamage == 1f && l.OpponentDamage == 1f);
            Assert.AreEqual((int)Math.Round(100 * armed.OpponentScale * armed.OpponentScale * PowerRating.ArmedFactor), armed.RecommendedPower);
        }
    }

    public class ShopTests
    {
        private static SaveData Rich()
        {
            var s = ProgressionService.CreateNew();
            s.Currency = 100000;
            return s;
        }

        private static void ClearLevelUnlocking(SaveData s, string weapon) =>
            s.GetOrCreateLevel(CampaignDatabase.Levels.First(l => l.UnlocksWeaponId == weapon).Id).Completed = true;

        [Test]
        public void Weapons_AreSold_OnlyAfterTheirLevel_AndCostCoins()
        {
            var s = Rich();
            foreach (var w in WeaponCatalog.All)
            {
                Assert.IsFalse(ProgressionService.IsWeaponInShop(s, w), w);
                Assert.AreEqual(PurchaseResult.Locked, ProgressionService.TryBuyWeapon(s, w), w);
            }
            ClearLevelUnlocking(s, WeaponCatalog.Sword);
            int before = s.Currency;
            Assert.AreEqual(PurchaseResult.Success, ProgressionService.TryBuyWeapon(s, WeaponCatalog.Sword));
            Assert.AreEqual(before - Shop.WeaponPrice(WeaponCatalog.Sword), s.Currency);
            Assert.AreEqual(1, ProgressionService.WeaponLevel(s, WeaponCatalog.Sword));
            Assert.AreEqual(PurchaseResult.AlreadyOwned, ProgressionService.TryBuyWeapon(s, WeaponCatalog.Sword));

            var poor = ProgressionService.CreateNew();
            ClearLevelUnlocking(poor, WeaponCatalog.Knife);
            poor.Currency = Shop.WeaponPrice(WeaponCatalog.Knife) - 1;
            Assert.AreEqual(PurchaseResult.InsufficientFunds, ProgressionService.TryBuyWeapon(poor, WeaponCatalog.Knife));
            Assert.AreEqual(Shop.WeaponPrice(WeaponCatalog.Knife) - 1, poor.Currency, "a failed purchase costs nothing");
            Assert.IsFalse(ProgressionService.OwnsWeapon(poor, WeaponCatalog.Knife));
        }

        [Test]
        public void Prices_RiseWithWeaponReach()
        {
            Assert.Less(Shop.WeaponPrice(WeaponCatalog.Knife), Shop.WeaponPrice(WeaponCatalog.Staff));
            Assert.Less(Shop.WeaponPrice(WeaponCatalog.Staff), Shop.WeaponPrice(WeaponCatalog.Sword));
            for (int i = 1; i < Shop.Armor.Count; i++)
            {
                Assert.Greater(Shop.Armor[i].Price, Shop.Armor[i - 1].Price);
                Assert.Less(Shop.Armor[i].DamageTaken, Shop.Armor[i - 1].DamageTaken);
            }
        }

        [Test]
        public void WeaponUpgrades_RaiseStrikeDamage_UpToMax()
        {
            var s = Rich();
            ClearLevelUnlocking(s, WeaponCatalog.Knife);
            ProgressionService.TryBuyWeapon(s, WeaponCatalog.Knife);
            ProgressionService.TryEquipWeapon(s, FighterRoster.Mara, WeaponCatalog.Knife);
            int Jab() => ProgressionService.BuildLoadout(s, FighterRoster.Mara, true).Moves.First(m => m.Id == "knife_jab").Damage;
            int baseDamage = Jab();
            int power1 = ProgressionService.PlayerPower(s, FighterRoster.Mara, true);
            for (int lvl = 2; lvl <= Shop.MaxWeaponLevel; lvl++)
            {
                int cost = ProgressionService.WeaponUpgradeCost(s, WeaponCatalog.Knife);
                int before = s.Currency;
                Assert.AreEqual(PurchaseResult.Success, ProgressionService.TryUpgradeWeapon(s, WeaponCatalog.Knife));
                Assert.AreEqual(before - cost, s.Currency);
                Assert.AreEqual(lvl, ProgressionService.WeaponLevel(s, WeaponCatalog.Knife));
            }
            Assert.AreEqual(PurchaseResult.Maxed, ProgressionService.TryUpgradeWeapon(s, WeaponCatalog.Knife));
            Assert.AreEqual((int)Math.Round(baseDamage * PowerRating.WeaponLevelScale(Shop.MaxWeaponLevel)), Jab(), "+6% per level above 1");
            Assert.Greater(ProgressionService.PlayerPower(s, FighterRoster.Mara, true), power1);
            Assert.AreEqual(ProgressionService.PlayerPower(s, FighterRoster.Mara, false), 100, "weapon upgrades don't count in unarmed levels");
        }

        [Test]
        public void Armour_ReducesDamageTaken_InTheSimulation()
        {
            var s = Rich();
            for (int i = 0; i < Shop.MaxArmorTier; i++) Assert.AreEqual(PurchaseResult.Success, ProgressionService.TryUpgradeArmor(s));
            Assert.AreEqual(PurchaseResult.Maxed, ProgressionService.TryUpgradeArmor(s));
            Assert.Contains("armor_max", s.UnlockedAchievements);
            var armoured = ProgressionService.BuildLoadout(s, FighterRoster.Mara);
            Assert.AreEqual(0.75f, armoured.DamageTakenMultiplier, 1e-4);
            Assert.AreEqual(133, ProgressionService.PlayerPower(s, FighterRoster.Mara, false));

            int DamageFromJab(FighterDefinition victim)
            {
                var sim = new FightSimulation(FighterRoster.Get(FighterRoster.Mara), victim, new MatchConfig { IntroTicks = 2 });
                var h = new TestHarness(sim);
                h.UntilFighting();
                h.SetDistance(0.8f);
                h.Step(InputButtons.LightPunch);
                h.RunUntil(() => h.P1.Health < victim.MaxHealth, 30);
                return victim.MaxHealth - h.P1.Health;
            }
            int plain = DamageFromJab(FighterRoster.Get(FighterRoster.Mara));
            int reduced = DamageFromJab(armoured);
            Assert.Greater(plain, 0);
            Assert.AreEqual((int)Math.Round(plain * 0.75f), reduced, 1);
        }

        [Test]
        public void FighterLevels_AddHealthAndDamage()
        {
            var s = Rich();
            var before = ProgressionService.BuildLoadout(s, FighterRoster.Mara);
            for (int i = 1; i < Catalog.MaxFighterLevel; i++) Assert.IsTrue(ProgressionService.TryUpgradeFighter(s, FighterRoster.Mara));
            var after = ProgressionService.BuildLoadout(s, FighterRoster.Mara);
            Assert.Greater(after.MaxHealth, before.MaxHealth);
            Assert.AreEqual(Shop.FighterDamageScale(10), after.DamageDealtMultiplier, 1e-4);
            Assert.AreEqual((int)Math.Round(100 * Shop.FighterHealthScale(10) * Shop.FighterDamageScale(10)), ProgressionService.PlayerPower(s, FighterRoster.Mara, false));
        }

        [Test]
        public void CoinPacks_CreditCoins_OnlyForKnownProducts()
        {
            var s = ProgressionService.CreateNew();
            var pack = Shop.CoinPacks[0];
            Assert.AreEqual(pack.Coins, ProgressionService.GrantCoinPack(s, pack.ProductId));
            Assert.AreEqual(pack.Coins, s.Currency);
            Assert.AreEqual(pack.Coins, s.PurchasedCoins);
            Assert.AreEqual(0, ProgressionService.GrantCoinPack(s, "not_a_product"));
            Assert.AreEqual(pack.Coins, s.Currency);

            BillingOutcome outcome = BillingOutcome.Failed;
            new UnconfiguredBillingService().Purchase(pack.ProductId, o => outcome = o);
            Assert.AreEqual(BillingOutcome.Unavailable, outcome, "no fake success without a configured store");
            var mock = new DevelopmentMockBillingService();
            Assert.AreEqual(IntegrationStatus.DevelopmentMock, mock.Status);
            Assert.AreEqual(Shop.CoinPacks.Count, mock.Products.Count);
            mock.Purchase(pack.ProductId, o => outcome = o);
            Assert.AreEqual(BillingOutcome.Purchased, outcome);
        }

        [Test]
        public void Versus_PaysMore_AgainstHarderAI()
        {
            var s = ProgressionService.CreateNew();
            var easy = ProgressionService.ApplyQuickFightResult(s, new MatchSummary { Result = MatchResultKind.Victory }, AIDifficulty.Easy);
            var expert = ProgressionService.ApplyQuickFightResult(s, new MatchSummary { Result = MatchResultKind.Victory }, AIDifficulty.Expert);
            Assert.Greater(expert.Currency - expert.UnlockedAchievements.Sum(a => Catalog.GetAchievement(a).RewardCurrency),
                easy.Currency - easy.UnlockedAchievements.Sum(a => Catalog.GetAchievement(a).RewardCurrency));
            var bestOf3 = ProgressionService.ApplyQuickFightResult(s, new MatchSummary { Result = MatchResultKind.Victory, RoundsWon = 2 }, AIDifficulty.Easy);
            Assert.AreEqual(2 * Shop.QuickFightCoins(AIDifficulty.Easy), bestOf3.Currency - bestOf3.UnlockedAchievements.Sum(a => Catalog.GetAchievement(a).RewardCurrency),
                "paid per round won, so one-round fights are not a shortcut");
        }

        [Test]
        public void V2Save_KeepsWonWeapons_AsLevelOne()
        {
            var codec = new TestJsonCodec();
            var old = ProgressionService.CreateNew();
            old.Version = 2;
            old.UnlockedWeapons.Add(WeaponCatalog.Knife);
            old.WeaponLevels = null;
            old.ArmorTier = 99;
            var repo = new SaveRepository(new MemoryStore { Text = codec.Encode(old) }, codec);
            var s = repo.Load();
            Assert.AreEqual(SaveLoadStatus.Migrated, repo.Status);
            Assert.AreEqual(3, s.Version);
            Assert.IsTrue(ProgressionService.OwnsWeapon(s, WeaponCatalog.Knife));
            Assert.AreEqual(1, ProgressionService.WeaponLevel(s, WeaponCatalog.Knife));
            Assert.AreEqual(Shop.MaxArmorTier, s.ArmorTier, "out-of-range armour is clamped");
        }
    }

    public class EconomyTests
    {
        public sealed class LevelLog
        {
            public LevelDefinition Level;
            public int PowerOnArrival, PowerBefore, Recommended, Replays, CoinsAfter;
        }

        /// <summary>
        /// A free player (no coin packs) who plays the campaign in order with one fighter, wins every level
        /// at the first attempt, completes one objective per level, and spends greedily on whatever raises
        /// power most per coin. When under the recommended power they replay their best-paying cleared level.
        /// </summary>
        public static List<LevelLog> SimulateFreePlayer()
        {
            var s = ProgressionService.CreateNew();
            var me = FighterRoster.Mara;
            var log = new List<LevelLog>();
            foreach (var level in CampaignDatabase.Levels)
            {
                Spend(s, me, level);
                var entry = new LevelLog { Level = level, Recommended = level.RecommendedPower, PowerOnArrival = ProgressionService.PlayerPower(s, me, level.PlayerWeaponsAllowed) };
                while (ProgressionService.PlayerPower(s, me, level.PlayerWeaponsAllowed) < level.RecommendedPower && entry.Replays < 100)
                {
                    var best = CampaignDatabase.Levels.Where(l => s.IsLevelCompleted(l.Id)).OrderByDescending(l => l.RewardCurrency).FirstOrDefault();
                    if (best == null) break;
                    ProgressionService.ApplyMatchResult(s, best, new MatchSummary { Result = MatchResultKind.Victory, FightSeconds = 90 });
                    entry.Replays++;
                    Spend(s, me, level);
                }
                entry.PowerBefore = ProgressionService.PlayerPower(s, me, level.PlayerWeaponsAllowed);
                var obj = level.Objectives[0];
                var summary = new MatchSummary { Result = MatchResultKind.Victory, FightSeconds = 90, RoundsWon = level.RoundsToWin };
                switch (obj.Type)
                {
                    case ObjectiveType.LandParries: summary.Parries = obj.Value; break;
                    case ObjectiveType.LandCombo: summary.MaxCombo = obj.Value; break;
                    case ObjectiveType.LandThrows: summary.Throws = obj.Value; break;
                    case ObjectiveType.LandSpecial: summary.SpecialsLanded = 1; break;
                    case ObjectiveType.PerfectRound: summary.PerfectRounds = 1; break;
                    case ObjectiveType.LandClashes: summary.Clashes = obj.Value; break;
                    case ObjectiveType.DisarmOpponent: summary.Disarms = 1; break;
                    case ObjectiveType.WinWithinSeconds: summary.FightSeconds = obj.Value; break;
                }
                ProgressionService.ApplyMatchResult(s, level, summary);
                entry.CoinsAfter = s.Currency;
                log.Add(entry);
            }
            return log;
        }

        private static void Spend(SaveData s, string me, LevelDefinition level)
        {
            bool armedLevel = level.PlayerWeaponsAllowed;
            while (true)
            {
                int now = ProgressionService.PlayerPower(s, me, armedLevel);
                var options = new List<(float value, Action buy)>();
                void Consider(int cost, Func<SaveData, bool> apply)
                {
                    if (cost <= 0 || cost > s.Currency) return;
                    var trial = Clone(s);
                    if (!apply(trial)) return;
                    int gain = ProgressionService.PlayerPower(trial, me, armedLevel) - now;
                    if (gain > 0) options.Add((gain / (float)cost, () => apply(s)));
                }
                Consider(ProgressionService.ArmorUpgradeCost(s), t => ProgressionService.TryUpgradeArmor(t) == PurchaseResult.Success);
                int lvl = s.GetFighter(me).Level;
                if (lvl < Catalog.MaxFighterLevel) Consider(Catalog.UpgradeCost(lvl), t => ProgressionService.TryUpgradeFighter(t, me));
                foreach (var w in WeaponCatalog.All)
                {
                    var id = w;
                    if (!ProgressionService.OwnsWeapon(s, id) && ProgressionService.IsWeaponInShop(s, id))
                        Consider(Shop.WeaponPrice(id), t => ProgressionService.TryBuyWeapon(t, id) == PurchaseResult.Success && ProgressionService.TryEquipWeapon(t, me, id));
                    else if (ProgressionService.OwnsWeapon(s, id) && s.GetFighter(me).EquippedWeapon == id)
                        Consider(ProgressionService.WeaponUpgradeCost(s, id), t => ProgressionService.TryUpgradeWeapon(t, id) == PurchaseResult.Success);
                }
                if (options.Count == 0) return;
                options.OrderByDescending(o => o.value).First().buy();
            }
        }

        private static SaveData Clone(SaveData s)
        {
            var codec = new TestJsonCodec();
            return codec.Decode(codec.Encode(s));
        }

        [Test]
        public void FreePlayer_CanReachRecommendedPower_ByPlaying()
        {
            var log = SimulateFreePlayer();
            foreach (var e in log)
            {
                Assert.GreaterOrEqual(e.PowerBefore, e.Recommended, $"{e.Level.Id} unreachable without paying");
                if (e.Level.Tier == LevelTier.Easy || e.Level.Tier == LevelTier.Medium)
                    Assert.AreEqual(0, e.Replays, $"{e.Level.Id} ({e.Level.Tier}) should not need grinding");
                Assert.LessOrEqual(e.Replays, 12, $"{e.Level.Id} needs too much grinding");
            }
            // Extreme levels sit at the edge of what even this ideal player has: real players (who lose fights and miss
            // objectives) will need replays, Versus coins or a coin pack. Easy levels leave a clear margin.
            foreach (var e in log.Where(e => e.Level.Tier == LevelTier.Extreme))
                Assert.GreaterOrEqual(e.Recommended, e.PowerOnArrival * 0.85f, $"{e.Level.Id} is too comfortable for an Extreme level");
            foreach (var e in log.Where(e => e.Level.Tier == LevelTier.Easy))
                Assert.LessOrEqual(e.Recommended, e.PowerOnArrival * 0.9f, $"{e.Level.Id} is too tight for an Easy level");
            Assert.LessOrEqual(log.Sum(e => e.Replays), 60, "total grind for a free player");
        }

        [Test, Explicit("Diagnostic report, run on demand")]
        public void PrintEconomyReport()
        {
            var sb = new StringBuilder("level   tier     rec  arrive  power  replays  coins-after\n");
            foreach (var e in SimulateFreePlayer())
                sb.AppendLine($"{e.Level.Id,-7} {e.Level.Tier,-8} {e.Recommended,4} {e.PowerOnArrival,7} {e.PowerBefore,6} {e.Replays,8} {e.CoinsAfter,12}");
            TestContext.WriteLine(sb.ToString());
        }

        [Test]
        public void HigherPower_WinsMoreOften_InTheSimulation()
        {
            // Same body, same AI skill: only stats differ. Opponent at Extreme late-campaign scale.
            float scale = TierRules.OpponentScale(LevelTier.Extreme, 25);
            var opponent = FighterRoster.Get(FighterRoster.Mara);
            opponent.MaxHealth = (int)Math.Round(opponent.MaxHealth * scale);
            opponent.DamageDealtMultiplier *= scale;

            var geared = ProgressionService.CreateNew();
            geared.Currency = 100000;
            for (int i = 0; i < Shop.MaxArmorTier; i++) ProgressionService.TryUpgradeArmor(geared);
            for (int i = 1; i < Catalog.MaxFighterLevel; i++) ProgressionService.TryUpgradeFighter(geared, FighterRoster.Mara);

            int Wins(FighterDefinition player)
            {
                int wins = 0;
                for (uint seed = 1; seed <= 16; seed++)
                {
                    var sim = new FightSimulation(player, opponent.Clone(), new MatchConfig());
                    var a = new FightAI(0, AIDifficulty.Hard, seed);
                    var b = new FightAI(1, AIDifficulty.Hard, seed * 13 + 5);
                    for (int t = 0; t < 60 * 60 * 10 && !sim.IsMatchOver; t++) sim.Step(a.Think(sim), b.Think(sim));
                    if (sim.MatchOutcome == RoundOutcome.Player0) wins++;
                }
                return wins;
            }
            int fresh = Wins(ProgressionService.BuildLoadout(ProgressionService.CreateNew(), FighterRoster.Mara));
            int strong = Wins(ProgressionService.BuildLoadout(geared, FighterRoster.Mara));
            TestContext.WriteLine($"fresh {fresh}/16, geared {strong}/16 (opponent scale {scale:0.00})");
            Assert.Less(fresh, 8, "an under-powered player should usually lose");
            Assert.Greater(strong, fresh + 3, "gear must matter");
        }
    }
}
