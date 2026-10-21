using System;
using System.Collections.Generic;
using NUnit.Framework;
using Undercurrent.Core.Combat;
using Undercurrent.Core.Content;
using Undercurrent.Core.Progression;

namespace Undercurrent.Tests
{
    /// <summary>JSON codec used by tests: Unity's JsonUtility inside the Editor, System.Text.Json in the standalone runner.</summary>
    public sealed class TestJsonCodec : ISaveCodec
    {
#if UNITY_5_3_OR_NEWER
        public string Encode(SaveData data) => UnityEngine.JsonUtility.ToJson(data);
        public SaveData Decode(string text) => UnityEngine.JsonUtility.FromJson<SaveData>(text);
#else
        private static readonly System.Text.Json.JsonSerializerOptions Options = new System.Text.Json.JsonSerializerOptions { IncludeFields = true };
        public string Encode(SaveData data) => System.Text.Json.JsonSerializer.Serialize(data, Options);
        public SaveData Decode(string text) => System.Text.Json.JsonSerializer.Deserialize<SaveData>(text, Options);
#endif
    }

    public sealed class MemoryStore : ISaveStore
    {
        public string Text;
        public readonly List<string> Backups = new List<string>();
        public int Writes;

        public bool TryRead(out string text) { text = Text; return Text != null; }
        public void Write(string text) { Text = text; Writes++; }
        public void WriteBackup(string text, string reason) => Backups.Add(reason + ":" + text);
    }

    public class ProgressionTests
    {
        private static MatchSummary Win(int seconds = 90, int roundsLost = 0) => new MatchSummary
        {
            Result = MatchResultKind.Victory, FightSeconds = seconds, RoundsWon = 2, RoundsLost = roundsLost,
        };

        [Test]
        public void NewSave_HasStartingRosterAndFirstLevelOnly()
        {
            var s = ProgressionService.CreateNew();
            CollectionAssert.AreEquivalent(new[] { FighterRoster.Mara, FighterRoster.Dima }, s.UnlockedFighters);
            Assert.IsFalse(ProgressionService.IsFighterUnlocked(s, FighterRoster.Haruto));
            Assert.IsTrue(ProgressionService.IsLevelUnlocked(s, CampaignDatabase.GetLevel("c1_l1")));
            Assert.IsFalse(ProgressionService.IsLevelUnlocked(s, CampaignDatabase.GetLevel("c1_l2")));
            Assert.AreEqual("c1_l1", ProgressionService.NextLevel(s).Id);
            Assert.AreEqual(SaveMigrator.CurrentVersion, s.Version);
        }

        [Test]
        public void Victory_FirstClear_GrantsRewards_AndUnlocksNextLevel()
        {
            var s = ProgressionService.CreateNew();
            var l1 = CampaignDatabase.GetLevel("c1_l1");
            var r = ProgressionService.ApplyMatchResult(s, l1, Win());
            Assert.IsTrue(r.FirstClear);
            Assert.GreaterOrEqual(r.Currency, l1.RewardCurrency + l1.FirstClearBonusCurrency);
            Assert.AreEqual(s.Currency, r.Currency);
            Assert.Contains("c1_l2", r.UnlockedLevels);
            Assert.IsTrue(ProgressionService.IsLevelUnlocked(s, CampaignDatabase.GetLevel("c1_l2")));
            Assert.Contains("first_win", r.UnlockedAchievements);
            Assert.AreEqual("c1_l2", ProgressionService.NextLevel(s).Id);
        }

        [Test]
        public void Replay_GivesReducedCurrency_NoDuplicateUnlocks()
        {
            var s = ProgressionService.CreateNew();
            var l1 = CampaignDatabase.GetLevel("c1_l1");
            ProgressionService.ApplyMatchResult(s, l1, Win());
            var r = ProgressionService.ApplyMatchResult(s, l1, Win());
            Assert.IsFalse(r.FirstClear);
            Assert.AreEqual((int)(l1.RewardCurrency * ProgressionService.ReplayCurrencyFactor), r.Currency);
            Assert.IsEmpty(r.UnlockedLevels);
            Assert.AreEqual(2, s.GetLevel("c1_l1").Wins);
        }

        [Test]
        public void Defeat_GivesPartialXp_NoCurrency_NoCompletion()
        {
            var s = ProgressionService.CreateNew();
            var l1 = CampaignDatabase.GetLevel("c1_l1");
            var r = ProgressionService.ApplyMatchResult(s, l1, new MatchSummary { Result = MatchResultKind.Defeat, RoundsLost = 2 });
            Assert.AreEqual(0, r.Currency);
            Assert.AreEqual((int)(l1.RewardXp * ProgressionService.DefeatXpFactor), r.Xp);
            Assert.IsFalse(s.IsLevelCompleted("c1_l1"));
            Assert.AreEqual(1, s.GetLevel("c1_l1").Attempts);
        }

        [Test]
        public void Objectives_AreAwardedOnce()
        {
            var s = ProgressionService.CreateNew();
            var l1 = CampaignDatabase.GetLevel("c1_l1");
            var summary = Win();
            summary.Parries = 2;
            summary.MaxCombo = 3;
            var r1 = ProgressionService.ApplyMatchResult(s, l1, summary);
            Assert.AreEqual(2, r1.CompletedObjectives.Count);
            var r2 = ProgressionService.ApplyMatchResult(s, l1, summary);
            Assert.AreEqual(0, r2.CompletedObjectives.Count);
        }

        [Test]
        public void ObjectiveRules_AreEvaluatedCorrectly()
        {
            var w = Win(100, 1);
            w.Throws = 2; w.SpecialsLanded = 1; w.PerfectRounds = 1; w.Parries = 3; w.MaxCombo = 4;
            Assert.IsFalse(ProgressionService.IsObjectiveMet(new ObjectiveDefinition { Type = ObjectiveType.WinWithoutLosingRound }, w));
            Assert.IsTrue(ProgressionService.IsObjectiveMet(new ObjectiveDefinition { Type = ObjectiveType.WinWithinSeconds, Value = 100 }, w));
            Assert.IsFalse(ProgressionService.IsObjectiveMet(new ObjectiveDefinition { Type = ObjectiveType.WinWithinSeconds, Value = 99 }, w));
            Assert.IsTrue(ProgressionService.IsObjectiveMet(new ObjectiveDefinition { Type = ObjectiveType.LandThrows, Value = 2 }, w));
            Assert.IsTrue(ProgressionService.IsObjectiveMet(new ObjectiveDefinition { Type = ObjectiveType.LandSpecial }, w));
            Assert.IsTrue(ProgressionService.IsObjectiveMet(new ObjectiveDefinition { Type = ObjectiveType.PerfectRound }, w));
            Assert.IsFalse(ProgressionService.IsObjectiveMet(new ObjectiveDefinition { Type = ObjectiveType.LandCombo, Value = 5 }, w));
            w.Result = MatchResultKind.Defeat;
            Assert.IsFalse(ProgressionService.IsObjectiveMet(new ObjectiveDefinition { Type = ObjectiveType.LandThrows, Value = 2 }, w), "objectives require victory");
        }

        [Test]
        public void BossVictory_UnlocksHaruto_AndCompletesChapter()
        {
            var s = ProgressionService.CreateNew();
            foreach (var l in CampaignDatabase.LevelsInChapter(1)) ProgressionService.ApplyMatchResult(s, l, Win());
            Assert.IsTrue(ProgressionService.IsFighterUnlocked(s, FighterRoster.Haruto));
            Assert.Contains("outfit_sable_tribute", s.OwnedCosmetics);
            Assert.Contains("chapter1", s.UnlockedAchievements);
            Assert.IsNotNull(s.GetFighter(FighterRoster.Haruto));
        }

        [Test]
        public void Xp_LevelsUpPlayer()
        {
            var s = ProgressionService.CreateNew();
            int gained = ProgressionService.AddXp(s, Catalog.XpToNextLevel(1) + Catalog.XpToNextLevel(2) + 5);
            Assert.AreEqual(2, gained);
            Assert.AreEqual(3, s.PlayerLevel);
            Assert.AreEqual(5, s.PlayerXp);
        }

        [Test]
        public void FighterUpgrade_CostsCurrency_AndIsCapped()
        {
            var s = ProgressionService.CreateNew();
            Assert.IsFalse(ProgressionService.TryUpgradeFighter(s, FighterRoster.Mara), "no money yet");
            s.Currency = 100000;
            int before = s.Currency;
            Assert.IsTrue(ProgressionService.TryUpgradeFighter(s, FighterRoster.Mara));
            Assert.AreEqual(before - Catalog.UpgradeCost(1), s.Currency);
            Assert.AreEqual(2, s.GetFighter(FighterRoster.Mara).Level);
            while (ProgressionService.TryUpgradeFighter(s, FighterRoster.Mara)) { }
            Assert.AreEqual(Catalog.MaxFighterLevel, s.GetFighter(FighterRoster.Mara).Level);
            Assert.IsFalse(ProgressionService.TryUpgradeFighter(s, FighterRoster.Haruto), "locked fighters cannot be upgraded");
        }

        [Test]
        public void Loadout_AppliesLevelAndSkillBonuses_Modestly()
        {
            var s = ProgressionService.CreateNew();
            var baseDef = FighterRoster.Get(FighterRoster.Mara);
            var lvl1 = ProgressionService.BuildLoadout(s, FighterRoster.Mara);
            Assert.AreEqual(baseDef.MaxHealth, lvl1.MaxHealth);
            s.GetFighter(FighterRoster.Mara).Level = 10;
            var lvl10 = ProgressionService.BuildLoadout(s, FighterRoster.Mara);
            Assert.Greater(lvl10.MaxHealth, baseDef.MaxHealth);
            Assert.LessOrEqual(lvl10.MaxHealth, baseDef.MaxHealth * 1.2f, "upgrades must not trivialise fights");
            Assert.Greater(lvl10.StaminaRegen, baseDef.StaminaRegen);
            Assert.Less(lvl10.GuardDamageTakenMultiplier, 1f);
            Assert.Less(lvl10.GetUpFrames, baseDef.GetUpFrames);
            Assert.AreEqual(baseDef.MaxHealth, FighterRoster.Get(FighterRoster.Mara).MaxHealth, "roster data must not be mutated");
        }

        [Test]
        public void Cosmetics_PurchaseAndEquip()
        {
            var s = ProgressionService.CreateNew();
            Assert.AreEqual(PurchaseResult.InsufficientFunds, ProgressionService.TryPurchaseCosmetic(s, "outfit_harbor"));
            s.Currency = 1000;
            Assert.AreEqual(PurchaseResult.Success, ProgressionService.TryPurchaseCosmetic(s, "outfit_harbor"));
            Assert.AreEqual(700, s.Currency);
            Assert.AreEqual(PurchaseResult.AlreadyOwned, ProgressionService.TryPurchaseCosmetic(s, "outfit_harbor"));
            Assert.AreEqual(PurchaseResult.NotForSale, ProgressionService.TryPurchaseCosmetic(s, "outfit_sable_tribute"));
            Assert.AreEqual(PurchaseResult.Unknown, ProgressionService.TryPurchaseCosmetic(s, "nope"));
            Assert.IsTrue(ProgressionService.TryEquipCosmetic(s, FighterRoster.Mara, "outfit_harbor"));
            Assert.AreEqual("outfit_harbor", s.GetFighter(FighterRoster.Mara).EquippedOutfit);
            Assert.IsFalse(ProgressionService.TryEquipCosmetic(s, FighterRoster.Mara, "wraps_gold"), "cannot equip unowned item");
        }

        [Test]
        public void QuickFight_GivesSmallRewards_AndCountsStats()
        {
            var s = ProgressionService.CreateNew();
            var win = ProgressionService.ApplyQuickFightResult(s, new MatchSummary { Result = MatchResultKind.Victory, Parries = 3, MaxCombo = 5 });
            Assert.GreaterOrEqual(win.Currency, ProgressionService.QuickFightWinCurrency);
            Assert.Contains("first_win", win.UnlockedAchievements);
            Assert.Contains("combo_5", win.UnlockedAchievements);
            Assert.AreEqual(3, s.TotalParries);
            var loss = ProgressionService.ApplyQuickFightResult(s, new MatchSummary { Result = MatchResultKind.Defeat });
            Assert.AreEqual(0, loss.Currency);
            Assert.AreEqual(ProgressionService.QuickFightLossXp, loss.Xp);
            Assert.IsEmpty(s.Levels, "quick fights never complete campaign levels");
            Assert.AreEqual(2, s.MatchesPlayed);
        }

        [Test]
        public void MatchSummary_IsBuiltFromSimulation()
        {
            var h = TestHarness.Fighting(distance: 0.8f, config: new MatchConfig { IntroTicks = 2, RoundOverTicks = 5, RoundsToWin = 1, MaxRounds = 1 });
            h.P1.Health = 5;
            h.Step(InputButtons.LightPunch);
            h.RunUntil(() => h.Sim.IsMatchOver, 100);
            var summary = MatchSummary.FromSimulation(h.Sim, 0);
            Assert.AreEqual(MatchResultKind.Victory, summary.Result);
            Assert.AreEqual(1, summary.RoundsWon);
            Assert.AreEqual(1, summary.PerfectRounds);
            Assert.AreEqual(MatchResultKind.Defeat, MatchSummary.FromSimulation(h.Sim, 1).Result);
        }
    }

    public class SaveTests
    {
        [Test]
        public void SaveAndLoad_RoundTrips()
        {
            var store = new MemoryStore();
            var repo = new SaveRepository(store, new TestJsonCodec());
            repo.Load();
            Assert.AreEqual(SaveLoadStatus.NewSave, repo.Status);
            var l1 = CampaignDatabase.GetLevel("c1_l1");
            ProgressionService.ApplyMatchResult(repo.Data, l1, new MatchSummary { Result = MatchResultKind.Victory, FightSeconds = 50, Parries = 1, MaxCombo = 3 });
            repo.Data.Settings.MusicVolume = 0.3f;
            repo.Data.Settings.SetOffset("btn_lp", 12f, -4f);
            ProgressionService.TryEquipCosmetic(repo.Data, FighterRoster.Dima, "wraps_white");
            Assert.IsTrue(repo.Save());

            var repo2 = new SaveRepository(store, new TestJsonCodec());
            var loaded = repo2.Load();
            Assert.AreEqual(SaveLoadStatus.Loaded, repo2.Status);
            Assert.AreEqual(repo.Data.Currency, loaded.Currency);
            Assert.IsTrue(loaded.IsLevelCompleted("c1_l1"));
            CollectionAssert.AreEquivalent(repo.Data.GetLevel("c1_l1").CompletedObjectives, loaded.GetLevel("c1_l1").CompletedObjectives);
            Assert.AreEqual(0.3f, loaded.Settings.MusicVolume, 1e-5);
            Assert.AreEqual(12f, loaded.Settings.GetOffset("btn_lp").X, 1e-5);
            CollectionAssert.AreEquivalent(repo.Data.UnlockedAchievements, loaded.UnlockedAchievements);
        }

        [Test]
        public void CorruptSave_IsBackedUp_AndReplaced()
        {
            var store = new MemoryStore { Text = "{ this is not json" };
            var repo = new SaveRepository(store, new TestJsonCodec());
            repo.Load();
            Assert.AreEqual(SaveLoadStatus.RecoveredFromCorruption, repo.Status);
            Assert.AreEqual(1, store.Backups.Count);
            StringAssert.StartsWith("corrupt:", store.Backups[0]);
            Assert.IsNotNull(repo.Data);
            Assert.Contains(FighterRoster.Mara, repo.Data.UnlockedFighters);
        }

        [Test]
        public void NewerVersionSave_IsNeverOverwritten()
        {
            var codec = new TestJsonCodec();
            var future = ProgressionService.CreateNew();
            future.Version = SaveMigrator.CurrentVersion + 5;
            future.Currency = 1234;
            var store = new MemoryStore { Text = codec.Encode(future) };
            string original = store.Text;
            var repo = new SaveRepository(store, codec);
            repo.Load();
            Assert.AreEqual(SaveLoadStatus.NewerVersionReadOnly, repo.Status);
            Assert.IsFalse(repo.Save());
            Assert.AreEqual(original, store.Text);
            Assert.AreEqual(1234, repo.Data.Currency);
        }

        [Test]
        public void UnversionedSave_IsMigrated_WithBackup()
        {
            var codec = new TestJsonCodec();
            var old = ProgressionService.CreateNew();
            old.Version = 0;
            old.Currency = 77;
            old.UnlockedFighters.Clear();
            old.UnlockedFighters.Add("deleted_fighter");
            old.SelectedFighterId = "deleted_fighter";
            old.Settings.MusicVolume = 7f;
            old.Settings.TargetFps = 144;
            var store = new MemoryStore { Text = codec.Encode(old) };
            var repo = new SaveRepository(store, codec);
            var s = repo.Load();
            Assert.AreEqual(SaveLoadStatus.Migrated, repo.Status);
            Assert.AreEqual(SaveMigrator.CurrentVersion, s.Version);
            Assert.AreEqual(77, s.Currency);
            Assert.IsFalse(s.UnlockedFighters.Contains("deleted_fighter"));
            Assert.Contains(FighterRoster.Mara, s.UnlockedFighters);
            Assert.AreEqual(FighterRoster.Mara, s.SelectedFighterId);
            Assert.AreEqual(1f, s.Settings.MusicVolume);
            Assert.AreEqual(60, s.Settings.TargetFps);
            Assert.AreEqual(1, store.Backups.Count);
            Assert.Greater(store.Writes, 0, "migrated save is persisted");
        }

        [Test]
        public void ResetProgress_KeepsSettings()
        {
            var store = new MemoryStore();
            var repo = new SaveRepository(store, new TestJsonCodec());
            repo.Load();
            repo.Data.Currency = 999;
            repo.Data.Settings.SfxVolume = 0.1f;
            repo.ResetProgress();
            Assert.AreEqual(0, repo.Data.Currency);
            Assert.AreEqual(0.1f, repo.Data.Settings.SfxVolume, 1e-5);
        }

        [Test]
        public void OnlineServices_ReportNotConfigured_AndNeverFakeSuccess()
        {
            var billing = new UnconfiguredBillingService();
            Assert.AreEqual(IntegrationStatus.NotConfigured, billing.Status);
            BillingOutcome? outcome = null;
            billing.Purchase("anything", o => outcome = o);
            Assert.AreEqual(BillingOutcome.Unavailable, outcome);
            bool? uploaded = null;
            new UnconfiguredCloudSaveService().Upload("x", ok => uploaded = ok);
            Assert.IsFalse(uploaded.Value);
            Assert.IsFalse(new OfflineAccountService().IsSignedIn);
        }
    }

    public class TutorialTests
    {
        /// <summary>A scripted "player" that performs each tutorial goal, proving every step is completable.</summary>
        [Test]
        public void EveryTutorialStep_IsCompletableInTheLiveSimulation()
        {
            var sim = new FightSimulation(FighterRoster.Get(FighterRoster.Mara), FighterRoster.Get(FighterRoster.Ines), TutorialTracker.CreateConfig());
            var tracker = new TutorialTracker(0);
            int stepTicks = 0;
            int maxTicksPerStep = 1200;
            var completed = new List<TutorialGoal>();
            int lastDummySerial = -1;
            int waitParry = 0;

            for (int t = 0; t < 20000 && !tracker.IsComplete; t++)
            {
                var p = sim.Fighters[0];
                var dummy = sim.Fighters[1];
                var input = InputButtons.None;
                var goal = tracker.Current.Goal;
                float dist = dummy.X - p.X;
                bool even = (t % 2) == 0;

                switch (goal)
                {
                    case TutorialGoal.WalkForward: input = dist > 1.2f ? InputButtons.Right : InputButtons.None; if (dist <= 1.2f) p.X -= 1f; break;
                    case TutorialGoal.WalkBack: input = InputButtons.Left; break;
                    case TutorialGoal.Crouch: input = InputButtons.Down; break;
                    case TutorialGoal.Jump: input = even ? InputButtons.Up : InputButtons.None; break;
                    case TutorialGoal.LightPunch: input = even ? InputButtons.LightPunch : InputButtons.None; break;
                    case TutorialGoal.HeavyPunch: input = even ? InputButtons.HeavyPunch : InputButtons.None; break;
                    case TutorialGoal.LightKick: input = even ? InputButtons.LightKick : InputButtons.None; break;
                    case TutorialGoal.HeavyKick: input = even ? InputButtons.HeavyKick : InputButtons.None; break;
                    case TutorialGoal.Combo:
                        if (dist > 0.85f) input = InputButtons.Right;
                        else if (p.IsActionable) input = even ? InputButtons.LightPunch : InputButtons.None;
                        else if (p.State == FighterStateId.Attack && p.MoveConnected) input = even ? InputButtons.HeavyPunch : InputButtons.None;
                        break;
                    case TutorialGoal.Block: input = InputButtons.Block; break;
                    case TutorialGoal.Dodge:
                    case TutorialGoal.Parry:
                        // React to the visible wind-up of the partner's kick.
                        if (dummy.State == FighterStateId.Attack && dummy.AttackSerial != lastDummySerial && dummy.MoveFrame >= 2)
                        {
                            lastDummySerial = dummy.AttackSerial;
                            waitParry = 1;
                        }
                        if (waitParry > 0)
                        {
                            input = goal == TutorialGoal.Dodge ? InputButtons.Dodge : InputButtons.Parry;
                            waitParry = 0;
                        }
                        break;
                    case TutorialGoal.Throw:
                        if (dist > 0.8f) input = InputButtons.Right;
                        else input = even ? InputButtons.Throw : InputButtons.None;
                        break;
                    case TutorialGoal.BreakThrow:
                        input = p.State == FighterStateId.Grabbed && even ? InputButtons.Throw : InputButtons.None;
                        break;
                    case TutorialGoal.Special: input = even ? InputButtons.Special : InputButtons.None; break;
                }

                sim.Step(new FrameInput(input), tracker.DummyInput(sim));
                var before = tracker.Current?.Goal;
                if (tracker.Observe(sim))
                {
                    completed.Add(before.Value);
                    stepTicks = 0;
                }
                else if (sim.Phase == MatchPhase.Fighting && ++stepTicks > maxTicksPerStep)
                {
                    Assert.Fail($"Tutorial step {tracker.Current.Goal} not completed within {maxTicksPerStep} ticks");
                }
            }

            Assert.IsTrue(tracker.IsComplete);
            Assert.AreEqual(TutorialTracker.Steps.Count, completed.Count);
            Assert.IsFalse(sim.IsMatchOver, "training must never end the match by KO");
        }
    }
}
