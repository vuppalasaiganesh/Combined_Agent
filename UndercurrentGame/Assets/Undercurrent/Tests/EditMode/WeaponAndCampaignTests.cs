using System;
using System.Collections.Generic;
using System.Linq;
using NUnit.Framework;
using Undercurrent.Core.AI;
using Undercurrent.Core.Combat;
using Undercurrent.Core.Content;
using Undercurrent.Core.Progression;

namespace Undercurrent.Tests
{
    public class WeaponTests
    {
        private static FighterDefinition Armed(string fighter, string weapon) => WeaponLoadout.Apply(FighterRoster.Get(fighter), WeaponCatalog.Get(weapon));

        private static TestHarness Duel(FighterDefinition a, FighterDefinition b, float distance)
        {
            var sim = new FightSimulation(a, b, new MatchConfig { IntroTicks = 2, RoundOverTicks = 10 });
            var h = new TestHarness(sim);
            h.UntilFighting();
            h.SetDistance(distance);
            return h;
        }

        [Test]
        public void Loadout_ReplacesPunches_KeepsKicksAndThrow()
        {
            foreach (var w in WeaponCatalog.All)
            {
                var armed = Armed(FighterRoster.Mara, w);
                Assert.AreEqual(w, armed.Weapon.Id);
                Assert.IsNotNull(armed.Unarmed);
                Assert.IsTrue(armed.Resolve(AttackButton.LightPunch, MoveDirection.Neutral).IsWeaponStrike, w);
                Assert.IsTrue(armed.Resolve(AttackButton.HeavyPunch, MoveDirection.Neutral).IsWeaponStrike, w);
                Assert.IsTrue(armed.Resolve(AttackButton.Special, MoveDirection.Neutral).IsWeaponStrike, w);
                Assert.IsFalse(armed.Resolve(AttackButton.HeavyKick, MoveDirection.Neutral).IsWeaponStrike, w + " kicks stay unarmed");
                Assert.IsNotNull(armed.Resolve(AttackButton.Throw, MoveDirection.Neutral));
                var ids = new HashSet<string>(armed.Moves.Select(m => m.Id));
                Assert.AreEqual(armed.Moves.Count, ids.Count, w + " duplicate move ids");
                foreach (var m in armed.Moves) foreach (var c in m.CancelInto) Assert.IsTrue(ids.Contains(c), $"{w}/{m.Id} -> {c}");
                foreach (var combo in armed.Combos)
                    for (int i = 1; i < combo.MoveIds.Length; i++)
                        Assert.Contains(combo.MoveIds[i], armed.FindMove(combo.MoveIds[i - 1]).CancelInto, $"{w} combo {combo.Name}");
                Assert.IsNull(FighterRoster.Get(FighterRoster.Mara).Weapon, "roster data untouched");
            }
        }

        [Test]
        public void WeaponTiers_TradeReachForSpeed()
        {
            float Reach(string w) { var m = WeaponCatalog.Get(w).Moves.First(x => x.Button == AttackButton.LightPunch && x.Direction == MoveDirection.Neutral); return m.HitFrontOffset + m.HitLength; }
            int Startup(string w) => WeaponCatalog.Get(w).Moves.First(x => x.Button == AttackButton.LightPunch && x.Direction == MoveDirection.Neutral).Startup;
            Assert.Less(Reach(WeaponCatalog.Knife), Reach(WeaponCatalog.Sword));
            Assert.Less(Reach(WeaponCatalog.Sword), Reach(WeaponCatalog.Staff));
            Assert.Less(Startup(WeaponCatalog.Knife), Startup(WeaponCatalog.Sword));
            Assert.Less(Startup(WeaponCatalog.Sword), Startup(WeaponCatalog.Staff));
            var jab = FighterRoster.Get(FighterRoster.Mara).Resolve(AttackButton.LightPunch, MoveDirection.Neutral);
            Assert.Greater(Reach(WeaponCatalog.Knife), jab.HitFrontOffset + jab.HitLength, "even a knife out-reaches a jab");
        }

        [Test]
        public void StaffPoke_HitsFromRangeWhereJabWhiffs()
        {
            var unarmed = Duel(FighterRoster.Get(FighterRoster.Mara), FighterRoster.Get(FighterRoster.Dima), 1.6f);
            unarmed.Step(InputButtons.LightPunch);
            unarmed.Hold(20);
            Assert.AreEqual(0, unarmed.Count(CombatEventType.Hit));

            var staff = Duel(Armed(FighterRoster.Mara, WeaponCatalog.Staff), FighterRoster.Get(FighterRoster.Dima), 1.6f);
            staff.Step(InputButtons.LightPunch);
            staff.Hold(20);
            Assert.AreEqual(1, staff.Count(CombatEventType.Hit, 0, 1));
        }

        [Test]
        public void SimultaneousWeaponStrikes_Clash()
        {
            var h = Duel(Armed(FighterRoster.Mara, WeaponCatalog.Sword), Armed(FighterRoster.Dima, WeaponCatalog.Sword), 1.6f);
            int hp0 = h.P0.Health, hp1 = h.P1.Health;
            h.Step(InputButtons.LightPunch, InputButtons.LightPunch);
            h.Hold(15);
            Assert.AreEqual(1, h.Count(CombatEventType.Clash));
            Assert.AreEqual(hp0, h.P0.Health);
            Assert.AreEqual(hp1, h.P1.Health);
            Assert.AreEqual(1, h.P0.Stats.Clashes);
        }

        [Test]
        public void GuardBreak_DisarmsWeaponUser_ThenWeaponReturns()
        {
            var h = Duel(FighterRoster.Get(FighterRoster.Mara), Armed(FighterRoster.Dima, WeaponCatalog.Knife), 1.0f);
            Assert.IsTrue(h.P1.IsArmed);
            h.Hold(2, InputButtons.None, InputButtons.Block);
            h.P1.Stamina = 1f;
            h.Step(InputButtons.LightPunch, InputButtons.Block);
            h.Hold(6, InputButtons.None, InputButtons.Block);
            Assert.AreEqual(1, h.Count(CombatEventType.GuardBreak));
            Assert.AreEqual(1, h.Count(CombatEventType.Disarmed, -2, 1));
            Assert.IsFalse(h.P1.IsArmed);
            Assert.IsFalse(h.P1.Def.Resolve(AttackButton.LightPunch, MoveDirection.Neutral).IsWeaponStrike, "fights unarmed while disarmed");
            Assert.AreEqual(1, h.P0.Stats.Disarms);
            h.RunUntil(() => h.Count(CombatEventType.Rearmed) > 0, WeaponCatalog.DisarmTicks + 120);
            Assert.AreEqual(1, h.Count(CombatEventType.Rearmed, 1));
            Assert.IsTrue(h.P1.IsArmed);
        }

        [Test]
        public void Throw_DisarmsWeaponUser()
        {
            var h = Duel(FighterRoster.Get(FighterRoster.Mara), Armed(FighterRoster.Dima, WeaponCatalog.Sword), 0.8f);
            h.Step(InputButtons.Throw);
            h.RunUntil(() => h.Count(CombatEventType.ThrowConnected) > 0, 60);
            Assert.AreEqual(1, h.Count(CombatEventType.Disarmed, -2, 1));
            Assert.IsFalse(h.P1.IsArmed);
        }

        [Test]
        public void WeaponIsRestoredAtNextRound()
        {
            var h = Duel(FighterRoster.Get(FighterRoster.Mara), Armed(FighterRoster.Dima, WeaponCatalog.Sword), 0.8f);
            h.Step(InputButtons.Throw);
            h.RunUntil(() => h.Count(CombatEventType.ThrowConnected) > 0, 60);
            h.P1.Health = 1;
            h.RunUntil(() => h.P1.State == FighterStateId.Idle, 200);
            h.SetDistance(0.8f);
            h.Step(InputButtons.LightPunch);
            h.RunUntil(() => h.Sim.Round == 2, 400);
            Assert.AreEqual(2, h.Sim.Round);
            Assert.IsTrue(h.P1.IsArmed);
        }

        [Test]
        public void KnifeCounterHit_DoesExtraDamage()
        {
            var knife = Duel(Armed(FighterRoster.Mara, WeaponCatalog.Knife), FighterRoster.Get(FighterRoster.Dima), 1.0f);
            int hp = knife.P1.Health;
            knife.Step(InputButtons.LightPunch, InputButtons.HeavyPunch);
            knife.Hold(6);
            Assert.AreEqual(1, knife.Count(CombatEventType.CounterHit, 0, 1));
            var move = knife.P0.Def.FindMove("knife_jab");
            float expected = move.Damage * (knife.Sim.Config.CounterHitMultiplier + WeaponCatalog.Get(WeaponCatalog.Knife).CounterHitBonus);
            Assert.AreEqual((int)Math.Round(expected, MidpointRounding.AwayFromZero), hp - knife.P1.Health);
        }

        [Test]
        public void Blades_ChipThroughBareHandedBlock_MoreThanFists()
        {
            var fist = FighterRoster.Get(FighterRoster.Mara).Resolve(AttackButton.HeavyPunch, MoveDirection.Neutral);
            var blade = WeaponCatalog.Get(WeaponCatalog.Sword).Moves.First(m => m.Button == AttackButton.HeavyPunch && m.Direction == MoveDirection.Neutral);
            Assert.Greater(blade.ChipDamage / (float)blade.Damage, fist.ChipDamage / (float)fist.Damage);
        }

        [Test]
        public void ArmedAI_MatchesComplete_AndUseWeaponMechanics()
        {
            int clashes = 0, disarms = 0;
            uint seed = 1;
            foreach (var wa in WeaponCatalog.All)
            foreach (var wb in WeaponCatalog.All)
            {
                var sim = new FightSimulation(Armed(FighterRoster.Lin, wa), Armed(FighterRoster.Brandt, wb), new MatchConfig());
                var a = new FightAI(0, AIDifficulty.Hard, seed++);
                var b = new FightAI(1, AIDifficulty.Hard, seed++ * 13);
                for (int t = 0; t < 60 * 60 * 8 && !sim.IsMatchOver; t++) sim.Step(a.Think(sim), b.Think(sim));
                Assert.IsTrue(sim.IsMatchOver, $"{wa} vs {wb}");
                foreach (var f in sim.Fighters) { clashes += f.Stats.Clashes; disarms += f.Stats.Disarms; }
            }
            TestContext.WriteLine($"clashes={clashes / 2} disarms={disarms}");
            Assert.Greater(clashes, 0);
            Assert.Greater(disarms, 0);
        }

        [Test]
        public void Modifiers_ScaleDamage_HealthAndStamina()
        {
            var cfg = new MatchConfig { IntroTicks = 2, RoundOverTicks = 10 };
            cfg.DamageDealt[0] = 2f;
            cfg.StartHealth[1] = 0.5f;
            cfg.StaminaRegen[0] = 0f;
            var sim = new FightSimulation(FighterRoster.Get(FighterRoster.Mara), FighterRoster.Get(FighterRoster.Dima), cfg);
            var h = new TestHarness(sim);
            h.UntilFighting();
            Assert.AreEqual(h.P1.Def.MaxHealth / 2, h.P1.Health);
            h.SetDistance(1.0f);
            int hp = h.P1.Health;
            h.Step(InputButtons.LightPunch);
            h.Hold(5);
            Assert.AreEqual(hp - h.P0.Def.FindMove("jab").Damage * 2, h.P1.Health);
            float stamina = h.P0.Stamina;
            h.Hold(120);
            Assert.AreEqual(stamina, h.P0.Stamina, 1e-4, "no regeneration with multiplier 0");
        }
    }

    public class CampaignTests
    {
        [Test]
        public void Campaign_Has30Levels_InSixPlayableChapters()
        {
            Assert.AreEqual(30, CampaignDatabase.Levels.Count);
            Assert.AreEqual(6, CampaignDatabase.Chapters.Count);
            foreach (var ch in CampaignDatabase.Chapters)
            {
                Assert.IsTrue(ch.ContentAvailable, "chapter " + ch.Number);
                Assert.GreaterOrEqual(CampaignDatabase.LevelsInChapter(ch.Number).Count(), 3);
            }
            Assert.AreEqual(30, CampaignDatabase.Levels.Select(l => l.Id).Distinct().Count());
            Assert.IsTrue(CampaignDatabase.Levels.Last().IsBoss);
            Assert.AreEqual(FighterRoster.Crane, CampaignDatabase.Levels.Last().OpponentId);
        }

        [Test]
        public void EveryLevel_ReferencesValidContent()
        {
            foreach (var l in CampaignDatabase.Levels)
            {
                Assert.AreEqual(l.ArenaId, CampaignDatabase.GetArena(l.ArenaId).Id, l.Id);
                Assert.AreEqual(2, l.Objectives.Count, l.Id + " needs two objectives for 3 stars");
                foreach (var bout in l.Bouts())
                {
                    Assert.IsNotNull(FighterRoster.Get(bout.OpponentId), l.Id);
                    if (bout.WeaponId != null) Assert.IsNotNull(WeaponCatalog.Get(bout.WeaponId), l.Id);
                }
                if (l.Mode == LevelMode.Gauntlet) Assert.GreaterOrEqual(l.Gauntlet.Count, 3, l.Id);
                if (l.UnlocksWeaponId != null) Assert.IsNotNull(WeaponCatalog.Get(l.UnlocksWeaponId));
                if (l.PlayerDamage != 1f || l.OpponentDamage > 1.1f || l.StartHealth < 1f || l.StaminaRegen != 1f || l.IsGauntlet)
                    Assert.IsTrue(l.RuleText.Length > 0 || l.OpponentDamage <= 1.15f, l.Id + " special rules must be explained");
            }
        }

        [Test]
        public void AllPlayableFighters_AndWeapons_AreUnlockable()
        {
            var unlockable = new HashSet<string>(ProgressionService.StartingFighters);
            var weapons = new HashSet<string>();
            foreach (var l in CampaignDatabase.Levels)
            {
                if (l.UnlocksFighterId != null) unlockable.Add(l.UnlocksFighterId);
                if (l.UnlocksWeaponId != null) weapons.Add(l.UnlocksWeaponId);
            }
            foreach (var id in FighterRoster.PlayableIds) Assert.IsTrue(unlockable.Contains(id), id);
            foreach (var w in WeaponCatalog.All) Assert.IsTrue(weapons.Contains(w), w);
            Assert.GreaterOrEqual(FighterRoster.All.Count, 8, "roster expansion target");
        }

        [Test]
        public void Difficulty_RisesAcrossChapters()
        {
            float Strength(LevelDefinition l) => (int)l.Difficulty + (l.OpponentDamage - 1f) * 10f + (l.IsGauntlet ? 1f : 0f);
            float previous = -1f;
            foreach (var ch in CampaignDatabase.Chapters)
            {
                var avg = (float)CampaignDatabase.LevelsInChapter(ch.Number).Average(Strength);
                Assert.GreaterOrEqual(avg, previous, "chapter " + ch.Number);
                previous = avg;
            }
        }

        [Test]
        public void EveryLevel_PlaysThroughAllBouts_WithWeapons()
        {
            uint seed = 3;
            foreach (var level in CampaignDatabase.Levels)
            {
                var run = new GauntletRun(level);
                int guard = 0;
                while (!run.IsComplete && guard++ < 10)
                {
                    var bout = run.Current;
                    var player = FighterRoster.Get(FighterRoster.Mara);
                    if (level.PlayerWeaponsAllowed) player = WeaponLoadout.Apply(player, WeaponCatalog.Get(WeaponCatalog.Sword));
                    var opp = FighterRoster.Get(bout.OpponentId);
                    if (bout.WeaponId != null) opp = WeaponLoadout.Apply(opp, WeaponCatalog.Get(bout.WeaponId));
                    var cfg = run.CurrentConfig();
                    var sim = new FightSimulation(player, opp, cfg);
                    // A strong reference player so gauntlets usually progress past the first bout.
                    var p = new FightAI(0, AIDifficulty.Expert, seed++);
                    var o = new FightAI(1, level.Difficulty, seed++ * 7);
                    int limit = (cfg.RoundSeconds * 60 + cfg.IntroTicks + cfg.RoundOverTicks + 60) * cfg.MaxRounds;
                    for (int t = 0; t < limit && !sim.IsMatchOver; t++) sim.Step(p.Think(sim), o.Think(sim));
                    Assert.IsTrue(sim.IsMatchOver, $"{level.Id} bout {run.Index} did not finish");
                    run.Record(sim);
                }
                Assert.IsTrue(run.IsComplete, level.Id);
            }
        }
    }

    public class WeaponProgressionTests
    {
        [Test]
        public void GauntletRun_CarriesHealth_AndRequiresEveryBout()
        {
            var level = CampaignDatabase.Levels.First(l => l.IsGauntlet);
            var run = new GauntletRun(level);
            Assert.AreEqual(level.Gauntlet.Count, run.Bouts.Count);
            var sim = new FightSimulation(FighterRoster.Get(FighterRoster.Mara), FighterRoster.Get(run.Current.OpponentId), run.CurrentConfig());
            var h = new TestHarness(sim);
            h.UntilFighting();
            h.SetDistance(0.8f);
            h.P0.Health = h.P0.Def.MaxHealth / 2;
            h.P1.Health = 5;
            h.Step(InputButtons.LightPunch);
            h.RunUntil(() => sim.IsMatchOver, 400);
            Assert.IsTrue(run.Record(sim));
            Assert.AreEqual(0.5f + level.GauntletRecovery, run.PlayerHealth, 0.01f);
            Assert.AreEqual(0.5f + level.GauntletRecovery, run.CurrentConfig().StartHealth[0], 0.01f);
            Assert.AreEqual(1, run.Index);
        }

        [Test]
        public void WeaponUnlock_Equip_AndLoadoutRespectsRules()
        {
            var s = ProgressionService.CreateNew();
            Assert.IsFalse(ProgressionService.TryEquipWeapon(s, FighterRoster.Mara, WeaponCatalog.Knife), "locked weapon");
            var knifeLevel = CampaignDatabase.Levels.First(l => l.UnlocksWeaponId == WeaponCatalog.Knife);
            var r = ProgressionService.ApplyMatchResult(s, knifeLevel, new MatchSummary { Result = MatchResultKind.Victory });
            Assert.Contains(WeaponCatalog.Knife, r.UnlockedWeapons);
            Assert.IsTrue(ProgressionService.TryEquipWeapon(s, FighterRoster.Mara, WeaponCatalog.Knife));
            Assert.AreEqual(WeaponCatalog.Knife, ProgressionService.BuildLoadout(s, FighterRoster.Mara, true).Weapon.Id);
            Assert.IsNull(ProgressionService.BuildLoadout(s, FighterRoster.Mara, false).Weapon, "standard levels are unarmed");
            Assert.IsTrue(ProgressionService.TryEquipWeapon(s, FighterRoster.Mara, ""));
            Assert.IsNull(ProgressionService.BuildLoadout(s, FighterRoster.Mara, true).Weapon);
        }

        [Test]
        public void Stars_CountWinPlusObjectives()
        {
            var s = ProgressionService.CreateNew();
            var l1 = CampaignDatabase.GetLevel("c1_l1");
            var r = ProgressionService.ApplyMatchResult(s, l1, new MatchSummary { Result = MatchResultKind.Victory });
            Assert.AreEqual(1, r.Stars);
            r = ProgressionService.ApplyMatchResult(s, l1, new MatchSummary { Result = MatchResultKind.Victory, Parries = 1, MaxCombo = 3 });
            Assert.AreEqual(3, r.Stars);
            Assert.IsTrue(r.NewBestStars);
            Assert.AreEqual(3, ProgressionService.TotalStars(s));
            ProgressionService.ApplyMatchResult(s, l1, new MatchSummary { Result = MatchResultKind.Defeat });
            Assert.AreEqual(3, s.GetLevel(l1.Id).Stars, "best rating is kept");
        }

        [Test]
        public void NewObjectives_ClashAndDisarm()
        {
            var win = new MatchSummary { Result = MatchResultKind.Victory, Clashes = 2, Disarms = 1 };
            Assert.IsTrue(ProgressionService.IsObjectiveMet(new ObjectiveDefinition { Type = ObjectiveType.LandClashes, Value = 2 }, win));
            Assert.IsFalse(ProgressionService.IsObjectiveMet(new ObjectiveDefinition { Type = ObjectiveType.LandClashes, Value = 3 }, win));
            Assert.IsTrue(ProgressionService.IsObjectiveMet(new ObjectiveDefinition { Type = ObjectiveType.DisarmOpponent }, win));
        }

        [Test]
        public void V1Save_MigratesToV2_WithStars()
        {
            var codec = new TestJsonCodec();
            var old = ProgressionService.CreateNew();
            old.Version = 1;
            var lp = old.GetOrCreateLevel("c1_l1");
            lp.Completed = true;
            lp.CompletedObjectives.Add("c1_l1_o1");
            old.UnlockedWeapons = null;
            old.GetFighter(FighterRoster.Mara).EquippedWeapon = "sword"; // not unlocked: must be cleared
            var store = new MemoryStore { Text = codec.Encode(old) };
            var repo = new SaveRepository(store, codec);
            var s = repo.Load();
            Assert.AreEqual(SaveLoadStatus.Migrated, repo.Status);
            Assert.AreEqual(2, s.Version);
            Assert.AreEqual(2, s.GetLevel("c1_l1").Stars);
            Assert.IsNotNull(s.UnlockedWeapons);
            Assert.AreEqual("", s.GetFighter(FighterRoster.Mara).EquippedWeapon);
        }
    }
}
