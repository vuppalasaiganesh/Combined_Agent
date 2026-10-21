using System;
using System.Collections.Generic;
using NUnit.Framework;
using Undercurrent.Core.AI;
using Undercurrent.Core.Combat;
using Undercurrent.Core.Content;

namespace Undercurrent.Tests
{
    public class ContentTests
    {
        [Test]
        public void Roster_FrameData_IsValid()
        {
            foreach (var kv in FighterRoster.All)
            {
                var f = kv.Value;
                Assert.IsNotEmpty(f.DisplayName, f.Id);
                Assert.Greater(f.MaxHealth, 0, f.Id);
                var ids = new HashSet<string>();
                foreach (var m in f.Moves)
                {
                    string ctx = $"{f.Id}/{m.Id}";
                    Assert.IsTrue(ids.Add(m.Id), "duplicate move id " + ctx);
                    Assert.Greater(m.Startup, 0, ctx);
                    Assert.Greater(m.Active, 0, ctx);
                    Assert.GreaterOrEqual(m.Recovery, 0, ctx);
                    Assert.Greater(m.Damage, 0, ctx);
                    Assert.GreaterOrEqual(m.StaminaCost, 0f, ctx);
                    if (!m.IsThrow)
                    {
                        Assert.Greater(m.HitLength, 0f, ctx);
                        Assert.Greater(m.HitHeight, 0f, ctx);
                        Assert.IsTrue(m.CausesKnockdown || m.HitStun > 0, "non-knockdown move needs hitstun " + ctx);
                        Assert.Greater(m.BlockStun, 0, ctx);
                    }
                    Assert.LessOrEqual(m.ChipDamage, m.Damage / 4, "chip damage too high " + ctx);
                    Assert.IsNotEmpty(m.AnimationId, ctx);
                    Assert.IsNotEmpty(m.ImpactSoundId, ctx);
                    Assert.IsNotEmpty(m.ImpactVfxId, ctx);
                }
                foreach (var m in f.Moves)
                    foreach (var c in m.CancelInto)
                        Assert.IsNotNull(f.FindMove(c), $"{f.Id}/{m.Id} cancels into missing move {c}");

                foreach (var b in new[] { AttackButton.LightPunch, AttackButton.HeavyPunch, AttackButton.LightKick, AttackButton.HeavyKick, AttackButton.Throw, AttackButton.Special })
                    Assert.IsNotNull(f.Resolve(b, MoveDirection.Neutral), $"{f.Id} missing neutral {b}");
                Assert.IsNotNull(f.Resolve(AttackButton.LightKick, MoveDirection.Air), f.Id + " missing air attack");

                foreach (var combo in f.Combos)
                {
                    for (int i = 0; i < combo.MoveIds.Length; i++)
                    {
                        var m = f.FindMove(combo.MoveIds[i]);
                        Assert.IsNotNull(m, $"{f.Id} combo {combo.Name} references {combo.MoveIds[i]}");
                        if (i > 0)
                            Assert.Contains(m.Id, f.FindMove(combo.MoveIds[i - 1]).CancelInto, $"{f.Id} combo {combo.Name} link {i} is not a legal cancel");
                    }
                }
            }
        }

        [Test]
        public void LightAttacks_ArePlusOnHit_HeavyKicksAreUnsafeOnBlock()
        {
            foreach (var id in FighterRoster.PlayableIds)
            {
                var f = FighterRoster.Get(id);
                var lp = f.Resolve(AttackButton.LightPunch, MoveDirection.Neutral);
                Assert.Greater(lp.AdvantageOnHit, 0, id + " jab should be plus on hit");
                var hk = f.Resolve(AttackButton.HeavyKick, MoveDirection.Neutral);
                Assert.Less(hk.AdvantageOnBlock, -4, id + " heavy kick should be punishable on block");
            }
        }

        [Test]
        public void PlayableFighters_HaveDistinctIdentities()
        {
            var a = FighterRoster.Get(FighterRoster.Mara);
            var b = FighterRoster.Get(FighterRoster.Dima);
            var c = FighterRoster.Get(FighterRoster.Haruto);
            Assert.AreNotEqual(a.Style, b.Style);
            Assert.AreNotEqual(b.Style, c.Style);
            // Boxer has the fastest jab; karateka has the longest kick; Muay Thai fighter has the most stamina.
            Assert.Less(b.Resolve(AttackButton.LightPunch, MoveDirection.Neutral).Startup, a.Resolve(AttackButton.LightPunch, MoveDirection.Neutral).Startup);
            Assert.Greater(c.Resolve(AttackButton.HeavyKick, MoveDirection.Neutral).HitLength, a.Resolve(AttackButton.HeavyKick, MoveDirection.Neutral).HitLength);
            Assert.Greater(a.MaxStamina, c.MaxStamina);
        }

        [Test]
        public void Campaign_Levels_AreCompleteAndChained()
        {
            Assert.AreEqual(6, CampaignDatabase.Chapters.Count);
            var chapter1 = new List<LevelDefinition>(CampaignDatabase.LevelsInChapter(1));
            Assert.AreEqual(3, chapter1.Count);
            string previous = null;
            foreach (var l in CampaignDatabase.Levels)
            {
                Assert.IsNotNull(FighterRoster.Get(l.OpponentId), l.Id + " opponent");
                Assert.IsNotNull(CampaignDatabase.GetArena(l.ArenaId), l.Id + " arena");
                Assert.AreEqual(l.ArenaId, CampaignDatabase.GetArena(l.ArenaId).Id, l.Id + " arena id resolves exactly");
                Assert.AreEqual(previous, l.RequiresLevelId, l.Id + " unlock chain");
                Assert.Greater(l.RewardCurrency, 0);
                Assert.Greater(l.RewardXp, 0);
                Assert.IsNotEmpty(l.Objectives, l.Id);
                Assert.IsNotEmpty(l.StoryIntro);
                Assert.IsNotEmpty(l.StoryOutro);
                if (l.UnlocksFighterId != null) Assert.IsNotNull(FighterRoster.Get(l.UnlocksFighterId));
                previous = l.Id;
            }
            Assert.IsTrue(CampaignDatabase.GetLevel("c1_l3").IsBoss);
        }

        [Test]
        public void EveryLevel_PlaysToACompleteMatch()
        {
            foreach (var level in CampaignDatabase.Levels)
            {
                var arena = CampaignDatabase.GetArena(level.ArenaId);
                var cfg = level.CreateMatchConfig(arena);
                var sim = new FightSimulation(FighterRoster.Get(FighterRoster.Mara), FighterRoster.Get(level.OpponentId), cfg);
                var p = new FightAI(0, AIDifficulty.Normal, 11);
                var o = new FightAI(1, level.Difficulty, 23);
                int limit = (cfg.RoundSeconds * 60 + cfg.IntroTicks + cfg.RoundOverTicks + 60) * cfg.MaxRounds;
                for (int t = 0; t < limit && !sim.IsMatchOver; t++) sim.Step(p.Think(sim), o.Think(sim));
                Assert.IsTrue(sim.IsMatchOver, level.Id + " match did not finish");
                Assert.LessOrEqual(Math.Max(sim.Fighters[0].RoundsWon, sim.Fighters[1].RoundsWon), cfg.RoundsToWin);
            }
        }
    }

    public class AITests
    {
        private static FightSimulation RunMatch(string p0, AIDifficulty d0, string p1, AIDifficulty d1, uint seed, out int ticks)
        {
            var sim = new FightSimulation(FighterRoster.Get(p0), FighterRoster.Get(p1), new MatchConfig());
            var a = new FightAI(0, d0, seed);
            var b = new FightAI(1, d1, seed * 31 + 7);
            ticks = 0;
            while (!sim.IsMatchOver && ticks < 60 * 60 * 8)
            {
                sim.Step(a.Think(sim), b.Think(sim));
                ticks++;
            }
            return sim;
        }

        [Test]
        public void AllDifficulties_AndFighters_CompleteMatches()
        {
            var ids = new List<string>(FighterRoster.All.Keys);
            uint seed = 1;
            foreach (AIDifficulty d in Enum.GetValues(typeof(AIDifficulty)))
            {
                for (int i = 0; i < ids.Count; i++)
                {
                    var sim = RunMatch(ids[i], d, ids[(i + 1) % ids.Count], AIDifficulty.Normal, seed++, out _);
                    Assert.IsTrue(sim.IsMatchOver, $"{ids[i]} at {d} did not finish");
                    foreach (var f in sim.Fighters)
                    {
                        Assert.GreaterOrEqual(f.Stamina, 0f);
                        Assert.LessOrEqual(f.Stamina, f.Def.MaxStamina + 1e-3f);
                        Assert.GreaterOrEqual(f.Health, 0);
                    }
                }
            }
        }

        [Test]
        public void HigherDifficulty_WinsMoreOften()
        {
            int expertWins = 0, beginnerWins = 0;
            for (uint seed = 1; seed <= 16; seed++)
            {
                bool expertFirst = seed % 2 == 0;
                var sim = expertFirst
                    ? RunMatch(FighterRoster.Mara, AIDifficulty.Expert, FighterRoster.Mara, AIDifficulty.Beginner, seed, out _)
                    : RunMatch(FighterRoster.Mara, AIDifficulty.Beginner, FighterRoster.Mara, AIDifficulty.Expert, seed, out _);
                var expertSide = expertFirst ? RoundOutcome.Player0 : RoundOutcome.Player1;
                var beginnerSide = expertFirst ? RoundOutcome.Player1 : RoundOutcome.Player0;
                if (sim.MatchOutcome == expertSide) expertWins++;
                if (sim.MatchOutcome == beginnerSide) beginnerWins++;
            }
            TestContext.WriteLine($"Expert {expertWins} - Beginner {beginnerWins}");
            Assert.GreaterOrEqual(expertWins, 13);
        }

        [Test]
        public void AI_UsesDefense_AttacksAndCombos()
        {
            int blocks = 0, hits = 0, parries = 0, maxCombo = 0;
            for (uint seed = 1; seed <= 6; seed++)
            {
                var sim = RunMatch(FighterRoster.Sable, AIDifficulty.Hard, FighterRoster.Dima, AIDifficulty.Hard, seed, out _);
                foreach (var f in sim.Fighters)
                {
                    blocks += f.Stats.AttacksBlocked;
                    hits += f.Stats.HitsLanded;
                    parries += f.Stats.Parries;
                    maxCombo = Math.Max(maxCombo, f.Stats.MaxCombo);
                }
            }
            TestContext.WriteLine($"blocks={blocks} hits={hits} parries={parries} maxCombo={maxCombo}");
            Assert.Greater(blocks, 0);
            Assert.Greater(hits, 0);
            Assert.Greater(parries, 0);
            Assert.GreaterOrEqual(maxCombo, 2);
        }

        [Test]
        public void AI_DoesNotReadOpponentInput()
        {
            // Two identical worlds. In world A the player starts a slow spinning attack; in world B the player does nothing.
            // The AI thinks before each step resolves, so on the press tick it cannot know about the press at all, and
            // until its reaction delay has elapsed (and before the attack can connect) it must behave identically.
            var a = TestHarness.Fighting(distance: 1.2f);
            var b = TestHarness.Fighting(distance: 1.2f);
            var aiA = new FightAI(1, AIDifficulty.Expert, 5);
            var aiB = new FightAI(1, AIDifficulty.Expert, 5);
            for (int t = 0; t < 30; t++)
            {
                a.Sim.Step(FrameInput.Empty, aiA.Think(a.Sim));
                b.Sim.Step(FrameInput.Empty, aiB.Think(b.Sim));
            }
            var spin = a.P0.Def.FindMove("spin_elbow");
            int delay = aiA.Profile.ReactionTicks;
            Assert.Greater(spin.Startup, delay - 1, "test needs an attack that cannot connect inside the reaction window");

            var outA = aiA.Think(a.Sim);
            var outB = aiB.Think(b.Sim);
            Assert.AreEqual(outA.Held, outB.Held, "AI decision differed on the tick the player pressed a button");
            a.Sim.Step(new FrameInput(InputButtons.Right | InputButtons.HeavyKick), outA);
            b.Sim.Step(FrameInput.Empty, outB);
            Assert.AreEqual(FighterStateId.Attack, a.P0.State);

            for (int t = 0; t < delay - 1; t++)
            {
                var ia = aiA.Think(a.Sim);
                var ib = aiB.Think(b.Sim);
                Assert.AreEqual(ia.Held, ib.Held, $"AI reacted to an attack {t + 1} ticks after it started (reaction {delay})");
                a.Sim.Step(FrameInput.Empty, ia);
                b.Sim.Step(FrameInput.Empty, ib);
            }
            Assert.AreEqual(a.P1.Health, b.P1.Health, "attack must not have connected inside the window");
        }

        [Test]
        public void Observation_IsDelayedByReactionTime()
        {
            var h = TestHarness.Fighting(distance: 3f);
            var ai = new FightAI(1, AIDifficulty.Normal, 1);
            for (int t = 0; t < 40; t++)
            {
                h.Sim.Step(new FrameInput(InputButtons.Right), ai.Think(h.Sim));
            }
            var now = ai.Observe(0);
            var delayed = ai.Observe(ai.Profile.ReactionTicks);
            Assert.Less(delayed.X, now.X, "delayed observation should see the opponent further back");
        }

        [Test]
        public void Boss_EntersSecondPhase_BelowHalfHealth()
        {
            var h = TestHarness.Fighting(FighterRoster.Mara, FighterRoster.Sable, distance: 3f);
            var boss = new FightAI(1, AIDifficulty.Boss, 3);
            int reaction = boss.Profile.ReactionTicks;
            boss.Think(h.Sim);
            Assert.AreEqual(1, boss.BossPhase);
            h.P1.Health = h.P1.Def.MaxHealth / 2 - 1;
            boss.Think(h.Sim);
            Assert.AreEqual(2, boss.BossPhase);
            Assert.Less(boss.Profile.ReactionTicks, reaction);
            Assert.GreaterOrEqual(boss.Profile.ReactionTicks, 12, "phase two must stay within human reaction range");
        }

        [Test]
        public void ReactionTimes_StayWithinHumanRange()
        {
            foreach (AIDifficulty d in Enum.GetValues(typeof(AIDifficulty)))
            {
                var p = AIProfile.For(d);
                Assert.GreaterOrEqual(p.ReactionTicks, 12, d.ToString()); // >= 200 ms
                Assert.GreaterOrEqual(p.BossPhaseTwo().ReactionTicks, 12, d + " phase 2");
            }
            Assert.Greater(AIProfile.For(AIDifficulty.Beginner).ReactionTicks, AIProfile.For(AIDifficulty.Expert).ReactionTicks);
        }
    }
}
