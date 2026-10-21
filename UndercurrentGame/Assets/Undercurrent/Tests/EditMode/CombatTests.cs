using System;
using NUnit.Framework;
using Undercurrent.Core.AI;
using Undercurrent.Core.Combat;
using Undercurrent.Core.Content;

namespace Undercurrent.Tests
{
    public class CombatTests
    {
        private const InputButtons LP = InputButtons.LightPunch;
        private const InputButtons HP = InputButtons.HeavyPunch;
        private const InputButtons LK = InputButtons.LightKick;
        private const InputButtons HK = InputButtons.HeavyKick;
        private const InputButtons Block = InputButtons.Block;
        private const InputButtons Down = InputButtons.Down;

        [Test]
        public void Round_StartsAfterIntro_WithBothFightersIdle()
        {
            var h = new TestHarness();
            Assert.AreEqual(MatchPhase.PreRound, h.Sim.Phase);
            Assert.AreEqual(1, h.Count(CombatEventType.RoundAnnounce));
            h.UntilFighting();
            Assert.AreEqual(FighterStateId.Idle, h.P0.State);
            Assert.AreEqual(FighterStateId.Idle, h.P1.State);
            Assert.AreEqual(1, h.Count(CombatEventType.RoundStart));
            Assert.AreEqual(1, h.P0.Facing);
            Assert.AreEqual(-1, h.P1.Facing);
        }

        [Test]
        public void Inputs_AreIgnoredDuringIntro()
        {
            var h = new TestHarness(config: new MatchConfig { IntroTicks = 30 });
            float x = h.P0.X;
            h.Hold(10, InputButtons.Right | LP);
            Assert.AreEqual(x, h.P0.X);
            Assert.AreEqual(0, h.Count(CombatEventType.MoveStarted));
        }

        [Test]
        public void Walking_MovesForwardAndBack()
        {
            var h = TestHarness.Fighting(distance: 3f);
            float x = h.P0.X;
            h.Hold(10, InputButtons.Right);
            Assert.AreEqual(FighterStateId.WalkForward, h.P0.State);
            Assert.That(h.P0.X, Is.EqualTo(x + 10 * h.P0.Def.WalkForwardSpeed).Within(1e-4));
            x = h.P0.X;
            h.Hold(10, InputButtons.Left);
            Assert.AreEqual(FighterStateId.WalkBack, h.P0.State);
            Assert.That(h.P0.X, Is.EqualTo(x - 10 * h.P0.Def.WalkBackSpeed).Within(1e-4));
        }

        [Test]
        public void Jab_HitsOnFirstActiveFrame_AndNotBefore()
        {
            var h = TestHarness.Fighting(distance: 1.0f);
            var jab = h.P0.Def.FindMove("jab");
            int hp = h.P1.Health;
            h.Step(LP); // move frame 1
            for (int i = 2; i <= jab.Startup; i++)
            {
                h.Step();
                Assert.AreEqual(hp, h.P1.Health, $"damage during startup frame {i}");
            }
            h.Step(); // first active frame
            Assert.AreEqual(1, h.Count(CombatEventType.Hit, 0, 1));
            Assert.AreEqual(hp - jab.Damage, h.P1.Health);
            Assert.AreEqual(FighterStateId.HitStun, h.P1.State);
        }

        [Test]
        public void Attack_OutOfRange_Whiffs_WithNoDamage()
        {
            var h = TestHarness.Fighting(distance: 2.5f);
            int hp = h.P1.Health;
            h.Tap(LP);
            h.Hold(20);
            Assert.AreEqual(hp, h.P1.Health);
            Assert.AreEqual(0, h.Count(CombatEventType.Hit));
            Assert.AreEqual(1, h.Count(CombatEventType.Whiff, 0));
        }

        [Test]
        public void Move_HitsOnlyOncePerExecution()
        {
            var h = TestHarness.Fighting(distance: 1.0f);
            h.Tap(LK); // front kick, 3 active frames
            h.Hold(40);
            Assert.AreEqual(1, h.Count(CombatEventType.Hit, 0, 1));
        }

        [Test]
        public void Blocking_ReducesDamageToChip_AndDrainsStamina()
        {
            var h = TestHarness.Fighting(distance: 1.0f);
            var jab = h.P0.Def.FindMove("jab");
            h.Hold(2, InputButtons.None, Block);
            int hp = h.P1.Health;
            float stamina = h.P1.Stamina;
            h.Step(LP, Block);
            h.Hold(jab.Startup, InputButtons.None, Block);
            Assert.AreEqual(1, h.Count(CombatEventType.Blocked, 0, 1));
            Assert.AreEqual(hp - jab.ChipDamage, h.P1.Health);
            Assert.That(h.P1.Stamina, Is.LessThan(stamina - 1f));
            Assert.AreEqual(FighterStateId.BlockStun, h.P1.State);
        }

        [Test]
        public void LowAttack_BeatsStandingBlock_ButNotCrouchingBlock()
        {
            var standing = TestHarness.Fighting(distance: 1.0f);
            standing.Hold(2, InputButtons.None, Block);
            standing.Step(LK | Down, Block);
            standing.Hold(10, Down, Block);
            Assert.AreEqual(1, standing.Count(CombatEventType.Hit, 0, 1), "low kick should hit a standing blocker");

            var crouching = TestHarness.Fighting(distance: 1.0f);
            crouching.Hold(2, InputButtons.None, Block | Down);
            crouching.Step(LK | Down, Block | Down);
            crouching.Hold(10, Down, Block | Down);
            Assert.AreEqual(1, crouching.Count(CombatEventType.Blocked, 0, 1), "low kick should be blocked crouching");
        }

        [Test]
        public void HighAttack_WhiffsOverCrouchingOpponent()
        {
            var h = TestHarness.Fighting(distance: 1.0f);
            h.Hold(2, InputButtons.None, Down);
            int hp = h.P1.Health;
            h.Step(LP, Down);
            h.Hold(15, InputButtons.None, Down);
            Assert.AreEqual(hp, h.P1.Health);
            Assert.AreEqual(1, h.Count(CombatEventType.Whiff, 0));
        }

        [Test]
        public void Parry_NegatesDamage_AndStunsAttacker()
        {
            var h = TestHarness.Fighting(distance: 1.0f);
            int hp = h.P1.Health;
            h.Step(LP, InputButtons.Parry);
            h.Hold(6);
            Assert.AreEqual(1, h.Count(CombatEventType.Parried, 1, 0));
            Assert.AreEqual(hp, h.P1.Health);
            Assert.AreEqual(FighterStateId.Parried, h.P0.State);
            Assert.AreEqual(1, h.P1.Stats.Parries);
        }

        [Test]
        public void WhiffedParry_HasRecovery_ThatCanBeCounterHit()
        {
            var h = TestHarness.Fighting(distance: 1.0f);
            h.Tap(InputButtons.None, InputButtons.Parry);
            h.Hold(h.P1.Def.ParryWindow + 1);
            Assert.AreEqual(FighterStateId.Parry, h.P1.State);
            Assert.IsFalse(h.P1.IsParryActive);
            h.Step(LP);
            h.Hold(6);
            Assert.AreEqual(1, h.Count(CombatEventType.CounterHit, 0, 1));
        }

        [Test]
        public void Dodge_InvulnerabilityEvadesAttack()
        {
            var h = TestHarness.Fighting(distance: 1.0f);
            // Pin the defender to the wall so the dodge cannot simply move out of range.
            float wall = h.Sim.Config.StageHalfWidth - h.P1.Def.PushHalfWidth;
            h.P1.X = wall;
            h.P0.X = wall - 1.0f;
            int hp = h.P1.Health;
            h.Step(HK, InputButtons.Dodge); // roundhouse, 12f startup; dodge invulnerable frames 2..15
            h.Hold(20);
            Assert.AreEqual(hp, h.P1.Health);
            Assert.AreEqual(1, h.Count(CombatEventType.Evaded, 1, 0));
        }

        [Test]
        public void Throw_BeatsBlock_AndCausesKnockdown()
        {
            var h = TestHarness.Fighting(distance: 0.8f);
            h.Hold(2, InputButtons.None, Block);
            int hp = h.P1.Health;
            h.Step(InputButtons.Throw, Block);
            int ticks = h.RunUntil(() => h.Count(CombatEventType.ThrowConnected) > 0, 60, InputButtons.None, Block);
            Assert.GreaterOrEqual(ticks, 0, "throw never connected");
            var throwMove = h.P0.Def.Resolve(AttackButton.Throw, MoveDirection.Neutral);
            Assert.AreEqual(hp - throwMove.Damage, h.P1.Health);
            Assert.AreEqual(FighterStateId.KnockedDown, h.P1.State);
        }

        [Test]
        public void Throw_CanBeBrokenDuringTechWindow()
        {
            var h = TestHarness.Fighting(distance: 0.8f);
            int hp = h.P1.Health;
            h.Step(InputButtons.Throw);
            h.RunUntil(() => h.P1.State == FighterStateId.Grabbed, 20);
            Assert.AreEqual(FighterStateId.Grabbed, h.P1.State);
            h.Hold(3);
            h.Tap(InputButtons.None, InputButtons.Throw);
            Assert.AreEqual(1, h.Count(CombatEventType.ThrowBroken));
            h.Hold(30);
            Assert.AreEqual(hp, h.P1.Health);
            Assert.AreEqual(0, h.Count(CombatEventType.ThrowConnected));
        }

        [Test]
        public void Throw_CannotGrabOpponentInHitStun()
        {
            var h = TestHarness.Fighting(distance: 0.8f);
            h.P1.StartDefenderStun(FighterStateId.HitStun, 30);
            Assert.IsFalse(h.P1.IsThrowable);
        }

        [Test]
        public void CancelChain_ProducesComboWithDamageScaling()
        {
            var h = TestHarness.Fighting(FighterRoster.Mara, FighterRoster.Dima, distance: 0.8f);
            var jab = h.P0.Def.FindMove("jab");
            var elbow = h.P0.Def.FindMove("elbow");
            int hp = h.P1.Health;
            h.Step(LP);
            h.RunUntil(() => h.Count(CombatEventType.Hit) > 0, 10);
            h.Step(HP);
            h.RunUntil(() => h.Count(CombatEventType.Hit) > 1, 20);
            Assert.AreEqual(2, h.Count(CombatEventType.Hit, 0, 1));
            Assert.AreEqual(2, h.P1.ComboCount);
            int expectedSecond = (int)Math.Round(elbow.Damage * 0.9f, MidpointRounding.AwayFromZero);
            Assert.AreEqual(hp - jab.Damage - expectedSecond, h.P1.Health);
            Assert.AreEqual(2, h.P0.Stats.MaxCombo);
        }

        [Test]
        public void Cancel_IsRejectedWhenMoveWhiffs()
        {
            var h = TestHarness.Fighting(distance: 3f);
            h.Step(LP);
            h.Hold(5);
            h.Step(HP);
            Assert.AreEqual("jab", h.P0.CurrentMove.Id, "whiffed jab must not cancel");
        }

        [Test]
        public void CounterHit_DealsBonusDamage()
        {
            var h = TestHarness.Fighting(distance: 1.0f);
            var jab = h.P0.Def.FindMove("jab");
            int hp = h.P1.Health;
            h.Step(LP, HP); // Dima starts a 7-frame cross; Mara's 4-frame jab lands during its startup
            h.Hold(5);
            Assert.AreEqual(1, h.Count(CombatEventType.CounterHit, 0, 1));
            int expected = (int)Math.Round(jab.Damage * h.Sim.Config.CounterHitMultiplier, MidpointRounding.AwayFromZero);
            Assert.AreEqual(hp - expected, h.P1.Health);
        }

        [Test]
        public void Attacks_RequireStamina_PreventingSpam()
        {
            var h = TestHarness.Fighting(distance: 1.0f);
            h.P0.Stamina = 2f;
            h.Step(LP);
            Assert.AreNotEqual(FighterStateId.Attack, h.P0.State);
            Assert.AreEqual(1, h.Count(CombatEventType.Exhausted, 0));
        }

        [Test]
        public void Stamina_RegeneratesAfterDelay()
        {
            var h = TestHarness.Fighting(distance: 3f);
            h.Tap(LP);
            float after = h.P0.Stamina;
            h.Hold(10);
            Assert.AreEqual(after, h.P0.Stamina, 1e-4, "no regen inside the delay");
            h.Hold(h.P0.Def.StaminaRegenDelay + 10);
            Assert.Greater(h.P0.Stamina, after);
        }

        [Test]
        public void GuardBreak_WhenBlockingWithoutStamina()
        {
            var h = TestHarness.Fighting(distance: 1.0f);
            h.Hold(2, InputButtons.None, Block);
            h.P1.Stamina = 1f;
            h.Step(LP, Block);
            h.Hold(5, InputButtons.None, Block);
            Assert.AreEqual(1, h.Count(CombatEventType.GuardBreak, 0, 1));
            Assert.AreEqual(FighterStateId.GuardBroken, h.P1.State);
        }

        [Test]
        public void ChipDamage_NeverKnocksOut()
        {
            var h = TestHarness.Fighting(distance: 1.0f);
            h.Hold(2, InputButtons.None, Block);
            h.P1.Health = 1;
            h.Step(LP, Block);
            h.Hold(6, InputButtons.None, Block);
            Assert.AreEqual(1, h.P1.Health);
            Assert.AreEqual(MatchPhase.Fighting, h.Sim.Phase);
        }

        [Test]
        public void Sweep_KnocksDown_AndDownedFighterIsInvulnerable()
        {
            var h = TestHarness.Fighting(distance: 1.0f);
            h.Step(HK | Down);
            h.RunUntil(() => h.Count(CombatEventType.Knockdown) > 0, 30, Down);
            Assert.AreEqual(FighterStateId.KnockedDown, h.P1.State);
            Assert.IsTrue(h.P1.IsStrikeInvulnerable);
            Assert.IsFalse(h.P1.IsThrowable);
            int ticks = h.RunUntil(() => h.P1.State == FighterStateId.Idle, 200);
            Assert.GreaterOrEqual(ticks, 0, "fighter never got up");
            Assert.AreEqual(0, h.P1.ComboCount);
        }

        [Test]
        public void HitStop_FreezesBothFighters()
        {
            var h = TestHarness.Fighting(distance: 1.0f);
            h.Step(LP);
            h.RunUntil(() => h.Count(CombatEventType.Hit) > 0, 10);
            int stop = h.Sim.HitStopRemaining;
            Assert.Greater(stop, 0);
            int f0 = h.P0.MoveFrame;
            int stun = h.P1.StunRemaining;
            float x1 = h.P1.X;
            for (int i = 0; i < stop; i++) h.Step();
            Assert.AreEqual(f0, h.P0.MoveFrame);
            Assert.AreEqual(stun, h.P1.StunRemaining);
            Assert.AreEqual(x1, h.P1.X);
            h.Step();
            Assert.AreEqual(f0 + 1, h.P0.MoveFrame);
        }

        [Test]
        public void InputsDuringHitStop_AreBufferedForCancels()
        {
            var h = TestHarness.Fighting(distance: 0.8f);
            h.Step(LP);
            h.RunUntil(() => h.Count(CombatEventType.Hit) > 0, 10);
            Assert.Greater(h.Sim.HitStopRemaining, 0);
            h.Step(HP); // pressed while frozen
            h.ThroughHitStop();
            h.Step();
            Assert.AreEqual("elbow", h.P0.CurrentMove?.Id);
        }

        [Test]
        public void StageBoundaries_AndPushboxes_AreRespected()
        {
            var h = TestHarness.Fighting(distance: 2f);
            h.Hold(600, InputButtons.Left, InputButtons.None);
            float wall = h.Sim.Config.StageHalfWidth - h.P0.Def.PushHalfWidth;
            Assert.GreaterOrEqual(h.P0.X, -wall - 1e-4);

            var push = TestHarness.Fighting(distance: 2f);
            push.Hold(200, InputButtons.Right, InputButtons.Left);
            float minDist = push.P0.Def.PushHalfWidth + push.P1.Def.PushHalfWidth;
            Assert.GreaterOrEqual(push.P1.X - push.P0.X, minDist - 1e-3);
            Assert.Less(push.P0.X, push.P1.X, "fighters must never swap sides");
        }

        [Test]
        public void Separation_IsCappedByCameraLimit()
        {
            var h = TestHarness.Fighting(distance: 2f);
            h.Hold(400, InputButtons.Left, InputButtons.Right);
            Assert.LessOrEqual(h.P1.X - h.P0.X, h.Sim.Config.MaxSeparation + 1e-3);
        }

        [Test]
        public void Jump_LeavesGround_AndLands()
        {
            var h = TestHarness.Fighting(distance: 3f);
            h.Step(InputButtons.Up);
            h.RunUntil(() => h.P0.State == FighterStateId.Airborne, 10);
            Assert.AreEqual(FighterStateId.Airborne, h.P0.State);
            h.Hold(5);
            Assert.Greater(h.P0.Y, 0f);
            int t = h.RunUntil(() => h.P0.State == FighterStateId.Landing, 80);
            Assert.GreaterOrEqual(t, 0);
            Assert.AreEqual(0f, h.P0.Y);
            Assert.AreEqual(1, h.Count(CombatEventType.Land, 0));
        }

        [Test]
        public void JumpingAttack_IsOverhead_AndBeatsCrouchBlock()
        {
            bool anyHit = false;
            for (int delay = 6; delay <= 34 && !anyHit; delay++)
            {
                var h = TestHarness.Fighting(distance: 2.2f);
                h.Step(InputButtons.Up | InputButtons.Right, Block | Down);
                h.Hold(delay, InputButtons.Right, Block | Down);
                h.Step(LK | InputButtons.Right, Block | Down);
                h.Hold(40, InputButtons.None, Block | Down);
                Assert.AreEqual(0, h.Count(CombatEventType.Blocked), "crouch block must not stop an overhead");
                anyHit |= h.Count(CombatEventType.Hit, 0, 1) > 0;
            }
            Assert.IsTrue(anyHit, "jumping kick never connected at any timing");
        }

        [Test]
        public void AntiAir_KnocksDownAirborneOpponent()
        {
            bool knockedDown = false;
            for (int delay = 0; delay <= 30 && !knockedDown; delay++)
            {
                var h = TestHarness.Fighting(distance: 1.6f);
                h.Step(InputButtons.None, InputButtons.Up | InputButtons.Left);
                h.Hold(delay, InputButtons.None, InputButtons.Left);
                h.Step(HP | Down);
                h.Hold(30, Down);
                knockedDown |= h.Count(CombatEventType.Knockdown, 0, 1) > 0;
            }
            Assert.IsTrue(knockedDown);
        }

        [Test]
        public void ComboCap_ForcesKnockdown()
        {
            var h = TestHarness.Fighting(distance: 0.8f);
            h.P1.ComboCount = FighterRuntime.ComboKnockdownCap - 1;
            h.Step(LP);
            h.RunUntil(() => h.Count(CombatEventType.Hit) > 0, 10);
            Assert.AreEqual(FighterStateId.KnockedDown, h.P1.State);
        }

        [Test]
        public void Special_RequiresFullFocus_AndIsInvulnerableOnStartup()
        {
            var h = TestHarness.Fighting(distance: 1.0f);
            h.Tap(InputButtons.Special);
            Assert.AreNotEqual(FighterStateId.Attack, h.P0.State, "special must not start without focus");

            h.P0.Focus = FighterRuntime.FocusMax;
            h.Step(InputButtons.Special, LP); // opponent jabs into the special's invulnerable startup
            Assert.AreEqual("special", h.P0.CurrentMove.Id);
            Assert.AreEqual(0f, h.P0.Focus);
            h.Hold(30);
            Assert.AreEqual(0, h.Count(CombatEventType.Hit, 1, 0));
            Assert.AreEqual(1, h.Count(CombatEventType.SpecialStarted, 0));
            Assert.GreaterOrEqual(h.Count(CombatEventType.Hit, 0, 1) + h.Count(CombatEventType.CounterHit, 0, 1), 1);
        }

        [Test]
        public void KO_EndsRound_AndBestOfThreeEndsMatch()
        {
            var h = TestHarness.Fighting(distance: 0.8f);
            h.P1.Health = 10;
            h.Step(LP);
            h.RunUntil(() => h.Sim.Phase == MatchPhase.RoundOver, 20);
            Assert.AreEqual(MatchPhase.RoundOver, h.Sim.Phase);
            Assert.AreEqual(1, h.Count(CombatEventType.KO, 0, 1));
            Assert.AreEqual(1, h.P0.RoundsWon);
            Assert.AreEqual(FighterStateId.KO, h.P1.State);

            h.RunUntil(() => h.Sim.Round == 2, 200);
            Assert.AreEqual(2, h.Sim.Round);
            Assert.AreEqual(h.P1.Def.MaxHealth, h.P1.Health, "health resets between rounds");
            h.UntilFighting();
            h.SetDistance(0.8f);
            h.P1.Health = 10;
            h.Step(LP);
            h.RunUntil(() => h.Sim.IsMatchOver, 300);
            Assert.IsTrue(h.Sim.IsMatchOver);
            Assert.AreEqual(RoundOutcome.Player0, h.Sim.MatchOutcome);
            Assert.AreEqual(1, h.Count(CombatEventType.MatchEnd));
            var ko = h.AllEvents.FindLast(e => e.Type == CombatEventType.KO);
            Assert.IsTrue(ko.IsFinal, "final KO is flagged for cinematic presentation");
        }

        [Test]
        public void Timeout_AwardsRoundToHealthier()
        {
            var cfg = new MatchConfig { IntroTicks = 2, RoundOverTicks = 10, RoundSeconds = 1 };
            var h = TestHarness.Fighting(config: cfg, distance: 3f);
            h.P0.Health = 300;
            h.RunUntil(() => h.Sim.Phase == MatchPhase.RoundOver, 100);
            Assert.AreEqual(1, h.Count(CombatEventType.TimeOut));
            Assert.AreEqual(RoundOutcome.Player1, h.Sim.LastRoundOutcome);
        }

        [Test]
        public void Simulation_IsDeterministic()
        {
            for (int seed = 1; seed <= 3; seed++)
            {
                var a = new TestHarness(FighterRoster.Mara, FighterRoster.Haruto);
                var b = new TestHarness(FighterRoster.Mara, FighterRoster.Haruto);
                var aiA0 = new FightAI(0, AIDifficulty.Hard, (uint)seed);
                var aiA1 = new FightAI(1, AIDifficulty.Normal, (uint)seed * 7);
                var aiB0 = new FightAI(0, AIDifficulty.Hard, (uint)seed);
                var aiB1 = new FightAI(1, AIDifficulty.Normal, (uint)seed * 7);
                for (int t = 0; t < 4000; t++)
                {
                    var i0 = aiA0.Think(a.Sim); var i1 = aiA1.Think(a.Sim);
                    a.Sim.Step(i0, i1);
                    var j0 = aiB0.Think(b.Sim); var j1 = aiB1.Think(b.Sim);
                    b.Sim.Step(j0, j1);
                    Assert.AreEqual(a.Sim.StateHash(), b.Sim.StateHash(), $"diverged at tick {t} (seed {seed})");
                }
            }
        }

        [Test]
        public void FrameAdvantage_MatchesSimulation()
        {
            // Verifies MoveData.AdvantageOnHit against the actual simulation for Mara's jab.
            var h = TestHarness.Fighting(distance: 1.0f);
            var jab = h.P0.Def.FindMove("jab");
            h.Step(LP);
            h.RunUntil(() => h.Count(CombatEventType.Hit) > 0, 10);
            h.ThroughHitStop();
            int attackerFree = -1, defenderFree = -1;
            for (int t = 1; t < 60 && (attackerFree < 0 || defenderFree < 0); t++)
            {
                h.Step();
                if (attackerFree < 0 && h.P0.IsActionable) attackerFree = t;
                if (defenderFree < 0 && h.P1.IsActionable) defenderFree = t;
            }
            Assert.AreEqual(jab.AdvantageOnHit, defenderFree - attackerFree);
        }
    }
}
