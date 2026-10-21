using System.Collections.Generic;
using Undercurrent.Core.Combat;

namespace Undercurrent.Core.Content
{
    public enum TutorialGoal
    {
        WalkForward,
        WalkBack,
        Crouch,
        Jump,
        LightPunch,
        HeavyPunch,
        LightKick,
        HeavyKick,
        Combo,
        Block,
        Dodge,
        Parry,
        Throw,
        BreakThrow,
        Special,
    }

    /// <summary>What the training partner does while a step is active.</summary>
    public enum DummyBehavior
    {
        Idle,
        RepeatAttack,
        RepeatThrow,
    }

    public sealed class TutorialStep
    {
        public TutorialGoal Goal;
        public string Title;
        public string Instruction;
        public DummyBehavior Dummy;
        public int RequiredCount = 1;
    }

    /// <summary>
    /// Guided training: explains every core action, then waits until the player actually performs it in the live
    /// simulation (detected from fighter state and combat events, not from button presses alone).
    /// </summary>
    public sealed class TutorialTracker
    {
        public static readonly IReadOnlyList<TutorialStep> Steps = new List<TutorialStep>
        {
            new TutorialStep { Goal = TutorialGoal.WalkForward, Title = "Advance", Instruction = "Hold the forward arrow on the left pad to step towards your partner." },
            new TutorialStep { Goal = TutorialGoal.WalkBack, Title = "Retreat", Instruction = "Hold the back arrow to retreat and reset the distance." },
            new TutorialStep { Goal = TutorialGoal.Crouch, Title = "Crouch", Instruction = "Hold down to crouch. Crouching ducks under high attacks." },
            new TutorialStep { Goal = TutorialGoal.Jump, Title = "Jump", Instruction = "Tap up to jump. Jumping attacks must be blocked standing." },
            new TutorialStep { Goal = TutorialGoal.LightPunch, Title = "Light Punch", Instruction = "Tap LP. Fast, safe, and a great combo starter." },
            new TutorialStep { Goal = TutorialGoal.HeavyPunch, Title = "Heavy Punch", Instruction = "Tap HP. Slower, but hits much harder." },
            new TutorialStep { Goal = TutorialGoal.LightKick, Title = "Light Kick", Instruction = "Tap LK. Kicks reach further than punches." },
            new TutorialStep { Goal = TutorialGoal.HeavyKick, Title = "Heavy Kick", Instruction = "Tap HK. Powerful, but a blocked heavy kick leaves you open." },
            new TutorialStep { Goal = TutorialGoal.Combo, Title = "Combination", Instruction = "Land LP, then tap HP the moment it connects to chain a 2-hit combo.", RequiredCount = 2 },
            new TutorialStep { Goal = TutorialGoal.Block, Title = "Block", Instruction = "Your partner will attack. Hold BLOCK to guard. Blocking drains stamina — watch the green bar.", Dummy = DummyBehavior.RepeatAttack },
            new TutorialStep { Goal = TutorialGoal.Dodge, Title = "Dodge", Instruction = "Tap DODGE as the attack comes to slip back out of range.", Dummy = DummyBehavior.RepeatAttack },
            new TutorialStep { Goal = TutorialGoal.Parry, Title = "Parry", Instruction = "Tap PARRY just before the attack lands. A successful parry stuns your opponent.", Dummy = DummyBehavior.RepeatAttack },
            new TutorialStep { Goal = TutorialGoal.Throw, Title = "Throw", Instruction = "Get close and tap THROW. Throws beat blocking and parrying." },
            new TutorialStep { Goal = TutorialGoal.BreakThrow, Title = "Break a Throw", Instruction = "When you are grabbed, tap THROW quickly to break free.", Dummy = DummyBehavior.RepeatThrow },
            new TutorialStep { Goal = TutorialGoal.Special, Title = "Special", Instruction = "Your Focus meter is full. Tap SPECIAL to unleash your signature move." },
        };

        public readonly int PlayerIndex;
        public int StepIndex { get; private set; }
        public int Progress { get; private set; }
        public bool IsComplete => StepIndex >= Steps.Count;
        public TutorialStep Current => IsComplete ? null : Steps[StepIndex];

        private int _dummyTimer;
        private float _walkStartX;
        private bool _walkStarted;

        public TutorialTracker(int playerIndex = 0)
        {
            PlayerIndex = playerIndex;
        }

        public static MatchConfig CreateConfig() => new MatchConfig
        {
            RoundsToWin = 1, MaxRounds = 1, RoundSeconds = 99, IntroTicks = 60,
        };

        /// <summary>Call after each simulation step. Returns true when the current step was completed on this tick.</summary>
        public bool Observe(FightSimulation sim)
        {
            if (IsComplete || sim.Phase != MatchPhase.Fighting) return false;
            var p = sim.Fighters[PlayerIndex];
            var dummy = sim.Fighters[1 - PlayerIndex];

            // Training is never lethal and never exhausting.
            p.Health = p.Def.MaxHealth;
            dummy.Health = dummy.Def.MaxHealth;
            if (Current.Goal == TutorialGoal.Special && p.Focus < FighterRuntime.FocusMax) p.Focus = FighterRuntime.FocusMax;
            if (p.Stamina < p.Def.MaxStamina * 0.5f) p.Stamina = p.Def.MaxStamina * 0.5f;

            bool done = false;
            switch (Current.Goal)
            {
                case TutorialGoal.WalkForward:
                case TutorialGoal.WalkBack:
                    var walkState = Current.Goal == TutorialGoal.WalkForward ? FighterStateId.WalkForward : FighterStateId.WalkBack;
                    if (p.State == walkState)
                    {
                        if (!_walkStarted) { _walkStarted = true; _walkStartX = p.X; }
                        done = System.Math.Abs(p.X - _walkStartX) > 0.6f;
                    }
                    break;
                case TutorialGoal.Crouch: done = p.State == FighterStateId.Crouch && p.StateFrame > 20; break;
                case TutorialGoal.Jump: done = HasEvent(sim, CombatEventType.Jump, PlayerIndex, -2); break;
                case TutorialGoal.LightPunch: done = StartedMoveWith(sim, p, AttackButton.LightPunch); break;
                case TutorialGoal.HeavyPunch: done = StartedMoveWith(sim, p, AttackButton.HeavyPunch); break;
                case TutorialGoal.LightKick: done = StartedMoveWith(sim, p, AttackButton.LightKick); break;
                case TutorialGoal.HeavyKick: done = StartedMoveWith(sim, p, AttackButton.HeavyKick); break;
                case TutorialGoal.Combo: done = dummy.ComboCount >= Current.RequiredCount; break;
                case TutorialGoal.Block: done = HasEvent(sim, CombatEventType.Blocked, -2, PlayerIndex) || HasEvent(sim, CombatEventType.GuardBreak, -2, PlayerIndex); break;
                case TutorialGoal.Dodge: done = HasEvent(sim, CombatEventType.Evaded, PlayerIndex, -2); break;
                case TutorialGoal.Parry: done = HasEvent(sim, CombatEventType.Parried, PlayerIndex, -2); break;
                case TutorialGoal.Throw: done = HasEvent(sim, CombatEventType.ThrowConnected, PlayerIndex, -2); break;
                case TutorialGoal.BreakThrow: done = HasEvent(sim, CombatEventType.ThrowBroken, -2, -2); break;
                case TutorialGoal.Special: done = HasEvent(sim, CombatEventType.SpecialStarted, PlayerIndex, -2); break;
            }

            if (!done) return false;
            Progress = 0;
            StepIndex++;
            _walkStarted = false;
            _dummyTimer = 0;
            return true;
        }

        /// <summary>Lets the player skip a step they are stuck on.</summary>
        public void Skip()
        {
            if (IsComplete) return;
            StepIndex++;
            _walkStarted = false;
            _dummyTimer = 0;
        }

        /// <summary>Input for the training partner for the current step.</summary>
        public FrameInput DummyInput(FightSimulation sim)
        {
            if (IsComplete || sim.Phase != MatchPhase.Fighting) return FrameInput.Empty;
            var dummy = sim.Fighters[1 - PlayerIndex];
            var player = sim.Fighters[PlayerIndex];
            var toward = dummy.Facing > 0 ? InputButtons.Right : InputButtons.Left;
            float distance = System.Math.Abs(player.X - dummy.X);
            _dummyTimer++;

            switch (Current.Dummy)
            {
                case DummyBehavior.RepeatAttack:
                    if (distance > 1.2f) return new FrameInput(toward);
                    if (_dummyTimer >= 75 && dummy.IsActionable) { _dummyTimer = 0; return new FrameInput(InputButtons.LightKick); }
                    return FrameInput.Empty;
                case DummyBehavior.RepeatThrow:
                    if (distance > 0.9f) return new FrameInput(toward);
                    if (_dummyTimer >= 90 && dummy.IsActionable) { _dummyTimer = 0; return new FrameInput(InputButtons.Throw); }
                    return FrameInput.Empty;
                default:
                    return FrameInput.Empty;
            }
        }

        private static bool StartedMoveWith(FightSimulation sim, FighterRuntime p, AttackButton button)
        {
            foreach (var e in sim.Events)
            {
                if (e.Type != CombatEventType.MoveStarted || e.Source != p.Index) continue;
                var m = p.Def.FindMove(e.MoveId);
                if (m != null && m.Button == button) return true;
            }
            return false;
        }

        /// <summary>-2 acts as a wildcard for source/target.</summary>
        private static bool HasEvent(FightSimulation sim, CombatEventType type, int source, int target)
        {
            foreach (var e in sim.Events)
            {
                if (e.Type != type) continue;
                if (source != -2 && e.Source != source) continue;
                if (target != -2 && e.Target != target) continue;
                return true;
            }
            return false;
        }
    }
}
