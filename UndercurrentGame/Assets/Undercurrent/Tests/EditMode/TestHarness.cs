using System;
using System.Collections.Generic;
using Undercurrent.Core.Combat;
using Undercurrent.Core.Content;

namespace Undercurrent.Tests
{
    /// <summary>Helpers that drive a <see cref="FightSimulation"/> tick by tick in tests.</summary>
    public sealed class TestHarness
    {
        public readonly FightSimulation Sim;
        public readonly List<CombatEvent> AllEvents = new List<CombatEvent>();

        public TestHarness(FightSimulation sim)
        {
            Sim = sim;
            AllEvents.AddRange(Sim.Events);
        }

        public TestHarness(string p0 = FighterRoster.Mara, string p1 = FighterRoster.Dima, MatchConfig config = null)
        {
            config = config ?? new MatchConfig { IntroTicks = 2, RoundOverTicks = 10 };
            Sim = new FightSimulation(FighterRoster.Get(p0), FighterRoster.Get(p1), config);
            AllEvents.AddRange(Sim.Events);
        }

        public FighterRuntime P0 => Sim.Fighters[0];
        public FighterRuntime P1 => Sim.Fighters[1];

        public static TestHarness Fighting(string p0 = FighterRoster.Mara, string p1 = FighterRoster.Dima, float distance = 1.0f, MatchConfig config = null)
        {
            var h = new TestHarness(p0, p1, config);
            h.UntilFighting();
            h.SetDistance(distance);
            return h;
        }

        public void UntilFighting()
        {
            for (int i = 0; i < 1000 && Sim.Phase != MatchPhase.Fighting; i++) Step();
            if (Sim.Phase != MatchPhase.Fighting) throw new InvalidOperationException("Never reached fighting phase");
        }

        public void SetDistance(float d)
        {
            P0.X = -d * 0.5f;
            P1.X = d * 0.5f;
        }

        public void Step(InputButtons i0 = InputButtons.None, InputButtons i1 = InputButtons.None)
        {
            Sim.Step(new FrameInput(i0), new FrameInput(i1));
            AllEvents.AddRange(Sim.Events);
        }

        public void Hold(int ticks, InputButtons i0 = InputButtons.None, InputButtons i1 = InputButtons.None)
        {
            for (int i = 0; i < ticks; i++) Step(i0, i1);
        }

        /// <summary>Presses a button for one tick (then releases), optionally with held directions.</summary>
        public void Tap(InputButtons i0 = InputButtons.None, InputButtons i1 = InputButtons.None)
        {
            Step(i0, i1);
            Step(i0 & Directions, i1 & Directions);
        }

        public const InputButtons Directions = InputButtons.Left | InputButtons.Right | InputButtons.Up | InputButtons.Down;

        /// <summary>Steps until the hit-stop freeze is over, holding inputs.</summary>
        public void ThroughHitStop(InputButtons i0 = InputButtons.None, InputButtons i1 = InputButtons.None)
        {
            for (int i = 0; i < 60 && Sim.HitStopRemaining > 0; i++) Step(i0, i1);
        }

        public int Count(CombatEventType type, int source = -2, int target = -2)
        {
            int n = 0;
            foreach (var e in AllEvents)
            {
                if (e.Type != type) continue;
                if (source != -2 && e.Source != source) continue;
                if (target != -2 && e.Target != target) continue;
                n++;
            }
            return n;
        }

        public CombatEvent First(CombatEventType type)
        {
            foreach (var e in AllEvents) if (e.Type == type) return e;
            throw new InvalidOperationException("No event " + type);
        }

        /// <summary>Runs until the given predicate is true; returns ticks taken or -1.</summary>
        public int RunUntil(Func<bool> condition, int maxTicks, InputButtons i0 = InputButtons.None, InputButtons i1 = InputButtons.None)
        {
            for (int i = 0; i < maxTicks; i++)
            {
                if (condition()) return i;
                Step(i0, i1);
            }
            return condition() ? maxTicks : -1;
        }
    }
}
