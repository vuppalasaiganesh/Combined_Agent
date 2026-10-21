using System;
using System.Text;
using NUnit.Framework;
using Undercurrent.Core.AI;
using Undercurrent.Core.Combat;
using Undercurrent.Core.Content;

namespace Undercurrent.Tests
{
    /// <summary>
    /// Diagnostic AI-vs-AI report used for balancing. Explicit: run on demand, never part of the pass/fail gate.
    /// Numbers come from the simulation only; they say nothing about rendering performance or human enjoyment.
    /// </summary>
    public class BalanceReport
    {
        [Test, Explicit("Diagnostic report, run on demand")]
        public void PrintAIMatchReport()
        {
            var sb = new StringBuilder();
            foreach (var diff in new[] { AIDifficulty.Easy, AIDifficulty.Normal, AIDifficulty.Hard, AIDifficulty.Expert })
            {
                int matches = 0, rounds = 0, timeouts = 0, ticks = 0, hits = 0, blocks = 0, parries = 0, evades = 0, throws = 0, specials = 0, maxCombo = 0, knockdowns = 0;
                foreach (var a in FighterRoster.PlayableIds)
                foreach (var b in FighterRoster.PlayableIds)
                for (uint seed = 1; seed <= 3; seed++)
                {
                    var sim = new FightSimulation(FighterRoster.Get(a), FighterRoster.Get(b), new MatchConfig());
                    var ai0 = new FightAI(0, diff, seed);
                    var ai1 = new FightAI(1, diff, seed + 100);
                    int t = 0;
                    while (!sim.IsMatchOver && t < 60 * 60 * 10)
                    {
                        sim.Step(ai0.Think(sim), ai1.Think(sim));
                        foreach (var e in sim.Events)
                        {
                            if (e.Type == CombatEventType.TimeOut) timeouts++;
                            if (e.Type == CombatEventType.RoundEnd) rounds++;
                        }
                        t++;
                    }
                    matches++;
                    ticks += sim.FightingTicksElapsed;
                    foreach (var f in sim.Fighters)
                    {
                        hits += f.Stats.HitsLanded; blocks += f.Stats.AttacksBlocked; parries += f.Stats.Parries;
                        evades += f.Stats.Evades; throws += f.Stats.ThrowsLanded; specials += f.Stats.SpecialsLanded;
                        knockdowns += f.Stats.Knockdowns; maxCombo = Math.Max(maxCombo, f.Stats.MaxCombo);
                    }
                }
                sb.AppendLine($"{diff,-7} matches={matches} rounds={rounds} timeouts={timeouts} avgRoundSec={(ticks / 60f / Math.Max(1, rounds)):F1} " +
                              $"hits={hits} blocks={blocks} parries={parries} evades={evades} throws={throws} specials={specials} knockdowns={knockdowns} maxCombo={maxCombo}");
            }
            TestContext.WriteLine(sb.ToString());
        }
    }
}
