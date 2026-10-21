using System;
using System.Collections.Generic;
using Undercurrent.Core.Combat;
using Undercurrent.Core.Content;

namespace Undercurrent.Core.Progression
{
    /// <summary>
    /// Runs a level as a sequence of bouts (one bout for ordinary levels). For gauntlets, each bout is a single round;
    /// the player's health carries over with partial recovery, and the level is won only if every bout is won.
    /// </summary>
    public sealed class GauntletRun
    {
        public readonly LevelDefinition Level;
        public readonly List<Bout> Bouts;
        public int Index { get; private set; }
        public float PlayerHealth { get; private set; } = 1f;
        public bool Failed { get; private set; }
        public bool IsComplete => Failed || Index >= Bouts.Count;
        public Bout Current => IsComplete ? null : Bouts[Index];
        public readonly MatchSummary Total = new MatchSummary { Result = MatchResultKind.Victory };

        public GauntletRun(LevelDefinition level)
        {
            Level = level ?? throw new ArgumentNullException(nameof(level));
            Bouts = level.Bouts();
        }

        /// <summary>Match rules for the current bout (carried-over player health for gauntlets).</summary>
        public MatchConfig CurrentConfig()
        {
            var config = Level.CreateMatchConfig(CampaignDatabase.GetArena(Level.ArenaId));
            if (Level.IsGauntlet)
            {
                config.StartHealth[0] = Math.Min(config.StartHealth[0], PlayerHealth);
                config.StartHealthEveryRound = false;
            }
            return config;
        }

        /// <summary>Records a finished bout; returns true if the run continues with another bout.</summary>
        public bool Record(FightSimulation sim, int playerIndex = 0)
        {
            if (IsComplete) throw new InvalidOperationException("Run already complete");
            var bout = MatchSummary.FromSimulation(sim, playerIndex);
            Total.Accumulate(bout);
            if (bout.Result != MatchResultKind.Victory)
            {
                Failed = true;
                Total.Result = bout.Result;
                return false;
            }
            var p = sim.Fighters[playerIndex];
            PlayerHealth = Math.Min(1f, Math.Max(0.05f, p.HealthFraction) + Level.GauntletRecovery);
            Index++;
            return !IsComplete;
        }
    }
}
