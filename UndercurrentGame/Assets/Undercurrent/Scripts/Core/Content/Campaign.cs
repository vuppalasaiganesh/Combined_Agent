using System;
using System.Collections.Generic;
using Undercurrent.Core.AI;
using Undercurrent.Core.Combat;

namespace Undercurrent.Core.Content
{
    [Serializable]
    public sealed class ArenaDefinition
    {
        public string Id;
        public string DisplayName;
        public string Description;
        public float StageHalfWidth = 6.5f;
        /// <summary>Lighting/prop preset understood by the presentation layer.</summary>
        public string VisualPreset;
        public string AmbienceId;
        public string MusicId;
    }

    public enum ObjectiveType
    {
        WinWithoutLosingRound,
        WinWithinSeconds,
        LandParries,
        LandCombo,
        PerfectRound,
        LandThrows,
        LandSpecial,
    }

    [Serializable]
    public sealed class ObjectiveDefinition
    {
        public string Id;
        public ObjectiveType Type;
        public int Value;
        public int BonusCurrency;

        public string Describe()
        {
            switch (Type)
            {
                case ObjectiveType.WinWithoutLosingRound: return "Win without losing a round";
                case ObjectiveType.WinWithinSeconds: return $"Win in under {Value} seconds of fight time";
                case ObjectiveType.LandParries: return $"Land {Value} parries";
                case ObjectiveType.LandCombo: return $"Land a {Value}-hit combo";
                case ObjectiveType.PerfectRound: return "Win a round without taking damage";
                case ObjectiveType.LandThrows: return $"Land {Value} throws";
                case ObjectiveType.LandSpecial: return "Land a special move";
                default: return Type.ToString();
            }
        }
    }

    [Serializable]
    public sealed class LevelDefinition
    {
        public string Id;
        public int Chapter;
        public int Number;
        public string Title;
        public string StoryIntro;
        public string StoryOutro;
        public string OpponentId;
        public string ArenaId;
        public AIDifficulty Difficulty;
        public int RoundsToWin = 2;
        public int RoundSeconds = 60;
        public bool IsBoss;
        public bool StartsTutorial;
        /// <summary>Level that must be won first; null for the first level.</summary>
        public string RequiresLevelId;
        public int RewardCurrency;
        public int RewardXp;
        public int FirstClearBonusCurrency;
        public string UnlocksFighterId;
        public string UnlocksCosmeticId;
        public List<ObjectiveDefinition> Objectives = new List<ObjectiveDefinition>();

        public MatchConfig CreateMatchConfig(ArenaDefinition arena)
        {
            return new MatchConfig
            {
                RoundsToWin = RoundsToWin,
                MaxRounds = RoundsToWin * 2 + 1,
                RoundSeconds = RoundSeconds,
                StageHalfWidth = arena != null ? arena.StageHalfWidth : 6.5f,
            };
        }
    }

    [Serializable]
    public sealed class ChapterDefinition
    {
        public int Number;
        public string Title;
        public string Summary;
        /// <summary>False when the chapter's levels have not been built yet; the UI shows it as upcoming.</summary>
        public bool ContentAvailable;
    }

    /// <summary>Campaign content: chapters, levels, arenas. Only content marked available is playable.</summary>
    public static class CampaignDatabase
    {
        public static readonly IReadOnlyList<ArenaDefinition> Arenas = new List<ArenaDefinition>
        {
            new ArenaDefinition
            {
                Id = "pier9", DisplayName = "Pier 9 Net-Loft",
                Description = "A converted fishing net-loft above the Varrow Bay docks. Sodium lamps, salt-stained timber, rope coils.",
                VisualPreset = "warehouse", AmbienceId = "amb_harbor", MusicId = "music_pulse", StageHalfWidth = 6.0f,
            },
            new ArenaDefinition
            {
                Id = "freightyard", DisplayName = "Calloway Freight Yard",
                Description = "A ring of stacked shipping containers under floodlights, rain on corrugated steel.",
                VisualPreset = "freightyard", AmbienceId = "amb_rain", MusicId = "music_drive", StageHalfWidth = 6.5f,
            },
            new ArenaDefinition
            {
                Id = "cistern", DisplayName = "The Old Cistern",
                Description = "A flooded Victorian-era water cistern lit by work lights. Brick arches, ankle-deep water.",
                VisualPreset = "cistern", AmbienceId = "amb_drips", MusicId = "music_boss", StageHalfWidth = 6.5f,
            },
        };

        public static readonly IReadOnlyList<ChapterDefinition> Chapters = new List<ChapterDefinition>
        {
            new ChapterDefinition { Number = 1, Title = "The Invitation", Summary = "A sealed envelope, a sparring partner, and a door you can't unsee.", ContentAvailable = true },
            new ChapterDefinition { Number = 2, Title = "Regional Challengers", Summary = "Prove yourself across Varrow Bay's hidden venues.", ContentAvailable = false },
            new ChapterDefinition { Number = 3, Title = "Qualification", Summary = "The Ledger's qualifiers. Only eight advance.", ContentAvailable = false },
            new ChapterDefinition { Number = 4, Title = "Elite Warriors", Summary = "Fighters who have never lost on the circuit.", ContentAvailable = false },
            new ChapterDefinition { Number = 5, Title = "The Undercurrent", Summary = "The real tournament beneath the tournament.", ContentAvailable = false },
            new ChapterDefinition { Number = 6, Title = "Final Championship", Summary = "Face The Ledger's champion.", ContentAvailable = false },
        };

        public static readonly IReadOnlyList<LevelDefinition> Levels = new List<LevelDefinition>
        {
            new LevelDefinition
            {
                Id = "c1_l1", Chapter = 1, Number = 1, Title = "Sparring at the Net-Loft",
                StoryIntro = "Coach Ines tapes your hands. \"Someone slid an invitation under my door with your name on it. Show me you're ready before I let you answer it.\"",
                StoryOutro = "Ines studies the envelope's black wax seal. \"The Undercurrent. Bram Calloway collects for them. He'll find you before you find him.\"",
                OpponentId = FighterRoster.Ines, ArenaId = "pier9", Difficulty = AIDifficulty.Beginner,
                RoundsToWin = 1, RoundSeconds = 90, StartsTutorial = true,
                RewardCurrency = 100, RewardXp = 60, FirstClearBonusCurrency = 150,
                Objectives = new List<ObjectiveDefinition>
                {
                    new ObjectiveDefinition { Id = "c1_l1_parry", Type = ObjectiveType.LandParries, Value = 1, BonusCurrency = 50 },
                    new ObjectiveDefinition { Id = "c1_l1_combo", Type = ObjectiveType.LandCombo, Value = 3, BonusCurrency = 50 },
                },
            },
            new LevelDefinition
            {
                Id = "c1_l2", Chapter = 1, Number = 2, Title = "Collection Day",
                StoryIntro = "Rain on steel. Bram \"Rook\" Calloway cracks his knuckles: \"The Ledger wants to know if you're worth the ink.\"",
                StoryOutro = "Calloway spits and laughs. \"Fine. You're in the book. Sable's waiting under the city — she doesn't do warm-ups.\"",
                OpponentId = FighterRoster.Rook, ArenaId = "freightyard", Difficulty = AIDifficulty.Easy,
                RoundsToWin = 2, RoundSeconds = 60, RequiresLevelId = "c1_l1",
                RewardCurrency = 150, RewardXp = 90, FirstClearBonusCurrency = 200, UnlocksCosmeticId = "wraps_crimson",
                Objectives = new List<ObjectiveDefinition>
                {
                    new ObjectiveDefinition { Id = "c1_l2_flawless", Type = ObjectiveType.WinWithoutLosingRound, Value = 0, BonusCurrency = 75 },
                    new ObjectiveDefinition { Id = "c1_l2_throws", Type = ObjectiveType.LandThrows, Value = 2, BonusCurrency = 50 },
                },
            },
            new LevelDefinition
            {
                Id = "c1_l3", Chapter = 1, Number = 3, Title = "The Gatekeeper",
                StoryIntro = "Work lights hum over black water. Teodora \"Sable\" Vance rolls her shoulders. \"Everyone who reaches the qualifiers went through me. Most didn't.\"",
                StoryOutro = "Sable offers a hand up — to you, not to herself. \"Haruto Sen asked about you. Says the sponsor funding his dojo is The Ledger. Find him.\"",
                OpponentId = FighterRoster.Sable, ArenaId = "cistern", Difficulty = AIDifficulty.Boss,
                RoundsToWin = 2, RoundSeconds = 75, IsBoss = true, RequiresLevelId = "c1_l2",
                RewardCurrency = 250, RewardXp = 150, FirstClearBonusCurrency = 400, UnlocksFighterId = FighterRoster.Haruto,
                UnlocksCosmeticId = "outfit_sable_tribute",
                Objectives = new List<ObjectiveDefinition>
                {
                    new ObjectiveDefinition { Id = "c1_l3_special", Type = ObjectiveType.LandSpecial, Value = 1, BonusCurrency = 75 },
                    new ObjectiveDefinition { Id = "c1_l3_fast", Type = ObjectiveType.WinWithinSeconds, Value = 120, BonusCurrency = 100 },
                },
            },
        };

        public static LevelDefinition GetLevel(string id)
        {
            foreach (var l in Levels) if (l.Id == id) return l;
            return null;
        }

        public static ArenaDefinition GetArena(string id)
        {
            foreach (var a in Arenas) if (a.Id == id) return a;
            return Arenas[0];
        }

        public static IEnumerable<LevelDefinition> LevelsInChapter(int chapter)
        {
            foreach (var l in Levels) if (l.Chapter == chapter) yield return l;
        }
    }
}
