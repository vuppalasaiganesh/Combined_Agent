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
        LandClashes,
        DisarmOpponent,
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
                case ObjectiveType.LandParries: return $"Land {Value} parr{(Value == 1 ? "y" : "ies")}";
                case ObjectiveType.LandCombo: return $"Land a {Value}-hit combo";
                case ObjectiveType.PerfectRound: return "Win a round without taking damage";
                case ObjectiveType.LandThrows: return $"Land {Value} throw{(Value == 1 ? "" : "s")}";
                case ObjectiveType.LandSpecial: return "Land a special move";
                case ObjectiveType.LandClashes: return $"Win {Value} weapon clash{(Value == 1 ? "" : "es")}";
                case ObjectiveType.DisarmOpponent: return "Disarm your opponent (guard break or throw)";
                default: return Type.ToString();
            }
        }
    }

    public enum LevelMode
    {
        /// <summary>Unarmed for both fighters.</summary>
        Standard,
        /// <summary>The player may bring their equipped weapon; the opponent uses <see cref="LevelDefinition.OpponentWeaponId"/>.</summary>
        Weapons,
        /// <summary>Several single-round bouts in a row; the player's health carries over (partially restored between bouts).</summary>
        Gauntlet,
    }

    /// <summary>One opponent in a gauntlet.</summary>
    [Serializable]
    public sealed class Bout
    {
        public string OpponentId;
        public string WeaponId;

        public Bout() { }
        public Bout(string opponent, string weapon = null) { OpponentId = opponent; WeaponId = weapon; }
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
        public string UnlocksWeaponId;
        public List<ObjectiveDefinition> Objectives = new List<ObjectiveDefinition>();

        // ---- Rules
        public LevelMode Mode = LevelMode.Standard;
        public string OpponentWeaponId;
        public List<Bout> Gauntlet = new List<Bout>();
        /// <summary>Fraction of max health restored between gauntlet bouts.</summary>
        public float GauntletRecovery = 0.3f;
        public float PlayerDamage = 1f;
        public float OpponentDamage = 1f;
        public float StartHealth = 1f;
        public bool StartHealthEveryRound;
        public float StaminaRegen = 1f;
        /// <summary>Short rule description shown on the briefing screen (empty for ordinary fights).</summary>
        public string RuleText = "";

        public bool PlayerWeaponsAllowed => Mode != LevelMode.Standard;
        public bool IsGauntlet => Mode == LevelMode.Gauntlet && Gauntlet.Count > 0;

        public MatchConfig CreateMatchConfig(ArenaDefinition arena)
        {
            var c = new MatchConfig
            {
                RoundsToWin = IsGauntlet ? 1 : RoundsToWin,
                MaxRounds = IsGauntlet ? 3 : RoundsToWin * 2 + 1,
                RoundSeconds = RoundSeconds,
                StageHalfWidth = arena != null ? arena.StageHalfWidth : 6.5f,
                StartHealthEveryRound = StartHealthEveryRound,
            };
            c.DamageDealt[0] = PlayerDamage;
            c.DamageDealt[1] = OpponentDamage;
            c.StartHealth[0] = StartHealth;
            c.StartHealth[1] = StartHealth;
            c.StaminaRegen[0] = StaminaRegen;
            c.StaminaRegen[1] = StaminaRegen;
            return c;
        }

        /// <summary>Opponents in fight order (a single bout for non-gauntlet levels).</summary>
        public List<Bout> Bouts() => IsGauntlet ? Gauntlet : new List<Bout> { new Bout(OpponentId, Mode == LevelMode.Standard ? null : OpponentWeaponId) };
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

    /// <summary>
    /// Campaign content: 6 chapters, 30 levels, 7 arenas. Levels are declared as compact table rows (see
    /// <see cref="Row"/>) so the whole campaign can be read, balanced and extended in one place; rewards scale
    /// automatically with campaign position.
    /// </summary>
    public static class CampaignDatabase
    {
        public static readonly IReadOnlyList<ArenaDefinition> Arenas = new List<ArenaDefinition>
        {
            new ArenaDefinition { Id = "pier9", DisplayName = "Pier 9 Net-Loft", VisualPreset = "warehouse", AmbienceId = "amb_harbor", MusicId = "music_pulse", StageHalfWidth = 6.0f,
                Description = "A converted fishing net-loft above the Varrow Bay docks. Sodium lamps, salt-stained timber, rope coils." },
            new ArenaDefinition { Id = "freightyard", DisplayName = "Calloway Freight Yard", VisualPreset = "freightyard", AmbienceId = "amb_rain", MusicId = "music_drive", StageHalfWidth = 6.5f,
                Description = "A ring of stacked shipping containers under floodlights, rain on corrugated steel." },
            new ArenaDefinition { Id = "cistern", DisplayName = "The Old Cistern", VisualPreset = "cistern", AmbienceId = "amb_drips", MusicId = "music_boss", StageHalfWidth = 6.5f,
                Description = "A flooded Victorian-era water cistern lit by work lights. Brick arches, ankle-deep water." },
            new ArenaDefinition { Id = "market", DisplayName = "Lantern Row Night Market", VisualPreset = "market", AmbienceId = "amb_harbor", MusicId = "music_drive", StageHalfWidth = 6.0f,
                Description = "A shuttered night market strung with paper lanterns; stalls pushed back to make a ring." },
            new ArenaDefinition { Id = "rooftop", DisplayName = "Sen Rooftop Garden", VisualPreset = "rooftop", AmbienceId = "amb_rain", MusicId = "music_pulse", StageHalfWidth = 6.5f,
                Description = "The Sen family's rooftop dojo garden: wet stone, moss and the city skyline." },
            new ArenaDefinition { Id = "forge", DisplayName = "Brandt Forge Yard", VisualPreset = "forge", AmbienceId = "amb_drips", MusicId = "music_boss", StageHalfWidth = 6.5f,
                Description = "A working forge with glowing coals, anvils and racks of training blades." },
            new ArenaDefinition { Id = "finals", DisplayName = "The Undercurrent", VisualPreset = "finals", AmbienceId = "amb_drips", MusicId = "music_boss", StageHalfWidth = 7.0f,
                Description = "A drained reservoir turned arena: one ring of light, a hundred watching faces in the dark." },
        };

        public static readonly IReadOnlyList<ChapterDefinition> Chapters = new List<ChapterDefinition>
        {
            new ChapterDefinition { Number = 1, Title = "The Invitation", Summary = "A sealed envelope, a sparring partner, and a door you can't unsee.", ContentAvailable = true },
            new ChapterDefinition { Number = 2, Title = "Regional Challengers", Summary = "Varrow Bay's hidden venues — and the first time someone brings a blade.", ContentAvailable = true },
            new ChapterDefinition { Number = 3, Title = "Qualification", Summary = "The Ledger's qualifiers. Only eight advance.", ContentAvailable = true },
            new ChapterDefinition { Number = 4, Title = "Elite Warriors", Summary = "Fighters who have never lost on the circuit.", ContentAvailable = true },
            new ChapterDefinition { Number = 5, Title = "The Undercurrent", Summary = "The real tournament beneath the tournament.", ContentAvailable = true },
            new ChapterDefinition { Number = 6, Title = "Final Championship", Summary = "Face The Ledger's champion.", ContentAvailable = true },
        };

        private static readonly List<LevelDefinition> LevelList = BuildLevels();
        public static readonly IReadOnlyList<LevelDefinition> Levels = LevelList;

        private const string K = WeaponCatalog.Knife, S = WeaponCatalog.Sword, St = WeaponCatalog.Staff;

        private static ObjectiveDefinition O(ObjectiveType t, int value = 0) => new ObjectiveDefinition { Type = t, Value = value };

        /// <summary>One campaign row. Ids, unlock chain and rewards are derived from the row's position.</summary>
        private static LevelDefinition Row(List<LevelDefinition> list, int chapter, string title, string opponent, string arena,
            AIDifficulty difficulty, int roundsToWin, string intro, string outro, ObjectiveDefinition[] objectives, Action<LevelDefinition> rules = null)
        {
            int index = list.Count; // 0-based campaign position
            int number = 1;
            foreach (var l in list) if (l.Chapter == chapter) number++;
            var level = new LevelDefinition
            {
                Id = $"c{chapter}_l{number}",
                Chapter = chapter,
                Number = number,
                Title = title,
                OpponentId = opponent,
                ArenaId = arena,
                Difficulty = difficulty,
                RoundsToWin = roundsToWin,
                StoryIntro = intro,
                StoryOutro = outro,
                RequiresLevelId = index > 0 ? list[index - 1].Id : null,
                RewardCurrency = 100 + 25 * index,
                RewardXp = 60 + 15 * index,
                FirstClearBonusCurrency = 150 + 35 * index,
                IsBoss = difficulty == AIDifficulty.Boss,
            };
            for (int i = 0; i < objectives.Length; i++)
            {
                var o = objectives[i];
                o.Id = $"{level.Id}_o{i + 1}";
                o.BonusCurrency = 50 + 10 * index;
                level.Objectives.Add(o);
            }
            rules?.Invoke(level);
            list.Add(level);
            return level;
        }

        private static List<LevelDefinition> BuildLevels()
        {
            var L = new List<LevelDefinition>();
            const AIDifficulty Beg = AIDifficulty.Beginner, Easy = AIDifficulty.Easy, Norm = AIDifficulty.Normal,
                Hard = AIDifficulty.Hard, Exp = AIDifficulty.Expert, Boss = AIDifficulty.Boss;
            const ObjectiveType Parry = ObjectiveType.LandParries, Combo = ObjectiveType.LandCombo, Flawless = ObjectiveType.WinWithoutLosingRound,
                Throws = ObjectiveType.LandThrows, Special = ObjectiveType.LandSpecial, Fast = ObjectiveType.WinWithinSeconds,
                Perfect = ObjectiveType.PerfectRound, Clash = ObjectiveType.LandClashes, Disarm = ObjectiveType.DisarmOpponent;

            // ------------------------------------------------------------- Chapter 1: The Invitation
            Row(L, 1, "Sparring at the Net-Loft", FighterRoster.Ines, "pier9", Beg, 1,
                "Coach Ines tapes your hands. \"Someone slid an invitation under my door with your name on it. Show me you're ready before I let you answer it.\"",
                "Ines studies the envelope's black wax seal. \"The Undercurrent. Bram Calloway collects for them. He'll find you before you find him.\"",
                new[] { O(Parry, 1), O(Combo, 3) }, l => { l.RoundSeconds = 90; l.StartsTutorial = true; l.RewardCurrency = 100; l.RewardXp = 60; l.FirstClearBonusCurrency = 150; });
            Row(L, 1, "Collection Day", FighterRoster.Rook, "freightyard", Easy, 2,
                "Rain on steel. Bram \"Rook\" Calloway cracks his knuckles: \"The Ledger wants to know if you're worth the ink.\"",
                "Calloway spits and laughs. \"Fine. You're in the book. Sable's waiting under the city — she doesn't do warm-ups.\"",
                new[] { O(Flawless), O(Throws, 2) }, l => { l.UnlocksCosmeticId = "wraps_crimson"; l.RewardCurrency = 150; l.RewardXp = 90; l.FirstClearBonusCurrency = 200; });
            Row(L, 1, "The Gatekeeper", FighterRoster.Sable, "cistern", Boss, 2,
                "Work lights hum over black water. Teodora \"Sable\" Vance rolls her shoulders. \"Everyone who reaches the qualifiers went through me. Most didn't.\"",
                "Sable offers a hand up — to you, not to herself. \"Haruto Sen asked about you. Says the sponsor funding his dojo is The Ledger. Find him.\"",
                new[] { O(Special), O(Fast, 120) }, l => { l.RoundSeconds = 75; l.UnlocksFighterId = FighterRoster.Haruto; l.UnlocksCosmeticId = "outfit_sable_tribute";
                    l.RewardCurrency = 250; l.RewardXp = 150; l.FirstClearBonusCurrency = 400; });

            // ------------------------------------------------------------- Chapter 2: Regional Challengers
            Row(L, 2, "Dockside Heavyweight", FighterRoster.Kofi, "freightyard", Norm, 2,
                "A crowd of dockworkers forms a ring. Kofi \"Anvil\" Mensah lifts his gloves. \"They say you beat Sable. Let's see you take a real punch.\"",
                "Kofi grins through a split lip. \"Good hands. Watch the night market — somebody there fights with steel.\"",
                new[] { O(Combo, 4), O(Flawless) });
            Row(L, 2, "Hidden Blade", FighterRoster.Wren, "market", Norm, 2,
                "Lanterns sway over Lantern Row. Wren \"Switchblade\" Hollis flips a training knife. \"Empty hands against steel? Bold. Break my guard and it's yours.\"",
                "Wren tosses you the knife, handle first. \"Keep it. You'll need it where you're going.\"",
                new[] { O(Disarm), O(Parry, 2) },
                l => { l.Mode = LevelMode.Weapons; l.OpponentWeaponId = K; l.PlayerDamage = 1.15f; l.UnlocksWeaponId = K;
                    l.RuleText = "You start unarmed against a knife; your strikes deal +15%. Guard-break or throw her to knock the blade away."; });
            Row(L, 2, "Old Lessons", FighterRoster.Ines, "pier9", Hard, 2,
                "Ines is waiting with two cups of tea and no gloves. \"No coaching today. Show me what the street taught you.\"",
                "\"You're not my student any more,\" she says quietly. \"Be careful who you become.\"",
                new[] { O(Perfect), O(Fast, 90) }, l => { l.Mode = LevelMode.Weapons; l.RuleText = "Weapons allowed — Ines fights empty-handed."; });
            Row(L, 2, "Freight Gauntlet", FighterRoster.Rook, "freightyard", Norm, 1,
                "Calloway blocks the yard gate with two friends. \"Three fights, no breaks. The Ledger likes stamina.\"",
                "The yard falls quiet. Kofi steps out of the crowd and offers his hand. \"I'm in your corner now.\"",
                new[] { O(Flawless), O(Special) },
                l => { l.Mode = LevelMode.Gauntlet; l.Gauntlet = new List<Bout> { new Bout(FighterRoster.Rook), new Bout(FighterRoster.Wren, K), new Bout(FighterRoster.Kofi) };
                    l.UnlocksFighterId = FighterRoster.Kofi; l.RuleText = "Gauntlet: three single-round bouts. Your health carries over (30% restored between bouts)."; });
            Row(L, 2, "Switchblade", FighterRoster.Wren, "market", Boss, 2,
                "Wren has stopped smiling. \"The Ledger told me to finish you before the qualifiers. Nothing personal.\"",
                "Wren laughs, bleeding pride more than anything else. \"Fine. I'm done running their messages. Count me in.\"",
                new[] { O(Parry, 3), O(Combo, 5) },
                l => { l.Mode = LevelMode.Weapons; l.OpponentWeaponId = K; l.UnlocksFighterId = FighterRoster.Wren; });

            // ------------------------------------------------------------- Chapter 3: Qualification
            Row(L, 3, "Willow in the Rain", FighterRoster.Lin, "rooftop", Norm, 2,
                "Rain beads on the rooftop moss. Lin \"Willow\" Qiao spins a staff once, twice. \"Close the distance if you can.\"",
                "Lin bows and presents her spare staff. \"Reach is a conversation. Learn to speak it.\"",
                new[] { O(Throws, 1), O(Fast, 120) },
                l => { l.Mode = LevelMode.Weapons; l.OpponentWeaponId = St; l.UnlocksWeaponId = St; });
            Row(L, 3, "Metronome", FighterRoster.Dima, "pier9", Hard, 2,
                "Dmitri taps his wristwatch. \"Qualifier rules: hands and feet only. Keep time with me.\"",
                "\"Off-beat,\" Dima mutters, rubbing his jaw. \"Nobody's done that to me in years.\"",
                new[] { O(Perfect), O(Combo, 5) });
            Row(L, 3, "Last Breath", FighterRoster.Rook, "freightyard", Hard, 2,
                "The qualifier referee holds up a card: SUDDEN STAKES. Every round begins at forty percent health.",
                "Calloway sits on a container, wheezing. \"Remind me never to do a short-fuse round with you again.\"",
                new[] { O(Flawless), O(Throws, 1) },
                l => { l.StartHealth = 0.4f; l.StartHealthEveryRound = true; l.RuleText = "Last Breath: every round starts at 40% health."; });
            Row(L, 3, "The Smith", FighterRoster.Brandt, "forge", Hard, 2,
                "Sparks drift from the forge. Tomasz \"The Smith\" Brandt tests a blade's edge with his thumb. \"Steel tells the truth. Let's hear yours.\"",
                "Brandt hands you a sword still warm from the quench. \"Forged it this morning. It suits you better than it suits a wall.\"",
                new[] { O(Clash, 1), O(Disarm) },
                l => { l.Mode = LevelMode.Weapons; l.OpponentWeaponId = S; l.UnlocksWeaponId = S; });
            Row(L, 3, "Qualifier Final", FighterRoster.Sable, "cistern", Boss, 2,
                "Sable is back, a staff across her shoulders. \"Last time was a gate. This time it's a wall.\"",
                "\"Qualified,\" Sable says, almost proud. Lin is already waiting at the exit: \"You'll need a partner who sees the long game.\"",
                new[] { O(Special), O(Parry, 3) },
                l => { l.Mode = LevelMode.Weapons; l.OpponentWeaponId = St; l.UnlocksFighterId = FighterRoster.Lin; l.OpponentDamage = 1.05f; });

            // ------------------------------------------------------------- Chapter 4: Elite Warriors
            Row(L, 4, "Kestrel's Return", FighterRoster.Haruto, "rooftop", Hard, 2,
                "Haruto stands where his grandfather taught him, staff in hand. \"The Ledger offered to save my dojo if I stop you. Forgive me.\"",
                "Haruto kneels. \"I won't take their money. Whatever happens, this roof stays ours.\"",
                new[] { O(Flawless), O(Combo, 5) },
                l => { l.Mode = LevelMode.Weapons; l.OpponentWeaponId = St; l.OpponentDamage = 1.05f; });
            Row(L, 4, "Steel on Steel", FighterRoster.Brandt, "forge", Exp, 2,
                "Brandt draws his best blade. \"Elite bracket. No more lessons — only answers.\"",
                "\"Good steel,\" Brandt says, pointing at your heart, not your sword.",
                new[] { O(Clash, 2), O(Fast, 120) },
                l => { l.Mode = LevelMode.Weapons; l.OpponentWeaponId = S; l.OpponentDamage = 1.05f; });
            Row(L, 4, "Armory Gauntlet", FighterRoster.Lin, "cistern", Hard, 1,
                "Three elites, three weapons, one walkway over black water.",
                "The walkway is empty except for you and the echo of applause from somewhere above.",
                new[] { O(Flawless), O(Disarm) },
                l => { l.Mode = LevelMode.Gauntlet; l.Gauntlet = new List<Bout> { new Bout(FighterRoster.Lin, St), new Bout(FighterRoster.Wren, K), new Bout(FighterRoster.Brandt, S) };
                    l.OpponentDamage = 1.05f; l.RuleText = "Gauntlet: knife, staff and sword specialists back to back. Health carries over."; });
            Row(L, 4, "Exhaustion", FighterRoster.Kofi, "freightyard", Exp, 2,
                "The organisers cut the yard's power. Generators only, no water, no rest: stamina recovers at half speed.",
                "You and Kofi sit back to back on the wet concrete, too tired to talk, both laughing anyway.",
                new[] { O(Throws, 2), O(Perfect) },
                l => { l.StaminaRegen = 0.5f; l.RuleText = "Exhaustion: stamina regenerates at half speed for both fighters."; });
            Row(L, 4, "Tidebreaker", FighterRoster.Mara, "pier9", Exp, 2,
                "Mara Okonkwo blocks the pier stairs. \"Everyone keeps saying your name. Let's see if it's earned.\"",
                "Mara offers a fist bump. \"Earned. Barely.\"",
                new[] { O(Combo, 6), O(Parry, 3) }, l => { l.Mode = LevelMode.Weapons; l.OpponentDamage = 1.05f; });
            Row(L, 4, "Willow Unbroken", FighterRoster.Lin, "rooftop", Boss, 2,
                "The Ledger has bought Lin's debts. She won't meet your eyes. \"I have no choice tonight.\"",
                "Brandt arrives with a lawyer's letter that frees Lin's debts — and joins your corner himself.",
                new[] { O(Disarm), O(Special) },
                l => { l.Mode = LevelMode.Weapons; l.OpponentWeaponId = St; l.UnlocksFighterId = FighterRoster.Brandt; l.OpponentDamage = 1.1f; });

            // ------------------------------------------------------------- Chapter 5: The Undercurrent
            Row(L, 5, "Gatekeeper's Grudge", FighterRoster.Sable, "finals", Exp, 2,
                "The real Undercurrent arena. Sable walks out with a sword. \"Third time. This one counts.\"",
                "Sable sheathes the blade. \"Crane fights at the end. He's never lost. Don't let him be the first thing you can't beat.\"",
                new[] { O(Clash, 2), O(Flawless) },
                l => { l.Mode = LevelMode.Weapons; l.OpponentWeaponId = S; l.OpponentDamage = 1.1f; });
            Row(L, 5, "Iron Freight", FighterRoster.Rook, "freightyard", Exp, 2,
                "Calloway has strapped on padded armour and found a steel pipe. Your blows land at 80% power.",
                "\"Armour's for cowards,\" Calloway admits, dropping it in a puddle. \"Turns out so am I.\"",
                new[] { O(Throws, 2), O(Combo, 5) },
                l => { l.Mode = LevelMode.Weapons; l.OpponentWeaponId = St; l.PlayerDamage = 0.8f; l.RuleText = "Iron Body: your strikes deal 80% damage."; });
            Row(L, 5, "Survival", FighterRoster.Ines, "market", Hard, 1,
                "Four fighters in a row. The market crowd places bets on how far you get.",
                "The bookmakers pay out with sour faces. Somewhere above, The Ledger crosses out a line.",
                new[] { O(Special), O(Parry, 4) },
                l => { l.Mode = LevelMode.Gauntlet; l.GauntletRecovery = 0.25f; l.OpponentDamage = 1.05f;
                    l.Gauntlet = new List<Bout> { new Bout(FighterRoster.Ines), new Bout(FighterRoster.Dima), new Bout(FighterRoster.Wren, K), new Bout(FighterRoster.Kofi) };
                    l.RuleText = "Survival: four bouts, 25% health restored between bouts."; });
            Row(L, 5, "Metronome Unbound", FighterRoster.Dima, "pier9", Exp, 2,
                "Dima has broken his own rule: there's a knife in his hand. \"They own my debt. I'm sorry.\"",
                "Dima drops the knife into the bay. \"Keeping time is easier without a leash.\"",
                new[] { O(Disarm), O(Fast, 110) },
                l => { l.Mode = LevelMode.Weapons; l.OpponentWeaponId = K; l.OpponentDamage = 1.1f; });
            Row(L, 5, "Forge Trial", FighterRoster.Brandt, "forge", Exp, 2,
                "Brandt says The Ledger will hurt his apprentices if he holds back. He won't. His blows land 15% harder.",
                "\"Thank you,\" Brandt says, \"for not letting me win.\"",
                new[] { O(Clash, 2), O(Perfect) },
                l => { l.Mode = LevelMode.Weapons; l.OpponentWeaponId = S; l.OpponentDamage = 1.15f; l.RuleText = "Forge Trial: Brandt deals +15% damage."; });
            Row(L, 5, "Smith's Last Stand", FighterRoster.Brandt, "forge", Boss, 2,
                "The forge fires roar. Brandt fights with everything he has left.",
                "The finals bracket lights up. Two names remain under yours: Haruto and Wren. Then Crane.",
                new[] { O(Disarm), O(Special) },
                l => { l.Mode = LevelMode.Weapons; l.OpponentWeaponId = S; l.OpponentDamage = 1.15f; });

            // ------------------------------------------------------------- Chapter 6: Final Championship
            Row(L, 6, "Semifinal: Kestrel", FighterRoster.Haruto, "finals", Exp, 2,
                "Haruto and you meet in the light. \"No sponsors. No debts. Just us.\"",
                "\"Win it,\" Haruto says. \"For every roof they tried to buy.\"",
                new[] { O(Flawless), O(Combo, 6) },
                l => { l.Mode = LevelMode.Weapons; l.OpponentWeaponId = St; l.OpponentDamage = 1.1f; });
            Row(L, 6, "Semifinal: Switchblade", FighterRoster.Wren, "finals", Exp, 2,
                "Wren spins her knife and winks. \"Friends in the corner, enemies in the ring.\"",
                "Wren whistles. \"Go get him. And bring back The Ledger's book — I want to burn it.\"",
                new[] { O(Parry, 4), O(Disarm) },
                l => { l.Mode = LevelMode.Weapons; l.OpponentWeaponId = K; l.OpponentDamage = 1.1f; });
            Row(L, 6, "Champions' Gauntlet", FighterRoster.Mara, "finals", Exp, 1,
                "The Ledger changes the rules at the last minute: five champions, back to back, before the final.",
                "You stand alone in the ring. The lights go out. One spotlight finds Silas Crane.",
                new[] { O(Special), O(Flawless) },
                l => { l.Mode = LevelMode.Gauntlet; l.GauntletRecovery = 0.25f; l.OpponentDamage = 1.1f;
                    l.Gauntlet = new List<Bout> { new Bout(FighterRoster.Mara), new Bout(FighterRoster.Lin, St), new Bout(FighterRoster.Kofi), new Bout(FighterRoster.Sable, S), new Bout(FighterRoster.Brandt, S) };
                    l.RuleText = "Five champions back to back. 25% health restored between bouts."; });
            Row(L, 6, "The Gatekeeper's Last Stand", FighterRoster.Sable, "finals", Boss, 2,
                "Sable steps between you and Crane. \"He doesn't fight people who haven't beaten me at my best. So — my best.\"",
                "Sable lowers her sword and steps aside. \"Go.\"",
                new[] { O(Clash, 2), O(Perfect) },
                l => { l.Mode = LevelMode.Weapons; l.OpponentWeaponId = S; l.OpponentDamage = 1.15f; });
            Row(L, 6, "The Ledger's Champion", FighterRoster.Crane, "finals", Boss, 3,
                "Silas Crane draws a sword older than the city. \"Eleven years. Nobody has reached the third round. Entertain me.\"",
                "Crane kneels in the light. In his coat you find The Ledger's book — every debt, every bribe, every name. The Undercurrent ends tonight.",
                new[] { O(Disarm), O(Special) },
                l => { l.Mode = LevelMode.Weapons; l.OpponentWeaponId = S; l.OpponentDamage = 1.15f; l.RoundSeconds = 75;
                    l.RuleText = "Final: best of five rounds."; });
            return L;
        }

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

        public static LevelDefinition NextAfter(LevelDefinition level)
        {
            foreach (var l in Levels) if (l.RequiresLevelId == level.Id) return l;
            return null;
        }
    }
}
