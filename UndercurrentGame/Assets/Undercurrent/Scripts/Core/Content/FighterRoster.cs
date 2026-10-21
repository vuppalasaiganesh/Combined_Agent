using System.Collections.Generic;
using Undercurrent.Core.Combat;
using static Undercurrent.Core.Content.MoveFactory;

namespace Undercurrent.Core.Content
{
    /// <summary>
    /// Original fighters of the Undercurrent circuit. All names, biographies and movesets are original to this project;
    /// fighting styles draw only on general, real-world martial-arts disciplines.
    /// </summary>
    public static class FighterRoster
    {
        public const string Mara = "mara";
        public const string Dima = "dima";
        public const string Haruto = "haruto";
        public const string Ines = "ines";
        public const string Rook = "rook";
        public const string Sable = "sable";
        public const string Kofi = "kofi";
        public const string Wren = "wren";
        public const string Lin = "lin";
        public const string Brandt = "brandt";
        public const string Crane = "crane";

        private static Dictionary<string, FighterDefinition> _cache;

        public static IReadOnlyDictionary<string, FighterDefinition> All
        {
            get
            {
                if (_cache == null)
                {
                    _cache = new Dictionary<string, FighterDefinition>();
                    foreach (var f in new[] { BuildMara(), BuildDima(), BuildHaruto(), BuildInes(), BuildRook(), BuildSable(),
                                 BuildKofi(), BuildWren(), BuildLin(), BuildBrandt(), BuildCrane() })
                        _cache[f.Id] = f;
                }
                return _cache;
            }
        }

        /// <summary>Fighters the player can choose (some are unlocked through the campaign).</summary>
        public static readonly string[] PlayableIds = { Mara, Dima, Haruto, Kofi, Wren, Lin, Brandt };

        /// <summary>Returns a private copy so callers can apply loadout modifiers safely.</summary>
        public static FighterDefinition Get(string id) => All.TryGetValue(id, out var f) ? f.Clone() : null;

        private static List<MoveData> CommonGroundAndAir()
        {
            return new List<MoveData>
            {
                CrouchJab().Cancels("low_kick", "crouch_jab"),
                LowKick().Cancels("sweep"),
                Sweep().Knockdown().WithHurtExtension(0.5f),
                Uppercut().Cancels("special"),
                JumpKick(),
                JumpHeavyKick(),
                JumpPunch(),
            };
        }

        // ------------------------------------------------------------------ playable

        private static FighterDefinition BuildMara()
        {
            var f = new FighterDefinition
            {
                Id = Mara,
                DisplayName = "Mara Okonkwo",
                Nickname = "Tidebreaker",
                Style = "Muay Thai",
                Bio = "A former dockyard crane operator from Varrow Bay who learned to fight in the shipping-container gyms along Pier 9. " +
                      "She entered the Undercurrent to buy back her late mother's gym.",
                Strengths = "Punishing knees and elbows at close range; heavy kicks break guards.",
                Weaknesses = "Short punch reach; kicks are slow to recover if blocked.",
                MaxHealth = 1200,
                MaxStamina = 105f,
                StaminaRegen = 0.34f,
                WalkForwardSpeed = 0.042f,
                WalkBackSpeed = 0.033f,
                PreferredRange = 1.15f,
            };
            f.Moves.Add(Jab().Tune(reach: 0.65f).Cancels("elbow", "front_kick", "jab"));
            f.Moves.Add(Cross("elbow").Named("Slashing Elbow").Tune(startup: 7, damage: 95, reach: 0.75f).Cancels("special", "roundhouse"));
            f.Moves.Add(FrontKick("front_kick").Named("Teep").Tune(reach: 1.05f).Cancels("roundhouse"));
            f.Moves.Add(Roundhouse().Named("Shin Roundhouse").Tune(damage: 125, reach: 1.1f).WithHurtExtension(0.4f));
            f.Moves.Add(Strike("knee", "Rising Knee", AttackButton.LightKick, MoveDirection.Forward, 7, 3, 14, 75, 9f,
                0.15f, 0.6f, 0.8f, 0.6f, AttackHeight.Mid, 19, 13, 9, 0.5f, 0.4f).WithAdvance(0.3f).Cancels("elbow", "special"));
            f.Moves.Add(SpinningBackKick("spin_elbow").Named("Spinning Elbow").Tune(startup: 13, reach: 0.75f, damage: 120).Knockdown());
            f.Moves.Add(Throw("clinch_throw", 125).Named("Clinch Sweep"));
            f.Moves.Add(Special("special", "Monsoon Rush", 240, 0.1f, 0.9f, 0.5f, 1.2f, 0.9f));
            f.Moves.AddRange(CommonGroundAndAir());
            f.Combos.Add(new ComboRoute("Pier Breaker", "jab", "elbow", "roundhouse"));
            f.Combos.Add(new ComboRoute("Teep Chain", "front_kick", "roundhouse"));
            f.Combos.Add(new ComboRoute("Knee to Monsoon", "knee", "special"));
            f.Combos.Add(new ComboRoute("Low Tide", "crouch_jab", "low_kick", "sweep"));
            return f;
        }

        private static FighterDefinition BuildDima()
        {
            var f = new FighterDefinition
            {
                Id = Dima,
                DisplayName = "Dmitri Vasko",
                Nickname = "The Metronome",
                Style = "Boxing",
                Bio = "A disqualified amateur boxing champion who keeps time on a battered wristwatch. He fights in the Undercurrent " +
                      "to pay off a debt he refuses to talk about.",
                Strengths = "Fastest hands on the circuit; long, safe evasions; excellent footwork.",
                Weaknesses = "Kicks are basic and slow; lower health; weak against low attacks.",
                MaxHealth = 1140,
                MaxStamina = 100f,
                StaminaRegen = 0.38f,
                WalkForwardSpeed = 0.05f,
                WalkBackSpeed = 0.042f,
                DodgeInvulnTo = 15,
                DodgeDistance = 1.25f,
                DodgeStaminaCost = 12f,
                PreferredRange = 1.1f,
            };
            f.Moves.Add(Jab().Named("Snap Jab").Tune(startup: 3, damage: 38).Cancels("jab", "cross", "body_hook"));
            f.Moves.Add(Cross().Named("Straight Right").Tune(startup: 7, damage: 82, reach: 0.95f).Cancels("lead_hook", "special"));
            f.Moves.Add(Strike("lead_hook", "Lead Hook", AttackButton.LightPunch, MoveDirection.Forward, 6, 3, 13, 65, 8f,
                0.15f, 0.7f, 1.25f, 0.35f, AttackHeight.High, 19, 13, 9, 0.5f, 0.45f).Cancels("cross", "special"));
            f.Moves.Add(Strike("overhand", "Overhand Right", AttackButton.HeavyPunch, MoveDirection.Forward, 14, 3, 20, 120, 15f,
                0.2f, 0.85f, 1.15f, 0.55f, AttackHeight.High, 24, 15, 12, 0.9f, 0.6f).WithAdvance(0.35f));
            f.Moves.Add(Strike("body_hook", "Liver Hook", AttackButton.HeavyPunch, MoveDirection.Back, 8, 3, 15, 80, 10f,
                0.15f, 0.7f, 0.8f, 0.4f, AttackHeight.Mid, 21, 14, 10, 0.4f, 0.4f).Cancels("lead_hook", "special"));
            f.Moves.Add(FrontKick("push_kick").Named("Push Kick").Tune(startup: 9, damage: 45, recovery: 15));
            f.Moves.Add(Roundhouse("round_kick").Named("Clumsy Round Kick").Tune(startup: 15, damage: 95, recovery: 24));
            f.Moves.Add(Throw("clinch_shove", 110).Named("Clinch Shove"));
            f.Moves.Add(Special("special", "Metronome Barrage", 230, 0.1f, 0.85f, 1.0f, 0.7f, 0.7f));
            f.Moves.AddRange(CommonGroundAndAir());
            f.Combos.Add(new ComboRoute("One-Two-Hook", "jab", "cross", "lead_hook"));
            f.Combos.Add(new ComboRoute("Double Jab Body", "jab", "jab", "body_hook"));
            f.Combos.Add(new ComboRoute("Hook to Barrage", "lead_hook", "special"));
            return f;
        }

        private static FighterDefinition BuildHaruto()
        {
            var f = new FighterDefinition
            {
                Id = Haruto,
                DisplayName = "Haruto Sen",
                Nickname = "Kestrel",
                Style = "Karate / Taekwondo",
                Bio = "A quiet rooftop-garden caretaker who trained alone from his grandfather's notebooks. He joined the circuit " +
                      "after a sponsor offered to fund his family's ailing dojo — and he has started to doubt the sponsor's motives.",
                Strengths = "Longest kicks on the roster; widest parry window; strong anti-air.",
                Weaknesses = "Lower stamina pool; takes slightly more damage; weak up close once his kicks are blocked.",
                MaxHealth = 1180,
                MaxStamina = 92f,
                StaminaRegen = 0.36f,
                DamageTakenMultiplier = 1.04f,
                ParryWindow = 9,
                ParryRecovery = 16,
                WalkForwardSpeed = 0.046f,
                WalkBackSpeed = 0.04f,
                PreferredRange = 1.55f,
            };
            f.Moves.Add(Jab("reverse_punch").Named("Reverse Punch").Tune(startup: 5, damage: 45, reach: 0.8f).Cancels("lunge_punch", "side_kick", "axe_kick"));
            f.Moves.Add(Cross("lunge_punch").Named("Lunging Punch").Tune(startup: 10, damage: 85).WithAdvance(0.45f).Cancels("special"));
            f.Moves.Add(FrontKick("side_kick").Named("Side Kick").Tune(startup: 7, damage: 60, reach: 1.25f).WithHurtExtension(0.3f).Cancels("axe_kick", "special"));
            f.Moves.Add(Roundhouse("axe_kick").Named("Axe Kick").Tune(startup: 13, damage: 115, reach: 1.25f).WithHurtExtension(0.45f));
            f.Moves.Add(Strike("crescent_kick", "Crescent Kick", AttackButton.LightKick, MoveDirection.Forward, 9, 4, 14, 70, 9f,
                0.25f, 1.15f, 1.0f, 0.6f, AttackHeight.High, 19, 13, 9, 0.55f, 0.45f).Cancels("axe_kick"));
            f.Moves.Add(SpinningBackKick("spin_hook_kick").Named("Spinning Hook Kick").Tune(reach: 1.3f).Knockdown().WithHurtExtension(0.5f));
            f.Moves.Add(Throw("hip_throw", 115).Named("Hip Throw"));
            f.Moves.Add(Special("special", "Kestrel Descent", 235, 0.2f, 1.2f, 0.6f, 1.3f, 0.6f));
            f.Moves.AddRange(CommonGroundAndAir());
            f.Combos.Add(new ComboRoute("Kestrel Line", "reverse_punch", "side_kick", "axe_kick"));
            f.Combos.Add(new ComboRoute("Side Kick to Descent", "side_kick", "special"));
            f.Combos.Add(new ComboRoute("Crescent Axe", "crescent_kick", "axe_kick"));
            return f;
        }

        // ------------------------------------------------------------------ campaign opponents

        private static FighterDefinition BuildInes()
        {
            var f = new FighterDefinition
            {
                Id = Ines,
                DisplayName = "Ines Dunmore",
                Nickname = "Coach",
                Style = "Kickboxing",
                Bio = "Retired circuit veteran who now trains newcomers in a converted net-loft. Hits softer than she used to, on purpose.",
                Strengths = "Fundamentals; patient spacing.",
                Weaknesses = "Holds back in sparring.",
                MaxHealth = 1080,
                DamageDealtMultiplier = 0.8f,
                PreferredRange = 1.3f,
            };
            f.Moves.Add(Jab().Cancels("cross", "front_kick"));
            f.Moves.Add(Cross().Cancels("roundhouse"));
            f.Moves.Add(FrontKick().Cancels("roundhouse"));
            f.Moves.Add(Roundhouse());
            f.Moves.Add(SpinningBackKick().Knockdown());
            f.Moves.Add(Throw());
            f.Moves.Add(Special("special", "Old Lesson", 180, 0.1f, 1.0f, 0.6f, 1.0f, 0.5f));
            f.Moves.AddRange(CommonGroundAndAir());
            f.Combos.Add(new ComboRoute("Coach's One-Two", "jab", "cross"));
            f.Combos.Add(new ComboRoute("Kick Check", "front_kick", "roundhouse"));
            return f;
        }

        private static FighterDefinition BuildRook()
        {
            var f = new FighterDefinition
            {
                Id = Rook,
                DisplayName = "Bram Calloway",
                Nickname = "Rook",
                Style = "Catch Wrestling / Brawling",
                Bio = "Freight-yard enforcer who treats every fight like a debt collection. Owes his place on the circuit to The Ledger.",
                Strengths = "Devastating throws; armored shoulder charge; very durable.",
                Weaknesses = "Slow footwork; heavy stamina use; struggles against long kicks.",
                MaxHealth = 1380,
                MaxStamina = 110f,
                StaminaRegen = 0.3f,
                WalkForwardSpeed = 0.035f,
                WalkBackSpeed = 0.028f,
                DodgeDistance = 0.8f,
                PreferredRange = 0.9f,
            };
            f.Moves.Add(Jab("short_jab").Named("Short Jab").Tune(startup: 5, reach: 0.6f, damage: 42).Cancels("hammer_fist"));
            f.Moves.Add(Cross("hammer_fist").Named("Hammer Fist").Tune(startup: 11, damage: 105, reach: 0.75f).Cancels("special"));
            f.Moves.Add(FrontKick("stomp_kick").Named("Stomp Kick").Tune(startup: 8, damage: 62));
            f.Moves.Add(Roundhouse("big_boot").Named("Big Boot").Tune(startup: 15, damage: 120, recovery: 24));
            f.Moves.Add(Strike("shoulder_charge", "Shoulder Charge", AttackButton.HeavyPunch, MoveDirection.Forward, 16, 5, 22, 115, 18f,
                0.1f, 0.7f, 0.9f, 0.8f, AttackHeight.Mid, 0, 16, 12, 1.4f, 0.8f).WithArmor().WithAdvance(1.0f).Knockdown());
            f.Moves.Add(Throw("suplex", 175).Named("Freight Suplex"));
            f.Moves.Add(Special("special", "Collection Day", 250, 0.0f, 0.8f, 0.4f, 1.3f, 1.0f));
            f.Moves.AddRange(CommonGroundAndAir());
            f.Combos.Add(new ComboRoute("Jab Hammer", "short_jab", "hammer_fist"));
            f.Combos.Add(new ComboRoute("Hammer Collection", "hammer_fist", "special"));
            return f;
        }

        private static FighterDefinition BuildSable()
        {
            var f = new FighterDefinition
            {
                Id = Sable,
                DisplayName = "Teodora Vance",
                Nickname = "Sable",
                Style = "Mixed Martial Arts",
                Bio = "Reigning gatekeeper of the Undercurrent's first tier. Nobody reaches the qualifiers without going through her.",
                Strengths = "Complete toolkit; fast recovery; dangerous counters.",
                Weaknesses = "Commits hard to her finishing combination.",
                MaxHealth = 1380,
                MaxStamina = 110f,
                StaminaRegen = 0.38f,
                WalkForwardSpeed = 0.048f,
                WalkBackSpeed = 0.04f,
                ParryWindow = 8,
                PreferredRange = 1.25f,
            };
            f.Moves.Add(Jab().Tune(startup: 4).Cancels("cross", "front_kick", "jab"));
            f.Moves.Add(Cross().Tune(damage: 90).Cancels("roundhouse", "special"));
            f.Moves.Add(FrontKick("front_kick").Named("Oblique Kick").Cancels("roundhouse", "special"));
            f.Moves.Add(Roundhouse().Named("Head Kick").Tune(damage: 120));
            f.Moves.Add(Strike("superman_punch", "Leaping Punch", AttackButton.HeavyPunch, MoveDirection.Forward, 13, 3, 18, 105, 13f,
                0.2f, 0.85f, 1.2f, 0.45f, AttackHeight.High, 22, 15, 11, 0.8f, 0.5f).WithAdvance(0.8f).Cancels("special"));
            f.Moves.Add(SpinningBackKick().Knockdown());
            f.Moves.Add(Throw("double_leg", 140).Named("Double-Leg Slam"));
            f.Moves.Add(Special("special", "Eclipse Combination", 260, 0.1f, 1.0f, 0.5f, 1.2f, 0.8f));
            f.Moves.AddRange(CommonGroundAndAir());
            f.Combos.Add(new ComboRoute("Gatekeeper", "jab", "cross", "roundhouse"));
            f.Combos.Add(new ComboRoute("Oblique Finish", "front_kick", "special"));
            f.Combos.Add(new ComboRoute("Eclipse", "jab", "cross", "special"));
            return f;
        }

        // ------------------------------------------------------------------ chapter 2+ fighters

        private static FighterDefinition BuildKofi()
        {
            var f = new FighterDefinition
            {
                Id = Kofi, DisplayName = "Kofi Mensah", Nickname = "Anvil", Style = "Heavyweight Boxing / Clinch",
                Bio = "A former shipyard welder and amateur heavyweight from the Varrow Bay docks. Fights for the crew he lost when the yard closed.",
                Strengths = "Heaviest punches; absorbs light hits with armored hooks; strong clinch throws.",
                Weaknesses = "Slow footwork and jumps; few low attacks.",
                MaxHealth = 1340, MaxStamina = 112f, StaminaRegen = 0.32f, WalkForwardSpeed = 0.038f, WalkBackSpeed = 0.03f,
                JumpVelocity = 0.185f, PreferredRange = 1.0f,
            };
            f.Moves.Add(Jab().Named("Power Jab").Tune(startup: 5, damage: 48).Cancels("cross", "hook"));
            f.Moves.Add(Cross().Named("Straight Right").Tune(damage: 100).Cancels("special"));
            f.Moves.Add(Strike("hook", "Armored Hook", AttackButton.LightPunch, MoveDirection.Forward, 9, 3, 16, 90, 11f,
                0.15f, 0.7f, 1.2f, 0.4f, AttackHeight.High, 21, 14, 10, 0.55f, 0.45f).WithArmor().Cancels("cross", "special"));
            f.Moves.Add(Strike("body_blow", "Body Blow", AttackButton.HeavyPunch, MoveDirection.Forward, 11, 3, 18, 110, 13f,
                0.15f, 0.75f, 0.8f, 0.45f, AttackHeight.Mid, 23, 15, 12, 0.7f, 0.5f).WithAdvance(0.3f));
            f.Moves.Add(FrontKick("push_kick").Named("Push Kick").Tune(startup: 9, damage: 55));
            f.Moves.Add(Roundhouse("round_kick").Named("Heavy Round Kick").Tune(startup: 14, damage: 115, recovery: 22));
            f.Moves.Add(Throw("clinch_slam", 160).Named("Clinch Slam"));
            f.Moves.Add(Special("special", "Anvil Drop", 260, 0.1f, 0.85f, 0.8f, 0.9f, 0.6f));
            f.Moves.AddRange(CommonGroundAndAir());
            f.Combos.Add(new ComboRoute("Jab Hook", "jab", "hook", "cross"));
            f.Combos.Add(new ComboRoute("Hook to Anvil", "hook", "special"));
            return f;
        }

        private static FighterDefinition BuildWren()
        {
            var f = new FighterDefinition
            {
                Id = Wren, DisplayName = "Wren Hollis", Nickname = "Switchblade", Style = "Street Knife-Fighting / Kickboxing",
                Bio = "Grew up running messages for The Ledger's couriers. Quick, cheerful and impossible to pin down — and she always keeps one blade hidden.",
                Strengths = "Very fast; deadly counters with a knife; slippery evasions.",
                Weaknesses = "Low health; short reach unarmed; loses trades to heavy hitters.",
                MaxHealth = 1080, MaxStamina = 100f, StaminaRegen = 0.4f, WalkForwardSpeed = 0.052f, WalkBackSpeed = 0.045f,
                DodgeInvulnTo = 15, DodgeStaminaCost = 11f, PreferredRange = 1.05f, SignatureWeapon = WeaponCatalog.Knife,
            };
            f.Moves.Add(Jab().Tune(startup: 3, damage: 36).Cancels("jab", "cross", "knee_flick"));
            f.Moves.Add(Cross().Tune(startup: 7, damage: 78).Cancels("special"));
            f.Moves.Add(Strike("knee_flick", "Flick Knee", AttackButton.LightKick, MoveDirection.Forward, 6, 3, 12, 62, 7f,
                0.15f, 0.6f, 0.8f, 0.6f, AttackHeight.Mid, 18, 12, 8, 0.4f, 0.4f).WithAdvance(0.25f).Cancels("roundhouse"));
            f.Moves.Add(FrontKick().Tune(startup: 5, damage: 50).Cancels("roundhouse"));
            f.Moves.Add(Roundhouse().Tune(startup: 11, damage: 100));
            f.Moves.Add(Throw("trip", 110).Named("Ankle Trip"));
            f.Moves.Add(Special("special", "Courier's Rush", 220, 0.1f, 0.95f, 0.7f, 1.0f, 0.9f));
            f.Moves.AddRange(CommonGroundAndAir());
            f.Combos.Add(new ComboRoute("Double Jab Knee", "jab", "jab", "knee_flick"));
            f.Combos.Add(new ComboRoute("Teep Round", "front_kick", "roundhouse"));
            return f;
        }

        private static FighterDefinition BuildLin()
        {
            var f = new FighterDefinition
            {
                Id = Lin, DisplayName = "Lin Qiao", Nickname = "Willow", Style = "Wushu / Staff Arts",
                Bio = "A former touring stage performer who learned staff forms from her grandmother. She bends around attacks and answers from far away.",
                Strengths = "Longest reach with the staff; graceful evasions; strong knockdowns.",
                Weaknesses = "Average power unarmed; long recovery when her big swings miss.",
                MaxHealth = 1120, MaxStamina = 98f, StaminaRegen = 0.37f, WalkForwardSpeed = 0.047f, WalkBackSpeed = 0.042f,
                ParryWindow = 8, PreferredRange = 1.5f, SignatureWeapon = WeaponCatalog.Staff,
            };
            f.Moves.Add(Jab("palm").Named("Palm Strike").Tune(startup: 5, damage: 42).Cancels("side_kick", "cross"));
            f.Moves.Add(Cross().Named("Thrusting Palm").Tune(damage: 80).Cancels("special"));
            f.Moves.Add(FrontKick("side_kick").Named("Side Kick").Tune(reach: 1.15f).Cancels("roundhouse"));
            f.Moves.Add(Roundhouse().Named("Tornado Kick").Tune(damage: 112, reach: 1.15f).WithHurtExtension(0.4f));
            f.Moves.Add(SpinningBackKick().Named("Butterfly Kick").Knockdown());
            f.Moves.Add(Throw("redirect", 115).Named("Willow Redirect"));
            f.Moves.Add(Special("special", "Falling Petals", 225, 0.2f, 1.1f, 0.6f, 1.1f, 0.5f));
            f.Moves.AddRange(CommonGroundAndAir());
            f.Combos.Add(new ComboRoute("Palm Kick", "palm", "side_kick", "roundhouse"));
            return f;
        }

        private static FighterDefinition BuildBrandt()
        {
            var f = new FighterDefinition
            {
                Id = Brandt, DisplayName = "Tomasz Brandt", Nickname = "The Smith", Style = "Historical Fencing / Wrestling",
                Bio = "A blacksmith and fencing-club instructor who forges every blade he fights with. Calm, methodical, and very hard to rattle.",
                Strengths = "Sturdy guard; punishing heavy cuts with a sword; good grappling.",
                Weaknesses = "Slow unarmed strikes; predictable rhythm.",
                MaxHealth = 1250, MaxStamina = 108f, StaminaRegen = 0.34f, WalkForwardSpeed = 0.041f, WalkBackSpeed = 0.034f,
                GuardDamageTakenMultiplier = 0.9f, PreferredRange = 1.2f, SignatureWeapon = WeaponCatalog.Sword,
            };
            f.Moves.Add(Jab().Tune(startup: 5, damage: 44).Cancels("cross"));
            f.Moves.Add(Cross().Named("Pommel Strike").Tune(startup: 9, damage: 92).Cancels("special"));
            f.Moves.Add(FrontKick("stop_kick").Named("Stop Kick").Tune(startup: 7, damage: 58).Cancels("roundhouse"));
            f.Moves.Add(Roundhouse().Tune(startup: 13, damage: 110));
            f.Moves.Add(Throw("hip_toss", 150).Named("Hip Toss"));
            f.Moves.Add(Special("special", "Forge Hammer", 245, 0.1f, 0.9f, 0.7f, 1.1f, 0.6f));
            f.Moves.AddRange(CommonGroundAndAir());
            f.Combos.Add(new ComboRoute("Jab Pommel", "jab", "cross"));
            f.Combos.Add(new ComboRoute("Stop and Turn", "stop_kick", "roundhouse"));
            return f;
        }

        private static FighterDefinition BuildCrane()
        {
            var f = new FighterDefinition
            {
                Id = Crane, DisplayName = "Silas Crane", Nickname = "The Ledger's Champion", Style = "Mixed Weapons / Mixed Martial Arts",
                Bio = "Undefeated in eleven years of Undercurrent finals. Nobody knows who he fought before that, or why he answers to The Ledger.",
                Strengths = "No weaknesses in range or speed; reads patterns quickly.",
                Weaknesses = "Arrogant: overcommits to his finisher.",
                MaxHealth = 1450, MaxStamina = 115f, StaminaRegen = 0.4f, WalkForwardSpeed = 0.049f, WalkBackSpeed = 0.042f,
                ParryWindow = 8, PreferredRange = 1.3f, SignatureWeapon = WeaponCatalog.Sword,
            };
            f.Moves.Add(Jab().Tune(startup: 4).Cancels("cross", "front_kick", "jab"));
            f.Moves.Add(Cross().Tune(damage: 95).Cancels("roundhouse", "special"));
            f.Moves.Add(FrontKick().Cancels("roundhouse", "special"));
            f.Moves.Add(Roundhouse().Named("Head Kick").Tune(damage: 125));
            f.Moves.Add(SpinningBackKick().Knockdown());
            f.Moves.Add(Throw("champion_slam", 155).Named("Champion's Slam"));
            f.Moves.Add(Special("special", "Final Ledger", 270, 0.1f, 1.05f, 0.5f, 1.2f, 0.8f));
            f.Moves.AddRange(CommonGroundAndAir());
            f.Combos.Add(new ComboRoute("Champion", "jab", "cross", "roundhouse"));
            f.Combos.Add(new ComboRoute("Ledger Close", "front_kick", "special"));
            return f;
        }
    }
}
