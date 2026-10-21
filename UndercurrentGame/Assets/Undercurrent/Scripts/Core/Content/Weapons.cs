using System;
using System.Collections.Generic;
using Undercurrent.Core.Combat;
using static Undercurrent.Core.Content.MoveFactory;

namespace Undercurrent.Core.Combat
{
    public enum WeaponClass
    {
        Knife,
        Sword,
        Staff,
    }

    /// <summary>
    /// A melee weapon. Weapons replace the punch buttons (LP/HP and their directional variants) and the special;
    /// kicks and throws keep the fighter's own martial-arts style.
    /// </summary>
    [Serializable]
    public sealed class WeaponDefinition
    {
        public string Id;
        public string DisplayName;
        public WeaponClass Class;
        public string Description;
        /// <summary>Visual length in metres (presentation only; hitboxes live in the move data).</summary>
        public float Length;
        /// <summary>Extra frames added to the user's parry window.</summary>
        public int ParryWindowBonus;
        /// <summary>Multiplier on stamina lost when blocking while holding this weapon (staff/sword guards are sturdier).</summary>
        public float GuardDrainMultiplier = 1f;
        /// <summary>Extra damage on counter-hits (knives punish openings).</summary>
        public float CounterHitBonus;
        public List<MoveData> Moves = new List<MoveData>();
        public List<ComboRoute> Combos = new List<ComboRoute>();
    }
}

namespace Undercurrent.Core.Content
{
    /// <summary>
    /// The three weapon classes. Design follows broad genre conventions (reach vs speed trade-off, clashes, disarms)
    /// with original frame data: Knife = fastest and shortest, Sword = balanced and strong on block, Staff = longest
    /// reach and knockdowns but slow recovery.
    /// </summary>
    public static class WeaponCatalog
    {
        public const string Knife = "knife";
        public const string Sword = "sword";
        public const string Staff = "staff";

        /// <summary>Ticks a disarmed fighter fights unarmed before recovering the weapon (5 seconds).</summary>
        public const int DisarmTicks = 300;

        public static readonly string[] All = { Knife, Sword, Staff };

        private static Dictionary<string, WeaponDefinition> _cache;

        public static WeaponDefinition Get(string id)
        {
            if (string.IsNullOrEmpty(id)) return null;
            if (_cache == null)
            {
                _cache = new Dictionary<string, WeaponDefinition>();
                foreach (var w in new[] { BuildKnife(), BuildSword(), BuildStaff() }) _cache[w.Id] = w;
            }
            return _cache.TryGetValue(id, out var d) ? d : null;
        }

        private static MoveData Weapon(this MoveData m)
        {
            m.IsWeaponStrike = true;
            m.ChipDamage = m.Damage / 6;           // blades cut through a bare-handed guard
            m.ImpactSoundId = m.Damage >= 90 ? "blade_heavy" : "blade_light";
            m.ImpactVfxId = m.Damage >= 90 ? "spark_heavy" : "spark_light";
            return m;
        }

        private static WeaponDefinition BuildKnife()
        {
            var w = new WeaponDefinition
            {
                Id = Knife, DisplayName = "Knife", Class = WeaponClass.Knife, Length = 0.28f, CounterHitBonus = 0.12f,
                Description = "Fastest strikes and chains; punishes openings hardest. Short reach.",
            };
            w.Moves.Add(Strike("knife_jab", "Quick Thrust", AttackButton.LightPunch, MoveDirection.Neutral, 4, 2, 10, 40, 6f,
                0.2f, 0.8f, 1.25f, 0.32f, AttackHeight.High, 16, 11, 6, 0.12f, 0.22f).Weapon().Cancels("knife_slash", "knife_jab", "knife_backhand"));
            w.Moves.Add(Strike("knife_slash", "Cross Slash", AttackButton.HeavyPunch, MoveDirection.Neutral, 8, 3, 16, 80, 10f,
                0.2f, 0.95f, 1.1f, 0.5f, AttackHeight.High, 20, 13, 9, 0.35f, 0.45f).Weapon().Cancels("special", "knife_lunge"));
            w.Moves.Add(Strike("knife_backhand", "Backhand Cut", AttackButton.LightPunch, MoveDirection.Forward, 6, 3, 13, 58, 7f,
                0.2f, 0.9f, 1.15f, 0.45f, AttackHeight.High, 18, 12, 8, 0.3f, 0.35f).Weapon().Cancels("knife_slash", "special"));
            w.Moves.Add(Strike("knife_lunge", "Lunging Stab", AttackButton.HeavyPunch, MoveDirection.Forward, 12, 3, 20, 104, 13f,
                0.2f, 0.9f, 0.95f, 0.4f, AttackHeight.Mid, 22, 14, 11, 0.7f, 0.5f).Weapon().WithAdvance(0.6f));
            w.Moves.Add(Strike("knife_low", "Low Cut", AttackButton.LightPunch, MoveDirection.Down, 5, 2, 10, 40, 5f,
                0.2f, 0.85f, 0.05f, 0.4f, AttackHeight.Low, 15, 11, 6, 0.15f, 0.25f).Weapon().Cancels("low_kick", "knife_rising"));
            w.Moves.Add(Strike("knife_rising", "Rising Cut", AttackButton.HeavyPunch, MoveDirection.Down, 7, 4, 19, 84, 10f,
                0.1f, 0.8f, 0.9f, 1.4f, AttackHeight.Mid, 21, 13, 10, 0.45f, 0.4f).Weapon().Cancels("special"));
            w.Moves.Add(Special("special", "Thousand Cuts", 230, 0.1f, 1.0f, 0.6f, 1.1f, 0.8f).Weapon());
            w.Combos.Add(new ComboRoute("Switchblade Chain", "knife_jab", "knife_jab", "knife_slash"));
            w.Combos.Add(new ComboRoute("Backhand Finish", "knife_backhand", "knife_slash", "special"));
            w.Combos.Add(new ComboRoute("Low to High", "knife_low", "knife_rising"));
            return w;
        }

        private static WeaponDefinition BuildSword()
        {
            var w = new WeaponDefinition
            {
                Id = Sword, DisplayName = "Sword", Class = WeaponClass.Sword, Length = 0.95f, ParryWindowBonus = 2, GuardDrainMultiplier = 0.8f,
                Description = "Balanced reach and power; heavy cuts break guards; the crossguard makes parries easier.",
            };
            w.Moves.Add(Strike("sword_thrust", "Thrust", AttackButton.LightPunch, MoveDirection.Neutral, 5, 3, 9, 64, 6f,
                0.25f, 1.2f, 1.2f, 0.32f, AttackHeight.High, 19, 13, 7, 0.25f, 0.35f).Weapon().Cancels("sword_slash", "sword_riposte"));
            w.Moves.Add(Strike("sword_slash", "Horizontal Cut", AttackButton.HeavyPunch, MoveDirection.Neutral, 9, 4, 15, 130, 11f,
                0.2f, 1.2f, 0.85f, 0.7f, AttackHeight.Mid, 22, 15, 11, 0.6f, 0.55f).Weapon().Cancels("special").WithHurtExtension(0.3f));
            w.Moves.Add(Strike("sword_riposte", "Riposte", AttackButton.LightPunch, MoveDirection.Forward, 6, 2, 10, 76, 7f,
                0.25f, 1.15f, 1.15f, 0.4f, AttackHeight.High, 19, 12, 8, 0.35f, 0.4f).Weapon().Cancels("sword_slash"));
            w.Moves.Add(Strike("sword_overhead", "Overhead Cleave", AttackButton.HeavyPunch, MoveDirection.Forward, 14, 3, 20, 150, 15f,
                0.2f, 1.15f, 0.9f, 1.0f, AttackHeight.Overhead, 0, 18, 14, 1.2f, 0.7f).Weapon().Knockdown().WithHurtExtension(0.4f));
            w.Moves.Add(Strike("sword_low", "Leg Cut", AttackButton.LightPunch, MoveDirection.Down, 7, 3, 12, 58, 7f,
                0.25f, 1.1f, 0.05f, 0.4f, AttackHeight.Low, 17, 12, 7, 0.3f, 0.3f).Weapon().Cancels("sword_rising"));
            w.Moves.Add(Strike("sword_rising", "Rising Cut", AttackButton.HeavyPunch, MoveDirection.Down, 8, 4, 20, 100, 12f,
                0.1f, 1.0f, 0.9f, 1.5f, AttackHeight.Mid, 22, 14, 11, 0.55f, 0.45f).Weapon().Cancels("special"));
            foreach (var m in w.Moves) m.GuardDamage *= 1.35f; // heavy steel drains a guard faster
            w.Moves.Add(Special("special", "Crescent Cleave", 250, 0.2f, 1.25f, 0.5f, 1.3f, 0.6f).Weapon());
            w.Combos.Add(new ComboRoute("Thrust and Cut", "sword_thrust", "sword_slash"));
            w.Combos.Add(new ComboRoute("Riposte Cut", "sword_riposte", "sword_slash", "special"));
            w.Combos.Add(new ComboRoute("Leg to Rise", "sword_low", "sword_rising"));
            return w;
        }

        private static WeaponDefinition BuildStaff()
        {
            var w = new WeaponDefinition
            {
                Id = Staff, DisplayName = "Staff", Class = WeaponClass.Staff, Length = 1.7f, GuardDrainMultiplier = 0.85f,
                Description = "Longest reach; sweeps and vaulting strikes knock down. Slow to recover when whiffed.",
            };
            w.Moves.Add(Strike("staff_poke", "Long Poke", AttackButton.LightPunch, MoveDirection.Neutral, 9, 3, 17, 42, 8f,
                0.3f, 1.25f, 1.0f, 0.4f, AttackHeight.Mid, 17, 12, 7, 0.35f, 0.4f).Weapon().Cancels("staff_swing", "staff_poke"));
            w.Moves.Add(Strike("staff_swing", "Wide Swing", AttackButton.HeavyPunch, MoveDirection.Neutral, 15, 4, 26, 92, 15f,
                0.2f, 1.3f, 1.0f, 0.7f, AttackHeight.High, 22, 15, 11, 0.9f, 0.6f).Weapon().Cancels("special").WithHurtExtension(0.5f));
            w.Moves.Add(Strike("staff_rising", "Rising Butt", AttackButton.LightPunch, MoveDirection.Forward, 9, 3, 17, 62, 9f,
                0.2f, 1.1f, 0.8f, 0.8f, AttackHeight.Mid, 19, 13, 8, 0.4f, 0.4f).Weapon().Cancels("staff_swing"));
            w.Moves.Add(Strike("staff_vault", "Vaulting Crash", AttackButton.HeavyPunch, MoveDirection.Forward, 18, 3, 28, 115, 18f,
                0.2f, 1.3f, 0.9f, 1.1f, AttackHeight.Overhead, 0, 17, 14, 1.3f, 0.7f).Weapon().Knockdown().WithAdvance(0.4f));
            w.Moves.Add(Strike("staff_low", "Shin Rap", AttackButton.LightPunch, MoveDirection.Down, 9, 3, 17, 38, 8f,
                0.3f, 1.2f, 0.0f, 0.35f, AttackHeight.Low, 16, 12, 7, 0.3f, 0.3f).Weapon().Cancels("staff_sweep"));
            w.Moves.Add(Strike("staff_sweep", "Floor Sweep", AttackButton.HeavyPunch, MoveDirection.Down, 14, 4, 28, 78, 14f,
                0.2f, 1.35f, 0.0f, 0.4f, AttackHeight.Low, 0, 14, 10, 0.6f, 0.4f).Weapon().Knockdown().WithHurtExtension(0.6f));
            w.Moves.Add(Special("special", "Whirlwind", 240, 0.0f, 1.5f, 0.3f, 1.5f, 0.3f).Weapon());
            w.Combos.Add(new ComboRoute("Poke and Swing", "staff_poke", "staff_swing"));
            w.Combos.Add(new ComboRoute("Shin to Sweep", "staff_low", "staff_sweep"));
            w.Combos.Add(new ComboRoute("Rising Swing", "staff_rising", "staff_swing", "special"));
            return w;
        }
    }

    /// <summary>Combines a fighter with a weapon: weapon moves replace punch-button moves and the special.</summary>
    public static class WeaponLoadout
    {
        public static FighterDefinition Apply(FighterDefinition unarmed, WeaponDefinition weapon)
        {
            if (unarmed == null) throw new ArgumentNullException(nameof(unarmed));
            if (weapon == null) return unarmed;
            var armed = unarmed.Clone();
            armed.Moves.RemoveAll(m => m.Button == AttackButton.LightPunch || m.Button == AttackButton.HeavyPunch || m.Button == AttackButton.Special);
            foreach (var m in weapon.Moves) armed.Moves.Add(m.Clone());
            // Kicks that chained into removed punches now chain into the weapon equivalents.
            var ids = new HashSet<string>();
            foreach (var m in armed.Moves) ids.Add(m.Id);
            foreach (var m in armed.Moves)
                m.CancelInto.RemoveAll(c => !ids.Contains(c));
            armed.Combos = new List<ComboRoute>(weapon.Combos);
            foreach (var c in unarmed.Combos)
            {
                bool valid = true;
                foreach (var id in c.MoveIds) valid &= ids.Contains(id);
                if (valid) armed.Combos.Add(c);
            }
            armed.ParryWindow += weapon.ParryWindowBonus;
            armed.GuardDamageTakenMultiplier *= weapon.GuardDrainMultiplier;
            armed.PreferredRange = Math.Max(armed.PreferredRange, 0.6f + MaxReach(weapon) * 0.75f);
            armed.Weapon = weapon;
            armed.Unarmed = unarmed;
            return armed;
        }

        private static float MaxReach(WeaponDefinition w)
        {
            float r = 0f;
            foreach (var m in w.Moves) if (!m.IsSpecial) r = Math.Max(r, m.HitFrontOffset + m.HitLength);
            return r;
        }
    }
}
