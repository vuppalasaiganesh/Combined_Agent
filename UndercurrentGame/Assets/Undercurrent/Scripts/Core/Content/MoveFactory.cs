using Undercurrent.Core.Combat;

namespace Undercurrent.Core.Content
{
    /// <summary>
    /// Builds the shared martial-arts move archetypes. Each fighter starts from these and then tunes timing,
    /// reach and damage to express its own style. Values are frame counts at 60 ticks per second and metres.
    /// </summary>
    public static class MoveFactory
    {
        public static MoveData Strike(string id, string name, AttackButton button, MoveDirection dir,
            int startup, int active, int recovery, int damage, float stamina,
            float front, float length, float bottom, float height, AttackHeight attackHeight,
            int hitStun, int blockStun, int hitStop, float kbHit, float kbBlock)
        {
            return new MoveData
            {
                Id = id,
                DisplayName = name,
                Button = button,
                Direction = dir,
                Startup = startup,
                Active = active,
                Recovery = recovery,
                Damage = damage,
                ChipDamage = damage / 10,
                StaminaCost = stamina,
                GuardDamage = stamina * 1.4f + damage * 0.06f,
                HitFrontOffset = front,
                HitLength = length,
                HitBottom = bottom,
                HitHeight = height,
                Height = attackHeight,
                HitStun = hitStun,
                BlockStun = blockStun,
                HitStop = hitStop,
                KnockbackOnHit = kbHit,
                KnockbackOnBlock = kbBlock,
                ParriedStun = 24 + recovery / 2,
                AnimationId = id,
                ImpactSoundId = damage >= 80 ? "impact_heavy" : "impact_light",
                ImpactVfxId = damage >= 80 ? "spark_heavy" : "spark_light",
            };
        }

        public static MoveData Jab(string id = "jab") =>
            Strike(id, "Jab", AttackButton.LightPunch, MoveDirection.Neutral, 4, 2, 8, 40, 5f,
                0.2f, 0.75f, 1.3f, 0.3f, AttackHeight.High, 15, 11, 6, 0.15f, 0.25f);

        public static MoveData Cross(string id = "cross") =>
            Strike(id, "Cross", AttackButton.HeavyPunch, MoveDirection.Neutral, 8, 3, 16, 85, 10f,
                0.2f, 0.9f, 1.25f, 0.32f, AttackHeight.High, 20, 14, 9, 0.45f, 0.55f);

        public static MoveData FrontKick(string id = "front_kick") =>
            Strike(id, "Front Kick", AttackButton.LightKick, MoveDirection.Neutral, 6, 3, 12, 55, 7f,
                0.2f, 0.95f, 0.7f, 0.45f, AttackHeight.Mid, 17, 12, 7, 0.4f, 0.45f);

        public static MoveData Roundhouse(string id = "roundhouse") =>
            Strike(id, "Roundhouse", AttackButton.HeavyKick, MoveDirection.Neutral, 12, 3, 20, 110, 14f,
                0.25f, 1.05f, 1.1f, 0.5f, AttackHeight.High, 22, 16, 11, 0.7f, 0.6f);

        public static MoveData SpinningBackKick(string id = "spin_back_kick") =>
            Strike(id, "Spinning Back Kick", AttackButton.HeavyKick, MoveDirection.Forward, 16, 3, 22, 130, 18f,
                0.2f, 1.1f, 0.85f, 0.55f, AttackHeight.Mid, 26, 18, 13, 1.2f, 0.7f);

        public static MoveData CrouchJab(string id = "crouch_jab") =>
            Strike(id, "Low Jab", AttackButton.LightPunch, MoveDirection.Down, 4, 2, 9, 30, 4f,
                0.2f, 0.7f, 0.55f, 0.3f, AttackHeight.Mid, 14, 11, 5, 0.12f, 0.2f);

        public static MoveData LowKick(string id = "low_kick") =>
            Strike(id, "Low Kick", AttackButton.LightKick, MoveDirection.Down, 6, 3, 12, 45, 6f,
                0.2f, 0.95f, 0.0f, 0.4f, AttackHeight.Low, 16, 12, 7, 0.3f, 0.35f);

        public static MoveData Sweep(string id = "sweep") =>
            Strike(id, "Sweep", AttackButton.HeavyKick, MoveDirection.Down, 10, 3, 24, 80, 12f,
                0.25f, 1.05f, 0.0f, 0.35f, AttackHeight.Low, 0, 14, 10, 0.5f, 0.4f);

        public static MoveData Uppercut(string id = "uppercut") =>
            Strike(id, "Rising Uppercut", AttackButton.HeavyPunch, MoveDirection.Down, 7, 4, 20, 90, 11f,
                0.1f, 0.7f, 0.9f, 1.4f, AttackHeight.Mid, 22, 13, 10, 0.5f, 0.4f);

        public static MoveData JumpKick(string id = "jump_kick") =>
            Strike(id, "Jumping Kick", AttackButton.LightKick, MoveDirection.Air, 5, 8, 8, 60, 7f,
                0.0f, 0.85f, -0.2f, 0.9f, AttackHeight.Overhead, 17, 12, 7, 0.35f, 0.35f);

        public static MoveData JumpHeavyKick(string id = "jump_heavy_kick") =>
            Strike(id, "Flying Kick", AttackButton.HeavyKick, MoveDirection.Air, 7, 6, 10, 90, 12f,
                0.05f, 0.95f, -0.3f, 0.9f, AttackHeight.Overhead, 20, 14, 10, 0.6f, 0.45f);

        public static MoveData JumpPunch(string id = "jump_punch") =>
            Strike(id, "Jumping Hammer", AttackButton.HeavyPunch, MoveDirection.Air, 6, 5, 10, 75, 10f,
                0.05f, 0.7f, 0.3f, 0.9f, AttackHeight.Overhead, 18, 13, 9, 0.4f, 0.4f);

        public static MoveData Throw(string id = "throw", int damage = 120) =>
            new MoveData
            {
                Id = id,
                DisplayName = "Throw",
                Button = AttackButton.Throw,
                Direction = MoveDirection.Neutral,
                Startup = FighterRuntime.ThrowStartup,
                Active = FighterRuntime.ThrowActive,
                Recovery = FighterRuntime.ThrowWhiffRecovery,
                Damage = damage,
                StaminaCost = 10f,
                Height = AttackHeight.Throw,
                HitStop = 10,
                KnockbackOnHit = 1.4f,
                Parryable = false,
                AnimationId = id,
                ImpactSoundId = "throw_slam",
                ImpactVfxId = "dust",
            };

        public static MoveData Special(string id, string name, int damage, float front, float length, float bottom, float height, float advance)
        {
            var m = Strike(id, name, AttackButton.Special, MoveDirection.Neutral, 10, 5, 28, damage, 0f,
                front, length, bottom, height, AttackHeight.Mid, 0, 20, 16, 1.6f, 0.9f);
            m.MeterCost = 100f;
            m.CausesKnockdown = true;
            m.InvulnerableFrom = 1;
            m.InvulnerableTo = 10;
            m.Parryable = false;
            m.ChipDamage = damage / 6;
            m.GuardDamage = 30f;
            m.Advance = advance;
            m.ParriedStun = 40;
            m.ImpactSoundId = "impact_special";
            m.ImpactVfxId = "spark_special";
            return m;
        }

        /// <summary>Sweeps cause knockdown; called after construction so hitstun 0 is never used.</summary>
        public static MoveData Knockdown(this MoveData m)
        {
            m.CausesKnockdown = true;
            return m;
        }

        public static MoveData Cancels(this MoveData m, params string[] ids)
        {
            m.CancelInto.AddRange(ids);
            return m;
        }

        public static MoveData Tune(this MoveData m, int? startup = null, int? active = null, int? recovery = null,
            int? damage = null, float? reach = null, float? stamina = null, int? hitStun = null, int? blockStun = null)
        {
            if (startup.HasValue) m.Startup = startup.Value;
            if (active.HasValue) m.Active = active.Value;
            if (recovery.HasValue) m.Recovery = recovery.Value;
            if (damage.HasValue) { m.Damage = damage.Value; m.ChipDamage = damage.Value / 10; }
            if (reach.HasValue) m.HitLength = reach.Value;
            if (stamina.HasValue) m.StaminaCost = stamina.Value;
            if (hitStun.HasValue) m.HitStun = hitStun.Value;
            if (blockStun.HasValue) m.BlockStun = blockStun.Value;
            return m;
        }

        public static MoveData Named(this MoveData m, string displayName)
        {
            m.DisplayName = displayName;
            return m;
        }

        public static MoveData WithHurtExtension(this MoveData m, float ext)
        {
            m.HurtExtension = ext;
            return m;
        }

        public static MoveData WithAdvance(this MoveData m, float advance)
        {
            m.Advance = advance;
            return m;
        }

        public static MoveData WithArmor(this MoveData m)
        {
            m.Armored = true;
            return m;
        }
    }
}
