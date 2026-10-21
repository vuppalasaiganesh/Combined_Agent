using System;
using System.Collections.Generic;

namespace Undercurrent.Core.Combat
{
    /// <summary>
    /// Frame data for one attack. Frame numbering: frames 1..Startup are startup, the next Active frames can hit,
    /// the remaining Recovery frames are vulnerable. First active frame = Startup + 1.
    /// </summary>
    [Serializable]
    public sealed class MoveData
    {
        public string Id;
        public string DisplayName;
        public AttackButton Button;
        public MoveDirection Direction;

        // Timing (ticks at 60/s)
        public int Startup;
        public int Active;
        public int Recovery;

        // Damage & resources
        public int Damage;
        public int ChipDamage;
        public float StaminaCost;
        /// <summary>Stamina removed from a blocking defender.</summary>
        public float GuardDamage;
        /// <summary>Focus meter required (special moves only).</summary>
        public float MeterCost;

        // Hitbox, facing-relative: starts FrontOffset metres in front of the fighter's origin.
        public float HitFrontOffset;
        public float HitLength;
        public float HitBottom;
        public float HitHeight;
        /// <summary>Extra hurtbox length in front of the fighter during active+recovery (makes whiffs punishable).</summary>
        public float HurtExtension;

        public AttackHeight Height;

        // Reactions
        public int HitStun;
        public int BlockStun;
        public int HitStop;
        /// <summary>Total horizontal distance the defender is pushed on hit (metres).</summary>
        public float KnockbackOnHit;
        public float KnockbackOnBlock;
        public bool CausesKnockdown;
        /// <summary>Stun applied to this move's user when the defender parries it.</summary>
        public int ParriedStun = 30;
        public bool Parryable = true;

        /// <summary>Frames (1-based, inclusive) during which the fighter is strike-invulnerable. 0 = none.</summary>
        public int InvulnerableFrom;
        public int InvulnerableTo;
        /// <summary>If true the move is not interrupted by a light hit (damage still applies).</summary>
        public bool Armored;

        /// <summary>Forward movement applied evenly during startup+active (metres).</summary>
        public float Advance;

        // Cancels
        public List<string> CancelInto = new List<string>();
        /// <summary>Additional recovery frames after the last active frame during which a cancel is still accepted.</summary>
        public int CancelWindowRecovery = 6;

        public bool IsThrow => Height == AttackHeight.Throw;
        public bool IsSpecial => Button == AttackButton.Special;
        public bool IsAirMove => Direction == MoveDirection.Air;

        public int TotalFrames => Startup + Active + Recovery;
        public int FirstActiveFrame => Startup + 1;
        public int LastActiveFrame => Startup + Active;

        // Presentation hooks (resolved by the presentation layer)
        public string AnimationId;
        public string ImpactSoundId;
        public string ImpactVfxId;

        public MovePhase PhaseAt(int frame)
        {
            if (frame <= Startup) return MovePhase.Startup;
            if (frame <= Startup + Active) return MovePhase.Active;
            if (frame <= TotalFrames) return MovePhase.Recovery;
            return MovePhase.None;
        }

        /// <summary>
        /// Frame advantage when the move connects on its first active frame (positive = attacker acts first).
        /// A defender given N stun acts N ticks after the hit tick; the attacker acts Active + Recovery ticks after it.
        /// </summary>
        public int AdvantageOnHit => HitStun - (Active + Recovery);
        public int AdvantageOnBlock => BlockStun - (Active + Recovery);

        public MoveData Clone()
        {
            var m = (MoveData)MemberwiseClone();
            m.CancelInto = new List<string>(CancelInto);
            return m;
        }
    }
}
