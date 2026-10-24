using System;
using System.Collections.Generic;

namespace Undercurrent.Core.Combat
{
    /// <summary>Data-driven description of a fighter's body, stats and moveset.</summary>
    [Serializable]
    public sealed class FighterDefinition
    {
        public string Id;
        public string DisplayName;
        public string Nickname;
        public string Style;
        public string Bio;
        public string Strengths;
        public string Weaknesses;

        public int MaxHealth = 1000;
        public float MaxStamina = 100f;
        /// <summary>Stamina restored per tick once the regen delay has elapsed.</summary>
        public float StaminaRegen = 0.35f;
        public int StaminaRegenDelay = 30;
        public float FocusGainOnHit = 6f;
        public float FocusGainOnDamaged = 4f;
        public float DamageTakenMultiplier = 1f;
        public float DamageDealtMultiplier = 1f;
        public float GuardDamageTakenMultiplier = 1f;

        public float WalkForwardSpeed = 0.045f;
        public float WalkBackSpeed = 0.035f;
        public float JumpVelocity = 0.20f;
        public float JumpHorizontalSpeed = 0.05f;
        public float Gravity = 0.011f;
        public int JumpSquatFrames = 4;
        public int LandingFrames = 4;

        public int DodgeFrames = 22;
        public int DodgeInvulnFrom = 2;
        public int DodgeInvulnTo = 13;
        public float DodgeDistance = 1.1f;
        public float DodgeStaminaCost = 14f;

        public int ParryWindow = 7;
        public int ParryRecovery = 18;
        public float ParryStaminaCost = 8f;
        public float ParryStaminaRefund = 14f;

        public int KnockdownFrames = 40;
        public int GetUpFrames = 18;
        public int GuardBreakFrames = 50;

        // Body (metres)
        public float BodyHalfWidth = 0.28f;
        public float StandHeight = 1.78f;
        public float CrouchHeight = 1.05f;
        public float PushHalfWidth = 0.32f;

        public List<MoveData> Moves = new List<MoveData>();
        /// <summary>Named combo routes (sequences of move ids) used by the tutorial, move list and AI.</summary>
        public List<ComboRoute> Combos = new List<ComboRoute>();

        /// <summary>Preferred engagement distance for AI spacing (metres between origins).</summary>
        public float PreferredRange = 1.3f;

        public MoveData FindMove(string id)
        {
            for (int i = 0; i < Moves.Count; i++)
                if (Moves[i].Id == id) return Moves[i];
            return null;
        }

        /// <summary>Returns the move triggered by a button with the given direction, falling back to neutral.</summary>
        public MoveData Resolve(AttackButton button, MoveDirection direction)
        {
            MoveData neutral = null;
            for (int i = 0; i < Moves.Count; i++)
            {
                var m = Moves[i];
                if (m.Button != button) continue;
                if (m.Direction == direction) return m;
                if (m.Direction == MoveDirection.Neutral) neutral = m;
            }
            // Air and crouch moves never fall back to a grounded standing move
            if (direction == MoveDirection.Air) return null;
            return neutral;
        }

        public FighterDefinition Clone()
        {
            var f = (FighterDefinition)MemberwiseClone();
            f.Moves = new List<MoveData>(Moves.Count);
            foreach (var m in Moves) f.Moves.Add(m.Clone());
            f.Combos = new List<ComboRoute>(Combos);
            return f;
        }
    }

    [Serializable]
    public sealed class ComboRoute
    {
        public string Name;
        public string[] MoveIds;

        public ComboRoute() { }

        public ComboRoute(string name, params string[] moveIds)
        {
            Name = name;
            MoveIds = moveIds;
        }
    }
}
