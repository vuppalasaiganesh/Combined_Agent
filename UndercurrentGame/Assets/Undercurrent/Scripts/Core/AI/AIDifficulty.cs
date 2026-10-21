using System;

namespace Undercurrent.Core.AI
{
    public enum AIDifficulty
    {
        Beginner,
        Easy,
        Normal,
        Hard,
        Expert,
        Boss,
    }

    /// <summary>
    /// Tunables for one difficulty. Probabilities are rolled once per decision, never per frame, so higher difficulty
    /// means better choices and faster (but still human-range) reactions — never extra information.
    /// </summary>
    [Serializable]
    public sealed class AIProfile
    {
        /// <summary>Ticks between something happening on screen and the AI being able to see it.</summary>
        public int ReactionTicks;
        /// <summary>Ticks between neutral-game decisions.</summary>
        public int DecisionInterval;
        public float BlockChance;
        public float ParryChance;
        public float DodgeChance;
        public float PunishChance;
        public float ComboContinueChance;
        public float Aggression;
        public float AntiAirChance;
        public float ThrowTechChance;
        public float SpecialUsage;
        public float MistakeChance;
        /// <summary>Below this stamina fraction the AI plays defensively to recover.</summary>
        public float LowStaminaThreshold;

        public AIProfile Clone() => (AIProfile)MemberwiseClone();

        public static AIProfile For(AIDifficulty d)
        {
            switch (d)
            {
                case AIDifficulty.Beginner:
                    return new AIProfile { ReactionTicks = 32, DecisionInterval = 26, BlockChance = 0.12f, ParryChance = 0f, DodgeChance = 0.03f,
                        PunishChance = 0.1f, ComboContinueChance = 0.2f, Aggression = 0.22f, AntiAirChance = 0.05f, ThrowTechChance = 0f,
                        SpecialUsage = 0.2f, MistakeChance = 0.12f, LowStaminaThreshold = 0.15f };
                case AIDifficulty.Easy:
                    return new AIProfile { ReactionTicks = 26, DecisionInterval = 20, BlockChance = 0.3f, ParryChance = 0.03f, DodgeChance = 0.06f,
                        PunishChance = 0.25f, ComboContinueChance = 0.4f, Aggression = 0.35f, AntiAirChance = 0.15f, ThrowTechChance = 0.1f,
                        SpecialUsage = 0.35f, MistakeChance = 0.08f, LowStaminaThreshold = 0.2f };
                case AIDifficulty.Normal:
                    return new AIProfile { ReactionTicks = 20, DecisionInterval = 15, BlockChance = 0.5f, ParryChance = 0.08f, DodgeChance = 0.08f,
                        PunishChance = 0.45f, ComboContinueChance = 0.6f, Aggression = 0.45f, AntiAirChance = 0.35f, ThrowTechChance = 0.25f,
                        SpecialUsage = 0.5f, MistakeChance = 0.05f, LowStaminaThreshold = 0.25f };
                case AIDifficulty.Hard:
                    return new AIProfile { ReactionTicks = 16, DecisionInterval = 12, BlockChance = 0.65f, ParryChance = 0.15f, DodgeChance = 0.1f,
                        PunishChance = 0.65f, ComboContinueChance = 0.8f, Aggression = 0.55f, AntiAirChance = 0.55f, ThrowTechChance = 0.4f,
                        SpecialUsage = 0.65f, MistakeChance = 0.03f, LowStaminaThreshold = 0.25f };
                case AIDifficulty.Expert:
                    return new AIProfile { ReactionTicks = 13, DecisionInterval = 10, BlockChance = 0.75f, ParryChance = 0.22f, DodgeChance = 0.12f,
                        PunishChance = 0.85f, ComboContinueChance = 0.92f, Aggression = 0.6f, AntiAirChance = 0.7f, ThrowTechChance = 0.55f,
                        SpecialUsage = 0.8f, MistakeChance = 0.015f, LowStaminaThreshold = 0.3f };
                case AIDifficulty.Boss:
                default:
                    return new AIProfile { ReactionTicks = 17, DecisionInterval = 12, BlockChance = 0.6f, ParryChance = 0.14f, DodgeChance = 0.1f,
                        PunishChance = 0.6f, ComboContinueChance = 0.8f, Aggression = 0.55f, AntiAirChance = 0.5f, ThrowTechChance = 0.35f,
                        SpecialUsage = 0.7f, MistakeChance = 0.03f, LowStaminaThreshold = 0.25f };
            }
        }

        /// <summary>Second boss phase (below half health): faster and more aggressive, still bounded by human-range reactions.</summary>
        public AIProfile BossPhaseTwo()
        {
            var p = Clone();
            p.ReactionTicks = Math.Max(12, ReactionTicks - 4);
            p.DecisionInterval = Math.Max(8, DecisionInterval - 3);
            p.Aggression = Math.Min(0.85f, Aggression + 0.2f);
            p.ComboContinueChance = Math.Min(0.95f, ComboContinueChance + 0.1f);
            p.SpecialUsage = Math.Min(1f, SpecialUsage + 0.2f);
            p.ParryChance = Math.Min(0.3f, ParryChance + 0.08f);
            return p;
        }
    }

    /// <summary>Small deterministic PRNG (xorshift32) so AI decisions replay identically from a seed.</summary>
    public sealed class DeterministicRandom
    {
        private uint _state;

        public DeterministicRandom(uint seed) { _state = seed == 0 ? 0x9E3779B9u : seed; }

        public uint NextUInt()
        {
            uint x = _state;
            x ^= x << 13;
            x ^= x >> 17;
            x ^= x << 5;
            _state = x;
            return x;
        }

        /// <summary>Uniform value in [0, 1).</summary>
        public float NextFloat() => (NextUInt() >> 8) * (1f / 16777216f);

        public bool Chance(float p) => p > 0f && NextFloat() < p;

        public int Range(int minInclusive, int maxExclusive) =>
            maxExclusive <= minInclusive ? minInclusive : minInclusive + (int)(NextUInt() % (uint)(maxExclusive - minInclusive));
    }
}
