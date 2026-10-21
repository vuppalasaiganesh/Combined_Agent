using System;

namespace Undercurrent.Core.Combat
{
    /// <summary>Simulation runs at a fixed tick rate. All frame data is expressed in these ticks.</summary>
    public static class SimClock
    {
        public const int TicksPerSecond = 60;
        public const float SecondsPerTick = 1f / TicksPerSecond;
    }

    [Flags]
    public enum InputButtons : ushort
    {
        None = 0,
        Left = 1 << 0,
        Right = 1 << 1,
        Up = 1 << 2,
        Down = 1 << 3,
        LightPunch = 1 << 4,
        HeavyPunch = 1 << 5,
        LightKick = 1 << 6,
        HeavyKick = 1 << 7,
        Block = 1 << 8,
        Dodge = 1 << 9,
        Parry = 1 << 10,
        Throw = 1 << 11,
        Special = 1 << 12,
    }

    /// <summary>Raw controller state for one simulation tick (held buttons, screen-relative directions).</summary>
    public struct FrameInput
    {
        public InputButtons Held;

        public FrameInput(InputButtons held) { Held = held; }

        public bool Has(InputButtons b) => (Held & b) != 0;

        public static readonly FrameInput Empty = new FrameInput(InputButtons.None);
    }

    /// <summary>Direction modifier attached to an attack trigger, relative to the fighter's facing.</summary>
    public enum MoveDirection : byte
    {
        Neutral,
        Forward,
        Back,
        Down,
        Air,
    }

    /// <summary>Which button starts a move.</summary>
    public enum AttackButton : byte
    {
        LightPunch,
        HeavyPunch,
        LightKick,
        HeavyKick,
        Throw,
        Special,
    }

    public enum AttackHeight : byte
    {
        /// <summary>Blocked standing only; naturally whiffs over crouching hurtboxes.</summary>
        High,
        /// <summary>Blocked standing or crouching.</summary>
        Mid,
        /// <summary>Must be blocked crouching.</summary>
        Low,
        /// <summary>Jumping attacks: must be blocked standing.</summary>
        Overhead,
        /// <summary>Grab: cannot be blocked or parried; can be broken (teched).</summary>
        Throw,
    }

    public enum FighterStateId : byte
    {
        Idle,
        WalkForward,
        WalkBack,
        Crouch,
        Blocking,
        CrouchBlocking,
        JumpSquat,
        Airborne,
        Landing,
        Attack,
        Dodge,
        Parry,
        HitStun,
        BlockStun,
        Parried,
        GuardBroken,
        ThrowAttempt,
        Throwing,
        Grabbed,
        KnockedDown,
        GetUp,
        KO,
        Victory,
        Intro,
    }

    public enum MovePhase : byte
    {
        None,
        Startup,
        Active,
        Recovery,
    }

    public enum CombatEventType : byte
    {
        RoundAnnounce,
        RoundStart,
        Hit,
        CounterHit,
        Blocked,
        GuardBreak,
        Parried,
        Evaded,
        Whiff,
        MoveStarted,
        ThrowConnected,
        ThrowBroken,
        Knockdown,
        GetUp,
        Jump,
        Land,
        Dodge,
        ParryAttempt,
        Exhausted,
        SpecialReady,
        SpecialStarted,
        KO,
        TimeOut,
        RoundEnd,
        MatchEnd,
        Footstep,
        BossPhaseChange,
    }

    /// <summary>
    /// Presentation-facing event emitted by the simulation. Value type stored in a reused list so the
    /// combat loop does not allocate.
    /// </summary>
    public struct CombatEvent
    {
        public CombatEventType Type;
        /// <summary>Index (0/1) of the fighter that caused the event, -1 if none.</summary>
        public int Source;
        /// <summary>Index (0/1) of the fighter affected by the event, -1 if none.</summary>
        public int Target;
        public string MoveId;
        public int Amount;
        public float X;
        public float Y;
        public bool IsHeavy;
        public bool IsFinal;
        public int Round;

        public override string ToString() => $"{Type} src={Source} tgt={Target} move={MoveId} amt={Amount}";
    }

    /// <summary>Axis-aligned box in world metres (x horizontal, y vertical from floor).</summary>
    public struct Box
    {
        public float MinX, MinY, MaxX, MaxY;

        public Box(float minX, float minY, float maxX, float maxY)
        {
            MinX = minX; MinY = minY; MaxX = maxX; MaxY = maxY;
        }

        public bool Overlaps(in Box o) => MinX < o.MaxX && o.MinX < MaxX && MinY < o.MaxY && o.MinY < MaxY;

        public float CenterX => (MinX + MaxX) * 0.5f;
        public float CenterY => (MinY + MaxY) * 0.5f;

        /// <summary>Builds a world box from a facing-relative local box (offset measured towards the opponent).</summary>
        public static Box FromLocal(float originX, float originY, int facing, float frontOffset, float bottom, float length, float height)
        {
            float a = originX + facing * frontOffset;
            float b = originX + facing * (frontOffset + length);
            return new Box(Math.Min(a, b), originY + bottom, Math.Max(a, b), originY + bottom + height);
        }
    }

    public enum RoundOutcome : byte
    {
        None,
        Player0,
        Player1,
        Draw,
    }

    public enum MatchPhase : byte
    {
        PreRound,
        Fighting,
        RoundOver,
        MatchOver,
    }
}
