using System;

namespace Undercurrent.Core.Combat
{
    /// <summary>Per-match statistics used for rewards, objectives and achievements.</summary>
    public sealed class FighterMatchStats
    {
        public int HitsLanded;
        public int CounterHits;
        public int AttacksBlocked;
        public int Parries;
        public int Evades;
        public int ThrowsLanded;
        public int ThrowBreaks;
        public int MaxCombo;
        public int DamageDealt;
        public int DamageTaken;
        public int RoundsWon;
        public int PerfectRounds;
        public int SpecialsLanded;
        public int Knockdowns;
        public int Clashes;
        public int Disarms;

        public void Reset()
        {
            HitsLanded = CounterHits = AttacksBlocked = Parries = Evades = ThrowsLanded = ThrowBreaks = 0;
            MaxCombo = DamageDealt = DamageTaken = RoundsWon = PerfectRounds = SpecialsLanded = Knockdowns = 0;
            Clashes = Disarms = 0;
        }
    }

    /// <summary>
    /// Mutable runtime state of one fighter. Owned and stepped by <see cref="FightSimulation"/>;
    /// presentation and AI read it but never write it.
    /// </summary>
    public sealed class FighterRuntime
    {
        public const int InputBufferFrames = 6;
        public const int ThrowStartup = 5;
        public const int ThrowActive = 2;
        public const int ThrowWhiffRecovery = 24;
        public const int ThrowTechWindow = 12;
        public const int ThrowFollowThrough = 18;
        public const float FocusMax = 100f;
        public const int ComboKnockdownCap = 7;

        public readonly int Index;
        public FighterDefinition Def { get; private set; }
        private readonly FighterDefinition _armedDef;
        private readonly FighterDefinition _unarmedDef;
        /// <summary>The weapon this fighter brought (null = unarmed fighter).</summary>
        public WeaponDefinition Weapon => _armedDef.Weapon;
        public bool IsArmed => _armedDef.Weapon != null && ReferenceEquals(Def, _armedDef);
        /// <summary>Ticks until a disarmed fighter recovers their weapon.</summary>
        public int DisarmedTicks { get; private set; }

        // Kinematics
        public float X, Y, VX, VY;
        public float SlideVelocity;
        public int Facing = 1;

        // Resources
        public int Health;
        public float Stamina;
        public float Focus;

        // State machine
        public FighterStateId State { get; private set; } = FighterStateId.Intro;
        public int StateFrame { get; private set; }
        public int StunRemaining { get; internal set; }
        public MoveData CurrentMove { get; private set; }
        public int MoveFrame { get; private set; }
        public bool MoveConnected { get; internal set; }
        public bool MoveHit { get; internal set; }
        internal bool EvadeReported;
        internal bool WhiffReported;
        public bool AirAttackUsed { get; private set; }
        /// <summary>Increments every time a new attack starts; lets observers tell attacks apart.</summary>
        public int AttackSerial { get; private set; }
        internal int ThrowResolveFrame;
        internal bool ThrowDamageApplied;
        internal MoveData PendingThrowMove;
        /// <summary>The grab being attempted (visible wind-up), for observers.</summary>
        public MoveData PendingThrowMoveVisible => PendingThrowMove;
        internal int JumpDirection;

        // Combo bookkeeping (as defender)
        public int ComboCount { get; internal set; }
        public int ComboDamage { get; internal set; }

        // Input
        public FrameInput CurrentInput { get; private set; }
        private InputButtons _previousHeld;
        private readonly int[] _buffer = new int[16];
        internal int StaminaIdleFrames;
        private int _lastExhaustedTick = -1000;
        private int _footstepCounter;
        public bool SpecialReadyAnnounced { get; internal set; }

        public readonly FighterMatchStats Stats = new FighterMatchStats();
        public int RoundsWon;
        public int HealthAtRoundStart { get; internal set; }

        public FighterRuntime(int index, FighterDefinition def)
        {
            Index = index;
            Def = def ?? throw new ArgumentNullException(nameof(def));
            _armedDef = def;
            _unarmedDef = def.Weapon != null ? (def.Unarmed ?? def) : def;
        }

        public float HealthFraction => Def.MaxHealth <= 0 ? 0f : (float)Health / Def.MaxHealth;
        public float StaminaFraction => Def.MaxStamina <= 0 ? 0f : Stamina / Def.MaxStamina;
        public bool IsGrounded => Y <= 0f && State != FighterStateId.Airborne && !(State == FighterStateId.Attack && CurrentMove != null && CurrentMove.IsAirMove);
        public bool IsCrouching => State == FighterStateId.Crouch || State == FighterStateId.CrouchBlocking ||
                                   (State == FighterStateId.Attack && CurrentMove != null && CurrentMove.Direction == MoveDirection.Down) ||
                                   (State == FighterStateId.BlockStun && CurrentInput.Has(InputButtons.Down));
        public bool IsAirborne => State == FighterStateId.Airborne || (State == FighterStateId.Attack && CurrentMove != null && CurrentMove.IsAirMove);
        public MovePhase CurrentPhase => State == FighterStateId.Attack && CurrentMove != null ? CurrentMove.PhaseAt(MoveFrame) : MovePhase.None;
        public bool IsKO => State == FighterStateId.KO;

        /// <summary>True when the fighter can start a new action this tick.</summary>
        public bool IsActionable => State == FighterStateId.Idle || State == FighterStateId.WalkForward ||
                                    State == FighterStateId.WalkBack || State == FighterStateId.Crouch ||
                                    State == FighterStateId.Blocking || State == FighterStateId.CrouchBlocking;

        public bool IsStrikeInvulnerable
        {
            get
            {
                switch (State)
                {
                    case FighterStateId.KnockedDown:
                    case FighterStateId.GetUp:
                    case FighterStateId.KO:
                    case FighterStateId.Intro:
                    case FighterStateId.Victory:
                    case FighterStateId.Grabbed:
                    case FighterStateId.Throwing:
                        return true;
                    case FighterStateId.Dodge:
                        return StateFrame >= Def.DodgeInvulnFrom && StateFrame <= Def.DodgeInvulnTo;
                    case FighterStateId.Attack:
                        return CurrentMove.InvulnerableFrom > 0 && MoveFrame >= CurrentMove.InvulnerableFrom && MoveFrame <= CurrentMove.InvulnerableTo;
                    default:
                        return false;
                }
            }
        }

        public bool IsThrowable
        {
            get
            {
                if (!IsGrounded || IsStrikeInvulnerable) return false;
                switch (State)
                {
                    case FighterStateId.HitStun:
                    case FighterStateId.BlockStun:
                    case FighterStateId.Landing:
                    case FighterStateId.Dodge: // whole dodge is throw-invulnerable: it is an evasive commitment
                        return false;
                    default:
                        return true;
                }
            }
        }

        public bool IsParryActive => State == FighterStateId.Parry && StateFrame <= Def.ParryWindow;

        public bool IsBlockingStance(out bool crouching)
        {
            crouching = false;
            if (State == FighterStateId.Blocking) return true;
            if (State == FighterStateId.CrouchBlocking) { crouching = true; return true; }
            if (State == FighterStateId.BlockStun && CurrentInput.Has(InputButtons.Block))
            {
                crouching = CurrentInput.Has(InputButtons.Down);
                return true;
            }
            return false;
        }

        public Box GetHurtbox()
        {
            float hw = Def.BodyHalfWidth;
            if (State == FighterStateId.KnockedDown || State == FighterStateId.KO)
                return new Box(X - 0.8f, 0f, X + 0.8f, 0.35f);
            if (IsAirborne)
                return new Box(X - hw, Y + 0.25f, X + hw, Y + 1.45f);

            float h = IsCrouching ? Def.CrouchHeight : Def.StandHeight;
            var box = new Box(X - hw, Y, X + hw, Y + h);
            if (State == FighterStateId.Attack && CurrentMove.HurtExtension > 0f && MoveFrame >= CurrentMove.FirstActiveFrame)
            {
                if (Facing > 0) box.MaxX += CurrentMove.HurtExtension;
                else box.MinX -= CurrentMove.HurtExtension;
            }
            return box;
        }

        public Box GetActiveHitbox()
        {
            var m = CurrentMove;
            return Box.FromLocal(X, Y, Facing, m.HitFrontOffset, m.HitBottom, m.HitLength, m.HitHeight);
        }

        // ---------------------------------------------------------------- lifecycle

        internal void ResetForRound(float x, int facing, float startHealthFraction = 1f)
        {
            X = x; Y = 0f; VX = VY = 0f; SlideVelocity = 0f;
            Facing = facing;
            Def = _armedDef;
            DisarmedTicks = 0;
            Health = Math.Max(1, (int)Math.Round(Def.MaxHealth * Math.Max(0.01f, Math.Min(1f, startHealthFraction))));
            HealthAtRoundStart = Health;
            Stamina = Def.MaxStamina;
            CurrentMove = null;
            MoveFrame = 0;
            ComboCount = 0;
            ComboDamage = 0;
            StunRemaining = 0;
            AirAttackUsed = false;
            Array.Clear(_buffer, 0, _buffer.Length);
            _previousHeld = InputButtons.None;
            CurrentInput = FrameInput.Empty;
            SetState(FighterStateId.Intro);
        }

        internal void SetState(FighterStateId state)
        {
            State = state;
            StateFrame = 1;
            if (state != FighterStateId.Attack)
            {
                CurrentMove = null;
                MoveFrame = 0;
            }
        }

        // ---------------------------------------------------------------- input

        internal void FeedInput(FrameInput input, bool decayBuffer)
        {
            if (decayBuffer)
            {
                for (int i = 0; i < _buffer.Length; i++)
                    if (_buffer[i] > 0) _buffer[i]--;
            }

            var pressed = input.Held & ~_previousHeld;
            for (int bit = 0; bit < 16; bit++)
            {
                if (((int)pressed & (1 << bit)) != 0) _buffer[bit] = InputBufferFrames;
            }
            _previousHeld = input.Held;
            CurrentInput = input;
        }

        internal void ClearBuffer() => Array.Clear(_buffer, 0, _buffer.Length);

        private static int BitIndex(InputButtons b)
        {
            int v = (int)b;
            int i = 0;
            while (v > 1) { v >>= 1; i++; }
            return i;
        }

        public bool IsBuffered(InputButtons b) => _buffer[BitIndex(b)] > 0;

        internal bool ConsumeBuffered(InputButtons b)
        {
            int i = BitIndex(b);
            if (_buffer[i] <= 0) return false;
            _buffer[i] = 0;
            return true;
        }

        public bool HoldingForward => CurrentInput.Has(Facing > 0 ? InputButtons.Right : InputButtons.Left);
        public bool HoldingBack => CurrentInput.Has(Facing > 0 ? InputButtons.Left : InputButtons.Right);

        private MoveDirection CurrentDirection()
        {
            if (IsAirborne) return MoveDirection.Air;
            if (CurrentInput.Has(InputButtons.Down)) return MoveDirection.Down;
            if (HoldingForward) return MoveDirection.Forward;
            if (HoldingBack) return MoveDirection.Back;
            return MoveDirection.Neutral;
        }

        // ---------------------------------------------------------------- state update

        /// <summary>Drops the weapon (only while stunned or knocked down, never mid-attack).</summary>
        internal bool Disarm(FightSimulation sim, int ticks)
        {
            if (!IsArmed || State == FighterStateId.Attack) return false;
            Def = _unarmedDef;
            sim.Opponent(this).Stats.Disarms++;
            DisarmedTicks = ticks;
            sim.Emit(CombatEventType.Disarmed, -1, Index, Weapon.Id, ticks, X, 1f);
            return true;
        }

        internal void Tick(FightSimulation sim)
        {
            if (DisarmedTicks > 0)
            {
                if (DisarmedTicks > 1) DisarmedTicks--;
                else if (IsActionable) // timer done: pick the weapon back up once free to act
                {
                    DisarmedTicks = 0;
                    Def = _armedDef;
                    sim.Emit(CombatEventType.Rearmed, Index, -1, Weapon.Id, 0, X, 1f);
                }
            }
            StateFrame++;
            switch (State)
            {
                case FighterStateId.Idle:
                case FighterStateId.WalkForward:
                case FighterStateId.WalkBack:
                case FighterStateId.Crouch:
                case FighterStateId.Blocking:
                case FighterStateId.CrouchBlocking:
                    HandleNeutral(sim);
                    break;

                case FighterStateId.JumpSquat:
                    if (StateFrame > Def.JumpSquatFrames)
                    {
                        SetState(FighterStateId.Airborne);
                        VY = Def.JumpVelocity;
                        VX = JumpDirection * Def.JumpHorizontalSpeed;
                        sim.Emit(CombatEventType.Jump, Index, -1, null, 0, X, Y);
                    }
                    break;

                case FighterStateId.Airborne:
                    TryAirAttack(sim);
                    break;

                case FighterStateId.Landing:
                    if (StateFrame > Def.LandingFrames) BecomeNeutral(sim);
                    break;

                case FighterStateId.Attack:
                    TickAttack(sim);
                    break;

                case FighterStateId.Dodge:
                    if (StateFrame > Def.DodgeFrames) BecomeNeutral(sim);
                    break;

                case FighterStateId.Parry:
                    if (StateFrame > Def.ParryWindow + Def.ParryRecovery) BecomeNeutral(sim);
                    break;

                case FighterStateId.HitStun:
                case FighterStateId.BlockStun:
                case FighterStateId.Parried:
                case FighterStateId.GuardBroken:
                    StunRemaining--;
                    if (StunRemaining <= 0)
                    {
                        if (State == FighterStateId.HitStun) { ComboCount = 0; ComboDamage = 0; }
                        BecomeNeutral(sim);
                    }
                    break;

                case FighterStateId.ThrowAttempt:
                    if (StateFrame > ThrowStartup + ThrowActive + ThrowWhiffRecovery) BecomeNeutral(sim);
                    break;

                case FighterStateId.Throwing:
                case FighterStateId.Grabbed:
                    // Resolved by the simulation (tech window and damage).
                    break;

                case FighterStateId.KnockedDown:
                    if (StateFrame > Def.KnockdownFrames)
                    {
                        SetState(FighterStateId.GetUp);
                        sim.Emit(CombatEventType.GetUp, Index, -1, null, 0, X, Y);
                    }
                    break;

                case FighterStateId.GetUp:
                    if (StateFrame > Def.GetUpFrames)
                    {
                        ComboCount = 0;
                        ComboDamage = 0;
                        BecomeNeutral(sim);
                    }
                    break;
            }

            if (State == FighterStateId.WalkForward || State == FighterStateId.WalkBack)
            {
                if (++_footstepCounter >= 18)
                {
                    _footstepCounter = 0;
                    sim.Emit(CombatEventType.Footstep, Index, -1, null, 0, X, Y);
                }
            }
            else _footstepCounter = 9;
        }

        private void BecomeNeutral(FightSimulation sim)
        {
            SetState(FighterStateId.Idle);
            AirAttackUsed = false;
            HandleNeutral(sim);
        }

        private void HandleNeutral(FightSimulation sim)
        {
            // Face the opponent whenever we are free to act.
            var other = sim.Opponent(this);
            int desired = other.X >= X ? 1 : -1;
            if (Math.Abs(other.X - X) > 0.01f) Facing = desired;

            if (ConsumeBuffered(InputButtons.Special))
            {
                if (TryStartMove(sim, AttackButton.Special, MoveDirection.Neutral)) return;
            }

            if (ConsumeBuffered(InputButtons.Throw))
            {
                StartThrow(sim);
                return;
            }

            if (TryBufferedAttack(sim, CurrentDirection(), null)) return;

            if (ConsumeBuffered(InputButtons.Parry))
            {
                if (SpendStamina(sim, Def.ParryStaminaCost))
                {
                    SetState(FighterStateId.Parry);
                    sim.Emit(CombatEventType.ParryAttempt, Index, -1, null, 0, X, Y);
                    return;
                }
            }

            if (ConsumeBuffered(InputButtons.Dodge))
            {
                if (SpendStamina(sim, Def.DodgeStaminaCost))
                {
                    SetState(FighterStateId.Dodge);
                    sim.Emit(CombatEventType.Dodge, Index, -1, null, 0, X, Y);
                    return;
                }
            }

            if (CurrentInput.Has(InputButtons.Up))
            {
                JumpDirection = HoldingForward ? Facing : HoldingBack ? -Facing : 0;
                SetState(FighterStateId.JumpSquat);
                return;
            }

            bool down = CurrentInput.Has(InputButtons.Down);
            FighterStateId next;
            if (CurrentInput.Has(InputButtons.Block)) next = down ? FighterStateId.CrouchBlocking : FighterStateId.Blocking;
            else if (down) next = FighterStateId.Crouch;
            else if (HoldingForward) next = FighterStateId.WalkForward;
            else if (HoldingBack) next = FighterStateId.WalkBack;
            else next = FighterStateId.Idle;

            if (next != State)
            {
                State = next;
                StateFrame = 1;
            }
        }

        /// <summary>Starts a buffered attack. When <paramref name="allowed"/> is non-null only those move ids may start (cancels).</summary>
        private bool TryBufferedAttack(FightSimulation sim, MoveDirection dir, MoveData cancelFrom)
        {
            for (int b = 0; b < 4; b++)
            {
                var button = (AttackButton)b;
                var flag = ToInput(button);
                if (!IsBuffered(flag)) continue;
                var move = Def.Resolve(button, dir);
                if (move == null) continue;
                if (cancelFrom != null && !cancelFrom.CancelInto.Contains(move.Id)) continue;
                ConsumeBuffered(flag);
                if (TryStartMoveData(sim, move)) return true;
            }
            return false;
        }

        public static InputButtons ToInput(AttackButton button)
        {
            switch (button)
            {
                case AttackButton.LightPunch: return InputButtons.LightPunch;
                case AttackButton.HeavyPunch: return InputButtons.HeavyPunch;
                case AttackButton.LightKick: return InputButtons.LightKick;
                case AttackButton.HeavyKick: return InputButtons.HeavyKick;
                case AttackButton.Throw: return InputButtons.Throw;
                default: return InputButtons.Special;
            }
        }

        private bool TryStartMove(FightSimulation sim, AttackButton button, MoveDirection dir)
        {
            var move = Def.Resolve(button, dir);
            return move != null && TryStartMoveData(sim, move);
        }

        private bool TryStartMoveData(FightSimulation sim, MoveData move)
        {
            if (move.IsSpecial && Focus < move.MeterCost) return false;
            if (!SpendStamina(sim, move.StaminaCost)) return false;
            if (move.IsSpecial)
            {
                Focus -= move.MeterCost;
                SpecialReadyAnnounced = false;
                sim.Emit(CombatEventType.SpecialStarted, Index, -1, move.Id, 0, X, Y);
            }
            State = FighterStateId.Attack;
            StateFrame = 1;
            CurrentMove = move;
            MoveFrame = 1;
            MoveConnected = false;
            MoveHit = false;
            EvadeReported = false;
            WhiffReported = false;
            AttackSerial++;
            if (move.IsAirMove) AirAttackUsed = true;
            sim.Emit(CombatEventType.MoveStarted, Index, -1, move.Id, 0, X, Y);
            return true;
        }

        private bool SpendStamina(FightSimulation sim, float cost)
        {
            if (cost <= 0f) return true;
            if (Stamina < cost)
            {
                if (sim.Tick - _lastExhaustedTick > 30)
                {
                    _lastExhaustedTick = sim.Tick;
                    sim.Emit(CombatEventType.Exhausted, Index, -1, null, 0, X, Y);
                }
                return false;
            }
            Stamina -= cost;
            StaminaIdleFrames = 0;
            return true;
        }

        private void StartThrow(FightSimulation sim)
        {
            var throwMove = Def.Resolve(AttackButton.Throw, MoveDirection.Neutral);
            if (throwMove == null) return;
            if (!SpendStamina(sim, throwMove.StaminaCost)) return;
            SetState(FighterStateId.ThrowAttempt);
            PendingThrowMove = throwMove;
            AttackSerial++;
            sim.Emit(CombatEventType.MoveStarted, Index, -1, throwMove.Id, 0, X, Y);
        }

        private void TryAirAttack(FightSimulation sim)
        {
            if (AirAttackUsed) return;
            TryBufferedAttack(sim, MoveDirection.Air, null);
        }

        private void TickAttack(FightSimulation sim)
        {
            MoveFrame++;
            var m = CurrentMove;

            if (!WhiffReported && !MoveConnected && MoveFrame == m.LastActiveFrame + 1)
            {
                WhiffReported = true;
                sim.Emit(CombatEventType.Whiff, Index, -1, m.Id, 0, X, Y);
            }

            if (MoveConnected && MoveFrame >= m.FirstActiveFrame && MoveFrame <= m.LastActiveFrame + m.CancelWindowRecovery && m.CancelInto.Count > 0)
            {
                if (IsBuffered(InputButtons.Special) && m.CancelInto.Contains(SpecialId()))
                {
                    var special = Def.Resolve(AttackButton.Special, MoveDirection.Neutral);
                    if (special != null && Focus >= special.MeterCost)
                    {
                        ConsumeBuffered(InputButtons.Special);
                        if (TryStartMoveData(sim, special)) return;
                    }
                }
                var dir = CurrentDirection();
                if (TryBufferedAttack(sim, dir, m)) return;
                if (dir != MoveDirection.Neutral && TryBufferedAttack(sim, MoveDirection.Neutral, m)) return;
            }

            if (MoveFrame > m.TotalFrames)
            {
                if (m.IsAirMove && Y > 0f)
                {
                    // Finished the swing while still in the air: drift down without another attack.
                    State = FighterStateId.Airborne;
                    StateFrame = 1;
                    CurrentMove = null;
                    MoveFrame = 0;
                }
                else BecomeNeutral(sim);
            }
        }

        private string SpecialId()
        {
            var s = Def.Resolve(AttackButton.Special, MoveDirection.Neutral);
            return s?.Id;
        }

        internal void StartDefenderStun(FighterStateId state, int frames)
        {
            SetState(state);
            StunRemaining = Math.Max(1, frames);
        }

        internal void Land(FightSimulation sim)
        {
            Y = 0f; VY = 0f; VX = 0f;
            bool attacking = State == FighterStateId.Attack;
            SetState(FighterStateId.Landing);
            if (attacking) StateFrame = -2; // slightly longer landing after an air attack
            AirAttackUsed = false;
            sim.Emit(CombatEventType.Land, Index, -1, null, 0, X, Y);
        }

        internal void RegenerateStamina(float multiplier = 1f)
        {
            StaminaIdleFrames++;
            if (StaminaIdleFrames < Def.StaminaRegenDelay) return;
            float rate = Def.StaminaRegen * multiplier;
            switch (State)
            {
                case FighterStateId.Attack:
                case FighterStateId.Dodge:
                case FighterStateId.Parry:
                case FighterStateId.GuardBroken:
                    return;
                case FighterStateId.Blocking:
                case FighterStateId.CrouchBlocking:
                case FighterStateId.BlockStun:
                    rate *= 0.5f;
                    break;
            }
            Stamina = Math.Min(Def.MaxStamina, Stamina + rate);
        }

        internal void AddFocus(FightSimulation sim, float amount)
        {
            if (amount <= 0f) return;
            Focus = Math.Min(FocusMax, Focus + amount);
            var special = Def.Resolve(AttackButton.Special, MoveDirection.Neutral);
            if (special != null && !SpecialReadyAnnounced && Focus >= special.MeterCost)
            {
                SpecialReadyAnnounced = true;
                sim.Emit(CombatEventType.SpecialReady, Index, -1, special.Id, 0, X, Y);
            }
        }

        internal void ForceIdle()
        {
            SetState(FighterStateId.Idle);
        }

        internal void ForceVictory() => SetState(FighterStateId.Victory);

        internal void ForceKO() => SetState(FighterStateId.KO);

        internal void BeginThrowing(int resolveFrames)
        {
            SetState(FighterStateId.Throwing);
            ThrowResolveFrame = resolveFrames;
            ThrowDamageApplied = false;
        }

        /// <summary>Stable hash of gameplay-relevant state for determinism tests.</summary>
        public int StateHash()
        {
            unchecked
            {
                int h = 17;
                h = h * 31 + BitConverter.ToInt32(BitConverter.GetBytes(X), 0);
                h = h * 31 + BitConverter.ToInt32(BitConverter.GetBytes(Y), 0);
                h = h * 31 + Health;
                h = h * 31 + BitConverter.ToInt32(BitConverter.GetBytes(Stamina), 0);
                h = h * 31 + BitConverter.ToInt32(BitConverter.GetBytes(Focus), 0);
                h = h * 31 + (int)State;
                h = h * 31 + StateFrame;
                h = h * 31 + MoveFrame;
                h = h * 31 + (CurrentMove != null ? CurrentMove.Id.GetHashCode() : 0);
                return h;
            }
        }
    }
}
