using System;
using System.Collections.Generic;

namespace Undercurrent.Core.Combat
{
    [Serializable]
    public sealed class MatchConfig
    {
        public int RoundsToWin = 2;
        public int MaxRounds = 5;
        public int RoundSeconds = 60;
        public int IntroTicks = 120;
        public int RoundOverTicks = 180;
        public float StageHalfWidth = 6.5f;
        public float MaxSeparation = 7.0f;
        public float StartSeparation = 2.6f;
        /// <summary>Damage multiplier on counter hits.</summary>
        public float CounterHitMultiplier = 1.25f;
        public int CounterHitBonusStun = 4;
        public float ThrowRange = 1.05f;
        public float ParryFocusGain = 12f;

        // ---- Level rule modifiers (index = fighter 0 / 1) ----
        /// <summary>Health each fighter starts round 1 with, as a fraction of max (gauntlet carry-over, "last breath" rules).</summary>
        public float[] StartHealth = { 1f, 1f };
        /// <summary>Multiplier on damage dealt by each fighter.</summary>
        public float[] DamageDealt = { 1f, 1f };
        /// <summary>Multiplier on stamina regeneration ("exhaustion" rules).</summary>
        public float[] StaminaRegen = { 1f, 1f };
        /// <summary>Apply <see cref="StartHealth"/> to every round instead of round 1 only.</summary>
        public bool StartHealthEveryRound;
        public int ClashStun = 16;

        public MatchConfig Clone()
        {
            var c = (MatchConfig)MemberwiseClone();
            c.StartHealth = (float[])StartHealth.Clone();
            c.DamageDealt = (float[])DamageDealt.Clone();
            c.StaminaRegen = (float[])StaminaRegen.Clone();
            return c;
        }
    }

    /// <summary>
    /// Deterministic one-on-one fight simulation stepped at <see cref="SimClock.TicksPerSecond"/>.
    /// It knows nothing about rendering, audio or devices: inputs come in as <see cref="FrameInput"/>,
    /// presentation reads fighter state and the per-tick <see cref="Events"/> list.
    /// </summary>
    public sealed class FightSimulation
    {
        public readonly MatchConfig Config;
        public readonly FighterRuntime[] Fighters = new FighterRuntime[2];

        public MatchPhase Phase { get; private set; } = MatchPhase.PreRound;
        public int Round { get; private set; }
        public int Tick { get; private set; }
        public int PhaseTicks { get; private set; }
        public int RoundTicksRemaining { get; private set; }
        public int HitStopRemaining { get; private set; }
        public RoundOutcome LastRoundOutcome { get; private set; }
        public RoundOutcome MatchOutcome { get; private set; }
        public int FightingTicksElapsed { get; private set; }

        private readonly List<CombatEvent> _events = new List<CombatEvent>(32);
        public IReadOnlyList<CombatEvent> Events => _events;

        // Scratch, reused every tick to avoid allocation.
        private readonly bool[] _pendingStrike = new bool[2];
        private readonly float[] _prevX = new float[2];

        public FightSimulation(FighterDefinition player0, FighterDefinition player1, MatchConfig config = null)
        {
            Config = config ?? new MatchConfig();
            Fighters[0] = new FighterRuntime(0, player0);
            Fighters[1] = new FighterRuntime(1, player1);
            StartRound(1);
        }

        public FighterRuntime Opponent(FighterRuntime f) => Fighters[1 - f.Index];

        public int RoundSecondsRemaining => (RoundTicksRemaining + SimClock.TicksPerSecond - 1) / SimClock.TicksPerSecond;

        public bool IsMatchOver => Phase == MatchPhase.MatchOver;

        internal void Emit(CombatEventType type, int source, int target, string moveId, int amount, float x, float y,
            bool heavy = false, bool final = false)
        {
            _events.Add(new CombatEvent
            {
                Type = type, Source = source, Target = target, MoveId = moveId, Amount = amount,
                X = x, Y = y, IsHeavy = heavy, IsFinal = final, Round = Round,
            });
        }

        private void StartRound(int round)
        {
            Round = round;
            Phase = MatchPhase.PreRound;
            PhaseTicks = 0;
            HitStopRemaining = 0;
            RoundTicksRemaining = Config.RoundSeconds * SimClock.TicksPerSecond;
            LastRoundOutcome = RoundOutcome.None;
            float half = Config.StartSeparation * 0.5f;
            Fighters[0].ResetForRound(-half, 1, round == 1 || Config.StartHealthEveryRound ? Config.StartHealth[0] : 1f);
            Fighters[1].ResetForRound(half, -1, round == 1 || Config.StartHealthEveryRound ? Config.StartHealth[1] : 1f);
            Emit(CombatEventType.RoundAnnounce, -1, -1, null, round, 0f, 0f);
        }

        /// <summary>Advances the simulation by one tick. Clears and repopulates <see cref="Events"/>.</summary>
        public void Step(FrameInput input0, FrameInput input1)
        {
            _events.Clear();
            Tick++;
            PhaseTicks++;

            switch (Phase)
            {
                case MatchPhase.PreRound:
                    Fighters[0].FeedInput(FrameInput.Empty, true);
                    Fighters[1].FeedInput(FrameInput.Empty, true);
                    if (PhaseTicks >= Config.IntroTicks)
                    {
                        Phase = MatchPhase.Fighting;
                        PhaseTicks = 0;
                        foreach (var f in Fighters)
                        {
                            f.ClearBuffer();
                            f.ForceIdle();
                        }
                        Emit(CombatEventType.RoundStart, -1, -1, null, Round, 0f, 0f);
                    }
                    return;

                case MatchPhase.Fighting:
                    StepFighting(input0, input1);
                    return;

                case MatchPhase.RoundOver:
                    // Let knockdowns and slides play out, no further control.
                    Fighters[0].FeedInput(FrameInput.Empty, true);
                    Fighters[1].FeedInput(FrameInput.Empty, true);
                    if (HitStopRemaining > 0) { HitStopRemaining--; return; }
                    int roundWinner = LastRoundOutcome == RoundOutcome.Player0 ? 0 : LastRoundOutcome == RoundOutcome.Player1 ? 1 : -1;
                    foreach (var f in Fighters)
                    {
                        if (f.State == FighterStateId.KO || f.State == FighterStateId.Victory) continue;
                        f.Tick(this);
                        if (f.IsActionable && f.Index == roundWinner) f.ForceVictory();
                    }
                    ApplyMovement();
                    if (PhaseTicks >= Config.RoundOverTicks) FinishRound();
                    return;

                case MatchPhase.MatchOver:
                    return;
            }
        }

        private void StepFighting(FrameInput input0, FrameInput input1)
        {
            bool frozen = HitStopRemaining > 0;
            Fighters[0].FeedInput(input0, !frozen);
            Fighters[1].FeedInput(input1, !frozen);
            if (frozen)
            {
                HitStopRemaining--;
                return;
            }

            FightingTicksElapsed++;
            RoundTicksRemaining--;

            Fighters[0].Tick(this);
            Fighters[1].Tick(this);

            ApplyMovement();
            ResolveThrows();
            ResolveStrikes();

            for (int i = 0; i < 2; i++) Fighters[i].RegenerateStamina(Config.StaminaRegen[i]);

            CheckRoundEnd();
        }

        // ---------------------------------------------------------------- movement & collision

        private void ApplyMovement()
        {
            for (int i = 0; i < 2; i++)
            {
                var f = Fighters[i];
                _prevX[i] = f.X;
                float dx = 0f;

                switch (f.State)
                {
                    case FighterStateId.WalkForward: dx += f.Facing * f.Def.WalkForwardSpeed; break;
                    case FighterStateId.WalkBack: dx -= f.Facing * f.Def.WalkBackSpeed; break;
                    case FighterStateId.Dodge:
                        if (f.StateFrame <= f.Def.DodgeInvulnTo)
                            dx -= f.Facing * f.Def.DodgeDistance / Math.Max(1, f.Def.DodgeInvulnTo);
                        break;
                    case FighterStateId.Attack:
                        var m = f.CurrentMove;
                        if (!m.IsAirMove && m.Advance != 0f && f.MoveFrame <= m.LastActiveFrame)
                            dx += f.Facing * m.Advance / Math.Max(1, m.LastActiveFrame);
                        break;
                }

                if (f.IsAirborne)
                {
                    dx += f.VX;
                    f.Y += f.VY;
                    f.VY -= f.Def.Gravity;
                    if (f.Y <= 0f && f.VY < 0f) f.Land(this);
                }
                else if (f.Y > 0f)
                {
                    // Was launched/knocked out of the air (e.g. hit while jumping): fall straight down.
                    f.Y = Math.Max(0f, f.Y + f.VY);
                    f.VY -= f.Def.Gravity;
                    if (f.Y <= 0f) { f.Y = 0f; f.VY = 0f; }
                }

                if (f.SlideVelocity != 0f)
                {
                    dx += f.SlideVelocity;
                    f.SlideVelocity *= 0.8f;
                    if (Math.Abs(f.SlideVelocity) < 0.002f) f.SlideVelocity = 0f;
                }

                f.X += dx;
            }

            ResolvePushboxes();
            ClampSeparation();
            ClampToStage(Fighters[0]);
            ClampToStage(Fighters[1]);
        }

        private float WallLimit(FighterRuntime f) => Config.StageHalfWidth - f.Def.PushHalfWidth;

        private void ClampToStage(FighterRuntime f)
        {
            float lim = WallLimit(f);
            if (f.X > lim) f.X = lim;
            if (f.X < -lim) f.X = -lim;
        }

        public bool IsAtWall(FighterRuntime f, int direction)
        {
            float lim = WallLimit(f);
            return direction > 0 ? f.X >= lim - 0.02f : f.X <= -lim + 0.02f;
        }

        private void ResolvePushboxes()
        {
            var a = Fighters[0];
            var b = Fighters[1];
            float minDist = a.Def.PushHalfWidth + b.Def.PushHalfWidth;
            float dx = b.X - a.X;
            // Fighters never swap sides; fighter 0 stays left of fighter 1 if they started that way.
            int side = a.Facing > 0 ? 1 : -1;
            if (dx * side >= minDist) return;
            float overlap = minDist - dx * side;
            float pushA = overlap * 0.5f;
            float pushB = overlap * 0.5f;

            // Whoever is cornered cannot move further back; the other one absorbs the push.
            if (IsAtWall(a, -side)) { pushB += pushA; pushA = 0f; }
            else if (IsAtWall(b, side)) { pushA += pushB; pushB = 0f; }

            a.X -= side * pushA;
            b.X += side * pushB;
        }

        private void ClampSeparation()
        {
            var a = Fighters[0];
            var b = Fighters[1];
            float dist = Math.Abs(b.X - a.X);
            float excess = dist - Config.MaxSeparation;
            if (excess <= 0f) return;
            float dirA = Math.Sign(a.X - b.X);
            float outA = Math.Max(0f, (a.X - _prevX[0]) * dirA);
            float outB = Math.Max(0f, (b.X - _prevX[1]) * -dirA);
            float total = outA + outB;
            if (total <= 0f) { outA = outB = 1f; total = 2f; }
            a.X -= dirA * excess * (outA / total);
            b.X += dirA * excess * (outB / total);
        }

        // ---------------------------------------------------------------- throws

        private void ResolveThrows()
        {
            var a = Fighters[0];
            var b = Fighters[1];
            bool aActive = IsThrowActive(a);
            bool bActive = IsThrowActive(b);

            if (aActive && bActive && InThrowRange(a, b))
            {
                // Simultaneous grabs cancel out.
                BreakThrow(a, b);
                return;
            }
            if (aActive) TryConnectThrow(a, b);
            if (bActive) TryConnectThrow(b, a);

            for (int i = 0; i < 2; i++)
            {
                var thrower = Fighters[i];
                if (thrower.State != FighterStateId.Throwing) continue;
                var victim = Opponent(thrower);
                if (thrower.ThrowDamageApplied)
                {
                    if (thrower.StateFrame > thrower.ThrowResolveFrame + FighterRuntime.ThrowFollowThrough) thrower.ForceIdle();
                    continue;
                }
                if (victim.State == FighterStateId.Grabbed && victim.ConsumeBuffered(InputButtons.Throw))
                {
                    victim.Stats.ThrowBreaks++;
                    BreakThrow(thrower, victim);
                    continue;
                }
                if (thrower.StateFrame > thrower.ThrowResolveFrame)
                {
                    var m = thrower.PendingThrowMove;
                    thrower.ThrowDamageApplied = true;
                    int dmg = ScaleDamage(thrower, victim, m.Damage, 1f);
                    ApplyDamage(thrower, victim, dmg);
                    thrower.Stats.ThrowsLanded++;
                    thrower.Stats.HitsLanded++;
                    victim.SetState(FighterStateId.KnockedDown);
                    victim.Stats.Knockdowns++;
                    victim.Disarm(this, Content.WeaponCatalog.DisarmTicks);
                    victim.SlideVelocity = thrower.Facing * m.KnockbackOnHit * 0.2f;
                    thrower.AddFocus(this, thrower.Def.FocusGainOnHit);
                    victim.AddFocus(this, victim.Def.FocusGainOnDamaged);
                    HitStopRemaining = Math.Max(HitStopRemaining, m.HitStop);
                    Emit(CombatEventType.ThrowConnected, thrower.Index, victim.Index, m.Id, dmg, victim.X, 1.0f, true);
                    Emit(CombatEventType.Knockdown, thrower.Index, victim.Index, m.Id, 0, victim.X, 0f);
                }
            }
        }

        private static bool IsThrowActive(FighterRuntime f) =>
            f.State == FighterStateId.ThrowAttempt &&
            f.StateFrame > FighterRuntime.ThrowStartup &&
            f.StateFrame <= FighterRuntime.ThrowStartup + FighterRuntime.ThrowActive;

        private bool InThrowRange(FighterRuntime a, FighterRuntime b) => Math.Abs(a.X - b.X) <= Config.ThrowRange;

        private void TryConnectThrow(FighterRuntime thrower, FighterRuntime victim)
        {
            if (!InThrowRange(thrower, victim) || !victim.IsThrowable) return;
            thrower.BeginThrowing(FighterRuntime.ThrowTechWindow);
            victim.SetState(FighterStateId.Grabbed);
            victim.ComboCount = 0;
        }

        private void BreakThrow(FighterRuntime a, FighterRuntime b)
        {
            a.ForceIdle();
            b.ForceIdle();
            int side = a.X <= b.X ? 1 : -1;
            a.SlideVelocity = -side * 0.12f;
            b.SlideVelocity = side * 0.12f;
            HitStopRemaining = Math.Max(HitStopRemaining, 8);
            Emit(CombatEventType.ThrowBroken, a.Index, b.Index, null, 0, (a.X + b.X) * 0.5f, 1.1f);
        }

        // ---------------------------------------------------------------- strikes

        private void ResolveStrikes()
        {
            for (int i = 0; i < 2; i++)
            {
                _pendingStrike[i] = false;
                var a = Fighters[i];
                if (a.State != FighterStateId.Attack || a.MoveConnected) continue;
                if (a.CurrentPhase != MovePhase.Active) continue;
                var d = Opponent(a);
                if (!a.GetActiveHitbox().Overlaps(d.GetHurtbox())) continue;
                _pendingStrike[i] = true;
            }

            // Evaluate both before applying either so trades are symmetric.
            bool strike0 = _pendingStrike[0];
            bool strike1 = _pendingStrike[1];

            // Weapon clash: the blades meet in the air, or two weapon strikes would trade hits on the same frame.
            if (TryClash(strike0 && strike1)) return;
            var snapshot0 = strike0 ? Fighters[0].CurrentMove : null;
            var snapshot1 = strike1 ? Fighters[1].CurrentMove : null;
            var state0 = Fighters[0].State;
            var state1 = Fighters[1].State;
            if (strike0) ApplyStrike(Fighters[0], Fighters[1], snapshot0, state1);
            if (strike1) ApplyStrike(Fighters[1], Fighters[0], snapshot1, state0);
        }

        private void ApplyStrike(FighterRuntime a, FighterRuntime d, MoveData m, FighterStateId defenderStateBefore)
        {
            var hitbox = Box.FromLocal(a.X, a.Y, a.Facing, m.HitFrontOffset, m.HitBottom, m.HitLength, m.HitHeight);
            float cx = (Math.Max(hitbox.MinX, d.X - d.Def.BodyHalfWidth) + Math.Min(hitbox.MaxX, d.X + d.Def.BodyHalfWidth)) * 0.5f;
            float cy = hitbox.CenterY;

            if (d.IsStrikeInvulnerable)
            {
                if (!a.EvadeReported)
                {
                    a.EvadeReported = true;
                    if (d.State == FighterStateId.Dodge || (d.State == FighterStateId.Attack && d.CurrentMove.InvulnerableFrom > 0))
                    {
                        d.Stats.Evades++;
                        Emit(CombatEventType.Evaded, d.Index, a.Index, m.Id, 0, cx, cy);
                    }
                }
                return;
            }

            a.MoveConnected = true;

            if (d.IsParryActive && m.Parryable)
            {
                a.StartDefenderStun(FighterStateId.Parried, m.ParriedStun);
                d.ForceIdle();
                d.Stamina = Math.Min(d.Def.MaxStamina, d.Stamina + d.Def.ParryStaminaRefund);
                d.AddFocus(this, Config.ParryFocusGain);
                d.Stats.Parries++;
                HitStopRemaining = Math.Max(HitStopRemaining, 12);
                Emit(CombatEventType.Parried, d.Index, a.Index, m.Id, 0, cx, cy, true);
                return;
            }

            if (d.IsBlockingStance(out bool crouchBlock) && CanBlock(m.Height, crouchBlock))
            {
                float guard = m.GuardDamage * d.Def.GuardDamageTakenMultiplier;
                int chip = m.ChipDamage;
                if (chip >= d.Health) chip = d.Health - 1; // chip damage never finishes a round
                if (chip > 0) ApplyDamage(a, d, chip);
                d.Stats.AttacksBlocked++;
                Push(a, d, m.KnockbackOnBlock);
                if (d.Stamina < guard)
                {
                    d.Stamina = 0f;
                    d.StaminaIdleFrames = 0;
                    d.StartDefenderStun(FighterStateId.GuardBroken, d.Def.GuardBreakFrames);
                    HitStopRemaining = Math.Max(HitStopRemaining, 14);
                    Emit(CombatEventType.GuardBreak, a.Index, d.Index, m.Id, chip, cx, cy, true);
                    d.Disarm(this, Content.WeaponCatalog.DisarmTicks);
                }
                else
                {
                    d.Stamina -= guard;
                    d.StaminaIdleFrames = 0;
                    d.StartDefenderStun(FighterStateId.BlockStun, m.BlockStun);
                    HitStopRemaining = Math.Max(HitStopRemaining, Math.Max(3, m.HitStop - 3));
                    Emit(CombatEventType.Blocked, a.Index, d.Index, m.Id, chip, cx, cy, m.Damage >= 80);
                }
                return;
            }

            // Clean hit.
            bool counter = defenderStateBefore == FighterStateId.Attack || defenderStateBefore == FighterStateId.ThrowAttempt ||
                           (defenderStateBefore == FighterStateId.Parry && !d.IsParryActive) ||
                           (defenderStateBefore == FighterStateId.Dodge);
            bool armorAbsorb = d.State == FighterStateId.Attack && d.CurrentMove.Armored &&
                               d.CurrentPhase != MovePhase.Recovery && !m.CausesKnockdown && m.Damage < 80;

            float mult = counter ? Config.CounterHitMultiplier + (m.IsWeaponStrike && a.IsArmed ? a.Weapon.CounterHitBonus : 0f) : 1f;
            int dmg = ScaleDamage(a, d, m.Damage, mult);
            ApplyDamage(a, d, dmg);
            a.MoveHit = true;
            a.Stats.HitsLanded++;
            if (counter) a.Stats.CounterHits++;
            if (m.IsSpecial) a.Stats.SpecialsLanded++;
            a.AddFocus(this, a.Def.FocusGainOnHit * (m.IsSpecial ? 0f : 1f));
            d.AddFocus(this, d.Def.FocusGainOnDamaged);

            d.ComboCount++;
            d.ComboDamage += dmg;
            if (d.ComboCount > a.Stats.MaxCombo) a.Stats.MaxCombo = d.ComboCount;

            int stop = m.HitStop + (counter ? 2 : 0);
            HitStopRemaining = Math.Max(HitStopRemaining, stop);

            var type = counter ? CombatEventType.CounterHit : CombatEventType.Hit;
            Emit(type, a.Index, d.Index, m.Id, dmg, cx, cy, m.Damage >= 80 || m.CausesKnockdown, false);

            if (armorAbsorb && d.Health > 0)
            {
                // Armor soaks the interruption, not the damage.
                return;
            }

            bool knockdown = m.CausesKnockdown || d.IsAirborne || d.ComboCount >= FighterRuntime.ComboKnockdownCap || d.Health <= 0;
            Push(a, d, m.KnockbackOnHit);
            if (knockdown)
            {
                if (d.IsAirborne && d.Y > 0f) d.VY = Math.Max(d.VY, 0.05f);
                d.SetState(FighterStateId.KnockedDown);
                d.Stats.Knockdowns++;
                Emit(CombatEventType.Knockdown, a.Index, d.Index, m.Id, 0, d.X, 0f);
            }
            else
            {
                int stun = m.HitStun + (counter ? Config.CounterHitBonusStun : 0) - Math.Min(d.ComboCount - 1, 6);
                d.StartDefenderStun(FighterStateId.HitStun, stun);
            }
        }

        /// <summary>A weapon swing that is about to land or already live (last 2 wind-up frames or active).</summary>
        private static bool SwingCommitted(FighterRuntime f) =>
            f.State == FighterStateId.Attack && f.IsArmed && f.CurrentMove.IsWeaponStrike && !f.CurrentMove.IsSpecial &&
            !f.MoveConnected && f.MoveFrame >= f.CurrentMove.Startup - 1 && f.CurrentPhase != MovePhase.Recovery;

        /// <summary>
        /// Weapon clash: both fighters are swinging weapons and either the blades meet in the air, or one blade lands while
        /// the other swing is already committed. Both recoil; nobody takes damage.
        /// </summary>
        private bool TryClash(bool mutualStrike)
        {
            var a = Fighters[0];
            var b = Fighters[1];
            if (!SwingCommitted(a) || !SwingCommitted(b)) return false;
            bool aLive = a.CurrentPhase == MovePhase.Active, bLive = b.CurrentPhase == MovePhase.Active;
            bool bladesMeet = aLive && bLive && a.GetActiveHitbox().Overlaps(b.GetActiveHitbox());
            bool landsIntoSwing = (_pendingStrike[0] && aLive) || (_pendingStrike[1] && bLive);
            if (!mutualStrike && !bladesMeet && !landsIntoSwing) return false;

            a.MoveConnected = b.MoveConnected = true;
            string moveId = a.CurrentMove.Id;
            a.StartDefenderStun(FighterStateId.Parried, Config.ClashStun);
            b.StartDefenderStun(FighterStateId.Parried, Config.ClashStun);
            a.SlideVelocity = -a.Facing * 0.08f;
            b.SlideVelocity = -b.Facing * 0.08f;
            HitStopRemaining = Math.Max(HitStopRemaining, 10);
            a.Stats.Clashes++;
            b.Stats.Clashes++;
            Emit(CombatEventType.Clash, 0, 1, moveId, 0, (a.X + b.X) * 0.5f, 1.2f, true);
            return true;
        }

        private static bool CanBlock(AttackHeight height, bool crouching)
        {
            switch (height)
            {
                case AttackHeight.High: return !crouching;
                case AttackHeight.Mid: return true;
                case AttackHeight.Low: return crouching;
                case AttackHeight.Overhead: return !crouching;
                default: return false;
            }
        }

        private int ScaleDamage(FighterRuntime a, FighterRuntime d, int baseDamage, float mult)
        {
            float comboScale = Math.Max(0.4f, 1f - 0.1f * d.ComboCount);
            float v = baseDamage * comboScale * mult * a.Def.DamageDealtMultiplier * d.Def.DamageTakenMultiplier * Config.DamageDealt[a.Index];
            return Math.Max(1, (int)Math.Round(v, MidpointRounding.AwayFromZero));
        }

        private void ApplyDamage(FighterRuntime a, FighterRuntime d, int dmg)
        {
            dmg = Math.Min(dmg, d.Health);
            d.Health -= dmg;
            a.Stats.DamageDealt += dmg;
            d.Stats.DamageTaken += dmg;
        }

        private void Push(FighterRuntime a, FighterRuntime d, float distance)
        {
            if (distance <= 0f) return;
            int dir = a.Facing;
            if (IsAtWall(d, dir))
            {
                // Cornered defender: the attacker is pushed back instead so pressure is not infinite.
                a.SlideVelocity = -dir * distance * 0.2f;
            }
            else
            {
                d.SlideVelocity = dir * distance * 0.2f;
            }
        }

        // ---------------------------------------------------------------- rounds

        private void CheckRoundEnd()
        {
            var a = Fighters[0];
            var b = Fighters[1];
            bool aDown = a.Health <= 0;
            bool bDown = b.Health <= 0;

            RoundOutcome outcome = RoundOutcome.None;
            bool timeout = false;
            if (aDown || bDown)
            {
                outcome = aDown && bDown ? RoundOutcome.Draw : aDown ? RoundOutcome.Player1 : RoundOutcome.Player0;
            }
            else if (RoundTicksRemaining <= 0)
            {
                timeout = true;
                float fa = a.HealthFraction, fb = b.HealthFraction;
                outcome = Math.Abs(fa - fb) < 0.0001f ? RoundOutcome.Draw : fa > fb ? RoundOutcome.Player0 : RoundOutcome.Player1;
            }
            if (outcome == RoundOutcome.None) return;

            LastRoundOutcome = outcome;
            if (outcome == RoundOutcome.Player0) a.RoundsWon++;
            if (outcome == RoundOutcome.Player1) b.RoundsWon++;
            if (outcome == RoundOutcome.Player0 || outcome == RoundOutcome.Player1)
            {
                var winner = outcome == RoundOutcome.Player0 ? a : b;
                winner.Stats.RoundsWon++;
                if (winner.Health >= winner.HealthAtRoundStart) winner.Stats.PerfectRounds++;
            }

            bool matchDecided = a.RoundsWon >= Config.RoundsToWin || b.RoundsWon >= Config.RoundsToWin || Round >= Config.MaxRounds;
            if (timeout)
            {
                Emit(CombatEventType.TimeOut, -1, -1, null, 0, 0f, 0f);
            }
            else
            {
                int loser = outcome == RoundOutcome.Player0 ? 1 : outcome == RoundOutcome.Player1 ? 0 : -1;
                int winner = loser < 0 ? -1 : 1 - loser;
                Emit(CombatEventType.KO, winner, loser, null, 0, loser >= 0 ? Fighters[loser].X : 0f, 1f, true, matchDecided);
            }

            for (int i = 0; i < 2; i++)
            {
                var f = Fighters[i];
                if (f.Health <= 0) f.ForceKO();
            }
            Emit(CombatEventType.RoundEnd, outcome == RoundOutcome.Player0 ? 0 : outcome == RoundOutcome.Player1 ? 1 : -1, -1, null, (int)outcome, 0f, 0f, false, matchDecided);
            Phase = MatchPhase.RoundOver;
            PhaseTicks = 0;
        }

        private void FinishRound()
        {
            var a = Fighters[0];
            var b = Fighters[1];
            bool decided = a.RoundsWon >= Config.RoundsToWin || b.RoundsWon >= Config.RoundsToWin || Round >= Config.MaxRounds;
            if (decided)
            {
                MatchOutcome = a.RoundsWon > b.RoundsWon ? RoundOutcome.Player0 : b.RoundsWon > a.RoundsWon ? RoundOutcome.Player1 : RoundOutcome.Draw;
                Phase = MatchPhase.MatchOver;
                PhaseTicks = 0;
                Emit(CombatEventType.MatchEnd, MatchOutcome == RoundOutcome.Player0 ? 0 : MatchOutcome == RoundOutcome.Player1 ? 1 : -1, -1, null, (int)MatchOutcome, 0f, 0f);
                return;
            }
            StartRound(Round + 1);
        }

        /// <summary>Deterministic fingerprint of the full simulation state.</summary>
        public int StateHash()
        {
            unchecked
            {
                int h = Fighters[0].StateHash() * 397 ^ Fighters[1].StateHash();
                h = h * 31 + Tick;
                h = h * 31 + (int)Phase;
                h = h * 31 + RoundTicksRemaining;
                h = h * 31 + HitStopRemaining;
                return h;
            }
        }
    }
}
