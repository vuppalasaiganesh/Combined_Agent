using System;
using Undercurrent.Core.Combat;

namespace Undercurrent.Core.AI
{
    /// <summary>What the AI can see of its opponent on a given tick — the same information a human reads from the screen.</summary>
    public struct OpponentObservation
    {
        public bool Valid;
        public FighterStateId State;
        public MovePhase Phase;
        public MoveData Move;
        public int MoveFrame;
        public int AttackSerial;
        public float X;
        public float Y;
        public bool Airborne;
        public bool Crouching;
        public bool Blocking;
        public float HealthFraction;
    }

    /// <summary>
    /// Computer opponent. Each tick it records what is visible about the opponent, then decides using the observation
    /// from <see cref="AIProfile.ReactionTicks"/> ago. It never reads the opponent's controller input and produces an
    /// ordinary <see cref="FrameInput"/> that goes through the same rules as the player's.
    /// </summary>
    public sealed class FightAI
    {
        private const int HistorySize = 64;

        public readonly int SelfIndex;
        public AIDifficulty Difficulty { get; }
        public AIProfile Profile { get; private set; }
        public int BossPhase { get; private set; } = 1;
        public bool IsBoss => Difficulty == AIDifficulty.Boss;

        private readonly DeterministicRandom _rng;
        private readonly OpponentObservation[] _history = new OpponentObservation[HistorySize];
        private int _historyCount;
        private int _historyHead;

        private InputButtons _lastOutput;
        private int _nextDecisionTick;
        private InputButtons _neutralHold;
        private int _blockUntilTick;
        private bool _blockLow;
        private int _lastThreatSerial = -1;
        private int _lastPunishSerial = -1;
        private int _tick;

        // Combo plan
        private string[] _plan;
        private int _planIndex;
        private int _planSerial = -1;
        private bool _planRolled;

        // Simple adaptation: how often has the opponent been seen blocking recently.
        private float _opponentBlockRate;
        private int _techRolledForSerial = -1;
        private bool _wasUnderPressure;

        public FightAI(int selfIndex, AIDifficulty difficulty, uint seed)
        {
            SelfIndex = selfIndex;
            Difficulty = difficulty;
            Profile = AIProfile.For(difficulty);
            _rng = new DeterministicRandom(seed);
        }

        public void OverrideProfile(AIProfile profile) => Profile = profile ?? Profile;

        /// <summary>Produces this tick's input. Call once per simulation step, before <see cref="FightSimulation.Step"/>.</summary>
        public FrameInput Think(FightSimulation sim)
        {
            _tick++;
            var self = sim.Fighters[SelfIndex];
            var opp = sim.Fighters[1 - SelfIndex];
            Record(opp);

            if (sim.Phase != MatchPhase.Fighting)
            {
                ResetTactics();
                return Output(InputButtons.None);
            }

            UpdateBossPhase(self);

            var obs = Observe(Profile.ReactionTicks);
            if (!obs.Valid) return Output(InputButtons.None);

            _opponentBlockRate = _opponentBlockRate * 0.99f + (obs.Blocking ? 0.01f : 0f);

            InputButtons held = InputButtons.None;
            InputButtons toward = self.Facing > 0 ? InputButtons.Right : InputButtons.Left;
            InputButtons away = self.Facing > 0 ? InputButtons.Left : InputButtons.Right;
            float distance = Math.Abs(obs.X - self.X);

            // --- Grabbed: try to break the throw (requires seeing the grab in time).
            if (self.State == FighterStateId.Grabbed)
            {
                if (_techRolledForSerial != obs.AttackSerial && obs.State == FighterStateId.Throwing)
                {
                    _techRolledForSerial = obs.AttackSerial;
                    if (_rng.Chance(Profile.ThrowTechChance)) return Press(InputButtons.Throw);
                }
                return Output(InputButtons.None);
            }

            // --- Mid-attack: continue the combo plan inside the cancel window.
            if (self.State == FighterStateId.Attack)
            {
                var press = ContinueCombo(self, toward, away);
                return press.HasValue ? Press(press.Value) : Output(InputButtons.None);
            }

            // --- Stunned states: keep holding block if we were blocking.
            if (self.State == FighterStateId.BlockStun)
            {
                _wasUnderPressure = true;
                return Output(InputButtons.Block | (_blockLow ? InputButtons.Down : InputButtons.None));
            }

            if (!self.IsActionable)
            {
                if (self.State == FighterStateId.HitStun || self.State == FighterStateId.GuardBroken) _wasUnderPressure = true;
                return Output(InputButtons.None);
            }

            // --- Just recovered from being hit or blocking: humans tend to guard against continued pressure.
            if (_wasUnderPressure)
            {
                _wasUnderPressure = false;
                if (distance < 1.6f && _rng.Chance(Profile.BlockChance))
                {
                    _blockLow = _rng.Chance(0.25f);
                    _blockUntilTick = _tick + 14 + _rng.Range(0, 12);
                }
            }

            // --- Defense: react to a visible incoming attack.
            if (IsThreat(obs, self, distance) && obs.AttackSerial != _lastThreatSerial)
            {
                _lastThreatSerial = obs.AttackSerial;
                var m = obs.Move;
                int remaining = Math.Max(4, m.TotalFrames - obs.MoveFrame);
                if (m.Parryable && _rng.Chance(Profile.ParryChance) && obs.Phase == MovePhase.Startup)
                {
                    _blockUntilTick = 0;
                    return Press(InputButtons.Parry);
                }
                if (_rng.Chance(Profile.BlockChance))
                {
                    _blockLow = m.Height == AttackHeight.Low;
                    _blockUntilTick = _tick + remaining + 4;
                }
                else if (_rng.Chance(Profile.DodgeChance) && self.Stamina >= self.Def.DodgeStaminaCost)
                {
                    _blockUntilTick = 0;
                    return Press(InputButtons.Dodge);
                }
            }

            if (_tick < _blockUntilTick)
                return Output(InputButtons.Block | (_blockLow ? InputButtons.Down : InputButtons.None));

            // --- Anti-air.
            if (obs.Airborne && distance < 2.0f && obs.AttackSerial != _lastPunishSerial)
            {
                _lastPunishSerial = obs.AttackSerial;
                if (_rng.Chance(Profile.AntiAirChance))
                {
                    var aa = self.Def.Resolve(AttackButton.HeavyPunch, MoveDirection.Down);
                    if (aa != null && self.Stamina >= aa.StaminaCost) return PressMove(aa, toward, away);
                }
            }

            // --- Punish visible openings.
            if (IsPunishable(obs) && obs.AttackSerial != _lastPunishSerial)
            {
                _lastPunishSerial = obs.AttackSerial;
                if (_rng.Chance(Profile.PunishChance))
                {
                    var start = ChooseOpener(self, distance, punishing: true);
                    if (start != null) return PressMove(start, toward, away);
                }
            }

            // --- Neutral game, re-evaluated on an interval.
            if (_tick >= _nextDecisionTick)
            {
                _nextDecisionTick = _tick + Profile.DecisionInterval + _rng.Range(0, Profile.DecisionInterval / 2 + 1);
                _neutralHold = InputButtons.None;

                float preferred = self.Def.PreferredRange;
                bool tired = self.StaminaFraction < Profile.LowStaminaThreshold;

                if (_rng.Chance(Profile.MistakeChance))
                {
                    // Human-like mistake: a random swing at the wrong time.
                    var randomButton = (AttackButton)_rng.Range(0, 4);
                    var mistake = self.Def.Resolve(randomButton, MoveDirection.Neutral);
                    if (mistake != null && self.Stamina >= mistake.StaminaCost) return PressMove(mistake, toward, away);
                }

                // Rough threat range of a typical kick; inside it a human keeps their guard up some of the time,
                // because the fastest attacks cannot be blocked on reaction.
                bool inThreatRange = distance < 1.6f;

                if (tired)
                {
                    _neutralHold = distance < preferred + 1.2f ? away : InputButtons.None;
                    if (inThreatRange && _rng.Chance(Profile.BlockChance))
                    {
                        _blockLow = _rng.Chance(0.3f);
                        _blockUntilTick = _tick + Profile.DecisionInterval;
                        return Output(InputButtons.Block | (_blockLow ? InputButtons.Down : InputButtons.None));
                    }
                }
                else if (inThreatRange && _rng.Chance(Profile.BlockChance * 0.45f))
                {
                    _blockLow = _rng.Chance(0.3f);
                    _blockUntilTick = _tick + Profile.DecisionInterval + _rng.Range(0, Profile.DecisionInterval);
                    return Output(InputButtons.Block | (_blockLow ? InputButtons.Down : InputButtons.None));
                }
                else if (distance < 1.0f && _rng.Chance(Profile.Aggression * (0.06f + _opponentBlockRate * 1.2f)))
                {
                    var thr = self.Def.Resolve(AttackButton.Throw, MoveDirection.Neutral);
                    if (thr != null && self.Stamina >= thr.StaminaCost) return Press(InputButtons.Throw);
                }
                else if (distance > preferred + 0.35f)
                {
                    if (distance < 3.2f && _rng.Chance(Profile.Aggression * 0.12f))
                        return Press(InputButtons.Up | toward);
                    _neutralHold = toward;
                }
                else if (_rng.Chance(Profile.Aggression))
                {
                    var opener = ChooseOpener(self, distance, punishing: false);
                    if (opener != null) return PressMove(opener, toward, away);
                }
                else
                {
                    float r = _rng.NextFloat();
                    _neutralHold = r < 0.3f ? away : r < 0.55f ? toward : InputButtons.None;
                }
            }

            held |= _neutralHold;
            return Output(held);
        }

        private void ResetTactics()
        {
            _plan = null;
            _blockUntilTick = 0;
            _neutralHold = InputButtons.None;
            _nextDecisionTick = 0;
        }

        private void UpdateBossPhase(FighterRuntime self)
        {
            if (!IsBoss || BossPhase != 1) return;
            if (self.HealthFraction <= 0.5f)
            {
                BossPhase = 2;
                Profile = Profile.BossPhaseTwo();
            }
        }

        private static bool IsThreat(in OpponentObservation obs, FighterRuntime self, float distance)
        {
            if (obs.State != FighterStateId.Attack || obs.Move == null) return false;
            if (obs.Phase != MovePhase.Startup && obs.Phase != MovePhase.Active) return false;
            float reach = obs.Move.HitFrontOffset + obs.Move.HitLength + obs.Move.Advance + self.Def.BodyHalfWidth + 0.15f;
            return distance <= reach;
        }

        private static bool IsPunishable(in OpponentObservation obs)
        {
            switch (obs.State)
            {
                case FighterStateId.Parried:
                case FighterStateId.GuardBroken:
                case FighterStateId.Landing:
                    return true;
                case FighterStateId.Attack:
                    return obs.Phase == MovePhase.Recovery && obs.Move != null && obs.Move.Recovery >= 14;
                case FighterStateId.ThrowAttempt:
                    return true;
                default:
                    return false;
            }
        }

        private MoveData ChooseOpener(FighterRuntime self, float distance, bool punishing)
        {
            var def = self.Def;

            // Special as a punish or finisher when the meter is ready.
            var special = def.Resolve(AttackButton.Special, MoveDirection.Neutral);
            if (special != null && self.Focus >= special.MeterCost && punishing && _rng.Chance(Profile.SpecialUsage))
            {
                float reach = special.HitFrontOffset + special.HitLength + special.Advance;
                if (distance <= reach + 0.3f) { _plan = null; return special; }
            }

            // Pick a combo route whose opener reaches.
            if (def.Combos.Count > 0)
            {
                int start = _rng.Range(0, def.Combos.Count);
                for (int k = 0; k < def.Combos.Count; k++)
                {
                    var route = def.Combos[(start + k) % def.Combos.Count];
                    if (route.MoveIds == null || route.MoveIds.Length == 0) continue;
                    var opener = def.FindMove(route.MoveIds[0]);
                    if (opener == null || self.Stamina < opener.StaminaCost) continue;
                    float reach = opener.HitFrontOffset + opener.HitLength + opener.Advance + 0.25f;
                    if (distance <= reach)
                    {
                        _plan = route.MoveIds;
                        _planIndex = 0;
                        _planSerial = -1;
                        _planRolled = false;
                        return opener;
                    }
                }
            }

            // Fall back to any single poke that reaches.
            MoveData best = null;
            foreach (var m in def.Moves)
            {
                if (m.IsSpecial || m.IsThrow || m.IsAirMove || m.Direction == MoveDirection.Back) continue;
                if (self.Stamina < m.StaminaCost) continue;
                float reach = m.HitFrontOffset + m.HitLength + m.Advance + 0.25f;
                if (distance > reach) continue;
                if (best == null || m.Startup < best.Startup) best = m;
            }
            _plan = null;
            return best;
        }

        private InputButtons? ContinueCombo(FighterRuntime self, InputButtons toward, InputButtons away)
        {
            if (_plan == null || self.CurrentMove == null) return null;
            if (_planSerial < 0) _planSerial = self.AttackSerial;
            if (_planSerial != self.AttackSerial)
            {
                // A new attack started: if it matches the next plan entry, advance.
                if (_planIndex + 1 < _plan.Length && self.CurrentMove.Id == _plan[_planIndex + 1])
                {
                    _planIndex++;
                    _planSerial = self.AttackSerial;
                    _planRolled = false;
                }
                else
                {
                    _plan = null;
                    return null;
                }
            }
            if (_planIndex + 1 >= _plan.Length) { _plan = null; return null; }
            if (!self.MoveConnected) return null;

            if (!_planRolled)
            {
                _planRolled = true;
                float chance = self.MoveHit ? Profile.ComboContinueChance : Profile.ComboContinueChance * 0.5f;
                if (!_rng.Chance(chance)) { _plan = null; return null; }
            }

            var next = self.Def.FindMove(_plan[_planIndex + 1]);
            if (next == null) { _plan = null; return null; }
            if (next.IsSpecial && self.Focus < next.MeterCost) { _plan = null; return null; }
            return ButtonsFor(next, toward, away);
        }

        private static InputButtons ButtonsFor(MoveData m, InputButtons toward, InputButtons away)
        {
            InputButtons b = FighterRuntime.ToInput(m.Button);
            switch (m.Direction)
            {
                case MoveDirection.Forward: b |= toward; break;
                case MoveDirection.Back: b |= away; break;
                case MoveDirection.Down: b |= InputButtons.Down; break;
            }
            return b;
        }

        private FrameInput PressMove(MoveData m, InputButtons toward, InputButtons away) => Press(ButtonsFor(m, toward, away));

        /// <summary>Outputs a fresh press: if a button was already held last tick it is released first.</summary>
        private FrameInput Press(InputButtons buttons)
        {
            const InputButtons directions = InputButtons.Left | InputButtons.Right | InputButtons.Up | InputButtons.Down;
            var actionButtons = buttons & ~directions;
            if ((_lastOutput & actionButtons) != 0)
                return Output(buttons & directions);
            return Output(buttons);
        }

        private FrameInput Output(InputButtons held)
        {
            _lastOutput = held;
            return new FrameInput(held);
        }

        private void Record(FighterRuntime opp)
        {
            var o = new OpponentObservation
            {
                Valid = true,
                State = opp.State,
                Phase = opp.CurrentPhase,
                Move = opp.State == FighterStateId.Attack ? opp.CurrentMove : (opp.State == FighterStateId.ThrowAttempt ? opp.PendingThrowMoveVisible : null),
                MoveFrame = opp.State == FighterStateId.Attack ? opp.MoveFrame : opp.StateFrame,
                AttackSerial = opp.AttackSerial,
                X = opp.X,
                Y = opp.Y,
                Airborne = opp.IsAirborne,
                Crouching = opp.IsCrouching,
                Blocking = opp.State == FighterStateId.Blocking || opp.State == FighterStateId.CrouchBlocking || opp.State == FighterStateId.BlockStun,
                HealthFraction = opp.HealthFraction,
            };
            _history[_historyHead] = o;
            _historyHead = (_historyHead + 1) % HistorySize;
            if (_historyCount < HistorySize) _historyCount++;
        }

        /// <summary>Observation from <paramref name="delay"/> ticks ago (the most recent one is delay 0).</summary>
        public OpponentObservation Observe(int delay)
        {
            if (_historyCount == 0) return default;
            delay = Math.Min(Math.Max(0, delay), _historyCount - 1);
            int idx = (_historyHead - 1 - delay + HistorySize * 2) % HistorySize;
            return _history[idx];
        }
    }
}
