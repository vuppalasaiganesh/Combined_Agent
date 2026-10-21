using System;
using System.Text;
using Undercurrent.Core.AI;
using Undercurrent.Core.Combat;
using Undercurrent.Core.Content;
using Undercurrent.Core.Progression;
using Undercurrent.Runtime.Presentation;
using Undercurrent.Runtime.Services;
using Undercurrent.Runtime.UI;
using UnityEngine;
using UnityEngine.UI;

namespace Undercurrent.Runtime.Fight
{
    public sealed class FightSetup
    {
        public FighterDefinition Player;
        public FighterDefinition Opponent;
        public CharacterLook PlayerLook;
        public CharacterLook OpponentLook;
        public ArenaDefinition Arena;
        public MatchConfig Config;
        public AIDifficulty Difficulty = AIDifficulty.Normal;
        /// <summary>Campaign level, or null for Versus / Training.</summary>
        public LevelDefinition Level;
        public bool Tutorial;
        public uint Seed = 1;
        /// <summary>Gauntlet progress (bout index/count) for multi-bout levels; null otherwise.</summary>
        public GauntletRun Run;
    }

    public sealed class FightOutcome
    {
        public FightSetup Setup;
        public FightSimulation Sim;
        public MatchSummary Summary;
        public bool TutorialCompleted;
    }

    /// <summary>
    /// Runs one match: steps the deterministic simulation at a fixed 60 Hz, feeds it player and AI input, and turns
    /// combat events into presentation (VFX, audio, haptics, camera, HUD). Owns every object it creates.
    /// </summary>
    public sealed class FightController : MonoBehaviour
    {
        private const float Step = SimClock.SecondsPerTick;
        private const int MaxStepsPerFrame = 5;

        private FightSetup _setup;
        private FightSimulation _sim;
        private FightAI _ai;
        private TutorialTracker _tutorial;
        private Services.ServiceHub _services;
        private Action<FightOutcome> _onFinished;
        private Action _onQuit;

        private GameObject _world;
        private FighterView[] _views;
        private FightCamera _camera;
        private VfxPool _vfx;
        private RectTransform _uiRoot;
        private FightHud _hud;
        private TouchControls _touch;
        private GameObject _pausePanel;
        private Text _tutorialTitle;
        private Text _tutorialBody;
        private Text _tutorialStep;

        private float _accumulator;
        private float _timeScale = 1f;
        private float _slowTimer;
        private bool _paused;
        private float _endTimer = -1f;
        private bool _finished;
        private int _lastBossPhase = 1;

        public FightSimulation Simulation => _sim;
        public bool IsPaused => _paused;

        public static FightController Create(FightSetup setup, Services.ServiceHub services, RectTransform uiParent,
            Action<FightOutcome> onFinished, Action onQuit)
        {
            var go = new GameObject("Fight");
            var fc = go.AddComponent<FightController>();
            fc.Init(setup, services, uiParent, onFinished, onQuit);
            return fc;
        }

        private void Init(FightSetup setup, Services.ServiceHub services, RectTransform uiParent, Action<FightOutcome> onFinished, Action onQuit)
        {
            _setup = setup;
            _services = services;
            _onFinished = onFinished;
            _onQuit = onQuit;
            _sim = new FightSimulation(setup.Player, setup.Opponent, setup.Config);
            if (setup.Tutorial) _tutorial = new TutorialTracker(0);
            else _ai = new FightAI(1, setup.Difficulty, setup.Seed);

            _world = new GameObject("FightWorld");
            ArenaBuilder.Build(setup.Arena).transform.SetParent(_world.transform, false);

            _views = new FighterView[2];
            for (int i = 0; i < 2; i++)
            {
                var v = new GameObject("Fighter" + i).AddComponent<FighterView>();
                v.transform.SetParent(_world.transform, false);
                v.Setup(_sim.Fighters[i], i == 0 ? setup.PlayerLook : setup.OpponentLook);
                _views[i] = v;
            }

            var camGo = new GameObject("FightCamera");
            camGo.transform.SetParent(_world.transform, false);
            _camera = camGo.AddComponent<FightCamera>();
            _camera.Init(_views[0].transform, _views[1].transform, services.Settings.ScreenShakeScale);
            camGo.AddComponent<AudioListener>();

            _vfx = new GameObject("VFX").AddComponent<VfxPool>();
            _vfx.transform.SetParent(_world.transform, false);
            _vfx.Init(services.Settings.ReduceFlashes);

            _uiRoot = UIFactory.Stretch(UIFactory.Rect(uiParent, "FightUI"));
            _touch = new GameObject("Touch").AddComponent<TouchControls>();
            _touch.Build(_uiRoot, services.Settings, false);
            _hud = new GameObject("Hud").AddComponent<FightHud>();
            _hud.Build(_uiRoot, _sim, Title(setup.Player), Title(setup.Opponent), _camera.Camera);
            _hud.PauseButton.onClick.AddListener(() => SetPaused(true));
            if (_tutorial != null) BuildTutorialOverlay();

            services.Audio.PlayMusic(setup.Arena.MusicId);
            services.Audio.PlayAmbience(setup.Arena.AmbienceId);
            HandleEvents();
        }

        private static string Title(FighterDefinition f) => $"{f.DisplayName.ToUpperInvariant()}  \"{f.Nickname}\"";

        private void OnDestroy()
        {
            if (_world != null) Destroy(_world);
            if (_uiRoot != null) Destroy(_uiRoot.gameObject);
            if (_touch != null) Destroy(_touch.gameObject);
            if (_hud != null) Destroy(_hud.gameObject);
        }

        private void OnApplicationPause(bool pause)
        {
            // Backgrounding the app mid-fight should never cost the player a round.
            if (pause && !_finished) SetPaused(true);
        }

        public void SetPaused(bool paused)
        {
            if (_finished || _paused == paused) return;
            _paused = paused;
            _touch.ReleaseAll();
            if (paused) ShowPauseMenu();
            else if (_pausePanel != null) Destroy(_pausePanel);
        }

        private void Update()
        {
            if (_finished) return;
            if (PlayerInput.PausePressed()) SetPaused(!_paused);
            if (_paused) return;

            float dt = Mathf.Min(Time.unscaledDeltaTime, 0.1f);
            if (_slowTimer > 0f)
            {
                _slowTimer -= dt;
                if (_slowTimer <= 0f) _timeScale = 1f;
            }
            _accumulator += dt * _timeScale;

            int steps = 0;
            while (_accumulator >= Step && steps < MaxStepsPerFrame)
            {
                _accumulator -= Step;
                steps++;
                var playerInput = PlayerInput.Sample(_touch);
                var opponentInput = _tutorial != null ? _tutorial.DummyInput(_sim) : _ai.Think(_sim);
                _sim.Step(playerInput, opponentInput);
                HandleEvents();
                if (_tutorial != null) TickTutorial();
                CheckBossPhase();
            }
            if (steps == MaxStepsPerFrame) _accumulator = 0f; // device hitch: drop time rather than spiral

            foreach (var v in _views) v.Tick(_sim, dt * _timeScale);
            _touch.UpdateHints(_sim.Fighters[0]);

            if (_sim.IsMatchOver)
            {
                if (_endTimer < 0f) _endTimer = 1.6f;
                _endTimer -= dt;
                if (_endTimer <= 0f) Finish(false);
            }
        }

        private void CheckBossPhase()
        {
            if (_ai == null || !_ai.IsBoss || _ai.BossPhase == _lastBossPhase) return;
            _lastBossPhase = _ai.BossPhase;
            _hud.Banner($"{_setup.Opponent.Nickname.ToUpperInvariant()} TURNS UP THE PRESSURE");
            _services.Audio.PlaySfx("guard_break", 0.6f);
        }

        private void Finish(bool tutorialDone)
        {
            if (_finished) return;
            _finished = true;
            _touch.ReleaseAll();
            var outcome = new FightOutcome
            {
                Setup = _setup,
                Sim = _sim,
                Summary = MatchSummary.FromSimulation(_sim, 0),
                TutorialCompleted = tutorialDone,
            };
            _onFinished?.Invoke(outcome);
        }

        // ------------------------------------------------------------------ feedback

        private void HandleEvents()
        {
            var events = _sim.Events;
            for (int i = 0; i < events.Count; i++) HandleEvent(events[i]);
        }

        private void SlowMo(float scale, float seconds)
        {
            _timeScale = scale;
            _slowTimer = seconds;
        }

        private bool PlayerInvolved(in CombatEvent e) => e.Source == 0 || e.Target == 0;

        private float Pan(float x) => Mathf.Clamp(x / 5f, -1f, 1f) * 0.6f;

        private void HandleEvent(CombatEvent e)
        {
            var audio = _services.Audio;
            var haptics = _services.Haptics;
            var at = new Vector3(e.X, e.Y, -0.15f);
            MoveData move = e.Source >= 0 && e.MoveId != null ? _sim.Fighters[e.Source].Def.FindMove(e.MoveId) : null;

            switch (e.Type)
            {
                case CombatEventType.RoundAnnounce:
                    bool final = _sim.Fighters[0].RoundsWon == _sim.Config.RoundsToWin - 1 && _sim.Fighters[1].RoundsWon == _sim.Config.RoundsToWin - 1 && _sim.Config.RoundsToWin > 1;
                    string announceSub = _setup.Arena.DisplayName;
                    if (_setup.Run != null && _setup.Run.Bouts.Count > 1)
                        announceSub = $"BOUT {_setup.Run.Index + 1} OF {_setup.Run.Bouts.Count}  ·  {_setup.Opponent.DisplayName.ToUpperInvariant()}";
                    string title = _tutorial != null ? "TRAINING" : final ? "FINAL ROUND" : "ROUND " + e.Amount;
                    if (_setup.Run != null && _setup.Run.Bouts.Count > 1) title = "BOUT " + (_setup.Run.Index + 1);
                    _hud.Announce(title, announceSub, 1.6f);
                    audio.PlaySfx("round_bell", 0.8f, 0f);
                    _camera.Snap();
                    break;

                case CombatEventType.RoundStart:
                    _hud.Announce("FIGHT!", "", 0.8f, Theme.Text);
                    break;

                case CombatEventType.Hit:
                case CombatEventType.CounterHit:
                {
                    bool heavy = e.IsHeavy;
                    bool counter = e.Type == CombatEventType.CounterHit;
                    _vfx.Play(move?.ImpactVfxId ?? (heavy ? "spark_heavy" : "spark_light"), at);
                    if (heavy) _vfx.Play("sweat", at + Vector3.up * 0.1f);
                    audio.PlaySfx(move?.ImpactSoundId ?? "impact_light", heavy ? 1f : 0.8f, 0.07f, Pan(e.X));
                    if (heavy || UnityEngine.Random.value < 0.35f) audio.PlayVoice("hurt", heavy ? 0.9f : 0.6f);
                    _views[e.Target].React(heavy);
                    _camera.Shake(heavy ? 0.35f : 0.14f + (counter ? 0.1f : 0f));
                    if (PlayerInvolved(e)) haptics.Play(heavy ? HapticPattern.HeavyHit : HapticPattern.LightHit);
                    _hud.ShowCombo(e.Source, _sim.Fighters[e.Target].ComboCount, counter);
                    if (move != null && move.IsSpecial) SlowMo(0.35f, 0.35f);
                    break;
                }

                case CombatEventType.Blocked:
                    _vfx.Play("block_spark", at);
                    audio.PlaySfx("block", 0.8f, 0.08f, Pan(e.X));
                    _camera.Shake(0.05f);
                    if (e.Target == 0) haptics.Play(HapticPattern.Blocked);
                    break;

                case CombatEventType.GuardBreak:
                    _vfx.Play("parry_flash", at);
                    audio.PlaySfx("guard_break", 1f, 0f);
                    _camera.Shake(0.4f);
                    _hud.Callout("GUARD BREAK", _views[e.Target].HeadPosition, Theme.Danger);
                    if (PlayerInvolved(e)) haptics.Play(HapticPattern.HeavyHit);
                    break;

                case CombatEventType.Parried:
                    _vfx.Play("parry_flash", at);
                    audio.PlaySfx("parry", 1f, 0.03f, Pan(e.X));
                    _hud.Callout("PARRY!", _views[e.Source].HeadPosition, Theme.Teal);
                    if (PlayerInvolved(e)) haptics.Play(HapticPattern.Parry);
                    SlowMo(0.4f, 0.22f);
                    break;

                case CombatEventType.Evaded:
                    audio.PlaySfx("dodge", 0.6f);
                    _hud.Callout("EVADE", _views[e.Source].HeadPosition, Theme.Teal);
                    break;

                case CombatEventType.Whiff:
                    if (e.Source == 0 && move != null && move.Recovery >= 18) _hud.Callout("WHIFF", _views[0].HeadPosition, Theme.TextDim);
                    break;

                case CombatEventType.MoveStarted:
                    if (move != null && !move.IsThrow)
                    {
                        audio.PlaySfx(move.Damage >= 80 ? "whoosh_heavy" : "whoosh_light", 0.45f, 0.1f, Pan(e.X));
                        if (move.Damage >= 80 && UnityEngine.Random.value < 0.6f) audio.PlayVoice("effort_heavy", 0.7f);
                        else if (UnityEngine.Random.value < 0.25f) audio.PlayVoice("effort", 0.5f);
                    }
                    break;

                case CombatEventType.ThrowConnected:
                    _vfx.Play("dust", new Vector3(_sim.Fighters[e.Target].X, 0.05f, -0.1f));
                    audio.PlaySfx("throw_slam", 1f, 0.05f, Pan(e.X));
                    _camera.Shake(0.45f);
                    _views[e.Target].React(true);
                    if (PlayerInvolved(e)) haptics.Play(HapticPattern.Knockdown);
                    break;

                case CombatEventType.ThrowBroken:
                    audio.PlaySfx("block", 1f, 0f);
                    _hud.Callout("BREAK!", at + Vector3.up * 0.6f, Theme.Teal);
                    break;

                case CombatEventType.Clash:
                    _vfx.Play("spark_heavy", at);
                    _vfx.Play("parry_flash", at);
                    audio.PlaySfx("clash", 1f, 0.05f, Pan(e.X));
                    _camera.Shake(0.25f);
                    _hud.Callout("CLASH!", at + Vector3.up * 0.5f, Theme.Accent);
                    haptics.Play(HapticPattern.Parry);
                    SlowMo(0.45f, 0.18f);
                    break;

                case CombatEventType.Disarmed:
                    audio.PlaySfx("disarm", 0.9f, 0.05f, Pan(e.X));
                    _hud.Callout(e.Target == 0 ? "DISARMED!" : "WEAPON DOWN!", _views[e.Target].HeadPosition + Vector3.up * 0.25f, e.Target == 0 ? Theme.Danger : Theme.Success);
                    break;

                case CombatEventType.Rearmed:
                    audio.PlaySfx("disarm", 0.5f, 0.1f, Pan(e.X));
                    _hud.Callout("REARMED", _views[e.Source].HeadPosition + Vector3.up * 0.25f, Theme.TextDim);
                    break;

                case CombatEventType.Knockdown:
                    _vfx.Play("dust", new Vector3(e.X, 0.05f, -0.1f));
                    audio.PlaySfx("throw_slam", 0.5f, 0.1f, Pan(e.X));
                    break;

                case CombatEventType.Jump:
                case CombatEventType.Land:
                case CombatEventType.Footstep:
                    audio.PlaySfx("footstep", e.Type == CombatEventType.Footstep ? 0.25f : 0.45f, 0.15f, Pan(e.X));
                    break;

                case CombatEventType.Dodge:
                    audio.PlaySfx("dodge", 0.5f, 0.1f, Pan(e.X));
                    break;

                case CombatEventType.Exhausted:
                    if (e.Source == 0) _hud.Callout("EXHAUSTED", _views[0].HeadPosition, Theme.Danger);
                    break;

                case CombatEventType.SpecialReady:
                    if (e.Source == 0) audio.PlaySfx("special_ready", 0.7f, 0f);
                    break;

                case CombatEventType.SpecialStarted:
                    audio.PlaySfx("impact_special", 0.5f, 0f);
                    _camera.Focus(_views[e.Source].HeadPosition, 0.6f);
                    _hud.Callout(move?.DisplayName?.ToUpperInvariant() ?? "SPECIAL", _views[e.Source].HeadPosition + Vector3.up * 0.3f, Theme.Focus);
                    SlowMo(0.5f, 0.3f);
                    break;

                case CombatEventType.KO:
                    audio.PlaySfx("ko", 1f, 0f);
                    _camera.Shake(0.6f);
                    haptics.Play(HapticPattern.KO);
                    if (e.Target >= 0) _camera.Focus(_views[e.Target].transform.position + Vector3.up * 0.8f, e.IsFinal ? 1.6f : 0.8f);
                    if (e.IsFinal) SlowMo(0.25f, 1.3f);
                    _hud.Announce("K.O.", "", 1.6f, Theme.Danger);
                    break;

                case CombatEventType.TimeOut:
                    _hud.Announce("TIME", "", 1.4f, Theme.Text);
                    audio.PlaySfx("round_bell", 0.9f, 0f);
                    break;

                case CombatEventType.RoundEnd:
                {
                    if (_tutorial != null) break;
                    string who = e.Source == 0 ? _setup.Player.Nickname : e.Source == 1 ? _setup.Opponent.Nickname : null;
                    bool perfect = e.Source >= 0 && _sim.Fighters[e.Source].Health >= _sim.Fighters[e.Source].HealthAtRoundStart;
                    string sub = who == null ? "DRAW" : perfect ? $"{who.ToUpperInvariant()} WINS — PERFECT" : $"{who.ToUpperInvariant()} WINS THE ROUND";
                    _hud.Announce(e.Source == 0 ? "ROUND WON" : e.Source == 1 ? "ROUND LOST" : "DRAW", sub, 1.6f, e.Source == 0 ? Theme.Success : Theme.Danger);
                    break;
                }

                case CombatEventType.MatchEnd:
                {
                    bool won = e.Source == 0;
                    _hud.Announce(won ? "VICTORY" : e.Source == 1 ? "DEFEAT" : "DRAW", "", 2f, won ? Theme.Success : Theme.Danger);
                    audio.PlaySfx(won ? "victory" : "defeat", 0.9f, 0f);
                    break;
                }
            }
        }

        // ------------------------------------------------------------------ tutorial

        private void BuildTutorialOverlay()
        {
            var panel = UIFactory.Panel(_uiRoot, "TutorialPanel", new Color(0.05f, 0.06f, 0.08f, 0.88f));
            UIFactory.Fixed(panel.rectTransform, new Vector2(0.5f, 1f), new Vector2(0.5f, 1f), new Vector2(0, -230), new Vector2(980, 210));
            _tutorialStep = UIFactory.Label(panel.transform, "", 22, Theme.TextDim, TextAnchor.UpperLeft);
            UIFactory.Stretch(_tutorialStep.rectTransform, 28, 200, 14, 160);
            _tutorialTitle = UIFactory.Label(panel.transform, "", 38, Theme.Accent, TextAnchor.UpperLeft, FontStyle.Bold);
            UIFactory.Stretch(_tutorialTitle.rectTransform, 28, 200, 44, 110);
            _tutorialBody = UIFactory.Label(panel.transform, "", 28, Theme.Text, TextAnchor.UpperLeft);
            UIFactory.Stretch(_tutorialBody.rectTransform, 28, 200, 100, 14);
            var skip = UIFactory.Button(panel.transform, "SKIP STEP", () => { _tutorial.Skip(); RefreshTutorial(); }, ButtonStyle.Secondary, 24);
            UIFactory.Fixed((RectTransform)skip.transform, new Vector2(1f, 0.5f), new Vector2(1f, 0.5f), new Vector2(-20, 0), new Vector2(170, 70));
            RefreshTutorial();
        }

        private void RefreshTutorial()
        {
            if (_tutorial.IsComplete)
            {
                _tutorialStep.text = "TRAINING COMPLETE";
                _tutorialTitle.text = "Well done.";
                _tutorialBody.text = "You know every core technique. Your first real fight is waiting.";
                if (_endTimer < 0f)
                {
                    _endTimer = 0f;
                    Invoke(nameof(FinishTutorial), 2.5f);
                }
                return;
            }
            var step = _tutorial.Current;
            _tutorialStep.text = $"STEP {_tutorial.StepIndex + 1} / {TutorialTracker.Steps.Count}";
            _tutorialTitle.text = step.Title;
            _tutorialBody.text = step.Goal == TutorialGoal.Combo ? ComboInstruction() : step.Instruction;
        }

        private string ComboInstruction()
        {
            var route = _setup.Player.Combos.Count > 0 ? _setup.Player.Combos[0] : null;
            if (route == null || route.MoveIds.Length < 2) return TutorialTracker.Steps[(int)TutorialGoal.Combo].Instruction;
            var a = _setup.Player.FindMove(route.MoveIds[0]);
            var b = _setup.Player.FindMove(route.MoveIds[1]);
            return $"Land {a.DisplayName} ({Notation(a)}), then press {Notation(b)} the moment it connects to chain into {b.DisplayName}.";
        }

        private void FinishTutorial() => Finish(true);

        private void TickTutorial()
        {
            if (_tutorial.Observe(_sim))
            {
                _services.Audio.PlayUI("ui_confirm");
                _hud.Callout("NICE!", _views[0].HeadPosition, Theme.Success);
                RefreshTutorial();
            }
        }

        // ------------------------------------------------------------------ pause & move list

        private void ShowPauseMenu()
        {
            var overlay = UIFactory.Panel(_uiRoot, "Pause", new Color(0, 0, 0, 0.78f), rounded: false);
            UIFactory.Stretch(overlay.rectTransform);
            _pausePanel = overlay.gameObject;

            var col = UIFactory.VStack(overlay.transform, "Column", 22);
            UIFactory.Fixed((RectTransform)col.transform, new Vector2(0.5f, 0.5f), new Vector2(0.5f, 0.5f), Vector2.zero, new Vector2(560, 640));
            UIFactory.Size(UIFactory.Label(col.transform, "PAUSED", Theme.TitleSize, Theme.Accent, TextAnchor.MiddleCenter, FontStyle.Bold), 100);
            UIFactory.Size(UIFactory.Button(col.transform, "RESUME", () => SetPaused(false)), 96);
            UIFactory.Size(UIFactory.Button(col.transform, "MOVE LIST", ShowMoveList, ButtonStyle.Secondary), 90);
            if (_tutorial == null)
                UIFactory.Size(UIFactory.Button(col.transform, "RESTART MATCH", Restart, ButtonStyle.Secondary), 90);
            UIFactory.Size(UIFactory.Button(col.transform, "QUIT TO MENU", () => { _finished = true; _onQuit?.Invoke(); }, ButtonStyle.Danger), 90);
        }

        private void Restart()
        {
            var setup = _setup;
            var services = _services;
            var parent = (RectTransform)_uiRoot.parent;
            var finished = _onFinished;
            var quit = _onQuit;
            Destroy(gameObject);
            Create(setup, services, parent, finished, quit);
        }

        private void ShowMoveList()
        {
            var overlay = UIFactory.Panel(_pausePanel.transform, "MoveList", Theme.Background, rounded: false);
            UIFactory.Stretch(overlay.rectTransform);
            var title = UIFactory.Label(overlay.transform, $"{_setup.Player.DisplayName} — {_setup.Player.Style}", Theme.HeaderSize, Theme.Accent, TextAnchor.MiddleLeft, FontStyle.Bold);
            UIFactory.Anchor(title.rectTransform, 0.04f, 0.88f, 0.8f, 0.97f);
            var back = UIFactory.Button(overlay.transform, "BACK", () => Destroy(overlay.gameObject), ButtonStyle.Secondary);
            UIFactory.Fixed((RectTransform)back.transform, new Vector2(1f, 1f), new Vector2(1f, 1f), new Vector2(-40, -30), new Vector2(200, 80));

            var list = UIFactory.ScrollList(overlay.transform, "Moves", 6, 10);
            UIFactory.Anchor((RectTransform)list.parent, 0.04f, 0.04f, 0.96f, 0.86f);
            var header = UIFactory.Label(list, "MOVE                                   INPUT                 STARTUP   DAMAGE   PROPERTIES", 22, Theme.TextDim, TextAnchor.MiddleLeft);
            UIFactory.Size(header, 36);
            foreach (var m in _setup.Player.Moves)
            {
                var row = UIFactory.HStack(list, "Row", 10);
                UIFactory.Size(row, 52);
                UIFactory.Size(UIFactory.Label(row.transform, m.DisplayName, 26), -1, 520);
                UIFactory.Size(UIFactory.Label(row.transform, Notation(m), 26, Theme.Accent), -1, 380);
                UIFactory.Size(UIFactory.Label(row.transform, m.IsThrow ? "5f" : m.Startup + "f", 26, Theme.TextDim), -1, 160);
                UIFactory.Size(UIFactory.Label(row.transform, m.Damage.ToString(), 26, Theme.TextDim), -1, 150);
                UIFactory.Size(UIFactory.Label(row.transform, Properties(m), 22, Theme.TextDim), -1, 420);
            }
            UIFactory.Size(UIFactory.Label(list, "SIGNATURE COMBINATIONS", 30, Theme.Accent, TextAnchor.MiddleLeft, FontStyle.Bold), 70);
            foreach (var c in _setup.Player.Combos)
            {
                var sb = new StringBuilder();
                for (int i = 0; i < c.MoveIds.Length; i++)
                {
                    var m = _setup.Player.FindMove(c.MoveIds[i]);
                    if (i > 0) sb.Append("  >  ");
                    sb.Append(m != null ? Notation(m) : c.MoveIds[i]);
                }
                UIFactory.Size(UIFactory.Label(list, $"{c.Name}:   {sb}", 26), 50);
            }
        }

        public static string Notation(MoveData m)
        {
            string button;
            switch (m.Button)
            {
                case AttackButton.LightPunch: button = "LP"; break;
                case AttackButton.HeavyPunch: button = "HP"; break;
                case AttackButton.LightKick: button = "LK"; break;
                case AttackButton.HeavyKick: button = "HK"; break;
                case AttackButton.Throw: return "THROW (close)";
                default: return "SPECIAL (full Focus)";
            }
            switch (m.Direction)
            {
                case MoveDirection.Forward: return "Forward + " + button;
                case MoveDirection.Back: return "Back + " + button;
                case MoveDirection.Down: return "Down + " + button;
                case MoveDirection.Air: return "Jump, " + button;
                default: return button;
            }
        }

        private static string Properties(MoveData m)
        {
            var sb = new StringBuilder();
            switch (m.Height)
            {
                case AttackHeight.Low: sb.Append("Low  "); break;
                case AttackHeight.Overhead: sb.Append("Overhead  "); break;
                case AttackHeight.High: sb.Append("High  "); break;
                case AttackHeight.Throw: sb.Append("Unblockable  "); break;
            }
            if (m.CausesKnockdown) sb.Append("Knockdown  ");
            if (m.Armored) sb.Append("Armor  ");
            if (m.InvulnerableFrom > 0) sb.Append("Invulnerable start  ");
            if (m.CancelInto.Count > 0) sb.Append("Chains");
            return sb.ToString();
        }
    }
}
