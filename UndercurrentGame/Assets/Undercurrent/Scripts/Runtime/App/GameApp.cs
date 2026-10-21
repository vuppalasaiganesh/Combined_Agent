using System;
using System.Collections.Generic;
using Undercurrent.Core.AI;
using Undercurrent.Core.Combat;
using Undercurrent.Core.Content;
using Undercurrent.Core.Progression;
using Undercurrent.Runtime.Fight;
using Undercurrent.Runtime.Presentation;
using Undercurrent.Runtime.Services;
using Undercurrent.Runtime.UI;
using UnityEngine;
using UnityEngine.EventSystems;
using UnityEngine.UI;

namespace Undercurrent.Runtime.App
{
    /// <summary>
    /// Game entry point and flow controller. Boots automatically after the first scene loads, so the Boot scene can be
    /// empty. Owns services, the UI canvas, the menu backdrop and the active fight.
    /// </summary>
    public sealed class GameApp : MonoBehaviour
    {
        public const string BuildLabel = "Undercurrent (working title) — development build";

        public static GameApp Instance { get; private set; }

        public ServiceHub Services { get; private set; }
        public RectTransform SafeRoot { get; private set; }
        public Navigator Navigator { get; private set; }
        public MenuBackdrop Backdrop { get; private set; }

        private Canvas _canvas;
        private FightController _fight;
        private Rect _appliedSafeArea;

        [RuntimeInitializeOnLoadMethod(RuntimeInitializeLoadType.AfterSceneLoad)]
        private static void Boot()
        {
            if (Instance != null || FindAnyObjectByType<GameApp>() != null) return;
            if (GameObject.Find("UndercurrentDisableAutoBoot") != null) return; // escape hatch for test scenes
            var go = new GameObject("UndercurrentGame");
            DontDestroyOnLoad(go);
            go.AddComponent<GameApp>();
        }

        private void Awake()
        {
            if (Instance != null && Instance != this)
            {
                Destroy(gameObject);
                return;
            }
            Instance = this;
            CrashLog.Install();

            Screen.orientation = ScreenOrientation.AutoRotation;
            Screen.autorotateToPortrait = false;
            Screen.autorotateToPortraitUpsideDown = false;
            Screen.autorotateToLandscapeLeft = true;
            Screen.autorotateToLandscapeRight = true;
#if ENABLE_LEGACY_INPUT_MANAGER
            Input.multiTouchEnabled = true; // legacy API throws when only the Input System package backend is active
#endif

            Services = new ServiceHub
            {
                Save = new SaveRepository(new FileSaveStore(), new JsonUtilitySaveCodec()),
            };
            Services.Save.Load();
            if (Services.Save.Status == SaveLoadStatus.NewSave)
            {
                Services.Settings.GraphicsTier = (int)QualityService.SuggestDefault();
                Services.Persist();
            }
            QualityService.Apply(Services.Settings);

            Services.Audio = gameObject.AddComponent<AudioService>();
            Services.Audio.Init(Services.Settings);
            Services.Haptics = new HapticsService(Services.Settings);
            UIFactory.Clicked += () => Services.Audio.PlayUI("ui_click");

            EnsureEventSystem();
            BuildCanvas();
            Backdrop = new GameObject("MenuBackdrop").AddComponent<MenuBackdrop>();
            Backdrop.transform.SetParent(transform, false);
            Navigator = new Navigator(this);

            if (Services.Save.Status == SaveLoadStatus.RecoveredFromCorruption)
                Navigator.Toast("Your save file could not be read. A backup was kept and a new profile started.");
            if (Services.Save.ReadOnly)
                Navigator.Toast("This save was created by a newer version. Progress will not be saved in this version.");

            ShowMainMenu();
        }

        private void EnsureEventSystem()
        {
            if (FindAnyObjectByType<EventSystem>() != null) return;
            var es = new GameObject("EventSystem");
            es.transform.SetParent(transform, false);
            var system = es.AddComponent<EventSystem>();
            system.pixelDragThreshold = 6;
#if UC_INPUT_SYSTEM && ENABLE_INPUT_SYSTEM
            var module = es.AddComponent<UnityEngine.InputSystem.UI.InputSystemUIInputModule>();
            module.AssignDefaultActions();
#else
            es.AddComponent<StandaloneInputModule>();
#endif
        }

        private void BuildCanvas()
        {
            var go = new GameObject("UI", typeof(RectTransform));
            go.transform.SetParent(transform, false);
            go.layer = 5;
            _canvas = go.AddComponent<Canvas>();
            _canvas.renderMode = RenderMode.ScreenSpaceOverlay;
            _canvas.sortingOrder = 10;
            var scaler = go.AddComponent<CanvasScaler>();
            scaler.uiScaleMode = CanvasScaler.ScaleMode.ScaleWithScreenSize;
            scaler.referenceResolution = new Vector2(1920, 1080);
            scaler.screenMatchMode = CanvasScaler.ScreenMatchMode.MatchWidthOrHeight;
            scaler.matchWidthOrHeight = 1f; // landscape phones vary in width; keep vertical sizing stable
            go.AddComponent<GraphicRaycaster>();

            SafeRoot = UIFactory.Rect(go.transform, "SafeArea");
            ApplySafeArea();
        }

        private void ApplySafeArea()
        {
            var safe = Screen.safeArea;
            if (safe == _appliedSafeArea || Screen.width <= 0 || Screen.height <= 0) return;
            _appliedSafeArea = safe;
            SafeRoot.anchorMin = new Vector2(safe.xMin / Screen.width, safe.yMin / Screen.height);
            SafeRoot.anchorMax = new Vector2(safe.xMax / Screen.width, safe.yMax / Screen.height);
            SafeRoot.offsetMin = Vector2.zero;
            SafeRoot.offsetMax = Vector2.zero;
        }

        private void Update()
        {
            ApplySafeArea();
            if (_fight == null && PlayerInput.PausePressed()) Navigator.Back();
        }

        private void OnApplicationPause(bool paused)
        {
            if (paused) Services?.Persist();
        }

        private void OnApplicationQuit() => Services?.Persist();

        // ------------------------------------------------------------------ flow

        public void ShowMainMenu()
        {
            Screen.sleepTimeout = SleepTimeout.SystemSetting;
            Backdrop.Show(Services.Data.SelectedFighterId, Services.Data.GetFighter(Services.Data.SelectedFighterId));
            Services.Audio.PlayMusic("music_menu");
            Services.Audio.StopAmbience();
            Navigator.Reset(new MainMenuScreen());
        }

        /// <summary>Starts a campaign level; the very first level runs guided training first if it was never completed.</summary>
        public void StartLevel(LevelDefinition level)
        {
            if (level.StartsTutorial && !Services.Data.TutorialCompleted)
            {
                StartTraining(() => StartLevel(level));
                return;
            }
            var arena = CampaignDatabase.GetArena(level.ArenaId);
            var setup = new FightSetup
            {
                Player = ProgressionService.BuildLoadout(Services.Data, Services.Data.SelectedFighterId),
                Opponent = FighterRoster.Get(level.OpponentId),
                PlayerLook = CharacterLook.For(Services.Data.SelectedFighterId, Services.Data.GetFighter(Services.Data.SelectedFighterId)),
                OpponentLook = CharacterLook.For(level.OpponentId),
                Arena = arena,
                Config = level.CreateMatchConfig(arena),
                Difficulty = level.Difficulty,
                Level = level,
                Seed = (uint)Environment.TickCount,
            };
            StartFight(setup, outcome => Navigator.Reset(new ResultsScreen(outcome, ApplyCampaignResult(outcome))));
        }

        public void StartVersus(string opponentId, AIDifficulty difficulty, string arenaId, int roundsToWin)
        {
            var arena = CampaignDatabase.GetArena(arenaId);
            var config = new MatchConfig { RoundsToWin = roundsToWin, MaxRounds = roundsToWin * 2 + 1, StageHalfWidth = arena.StageHalfWidth };
            var playerId = Services.Data.SelectedFighterId;
            var opponentLook = CharacterLook.For(opponentId);
            if (opponentId == playerId)
            {
                // Mirror match: give the CPU an alternate outfit so the fighters are distinguishable.
                opponentLook.Outfit = new Color(0.55f, 0.12f, 0.12f);
                opponentLook.OutfitTrim = new Color(0.95f, 0.8f, 0.3f);
            }
            var setup = new FightSetup
            {
                Player = ProgressionService.BuildLoadout(Services.Data, playerId),
                Opponent = FighterRoster.Get(opponentId),
                PlayerLook = CharacterLook.For(playerId, Services.Data.GetFighter(playerId)),
                OpponentLook = opponentLook,
                Arena = arena,
                Config = config,
                Difficulty = difficulty,
                Seed = (uint)Environment.TickCount,
            };
            StartFight(setup, outcome =>
            {
                var reward = ProgressionService.ApplyQuickFightResult(Services.Data, outcome.Summary);
                Services.Persist();
                Navigator.Reset(new ResultsScreen(outcome, reward));
            });
        }

        public void StartTraining(Action afterCompletion = null)
        {
            var playerId = Services.Data.SelectedFighterId;
            var setup = new FightSetup
            {
                Player = ProgressionService.BuildLoadout(Services.Data, playerId),
                Opponent = FighterRoster.Get(FighterRoster.Ines),
                PlayerLook = CharacterLook.For(playerId, Services.Data.GetFighter(playerId)),
                OpponentLook = CharacterLook.For(FighterRoster.Ines),
                Arena = CampaignDatabase.GetArena("pier9"),
                Config = TutorialTracker.CreateConfig(),
                Tutorial = true,
            };
            StartFight(setup, outcome =>
            {
                if (outcome.TutorialCompleted && !Services.Data.TutorialCompleted)
                {
                    Services.Data.TutorialCompleted = true;
                    Services.Persist();
                }
                if (afterCompletion != null) afterCompletion();
                else ShowMainMenu();
            });
        }

        private void StartFight(FightSetup setup, Action<FightOutcome> onFinished)
        {
            EndFight();
            Navigator.Clear();
            Backdrop.Hide();
            Screen.sleepTimeout = SleepTimeout.NeverSleep;
            _fight = FightController.Create(setup, Services, SafeRoot, outcome =>
            {
                EndFight();
                Backdrop.Show(Services.Data.SelectedFighterId, Services.Data.GetFighter(Services.Data.SelectedFighterId));
                Services.Audio.PlayMusic("music_menu");
                Services.Audio.StopAmbience();
                Screen.sleepTimeout = SleepTimeout.SystemSetting;
                onFinished(outcome);
            }, () =>
            {
                EndFight();
                ShowMainMenu();
            });
        }

        private void EndFight()
        {
            if (_fight != null) Destroy(_fight.gameObject);
            // A restart from the pause menu replaces the controller; make sure no stray one survives.
            foreach (var fc in FindObjectsByType<FightController>(FindObjectsSortMode.None)) Destroy(fc.gameObject);
            _fight = null;
        }

        private RewardResult ApplyCampaignResult(FightOutcome outcome)
        {
            var reward = ProgressionService.ApplyMatchResult(Services.Data, outcome.Setup.Level, outcome.Summary);
            Services.Persist();
            return reward;
        }
    }

    /// <summary>Screen stack with back navigation and simple toasts.</summary>
    public sealed class Navigator
    {
        private readonly GameApp _app;
        private readonly Stack<ScreenBase> _stack = new Stack<ScreenBase>();
        private RectTransform _current;

        public Navigator(GameApp app) => _app = app;

        public GameApp App => _app;

        public void Reset(ScreenBase screen)
        {
            _stack.Clear();
            Push(screen);
        }

        public void Push(ScreenBase screen)
        {
            _stack.Push(screen);
            Render();
        }

        /// <summary>Re-renders the top screen (after data changes).</summary>
        public void Refresh() => Render();

        public void Back()
        {
            if (_stack.Count <= 1) return;
            _stack.Pop();
            _app.Services.Audio.PlayUI("ui_back");
            Render();
        }

        public void Clear()
        {
            _stack.Clear();
            if (_current != null) UnityEngine.Object.Destroy(_current.gameObject);
            _current = null;
        }

        private void Render()
        {
            if (_current != null) UnityEngine.Object.Destroy(_current.gameObject);
            var screen = _stack.Peek();
            _current = UIFactory.Stretch(UIFactory.Rect(_app.SafeRoot, screen.GetType().Name));
            screen.Build(_current, this);
        }

        public void Toast(string message, float seconds = 4f)
        {
            var panel = UIFactory.Panel(_app.SafeRoot, "Toast", new Color(0.08f, 0.09f, 0.11f, 0.95f));
            UIFactory.Fixed(panel.rectTransform, new Vector2(0.5f, 0f), new Vector2(0.5f, 0f), new Vector2(0, 40), new Vector2(1300, 110));
            var t = UIFactory.Label(panel.transform, message, Theme.BodySize, Theme.Text, TextAnchor.MiddleCenter);
            UIFactory.Stretch(t.rectTransform, 24, 24, 10, 10);
            UnityEngine.Object.Destroy(panel.gameObject, seconds);
        }

        public void Confirm(string message, string confirmLabel, Action onConfirm)
        {
            var overlay = UIFactory.Panel(_app.SafeRoot, "Confirm", new Color(0, 0, 0, 0.75f), rounded: false);
            UIFactory.Stretch(overlay.rectTransform);
            var box = UIFactory.Panel(overlay.transform, "Box", Theme.Panel);
            UIFactory.Fixed(box.rectTransform, new Vector2(0.5f, 0.5f), new Vector2(0.5f, 0.5f), Vector2.zero, new Vector2(1000, 420));
            var text = UIFactory.Label(box.transform, message, Theme.BodySize, Theme.Text, TextAnchor.MiddleCenter);
            UIFactory.Stretch(text.rectTransform, 40, 40, 30, 150);
            var row = UIFactory.HStack(box.transform, "Buttons", 30);
            UIFactory.Anchor((RectTransform)row.transform, 0.08f, 0.06f, 0.92f, 0.3f);
            UIFactory.Size(UIFactory.Button(row.transform, "CANCEL", () => UnityEngine.Object.Destroy(overlay.gameObject), ButtonStyle.Secondary), -1, 380, 1);
            UIFactory.Size(UIFactory.Button(row.transform, confirmLabel, () => { UnityEngine.Object.Destroy(overlay.gameObject); onConfirm(); }, ButtonStyle.Danger), -1, 380, 1);
        }
    }

    public abstract class ScreenBase
    {
        public abstract void Build(RectTransform root, Navigator nav);
    }

    /// <summary>
    /// 3D scene behind the menus: the selected fighter idling under arena lighting, driven by a tiny private
    /// simulation so the stance, breathing and cosmetics shown are exactly what appears in a fight.
    /// </summary>
    public sealed class MenuBackdrop : MonoBehaviour
    {
        private GameObject _root;
        private FightSimulation _sim;
        private FighterView _view;
        private float _acc;

        public void Show(string fighterId, FighterProgress progress)
        {
            Hide();
            _root = new GameObject("BackdropRoot");
            _root.transform.SetParent(transform, false);
            ArenaBuilder.Build(CampaignDatabase.GetArena("pier9")).transform.SetParent(_root.transform, false);

            var def = FighterRoster.Get(fighterId) ?? FighterRoster.Get(FighterRoster.Mara);
            _sim = new FightSimulation(def, FighterRoster.Get(FighterRoster.Ines),
                new MatchConfig { IntroTicks = 1, RoundSeconds = 9999, StartSeparation = 6f });
            _sim.Step(FrameInput.Empty, FrameInput.Empty);
            _view = new GameObject("BackdropFighter").AddComponent<FighterView>();
            _view.transform.SetParent(_root.transform, false);
            _view.Setup(_sim.Fighters[0], CharacterLook.For(def.Id, progress));

            var camGo = new GameObject("BackdropCamera");
            camGo.transform.SetParent(_root.transform, false);
            var cam = camGo.AddComponent<Camera>();
            cam.fieldOfView = 34f;
            cam.clearFlags = CameraClearFlags.SolidColor;
            cam.backgroundColor = new Color(0.02f, 0.02f, 0.03f);
            camGo.AddComponent<AudioListener>();
            var target = new Vector3(_sim.Fighters[0].X, 1.05f, 0f);
            camGo.transform.position = target + new Vector3(1.6f, 0.35f, -3.6f);
            camGo.transform.LookAt(target + new Vector3(1.0f, 0f, 0f));
        }

        public void Hide()
        {
            if (_root != null) Destroy(_root);
            _root = null;
            _sim = null;
            _view = null;
        }

        private void Update()
        {
            if (_sim == null) return;
            _acc += Time.unscaledDeltaTime;
            while (_acc >= SimClock.SecondsPerTick)
            {
                _acc -= SimClock.SecondsPerTick;
                _sim.Step(FrameInput.Empty, FrameInput.Empty);
            }
            _view.Tick(_sim, Time.unscaledDeltaTime);
        }
    }
}
