using Undercurrent.Core.Combat;
using Undercurrent.Runtime.UI;
using UnityEngine;
using UnityEngine.UI;

namespace Undercurrent.Runtime.Fight
{
    /// <summary>Match HUD: health (with damage trail), stamina, focus, round pips, timer, announcer, combo and callouts.</summary>
    public sealed class FightHud : MonoBehaviour
    {
        private sealed class Side
        {
            public RectTransform HealthFill;
            public RectTransform TrailFill;
            public RectTransform StaminaFill;
            public RectTransform FocusFill;
            public Text FocusLabel;
            public Image[] Pips;
            public Text Name;
            public Text Combo;
            public float ComboTimer;
            public float Trail = 1f;
            public float TrailDelay;
            public float LastHealth = 1f;
        }

        private sealed class Toast
        {
            public Text Text;
            public float Timer;
            public Vector3 World;
        }

        private readonly Side[] _sides = new Side[2];
        private readonly Toast[] _toasts = new Toast[6];
        private int _nextToast;
        private Text _timer;
        private Text _round;
        private Text _announce;
        private Text _announceSub;
        private float _announceTimer;
        private float _announceDuration;
        private Text _banner;
        private float _bannerTimer;
        private FightSimulation _sim;
        private Camera _camera;
        private RectTransform _root;

        public Button PauseButton { get; private set; }

        public void Build(RectTransform parent, FightSimulation sim, string name0, string name1, Camera camera)
        {
            _sim = sim;
            _camera = camera;
            _root = UIFactory.Stretch(UIFactory.Rect(parent, "HUD"));
            transform.SetParent(_root, false);

            _sides[0] = BuildSide(0, name0);
            _sides[1] = BuildSide(1, name1);

            var timerBg = UIFactory.Panel(_root, "TimerBg", new Color(0, 0, 0, 0.55f));
            UIFactory.Fixed(timerBg.rectTransform, new Vector2(0.5f, 1f), new Vector2(0.5f, 1f), new Vector2(0, -22), new Vector2(150, 108));
            _timer = UIFactory.Label(timerBg.transform, "60", 64, Theme.Text, TextAnchor.MiddleCenter, FontStyle.Bold);
            UIFactory.Stretch(_timer.rectTransform, 4, 4, 0, 26);
            _round = UIFactory.Label(timerBg.transform, "ROUND 1", 20, Theme.TextDim, TextAnchor.LowerCenter);
            UIFactory.Stretch(_round.rectTransform, 4, 4, 70, 6);

            PauseButton = UIFactory.Button(_root, "II", null, ButtonStyle.Ghost, 30);
            UIFactory.Fixed((RectTransform)PauseButton.transform, new Vector2(0.5f, 1f), new Vector2(0.5f, 1f), new Vector2(0, -140), new Vector2(84, 64));

            _announce = UIFactory.Label(_root, "", 120, Theme.Accent, TextAnchor.MiddleCenter, FontStyle.Bold);
            UIFactory.Fixed(_announce.rectTransform, new Vector2(0.5f, 0.5f), new Vector2(0.5f, 0.5f), new Vector2(0, 120), new Vector2(1400, 180));
            var shadow = _announce.gameObject.AddComponent<Outline>();
            shadow.effectColor = new Color(0, 0, 0, 0.8f);
            shadow.effectDistance = new Vector2(4, -4);
            _announceSub = UIFactory.Label(_root, "", 40, Theme.Text, TextAnchor.MiddleCenter);
            UIFactory.Fixed(_announceSub.rectTransform, new Vector2(0.5f, 0.5f), new Vector2(0.5f, 0.5f), new Vector2(0, 20), new Vector2(1400, 70));

            _banner = UIFactory.Label(_root, "", 36, Theme.Danger, TextAnchor.MiddleCenter, FontStyle.Bold);
            UIFactory.Fixed(_banner.rectTransform, new Vector2(0.5f, 1f), new Vector2(0.5f, 1f), new Vector2(0, -230), new Vector2(1200, 60));

            for (int i = 0; i < _toasts.Length; i++)
            {
                var t = UIFactory.Label(_root, "", 40, Color.white, TextAnchor.MiddleCenter, FontStyle.Bold);
                t.gameObject.AddComponent<Outline>().effectColor = new Color(0, 0, 0, 0.85f);
                UIFactory.Fixed(t.rectTransform, Vector2.zero, new Vector2(0.5f, 0.5f), Vector2.zero, new Vector2(420, 60));
                t.enabled = false;
                _toasts[i] = new Toast { Text = t };
            }
        }

        private Side BuildSide(int index, string displayName)
        {
            bool left = index == 0;
            var s = new Side();
            float ax = left ? 0f : 1f;
            float sign = left ? 1f : -1f;
            var pivot = new Vector2(left ? 0f : 1f, 1f);

            var frame = UIFactory.Panel(_root, "HealthFrame" + index, new Color(0, 0, 0, 0.6f), rounded: false);
            UIFactory.Fixed(frame.rectTransform, new Vector2(ax, 1f), pivot, new Vector2(sign * 50, -36), new Vector2(760, 50));
            var trail = UIFactory.Panel(frame.transform, "Trail", Theme.HealthTrail, rounded: false);
            UIFactory.Stretch(trail.rectTransform, 4, 4, 4, 4);
            s.TrailFill = trail.rectTransform;
            var health = UIFactory.Panel(frame.transform, "Health", Theme.Health, rounded: false);
            UIFactory.Stretch(health.rectTransform, 4, 4, 4, 4);
            s.HealthFill = health.rectTransform;

            s.Name = UIFactory.Label(_root, displayName, 30, Theme.Text, left ? TextAnchor.UpperLeft : TextAnchor.UpperRight, FontStyle.Bold);
            UIFactory.Fixed(s.Name.rectTransform, new Vector2(ax, 1f), pivot, new Vector2(sign * 52, -92), new Vector2(520, 40));

            var stamina = UIFactory.Bar(_root, "Stamina" + index, new Color(0, 0, 0, 0.55f), Theme.Stamina, out s.StaminaFill);
            UIFactory.Fixed(stamina, new Vector2(ax, 1f), pivot, new Vector2(sign * 50, -136), new Vector2(460, 16));
            var focus = UIFactory.Bar(_root, "Focus" + index, new Color(0, 0, 0, 0.55f), Theme.Focus, out s.FocusFill);
            UIFactory.Fixed(focus, new Vector2(ax, 1f), pivot, new Vector2(sign * 50, -160), new Vector2(340, 14));
            s.FocusLabel = UIFactory.Label(_root, "", 22, Theme.Focus, left ? TextAnchor.MiddleLeft : TextAnchor.MiddleRight, FontStyle.Bold);
            UIFactory.Fixed(s.FocusLabel.rectTransform, new Vector2(ax, 1f), pivot, new Vector2(sign * 400, -150), new Vector2(260, 30));

            int pips = Mathf.Max(1, _sim.Config.RoundsToWin);
            s.Pips = new Image[pips];
            for (int i = 0; i < pips; i++)
            {
                var pip = UIFactory.Panel(_root, "Pip", new Color(1, 1, 1, 0.18f));
                pip.sprite = Presentation.VisualAssets.Circle;
                pip.type = Image.Type.Simple;
                UIFactory.Fixed(pip.rectTransform, new Vector2(ax, 1f), pivot, new Vector2(sign * (770 - i * 34), -96), new Vector2(26, 26));
                s.Pips[i] = pip;
            }

            s.Combo = UIFactory.Label(_root, "", 54, Theme.Accent, left ? TextAnchor.MiddleLeft : TextAnchor.MiddleRight, FontStyle.Bold);
            s.Combo.gameObject.AddComponent<Outline>().effectColor = new Color(0, 0, 0, 0.8f);
            UIFactory.Fixed(s.Combo.rectTransform, new Vector2(ax, 0.62f), new Vector2(left ? 0f : 1f, 0.5f), new Vector2(sign * 60, 0), new Vector2(460, 150));
            return s;
        }

        public void Announce(string text, string sub = "", float seconds = 1.3f, Color? color = null)
        {
            _announce.text = text;
            _announce.color = color ?? Theme.Accent;
            _announceSub.text = sub ?? "";
            _announceSub.color = Theme.Text;
            _announceTimer = seconds;
            _announceDuration = seconds;
        }

        public void Banner(string text, float seconds = 2.5f)
        {
            _banner.text = text;
            _bannerTimer = seconds;
        }

        public void ShowCombo(int attacker, int hits, bool counter)
        {
            if (attacker < 0 || attacker > 1) return;
            var s = _sides[attacker];
            if (hits < 2 && !counter) return;
            s.Combo.text = (hits >= 2 ? hits + " HITS" : "") + (counter ? (hits >= 2 ? "\n" : "") + "COUNTER" : "");
            s.ComboTimer = 1.4f;
        }

        public void Callout(string text, Vector3 world, Color color)
        {
            var t = _toasts[_nextToast];
            _nextToast = (_nextToast + 1) % _toasts.Length;
            t.Text.text = text;
            t.Text.color = color;
            t.Text.enabled = true;
            t.Timer = 0.9f;
            t.World = world;
        }

        private void Update()
        {
            if (_sim == null) return;
            float dt = Time.unscaledDeltaTime;
            _timer.text = _sim.RoundSecondsRemaining.ToString();
            _timer.color = _sim.RoundSecondsRemaining <= 10 ? Theme.Danger : Theme.Text;
            _round.text = "ROUND " + _sim.Round;

            for (int i = 0; i < 2; i++)
            {
                var f = _sim.Fighters[i];
                var s = _sides[i];
                bool fromRight = i == 1;
                float hp = f.HealthFraction;
                if (hp < s.LastHealth) s.TrailDelay = 0.45f;
                if (hp > s.Trail) s.Trail = hp;
                s.LastHealth = hp;
                s.TrailDelay -= dt;
                if (s.TrailDelay <= 0f) s.Trail = Mathf.MoveTowards(s.Trail, hp, dt * 0.6f);
                UIFactory.SetFill(s.HealthFill, hp, fromRight);
                UIFactory.SetFill(s.TrailFill, s.Trail, fromRight);
                UIFactory.SetFill(s.StaminaFill, f.StaminaFraction, fromRight);
                UIFactory.SetFill(s.FocusFill, f.Focus / FighterRuntime.FocusMax, fromRight);
                var special = f.Def.Resolve(AttackButton.Special, MoveDirection.Neutral);
                s.FocusLabel.text = special != null && f.Focus >= special.MeterCost ? "SPECIAL READY" : "";
                s.StaminaFill.GetComponent<Image>().color = f.Stamina < 15f ? Color.Lerp(Theme.Stamina, Theme.Danger, 0.5f + 0.5f * Mathf.Sin(Time.unscaledTime * 10f)) : Theme.Stamina;

                for (int p = 0; p < s.Pips.Length; p++)
                    s.Pips[p].color = f.RoundsWon > p ? Theme.Accent : new Color(1, 1, 1, 0.18f);

                if (s.ComboTimer > 0f)
                {
                    s.ComboTimer -= dt;
                    s.Combo.color = new Color(Theme.Accent.r, Theme.Accent.g, Theme.Accent.b, Mathf.Clamp01(s.ComboTimer * 2f));
                    if (s.ComboTimer <= 0f) s.Combo.text = "";
                }
            }

            if (_announceTimer > 0f)
            {
                _announceTimer -= dt;
                float t = 1f - _announceTimer / Mathf.Max(0.01f, _announceDuration);
                float scale = t < 0.12f ? Mathf.Lerp(1.6f, 1f, t / 0.12f) : 1f;
                _announce.rectTransform.localScale = Vector3.one * scale;
                float a = _announceTimer < 0.25f ? _announceTimer / 0.25f : 1f;
                var c = _announce.color; c.a = a; _announce.color = c;
                var cs = _announceSub.color; cs.a = a; _announceSub.color = cs;
                if (_announceTimer <= 0f) { _announce.text = ""; _announceSub.text = ""; }
            }

            if (_bannerTimer > 0f)
            {
                _bannerTimer -= dt;
                if (_bannerTimer <= 0f) _banner.text = "";
            }

            foreach (var t in _toasts)
            {
                if (!t.Text.enabled) continue;
                t.Timer -= dt;
                if (t.Timer <= 0f || _camera == null) { t.Text.enabled = false; continue; }
                var screen = _camera.WorldToScreenPoint(t.World + Vector3.up * (0.9f - t.Timer) * 0.6f);
                if (RectTransformUtility.ScreenPointToLocalPointInRectangle(_root, screen, null, out var local))
                {
                    t.Text.rectTransform.anchorMin = t.Text.rectTransform.anchorMax = new Vector2(0.5f, 0.5f);
                    t.Text.rectTransform.anchoredPosition = local;
                }
                var c = t.Text.color; c.a = Mathf.Clamp01(t.Timer * 3f); t.Text.color = c;
            }
        }
    }
}
