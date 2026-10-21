using System;
using System.Collections.Generic;
using Undercurrent.Core.Combat;
using Undercurrent.Core.Progression;
using Undercurrent.Runtime.Presentation;
using Undercurrent.Runtime.UI;
using UnityEngine;
using UnityEngine.EventSystems;
using UnityEngine.UI;

namespace Undercurrent.Runtime.Fight
{
    /// <summary>
    /// Two-thumb landscape layout: an 8-way movement pad on the left, attack/defence cluster on the right.
    /// Every control can be repositioned in layout-edit mode; scale and opacity come from settings.
    /// </summary>
    public sealed class TouchControls : MonoBehaviour
    {
        private sealed class Spec
        {
            public string Id;
            public string Label;
            public string Sub;
            public InputButtons Buttons;
            public Vector2 Anchor;
            public Vector2 Position;
            public float Size;
            public Color Color;
        }

        private static readonly Color AttackColor = new Color(0.92f, 0.62f, 0.22f);
        private static readonly Color DefenseColor = new Color(0.22f, 0.62f, 0.75f);
        private static readonly Color UtilityColor = new Color(0.55f, 0.55f, 0.62f);

        private static readonly Spec[] Specs =
        {
            new Spec { Id = "btn_lp", Label = "LP", Sub = "Light Punch", Buttons = InputButtons.LightPunch, Anchor = new Vector2(1, 0), Position = new Vector2(-470, 330), Size = 150, Color = AttackColor },
            new Spec { Id = "btn_hp", Label = "HP", Sub = "Heavy Punch", Buttons = InputButtons.HeavyPunch, Anchor = new Vector2(1, 0), Position = new Vector2(-300, 400), Size = 150, Color = AttackColor },
            new Spec { Id = "btn_lk", Label = "LK", Sub = "Light Kick", Buttons = InputButtons.LightKick, Anchor = new Vector2(1, 0), Position = new Vector2(-520, 160), Size = 150, Color = AttackColor },
            new Spec { Id = "btn_hk", Label = "HK", Sub = "Heavy Kick", Buttons = InputButtons.HeavyKick, Anchor = new Vector2(1, 0), Position = new Vector2(-350, 225), Size = 150, Color = AttackColor },
            new Spec { Id = "btn_block", Label = "BLOCK", Buttons = InputButtons.Block, Anchor = new Vector2(1, 0), Position = new Vector2(-150, 300), Size = 165, Color = DefenseColor },
            new Spec { Id = "btn_dodge", Label = "DODGE", Buttons = InputButtons.Dodge, Anchor = new Vector2(1, 0), Position = new Vector2(-165, 115), Size = 125, Color = DefenseColor },
            new Spec { Id = "btn_parry", Label = "PARRY", Buttons = InputButtons.Parry, Anchor = new Vector2(1, 0), Position = new Vector2(-140, 490), Size = 125, Color = DefenseColor },
            new Spec { Id = "btn_throw", Label = "THROW", Buttons = InputButtons.Throw, Anchor = new Vector2(1, 0), Position = new Vector2(-680, 250), Size = 120, Color = UtilityColor },
            new Spec { Id = "btn_special", Label = "SPECIAL", Buttons = InputButtons.Special, Anchor = new Vector2(1, 0), Position = new Vector2(-470, 520), Size = 135, Color = new Color(0.45f, 0.55f, 1f) },
        };

        public const string PadId = "pad_move";
        private static readonly Vector2 PadAnchor = new Vector2(0, 0);
        private static readonly Vector2 PadPosition = new Vector2(270, 270);
        private const float PadSize = 380f;

        private readonly List<TouchButton> _buttons = new List<TouchButton>();
        private DirectionPad _pad;
        private SettingsData _settings;
        private TouchButton _special;
        private float _time;

        public bool EditMode { get; private set; }

        public InputButtons Held
        {
            get
            {
                if (EditMode) return InputButtons.None;
                var held = _pad != null ? _pad.Held : InputButtons.None;
                foreach (var b in _buttons) if (b.IsHeld) held |= b.Buttons;
                return held;
            }
        }

        public void Build(RectTransform parent, SettingsData settings, bool editMode)
        {
            _settings = settings;
            EditMode = editMode;
            var root = UIFactory.Stretch(UIFactory.Rect(parent, "TouchControls"));
            transform.SetParent(root, false);

            _pad = DirectionPad.Create(root, PadSize, settings.ControlOpacity);
            Place(_pad.Rect, PadId, PadAnchor, PadPosition, PadSize);
            _pad.EditMode = editMode;
            _pad.OnMoved = OnControlMoved;

            foreach (var s in Specs)
            {
                var b = TouchButton.Create(root, s.Label, settings.ShowInputHints ? s.Sub : null, s.Buttons, s.Size, s.Color, settings.ControlOpacity);
                b.Id = s.Id;
                b.EditMode = editMode;
                b.OnMoved = OnControlMoved;
                Place(b.Rect, s.Id, s.Anchor, s.Position, s.Size);
                _buttons.Add(b);
                if (s.Buttons == InputButtons.Special) _special = b;
            }
        }

        private void Place(RectTransform rt, string id, Vector2 anchor, Vector2 position, float size)
        {
            var offset = _settings.GetOffset(id);
            var pos = position + (offset != null ? new Vector2(offset.X, offset.Y) : Vector2.zero);
            float scale = _settings.ControlScale;
            UIFactory.Fixed(rt, anchor, new Vector2(0.5f, 0.5f), pos, Vector2.one * size * scale);
            rt.GetComponent<DraggableControl>().Init(id, anchor, position);
        }

        private void OnControlMoved(string id, Vector2 defaultPosition, Vector2 newPosition)
        {
            var delta = newPosition - defaultPosition;
            _settings.SetOffset(id, delta.x, delta.y);
        }

        public void ResetLayout()
        {
            _settings.ControlOffsets.Clear();
            Place(_pad.Rect, PadId, PadAnchor, PadPosition, PadSize);
            for (int i = 0; i < Specs.Length; i++) Place(_buttons[i].Rect, Specs[i].Id, Specs[i].Anchor, Specs[i].Position, Specs[i].Size);
        }

        public void ReleaseAll()
        {
            foreach (var b in _buttons) b.Release();
            _pad?.Release();
        }

        /// <summary>Visual state hints: special glows when ready, attacks dim when exhausted.</summary>
        public void UpdateHints(FighterRuntime player)
        {
            if (player == null) return;
            _time += Time.unscaledDeltaTime;
            var special = player.Def.Resolve(AttackButton.Special, MoveDirection.Neutral);
            bool ready = special != null && player.Focus >= special.MeterCost;
            if (_special != null) _special.SetHighlight(ready ? 0.5f + 0.5f * Mathf.Sin(_time * 6f) : 0f, ready);
            bool tired = player.Stamina < 8f;
            foreach (var b in _buttons)
                if ((b.Buttons & (InputButtons.LightPunch | InputButtons.HeavyPunch | InputButtons.LightKick | InputButtons.HeavyKick)) != 0)
                    b.SetDimmed(tired);
        }
    }

    /// <summary>Lets a control be dragged in layout-edit mode and reports its new anchored position.</summary>
    public class DraggableControl : MonoBehaviour, IDragHandler
    {
        public string Id;
        public bool EditMode;
        public Action<string, Vector2, Vector2> OnMoved;
        private Vector2 _default;
        private RectTransform _rt;

        public RectTransform Rect => _rt != null ? _rt : (_rt = (RectTransform)transform);

        public void Init(string id, Vector2 anchor, Vector2 defaultPosition)
        {
            Id = id;
            _default = defaultPosition;
        }

        public virtual void OnDrag(PointerEventData e)
        {
            if (!EditMode) return;
            var canvas = GetComponentInParent<Canvas>();
            float scale = canvas != null ? canvas.scaleFactor : 1f;
            Rect.anchoredPosition += e.delta / Mathf.Max(0.01f, scale);
            OnMoved?.Invoke(Id, _default, Rect.anchoredPosition);
        }
    }

    public sealed class TouchButton : DraggableControl, IPointerDownHandler, IPointerUpHandler
    {
        public InputButtons Buttons;
        private readonly HashSet<int> _pointers = new HashSet<int>();
        private Image _bg;
        private Image _glow;
        private Color _color;
        private float _opacity;
        private bool _dimmed;

        public bool IsHeld => _pointers.Count > 0;

        public static TouchButton Create(Transform parent, string label, string sub, InputButtons buttons, float size, Color color, float opacity)
        {
            var glow = UIFactory.Panel(parent, "Btn_" + label, new Color(1, 1, 1, 0));
            glow.sprite = VisualAssets.Circle;
            glow.type = Image.Type.Simple;
            var b = glow.gameObject.AddComponent<TouchButton>();
            b._glow = glow;
            b.Buttons = buttons;
            b._color = color;
            b._opacity = opacity;

            var bg = UIFactory.Panel(glow.transform, "Face", color);
            bg.sprite = VisualAssets.Circle;
            bg.type = Image.Type.Simple;
            UIFactory.Stretch(bg.rectTransform, 8, 8, 8, 8);
            bg.raycastTarget = false;
            b._bg = bg;

            var t = UIFactory.Label(bg.transform, label, label.Length <= 2 ? 46 : 26, Color.white, TextAnchor.MiddleCenter, FontStyle.Bold);
            UIFactory.Stretch(t.rectTransform, 6, 6, 6, string.IsNullOrEmpty(sub) ? 6 : 34);
            if (!string.IsNullOrEmpty(sub))
            {
                var st = UIFactory.Label(bg.transform, sub, 16, new Color(1, 1, 1, 0.85f), TextAnchor.LowerCenter);
                UIFactory.Stretch(st.rectTransform, 10, 10, 6, 22);
            }
            b.Apply();
            return b;
        }

        private void Apply()
        {
            var c = _color;
            float pressed = IsHeld ? 1.25f : 1f;
            float dim = _dimmed ? 0.45f : 1f;
            _bg.color = new Color(Mathf.Min(1f, c.r * pressed), Mathf.Min(1f, c.g * pressed), Mathf.Min(1f, c.b * pressed), _opacity * dim);
            _bg.rectTransform.localScale = Vector3.one * (IsHeld ? 0.93f : 1f);
        }

        public void SetHighlight(float amount, bool enabled)
        {
            if (_glow != null) _glow.color = new Color(0.6f, 0.75f, 1f, enabled ? 0.25f + 0.45f * amount : 0f);
            _dimmed = !enabled;
            Apply();
        }

        public void SetDimmed(bool dimmed)
        {
            if (_dimmed == dimmed) return;
            _dimmed = dimmed;
            Apply();
        }

        public void OnPointerDown(PointerEventData e)
        {
            if (EditMode) return;
            _pointers.Add(e.pointerId);
            Apply();
        }

        public void OnPointerUp(PointerEventData e)
        {
            _pointers.Remove(e.pointerId);
            Apply();
        }

        public void Release()
        {
            _pointers.Clear();
            Apply();
        }

        private void OnDisable() => _pointers.Clear();
    }

    /// <summary>
    /// Floating-knob movement pad. Up = jump, down = crouch; diagonals combine (e.g. up-forward jump).
    /// A 25% dead zone avoids accidental crouches when resting a thumb.
    /// </summary>
    public sealed class DirectionPad : DraggableControl, IPointerDownHandler, IPointerUpHandler
    {
        private RectTransform _knob;
        private int _pointer = int.MinValue;
        private Vector2 _vector;
        public InputButtons Held { get; private set; }

        public static DirectionPad Create(Transform parent, float size, float opacity)
        {
            var bg = UIFactory.Panel(parent, "MovePad", new Color(1f, 1f, 1f, 0.12f * opacity / 0.75f));
            bg.sprite = VisualAssets.Circle;
            bg.type = Image.Type.Simple;
            var pad = bg.gameObject.AddComponent<DirectionPad>();
            var knobImg = UIFactory.Panel(bg.transform, "Knob", new Color(1f, 1f, 1f, 0.35f * opacity / 0.75f));
            knobImg.sprite = VisualAssets.Circle;
            knobImg.type = Image.Type.Simple;
            knobImg.raycastTarget = false;
            pad._knob = knobImg.rectTransform;
            UIFactory.Fixed(pad._knob, new Vector2(0.5f, 0.5f), new Vector2(0.5f, 0.5f), Vector2.zero, Vector2.one * size * 0.38f);
            foreach (var (txt, anchor) in new[] { ("<", new Vector2(0.12f, 0.5f)), (">", new Vector2(0.88f, 0.5f)), ("^", new Vector2(0.5f, 0.88f)), ("v", new Vector2(0.5f, 0.12f)) })
            {
                var l = UIFactory.Label(bg.transform, txt, 44, new Color(1, 1, 1, 0.7f), TextAnchor.MiddleCenter, FontStyle.Bold);
                UIFactory.Fixed(l.rectTransform, anchor, new Vector2(0.5f, 0.5f), Vector2.zero, new Vector2(70, 70));
            }
            return pad;
        }

        public void OnPointerDown(PointerEventData e)
        {
            if (EditMode) return;
            _pointer = e.pointerId;
            UpdateVector(e);
        }

        public override void OnDrag(PointerEventData e)
        {
            if (EditMode) { base.OnDrag(e); return; }
            if (e.pointerId != _pointer) return;
            UpdateVector(e);
        }

        public void OnPointerUp(PointerEventData e)
        {
            if (e.pointerId != _pointer) return;
            Release();
        }

        public void Release()
        {
            _pointer = int.MinValue;
            _vector = Vector2.zero;
            Held = InputButtons.None;
            if (_knob != null) _knob.anchoredPosition = Vector2.zero;
        }

        private void UpdateVector(PointerEventData e)
        {
            if (!RectTransformUtility.ScreenPointToLocalPointInRectangle(Rect, e.position, e.pressEventCamera, out var local)) return;
            float radius = Rect.rect.width * 0.5f;
            _vector = Vector2.ClampMagnitude(local / Mathf.Max(1f, radius), 1f);
            _knob.anchoredPosition = _vector * radius * 0.6f;

            var held = InputButtons.None;
            if (_vector.magnitude > 0.25f)
            {
                float angle = Mathf.Atan2(_vector.y, _vector.x) * Mathf.Rad2Deg; // -180..180, 0 = right
                if (angle > -67.5f && angle < 67.5f) held |= InputButtons.Right;
                if (angle > 112.5f || angle < -112.5f) held |= InputButtons.Left;
                if (angle > 22.5f && angle < 157.5f) held |= InputButtons.Up;
                if (angle < -22.5f && angle > -157.5f) held |= InputButtons.Down;
            }
            Held = held;
        }

        private void OnDisable() => Release();
    }
}
