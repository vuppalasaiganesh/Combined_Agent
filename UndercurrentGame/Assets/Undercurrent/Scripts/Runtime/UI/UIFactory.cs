using System;
using Undercurrent.Runtime.Presentation;
using UnityEngine;
using UnityEngine.EventSystems;
using UnityEngine.UI;

namespace Undercurrent.Runtime.UI
{
    /// <summary>Visual language shared by every screen.</summary>
    public static class Theme
    {
        public static readonly Color Background = new Color(0.055f, 0.067f, 0.086f, 0.97f);
        public static readonly Color Panel = new Color(0.10f, 0.12f, 0.15f, 0.96f);
        public static readonly Color PanelLight = new Color(0.16f, 0.19f, 0.23f, 0.96f);
        public static readonly Color Accent = new Color(0.91f, 0.64f, 0.24f, 1f);
        public static readonly Color AccentDark = new Color(0.55f, 0.36f, 0.10f, 1f);
        public static readonly Color Teal = new Color(0.20f, 0.70f, 0.72f, 1f);
        public static readonly Color Text = new Color(0.93f, 0.94f, 0.95f, 1f);
        public static readonly Color TextDim = new Color(0.62f, 0.66f, 0.71f, 1f);
        public static readonly Color Danger = new Color(0.86f, 0.25f, 0.22f, 1f);
        public static readonly Color Success = new Color(0.35f, 0.78f, 0.42f, 1f);
        public static readonly Color Disabled = new Color(0.25f, 0.27f, 0.30f, 0.9f);
        public static readonly Color Health = new Color(0.86f, 0.22f, 0.18f, 1f);
        public static readonly Color HealthTrail = new Color(1f, 0.85f, 0.55f, 1f);
        public static readonly Color Stamina = new Color(0.33f, 0.80f, 0.45f, 1f);
        public static readonly Color Focus = new Color(0.45f, 0.62f, 1f, 1f);

        public const int TitleSize = 64;
        public const int HeaderSize = 44;
        public const int BodySize = 30;
        public const int SmallSize = 24;
    }

    public enum ButtonStyle
    {
        Primary,
        Secondary,
        Danger,
        Ghost,
    }

    /// <summary>Builds uGUI hierarchies in code so screens do not depend on hand-authored prefabs.</summary>
    public static class UIFactory
    {
        /// <summary>Raised on every button click; the audio service plays the UI sound.</summary>
        public static event Action Clicked;

        public static RectTransform Rect(Transform parent, string name)
        {
            var go = new GameObject(name, typeof(RectTransform));
            go.layer = 5; // UI
            var rt = (RectTransform)go.transform;
            rt.SetParent(parent, false);
            return rt;
        }

        public static RectTransform Stretch(RectTransform rt, float left = 0, float right = 0, float top = 0, float bottom = 0)
        {
            rt.anchorMin = Vector2.zero;
            rt.anchorMax = Vector2.one;
            rt.offsetMin = new Vector2(left, bottom);
            rt.offsetMax = new Vector2(-right, -top);
            return rt;
        }

        /// <summary>Anchors a rect to a normalized region of its parent.</summary>
        public static RectTransform Anchor(RectTransform rt, float xMin, float yMin, float xMax, float yMax)
        {
            rt.anchorMin = new Vector2(xMin, yMin);
            rt.anchorMax = new Vector2(xMax, yMax);
            rt.offsetMin = Vector2.zero;
            rt.offsetMax = Vector2.zero;
            return rt;
        }

        /// <summary>Fixed-size rect positioned relative to an anchor point.</summary>
        public static RectTransform Fixed(RectTransform rt, Vector2 anchor, Vector2 pivot, Vector2 position, Vector2 size)
        {
            rt.anchorMin = anchor;
            rt.anchorMax = anchor;
            rt.pivot = pivot;
            rt.anchoredPosition = position;
            rt.sizeDelta = size;
            return rt;
        }

        public static Image Panel(Transform parent, string name, Color color, bool rounded = true)
        {
            var rt = Rect(parent, name);
            var img = rt.gameObject.AddComponent<Image>();
            img.sprite = rounded ? VisualAssets.Rounded : VisualAssets.White;
            img.type = rounded ? Image.Type.Sliced : Image.Type.Simple;
            img.color = color;
            return img;
        }

        public static Text Label(Transform parent, string text, int size = Theme.BodySize, Color? color = null,
            TextAnchor align = TextAnchor.MiddleLeft, FontStyle style = FontStyle.Normal, bool bestFit = true)
        {
            var rt = Rect(parent, "Text");
            var t = rt.gameObject.AddComponent<Text>();
            t.font = VisualAssets.Font;
            t.text = text;
            t.fontSize = size;
            t.fontStyle = style;
            t.color = color ?? Theme.Text;
            t.alignment = align;
            t.horizontalOverflow = HorizontalWrapMode.Wrap;
            t.verticalOverflow = VerticalWrapMode.Truncate;
            t.raycastTarget = false;
            if (bestFit)
            {
                // Shrinks instead of cropping when a translation or long name does not fit.
                t.resizeTextForBestFit = true;
                t.resizeTextMinSize = Mathf.Max(10, size / 2);
                t.resizeTextMaxSize = size;
            }
            return t;
        }

        public static LayoutElement Size(Component c, float preferredHeight = -1, float preferredWidth = -1, float flexibleWidth = -1, float flexibleHeight = -1)
        {
            // Explicit null check: '??' bypasses UnityEngine.Object's overloaded null semantics.
            var le = c.GetComponent<LayoutElement>();
            if (le == null) le = c.gameObject.AddComponent<LayoutElement>();
            if (preferredHeight >= 0) { le.preferredHeight = preferredHeight; le.minHeight = preferredHeight; }
            if (preferredWidth >= 0) { le.preferredWidth = preferredWidth; le.minWidth = preferredWidth; }
            if (flexibleWidth >= 0) le.flexibleWidth = flexibleWidth;
            if (flexibleHeight >= 0) le.flexibleHeight = flexibleHeight;
            return le;
        }

        public static Button Button(Transform parent, string label, Action onClick, ButtonStyle style = ButtonStyle.Primary, int fontSize = Theme.BodySize)
        {
            Color bg;
            Color fg = Theme.Text;
            switch (style)
            {
                case ButtonStyle.Primary: bg = Theme.Accent; fg = new Color(0.08f, 0.07f, 0.06f); break;
                case ButtonStyle.Danger: bg = Theme.Danger; break;
                case ButtonStyle.Ghost: bg = new Color(1f, 1f, 1f, 0.06f); break;
                default: bg = Theme.PanelLight; break;
            }
            var img = Panel(parent, "Button_" + label, bg);
            var button = img.gameObject.AddComponent<Button>();
            button.targetGraphic = img;
            var colors = button.colors;
            colors.normalColor = Color.white;
            colors.highlightedColor = new Color(1.08f, 1.08f, 1.08f);
            colors.pressedColor = new Color(0.78f, 0.78f, 0.78f);
            colors.disabledColor = new Color(0.45f, 0.45f, 0.45f, 0.8f);
            colors.fadeDuration = 0.06f;
            button.colors = colors;
            var text = Label(img.transform, label, fontSize, fg, TextAnchor.MiddleCenter, FontStyle.Bold);
            Stretch(text.rectTransform, 16, 16, 6, 6);
            button.onClick.AddListener(() =>
            {
                Clicked?.Invoke();
                onClick?.Invoke();
            });
            return button;
        }

        public static void SetButtonLabel(Button b, string label)
        {
            var t = b.GetComponentInChildren<Text>();
            if (t != null) t.text = label;
        }

        public static VerticalLayoutGroup VStack(Transform parent, string name, float spacing = 16, int padding = 0,
            TextAnchor align = TextAnchor.UpperCenter, bool expandWidth = true)
        {
            var rt = Rect(parent, name);
            var v = rt.gameObject.AddComponent<VerticalLayoutGroup>();
            v.spacing = spacing;
            v.padding = new RectOffset(padding, padding, padding, padding);
            v.childAlignment = align;
            v.childControlWidth = true;
            v.childControlHeight = true;
            v.childForceExpandWidth = expandWidth;
            v.childForceExpandHeight = false;
            return v;
        }

        public static HorizontalLayoutGroup HStack(Transform parent, string name, float spacing = 16, int padding = 0,
            TextAnchor align = TextAnchor.MiddleCenter, bool expandHeight = true)
        {
            var rt = Rect(parent, name);
            var h = rt.gameObject.AddComponent<HorizontalLayoutGroup>();
            h.spacing = spacing;
            h.padding = new RectOffset(padding, padding, padding, padding);
            h.childAlignment = align;
            h.childControlWidth = true;
            h.childControlHeight = true;
            h.childForceExpandWidth = false;
            h.childForceExpandHeight = expandHeight;
            return h;
        }

        /// <summary>Vertical scrolling list. Returns the content transform to add rows to.</summary>
        public static RectTransform ScrollList(Transform parent, string name, float spacing = 14, int padding = 8)
        {
            var viewportImg = Panel(parent, name, new Color(0, 0, 0, 0.0f), rounded: false);
            var viewport = viewportImg.rectTransform;
            viewport.gameObject.AddComponent<RectMask2D>();
            viewportImg.raycastTarget = true;
            var scroll = viewport.gameObject.AddComponent<ScrollRect>();
            var content = VStack(viewport, "Content", spacing, padding).GetComponent<RectTransform>();
            content.anchorMin = new Vector2(0, 1);
            content.anchorMax = new Vector2(1, 1);
            content.pivot = new Vector2(0.5f, 1);
            content.offsetMin = Vector2.zero;
            content.offsetMax = Vector2.zero;
            var fitter = content.gameObject.AddComponent<ContentSizeFitter>();
            fitter.verticalFit = ContentSizeFitter.FitMode.PreferredSize;
            scroll.content = content;
            scroll.viewport = viewport;
            scroll.horizontal = false;
            scroll.vertical = true;
            scroll.movementType = ScrollRect.MovementType.Clamped;
            scroll.scrollSensitivity = 30f;
            return content;
        }

        public static UISlider Slider(Transform parent, string label, float value, Action<float> onChanged, string format = "{0:0}%", float displayScale = 100f)
        {
            var row = HStack(parent, "Slider_" + label, 20);
            Size(row, 72);
            var l = Label(row.transform, label, Theme.BodySize, Theme.Text);
            Size(l, -1, 340);
            var trackImg = Panel(row.transform, "Track", Theme.PanelLight);
            Size(trackImg, 40, -1, 1);
            var slider = trackImg.gameObject.AddComponent<UISlider>();
            var valueText = Label(row.transform, "", Theme.BodySize, Theme.Accent, TextAnchor.MiddleRight);
            Size(valueText, -1, 120);
            slider.Init(value, v =>
            {
                valueText.text = string.Format(format, v * displayScale);
                onChanged?.Invoke(v);
            });
            valueText.text = string.Format(format, value * displayScale);
            return slider;
        }

        public static Button Toggle(Transform parent, string label, bool value, Action<bool> onChanged)
        {
            var row = HStack(parent, "Toggle_" + label, 20);
            Size(row, 72);
            var l = Label(row.transform, label, Theme.BodySize, Theme.Text);
            Size(l, -1, 340, 1);
            bool state = value;
            Button b = null;
            b = Button(row.transform, state ? "ON" : "OFF", () =>
            {
                state = !state;
                SetButtonLabel(b, state ? "ON" : "OFF");
                b.GetComponent<Image>().color = state ? Theme.Teal : Theme.PanelLight;
                onChanged?.Invoke(state);
            }, ButtonStyle.Secondary);
            b.GetComponent<Image>().color = state ? Theme.Teal : Theme.PanelLight;
            Size(b, 64, 180);
            return b;
        }

        /// <summary>A row that cycles through named options with arrow buttons.</summary>
        public static void Selector(Transform parent, string label, string[] options, int index, Action<int> onChanged)
        {
            var row = HStack(parent, "Selector_" + label, 16);
            Size(row, 72);
            var l = Label(row.transform, label, Theme.BodySize, Theme.Text);
            Size(l, -1, 340, 1);
            int current = Mathf.Clamp(index, 0, options.Length - 1);
            Text valueLabel = null;
            var prev = Button(row.transform, "<", () =>
            {
                current = (current - 1 + options.Length) % options.Length;
                valueLabel.text = options[current];
                onChanged?.Invoke(current);
            }, ButtonStyle.Secondary);
            Size(prev, 64, 72);
            valueLabel = Label(row.transform, options[current], Theme.BodySize, Theme.Accent, TextAnchor.MiddleCenter, FontStyle.Bold);
            Size(valueLabel, -1, 220);
            var next = Button(row.transform, ">", () =>
            {
                current = (current + 1) % options.Length;
                valueLabel.text = options[current];
                onChanged?.Invoke(current);
            }, ButtonStyle.Secondary);
            Size(next, 64, 72);
        }

        /// <summary>Horizontal bar whose fill is driven by anchors (no sprite fill mode required).</summary>
        public static RectTransform Bar(Transform parent, string name, Color background, Color fill, out RectTransform fillRect)
        {
            var bg = Panel(parent, name, background, rounded: false);
            var f = Panel(bg.transform, "Fill", fill, rounded: false);
            fillRect = f.rectTransform;
            fillRect.anchorMin = Vector2.zero;
            fillRect.anchorMax = Vector2.one;
            fillRect.offsetMin = Vector2.zero;
            fillRect.offsetMax = Vector2.zero;
            return bg.rectTransform;
        }

        public static void SetFill(RectTransform fill, float fraction, bool fromRight = false)
        {
            fraction = Mathf.Clamp01(fraction);
            if (fromRight)
            {
                fill.anchorMin = new Vector2(1f - fraction, 0f);
                fill.anchorMax = Vector2.one;
            }
            else
            {
                fill.anchorMin = Vector2.zero;
                fill.anchorMax = new Vector2(fraction, 1f);
            }
        }
    }

    /// <summary>Minimal, touch-friendly 0..1 slider built from a single track image.</summary>
    public sealed class UISlider : MonoBehaviour, IPointerDownHandler, IDragHandler
    {
        private RectTransform _rect;
        private RectTransform _fill;
        private Action<float> _onChanged;
        public float Value { get; private set; }

        public void Init(float value, Action<float> onChanged)
        {
            _rect = (RectTransform)transform;
            var fillImg = UIFactory.Panel(transform, "Fill", Theme.Accent);
            _fill = fillImg.rectTransform;
            fillImg.raycastTarget = false;
            _onChanged = onChanged;
            SetValue(value, false);
        }

        public void SetValue(float v, bool notify)
        {
            Value = Mathf.Clamp01(v);
            _fill.anchorMin = Vector2.zero;
            _fill.anchorMax = new Vector2(Mathf.Max(0.04f, Value), 1f);
            _fill.offsetMin = Vector2.zero;
            _fill.offsetMax = Vector2.zero;
            if (notify) _onChanged?.Invoke(Value);
        }

        public void OnPointerDown(PointerEventData e) => UpdateFromPointer(e);
        public void OnDrag(PointerEventData e) => UpdateFromPointer(e);

        private void UpdateFromPointer(PointerEventData e)
        {
            if (!RectTransformUtility.ScreenPointToLocalPointInRectangle(_rect, e.position, e.pressEventCamera, out var local)) return;
            var r = _rect.rect;
            SetValue((local.x - r.xMin) / Mathf.Max(1f, r.width), true);
        }
    }
}
