using System;
using Undercurrent.Core.Combat;
using Undercurrent.Core.Content;
using Undercurrent.Core.Progression;
using Undercurrent.Runtime.App;
using Undercurrent.Runtime.Fight;
using Undercurrent.Runtime.Presentation;
using Undercurrent.Runtime.Services;
using UnityEngine;
using UnityEngine.UI;

namespace Undercurrent.Runtime.UI
{
    public sealed class FighterSelectScreen : ScreenBase
    {
        private static string _focused;
        private readonly Action _afterSelect;

        public FighterSelectScreen(Action afterSelect) => _afterSelect = afterSelect;

        public override void Build(RectTransform root, Navigator nav)
        {
            var content = ScreenFrame.Build(root, nav, "FIGHTERS", solidBackground: false);
            var data = nav.App.Services.Data;
            if (string.IsNullOrEmpty(_focused)) _focused = data.SelectedFighterId;

            var list = UIFactory.VStack(content, "Roster", 14);
            UIFactory.Anchor((RectTransform)list.transform, 0f, 0.15f, 0.26f, 1f);
            foreach (var id in FighterRoster.PlayableIds)
            {
                var f = FighterRoster.Get(id);
                bool unlocked = ProgressionService.IsFighterUnlocked(data, id);
                string label = unlocked ? $"{f.DisplayName.ToUpperInvariant()}\n<size=20>{f.Style}</size>" : $"LOCKED\n<size=20>{UnlockHint(id)}</size>";
                var fid = id;
                var b = UIFactory.Button(list.transform, label, () => { _focused = fid; nav.App.Backdrop.Show(fid, data.GetFighter(fid)); nav.Refresh(); },
                    id == _focused ? ButtonStyle.Primary : ButtonStyle.Secondary, 28);
                b.GetComponentInChildren<Text>().supportRichText = true;
                UIFactory.Size(b, 120);
            }

            var def = FighterRoster.Get(_focused);
            bool focusedUnlocked = ProgressionService.IsFighterUnlocked(data, _focused);
            var panel = UIFactory.Panel(content, "Detail", new Color(0.06f, 0.07f, 0.09f, 0.9f));
            UIFactory.Anchor(panel.rectTransform, 0.6f, 0f, 1f, 1f);
            var lvl = data.GetFighter(_focused)?.Level ?? 1;
            var text = UIFactory.Label(panel.transform,
                $"<b><size=40>{def.DisplayName}</size></b>\n<color=#E8A33D>\"{def.Nickname}\" · {def.Style}</color>   LV {lvl}\n\n{def.Bio}\n\n" +
                $"<color=#5AC76B>Strengths:</color> {def.Strengths}\n<color=#DC4038>Weaknesses:</color> {def.Weaknesses}",
                24, Theme.Text, TextAnchor.UpperLeft);
            text.supportRichText = true;
            UIFactory.Anchor(text.rectTransform, 0.05f, 0.42f, 0.95f, 0.97f);

            var bars = UIFactory.VStack(panel.transform, "Stats", 8);
            UIFactory.Anchor((RectTransform)bars.transform, 0.05f, 0.2f, 0.95f, 0.42f);
            StatBar(bars.transform, "Health", def.MaxHealth / 1400f);
            StatBar(bars.transform, "Stamina", def.MaxStamina / 115f);
            StatBar(bars.transform, "Speed", def.WalkForwardSpeed / 0.052f);
            StatBar(bars.transform, "Reach", def.PreferredRange / 1.6f);
            var jab = def.Resolve(AttackButton.LightPunch, MoveDirection.Neutral);
            StatBar(bars.transform, "Hand speed", 1f - (jab.Startup - 3) / 4f);

            var row = UIFactory.HStack(panel.transform, "Buttons", 12);
            UIFactory.Anchor((RectTransform)row.transform, 0.04f, 0.03f, 0.96f, 0.17f);
            bool selected = data.SelectedFighterId == _focused;
            var select = UIFactory.Button(row.transform, selected ? "SELECTED" : "SELECT", () =>
            {
                data.SelectedFighterId = _focused;
                nav.App.Services.Persist();
                if (_afterSelect != null) _afterSelect();
                else nav.Back();
            });
            select.interactable = focusedUnlocked && !selected;
            UIFactory.Size(select, -1, -1, 1);
            var custom = UIFactory.Button(row.transform, "STYLE", () => nav.Push(new CustomizeScreen(_focused)), ButtonStyle.Secondary, 26);
            custom.interactable = focusedUnlocked;
            UIFactory.Size(custom, -1, -1, 1);
            var up = UIFactory.Button(row.transform, "UPGRADE", () => nav.Push(new UpgradeScreen(_focused)), ButtonStyle.Secondary, 26);
            up.interactable = focusedUnlocked;
            UIFactory.Size(up, -1, -1, 1);
        }

        private static string UnlockHint(string id)
        {
            foreach (var l in CampaignDatabase.Levels)
                if (l.UnlocksFighterId == id) return $"Win chapter {l.Chapter}-{l.Number}";
            return "Coming later";
        }

        private static void StatBar(Transform parent, string label, float value)
        {
            var row = UIFactory.HStack(parent, label, 12);
            UIFactory.Size(row, 30);
            UIFactory.Size(UIFactory.Label(row.transform, label, 22, Theme.TextDim), -1, 200);
            var bar = UIFactory.Bar(row.transform, "Bar", new Color(1, 1, 1, 0.08f), Theme.Accent, out var fill);
            UIFactory.Size(bar.GetComponent<Image>(), 18, -1, 1);
            UIFactory.SetFill(fill, Mathf.Clamp01(value));
        }
    }

    public sealed class CustomizeScreen : ScreenBase
    {
        private readonly string _fighterId;

        public CustomizeScreen(string fighterId) => _fighterId = fighterId;

        public override void Build(RectTransform root, Navigator nav)
        {
            var data = nav.App.Services.Data;
            var def = FighterRoster.Get(_fighterId);
            var content = ScreenFrame.Build(root, nav, $"STYLE — {def.DisplayName.ToUpperInvariant()}", solidBackground: false);
            var fp = data.GetOrCreateFighter(_fighterId);
            nav.App.Backdrop.Show(_fighterId, fp);

            var panel = UIFactory.Panel(content, "Options", new Color(0.06f, 0.07f, 0.09f, 0.92f));
            UIFactory.Anchor(panel.rectTransform, 0.5f, 0f, 1f, 1f);
            var list = UIFactory.ScrollList(panel.transform, "List", 10, 16);
            UIFactory.Stretch((RectTransform)list.parent);

            var tones = CharacterLook.SkinToneNames;
            var toneNames = new string[tones.Length + 1];
            toneNames[0] = "Default";
            Array.Copy(tones, 0, toneNames, 1, tones.Length);
            UIFactory.Selector(list, "Skin tone", toneNames, fp.SkinToneIndex + 1, i =>
            {
                fp.SkinToneIndex = i - 1;
                nav.App.Services.Persist();
                nav.App.Backdrop.Show(_fighterId, fp);
            });

            foreach (CosmeticSlot slot in Enum.GetValues(typeof(CosmeticSlot)))
            {
                UIFactory.Size(UIFactory.Label(list, slot == CosmeticSlot.Outfit ? "OUTFITS" : "HAND WRAPS", 28, Theme.Accent, TextAnchor.MiddleLeft, FontStyle.Bold), 56);
                foreach (var c in Catalog.Cosmetics)
                {
                    if (c.Slot != slot) continue;
                    bool owned = data.OwnedCosmetics.Contains(c.Id);
                    bool equipped = slot == CosmeticSlot.Outfit ? fp.EquippedOutfit == c.Id : fp.EquippedWraps == c.Id;
                    var row = UIFactory.HStack(list, c.Id, 14);
                    UIFactory.Size(row, 70);
                    var swatch = UIFactory.Panel(row.transform, "Swatch", VisualAssets.Hex(c.PrimaryColor));
                    UIFactory.Size(swatch, 56, 56);
                    UIFactory.Size(UIFactory.Label(row.transform, c.DisplayName + (owned ? "" : c.RewardOnly ? "  (campaign reward)" : $"  ({c.Price} Marks in Store)"), 26, owned ? Theme.Text : Theme.TextDim), -1, -1, 1);
                    var id = c.Id;
                    var b = UIFactory.Button(row.transform, equipped ? "EQUIPPED" : owned ? "EQUIP" : "LOCKED", () =>
                    {
                        ProgressionService.TryEquipCosmetic(data, _fighterId, id);
                        nav.App.Services.Persist();
                        nav.Refresh();
                    }, equipped ? ButtonStyle.Primary : ButtonStyle.Secondary, 24);
                    b.interactable = owned && !equipped;
                    UIFactory.Size(b, 60, 220);
                }
            }
        }
    }

    public sealed class UpgradeScreen : ScreenBase
    {
        private readonly string _fighterId;

        public UpgradeScreen(string fighterId) => _fighterId = fighterId;

        public override void Build(RectTransform root, Navigator nav)
        {
            var data = nav.App.Services.Data;
            var def = FighterRoster.Get(_fighterId);
            var content = ScreenFrame.Build(root, nav, $"UPGRADE — {def.DisplayName.ToUpperInvariant()}");
            var fp = data.GetOrCreateFighter(_fighterId);
            var current = ProgressionService.BuildLoadout(data, _fighterId);

            var left = UIFactory.Panel(content, "Level", Theme.Panel);
            UIFactory.Anchor(left.rectTransform, 0f, 0f, 0.45f, 1f);
            bool canUpgrade = ProgressionService.CanUpgradeFighter(data, _fighterId, out int cost);
            bool maxed = fp.Level >= Catalog.MaxFighterLevel;
            var info = UIFactory.Label(left.transform,
                $"<b><size=44>LEVEL {fp.Level} / {Catalog.MaxFighterLevel}</size></b>\n\nHealth  {current.MaxHealth}\nStamina  {current.MaxStamina:0}\nStamina regen  {current.StaminaRegen * 60f:0.0}/s\n" +
                $"Guard drain  x{current.GuardDamageTakenMultiplier:0.00}\n\nEach level adds +2% health and +1.5% stamina.\nUpgrades are earned with Marks from fights — never sold for real money.",
                26, Theme.Text, TextAnchor.UpperLeft);
            info.supportRichText = true;
            UIFactory.Anchor(info.rectTransform, 0.06f, 0.25f, 0.94f, 0.95f);
            var btn = UIFactory.Button(left.transform, maxed ? "MAX LEVEL" : $"UPGRADE ({Catalog.UpgradeCost(fp.Level)} MARKS)", () =>
            {
                if (ProgressionService.TryUpgradeFighter(data, _fighterId))
                {
                    nav.App.Services.Persist();
                    nav.App.Services.Audio.PlayUI("ui_confirm");
                }
                nav.Refresh();
            });
            btn.interactable = canUpgrade;
            UIFactory.Anchor((RectTransform)btn.transform, 0.06f, 0.05f, 0.94f, 0.2f);

            var right = UIFactory.Panel(content, "Skills", Theme.Panel);
            UIFactory.Anchor(right.rectTransform, 0.47f, 0f, 1f, 1f);
            var list = UIFactory.VStack(right.transform, "SkillList", 16, 24);
            UIFactory.Stretch((RectTransform)list.transform);
            UIFactory.Size(UIFactory.Label(list.transform, "SKILLS (unlock with fighter level)", 30, Theme.Accent, TextAnchor.MiddleLeft, FontStyle.Bold), 56);
            foreach (var s in Catalog.Skills)
            {
                bool on = fp.Level >= s.RequiredFighterLevel;
                var row = UIFactory.Panel(list.transform, s.Id, on ? Theme.PanelLight : Theme.Disabled);
                UIFactory.Size(row, 110);
                var t = UIFactory.Label(row.transform, $"<b>{s.DisplayName}</b>   {(on ? "<color=#5AC76B>ACTIVE</color>" : $"<color=#9EA8B5>Level {s.RequiredFighterLevel}</color>")}\n{s.Description}", 24, Theme.Text, TextAnchor.MiddleLeft);
                t.supportRichText = true;
                UIFactory.Stretch(t.rectTransform, 20, 20, 8, 8);
            }
        }
    }

    public sealed class AchievementsScreen : ScreenBase
    {
        public override void Build(RectTransform root, Navigator nav)
        {
            var content = ScreenFrame.Build(root, nav, "AWARDS");
            var data = nav.App.Services.Data;
            var list = UIFactory.ScrollList(content, "List", 12, 6);
            UIFactory.Stretch((RectTransform)list.parent);
            var summary = UIFactory.Label(list, $"Matches {data.MatchesPlayed}  ·  Wins {data.TotalWins}  ·  Parries {data.TotalParries}  ·  Throws {data.TotalThrows}  ·  Best combo {data.BestCombo}", 26, Theme.TextDim, TextAnchor.MiddleLeft);
            UIFactory.Size(summary, 50);
            foreach (var a in Catalog.Achievements)
            {
                bool done = data.UnlockedAchievements.Contains(a.Id);
                var row = UIFactory.Panel(list, a.Id, done ? Theme.PanelLight : Theme.Panel);
                UIFactory.Size(row, 100);
                var t = UIFactory.Label(row.transform, $"<b>{a.DisplayName}</b>  {(done ? "<color=#5AC76B>UNLOCKED</color>" : "")}\n{a.Description}   (+{a.RewardCurrency} Marks)", 26, done ? Theme.Text : Theme.TextDim, TextAnchor.MiddleLeft);
                t.supportRichText = true;
                UIFactory.Stretch(t.rectTransform, 24, 24, 8, 8);
            }
        }
    }

    public sealed class StoreScreen : ScreenBase
    {
        private static bool _inventory;

        public override void Build(RectTransform root, Navigator nav)
        {
            var content = ScreenFrame.Build(root, nav, _inventory ? "INVENTORY" : "STORE");
            var services = nav.App.Services;
            var data = services.Data;

            var tabs = UIFactory.HStack(content, "Tabs", 14);
            UIFactory.Anchor((RectTransform)tabs.transform, 0f, 0.9f, 0.5f, 1f);
            UIFactory.Size(UIFactory.Button(tabs.transform, "STORE", () => { _inventory = false; nav.Refresh(); }, _inventory ? ButtonStyle.Secondary : ButtonStyle.Primary, 26), -1, -1, 1);
            UIFactory.Size(UIFactory.Button(tabs.transform, "INVENTORY", () => { _inventory = true; nav.Refresh(); }, _inventory ? ButtonStyle.Primary : ButtonStyle.Secondary, 26), -1, -1, 1);

            var list = UIFactory.ScrollList(content, "Items", 10, 6);
            UIFactory.Anchor((RectTransform)list.parent, 0f, 0.2f, 1f, 0.88f);
            foreach (var c in Catalog.Cosmetics)
            {
                bool owned = data.OwnedCosmetics.Contains(c.Id);
                if (_inventory != owned) continue;
                if (!_inventory && c.RewardOnly) continue;
                var row = UIFactory.HStack(list, c.Id, 16);
                UIFactory.Size(row, 76);
                var sw = UIFactory.Panel(row.transform, "Swatch", VisualAssets.Hex(c.PrimaryColor));
                UIFactory.Size(sw, 60, 60);
                UIFactory.Size(UIFactory.Label(row.transform, $"{c.DisplayName}   <color=#9EA8B5>{c.Slot}</color>", 28), -1, -1, 1).GetComponent<Text>().supportRichText = true;
                if (_inventory) continue;
                var id = c.Id;
                var b = UIFactory.Button(row.transform, $"{c.Price} MARKS", () =>
                {
                    var result = ProgressionService.TryPurchaseCosmetic(data, id);
                    if (result == PurchaseResult.Success)
                    {
                        services.Persist();
                        services.Audio.PlayUI("ui_confirm");
                        nav.Toast($"{c.DisplayName} added to your inventory. Equip it from Fighters > Style.");
                    }
                    else if (result == PurchaseResult.InsufficientFunds) nav.Toast("Not enough Marks yet — win fights and complete objectives to earn more.");
                    nav.Refresh();
                }, ButtonStyle.Primary, 24);
                b.interactable = data.Currency >= c.Price;
                UIFactory.Size(b, 64, 240);
            }

            var footer = UIFactory.Panel(content, "Premium", Theme.Panel);
            UIFactory.Anchor(footer.rectTransform, 0f, 0f, 1f, 0.17f);
            string status = services.Billing.Status == IntegrationStatus.Live ? "Premium cosmetics available." :
                "Real-money purchases are not available in this build (store billing is not configured). Everything in the game can be earned by playing.";
            var ft = UIFactory.Label(footer.transform, status, 24, Theme.TextDim, TextAnchor.MiddleLeft);
            UIFactory.Anchor(ft.rectTransform, 0.03f, 0f, 0.72f, 1f);
            var restore = UIFactory.Button(footer.transform, "RESTORE PURCHASES", () =>
                services.Billing.RestorePurchases(items => nav.Toast(services.Billing.Status == IntegrationStatus.Live
                    ? $"Restored {items.Count} purchases." : "Purchase restoration is unavailable: billing is not configured in this build.")), ButtonStyle.Secondary, 22);
            UIFactory.Anchor((RectTransform)restore.transform, 0.74f, 0.15f, 0.98f, 0.85f);
        }
    }

    public sealed class SettingsScreen : ScreenBase
    {
        public override void Build(RectTransform root, Navigator nav)
        {
            var content = ScreenFrame.Build(root, nav, "SETTINGS");
            var services = nav.App.Services;
            var s = services.Settings;
            var list = UIFactory.ScrollList(content, "List", 10, 10);
            UIFactory.Stretch((RectTransform)list.parent);

            void Section(string name) => UIFactory.Size(UIFactory.Label(list, name, 30, Theme.Accent, TextAnchor.MiddleLeft, FontStyle.Bold), 60);
            void Save() => services.Persist();

            Section("AUDIO");
            UIFactory.Slider(list, "Music", s.MusicVolume, v => { s.MusicVolume = v; services.Audio.ApplyVolumes(); });
            UIFactory.Slider(list, "Effects", s.SfxVolume, v => { s.SfxVolume = v; services.Audio.ApplyVolumes(); });
            UIFactory.Slider(list, "Voice", s.VoiceVolume, v => { s.VoiceVolume = v; services.Audio.ApplyVolumes(); });

            Section("FEEDBACK & ACCESSIBILITY");
            UIFactory.Toggle(list, "Vibration", s.HapticsEnabled, on => { s.HapticsEnabled = on; Save(); if (on) services.Haptics.Play(HapticPattern.HeavyHit); });
            UIFactory.Slider(list, "Vibration strength", s.HapticIntensity, v => s.HapticIntensity = v);
            UIFactory.Slider(list, "Screen shake", s.ScreenShakeScale, v => s.ScreenShakeScale = v);
            UIFactory.Toggle(list, "Reduce flashes", s.ReduceFlashes, on => { s.ReduceFlashes = on; Save(); });
            UIFactory.Toggle(list, "Show input hints", s.ShowInputHints, on => { s.ShowInputHints = on; Save(); });

            Section("GRAPHICS");
            UIFactory.Selector(list, "Quality", new[] { "Low", "Medium", "High", "Ultra" }, s.GraphicsTier, i => { s.GraphicsTier = i; QualityService.Apply(s); Save(); });
            UIFactory.Selector(list, "Frame rate", new[] { "30 FPS (battery)", "60 FPS" }, s.TargetFps == 30 ? 0 : 1, i => { s.TargetFps = i == 0 ? 30 : 60; QualityService.Apply(s); Save(); });

            Section("CONTROLS");
            UIFactory.Slider(list, "Button size", Mathf.InverseLerp(0.7f, 1.4f, s.ControlScale), v => s.ControlScale = Mathf.Lerp(0.7f, 1.4f, v), "{0:0}%", 100f);
            UIFactory.Slider(list, "Button opacity", s.ControlOpacity, v => s.ControlOpacity = Mathf.Max(0.2f, v));
            UIFactory.Size(UIFactory.Button(list, "EDIT BUTTON LAYOUT", () => { Save(); nav.Push(new LayoutEditorScreen()); }, ButtonStyle.Secondary), 84);

            Section("DATA");
            UIFactory.Size(UIFactory.Label(list, $"Profile: offline guest ({services.Account.Status}).  Cloud save: {services.CloudSave.Status}.", 22, Theme.TextDim), 40);
            UIFactory.Size(UIFactory.Button(list, "REPLAY TRAINING", () => { Save(); nav.App.StartTraining(); }, ButtonStyle.Secondary), 84);
            UIFactory.Size(UIFactory.Button(list, "RESET ALL PROGRESS", () =>
                nav.Confirm("Erase all progress, unlocks and Marks? Settings are kept. This cannot be undone.", "ERASE", () =>
                {
                    services.Save.ResetProgress();
                    nav.App.ShowMainMenu();
                }), ButtonStyle.Danger), 84);

            Section("ABOUT");
            UIFactory.Size(UIFactory.Label(list,
                $"{GameApp.BuildLabel}.\n\"Undercurrent\" is an internal working title and has not been cleared as a trademark. " +
                "Characters, story and moves are original to this project. Character models, animation, environment art and audio in this build are " +
                "procedural placeholders generated by code, not final production assets.", 22, Theme.TextDim, TextAnchor.UpperLeft), 150);

            // Persist slider changes when leaving the screen.
            root.gameObject.AddComponent<SaveOnDisable>().Services = services;
        }
    }

    /// <summary>Saves settings once when a screen closes instead of on every slider tick.</summary>
    public sealed class SaveOnDisable : MonoBehaviour
    {
        public ServiceHub Services;
        private void OnDestroy() => Services?.Persist();
    }

    public sealed class LayoutEditorScreen : ScreenBase
    {
        public override void Build(RectTransform root, Navigator nav)
        {
            var services = nav.App.Services;
            var bg = UIFactory.Panel(root, "Background", new Color(0.04f, 0.05f, 0.06f, 0.85f), rounded: false);
            UIFactory.Stretch(bg.rectTransform);
            var touch = new GameObject("LayoutTouch").AddComponent<TouchControls>();
            touch.Build(root, services.Settings, true);

            var hint = UIFactory.Label(root, "Drag any control to move it. Changes are saved when you press DONE.", 30, Theme.Text, TextAnchor.MiddleCenter);
            UIFactory.Anchor(hint.rectTransform, 0.2f, 0.88f, 0.8f, 0.98f);
            var row = UIFactory.HStack(root, "Buttons", 20);
            UIFactory.Fixed((RectTransform)row.transform, new Vector2(0.5f, 0.75f), new Vector2(0.5f, 0.5f), Vector2.zero, new Vector2(620, 90));
            UIFactory.Size(UIFactory.Button(row.transform, "RESET", () => { touch.ResetLayout(); nav.Refresh(); }, ButtonStyle.Secondary), -1, 280);
            UIFactory.Size(UIFactory.Button(row.transform, "DONE", () => { services.Persist(); nav.Back(); }), -1, 280);
        }
    }
}
