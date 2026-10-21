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

            var list = UIFactory.ScrollList(content, "Roster", 12, 4);
            UIFactory.Anchor((RectTransform)list.parent, 0f, 0f, 0.26f, 1f);
            foreach (var id in FighterRoster.PlayableIds)
            {
                var f = FighterRoster.Get(id);
                bool unlocked = ProgressionService.IsFighterUnlocked(data, id);
                string label = unlocked ? $"{f.DisplayName.ToUpperInvariant()}\n<size=20>{f.Style}</size>" : $"LOCKED\n<size=20>{UnlockHint(id)}</size>";
                var fid = id;
                var b = UIFactory.Button(list, label, () => { _focused = fid; nav.App.Backdrop.Show(fid, data.GetFighter(fid)); nav.Refresh(); },
                    id == _focused ? ButtonStyle.Primary : ButtonStyle.Secondary, 28);
                b.GetComponentInChildren<Text>().supportRichText = true;
                UIFactory.Size(b, 110);
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
            var armory = UIFactory.Button(row.transform, "ARMORY", () => nav.Push(new ArmoryScreen(_focused)), ButtonStyle.Secondary, 26);
            armory.interactable = focusedUnlocked;
            UIFactory.Size(armory, -1, -1, 1);
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
                    UIFactory.Size(UIFactory.Label(row.transform, c.DisplayName + (owned ? "" : c.RewardOnly ? "  (campaign reward)" : $"  ({GearUI.Coins(c.Price)} in Shop)"), 26, owned ? Theme.Text : Theme.TextDim), -1, -1, 1);
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
                $"Damage  x{current.DamageDealtMultiplier:0.00}\nGuard drain  x{current.GuardDamageTakenMultiplier:0.00}\n\nEach level adds +2% health, +2% damage and +1.5% stamina.\nPower (unarmed): {ProgressionService.PlayerPower(data, _fighterId, false)}",
                26, Theme.Text, TextAnchor.UpperLeft);
            info.supportRichText = true;
            UIFactory.Anchor(info.rectTransform, 0.06f, 0.25f, 0.94f, 0.95f);
            var btn = UIFactory.Button(left.transform, maxed ? "MAX LEVEL" : $"UPGRADE ({GearUI.Coins(Catalog.UpgradeCost(fp.Level))})", () =>
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

    /// <summary>Shared weapon/armour shop actions so the Armory and the Shop behave identically.</summary>
    public static class GearUI
    {
        public static string Coins(int amount) => $"{amount} {Shop.CurrencyName.ToUpperInvariant()}";

        public static string Stats(WeaponDefinition w)
        {
            var lp = w.Moves.Find(m => m.Button == AttackButton.LightPunch && m.Direction == MoveDirection.Neutral);
            string stats = $"{w.Description}\nReach {lp.HitFrontOffset + lp.HitLength:0.00} m  ·  quick strike {lp.Startup}f";
            if (w.ParryWindowBonus > 0) stats += $"  ·  parry +{w.ParryWindowBonus}f";
            if (w.CounterHitBonus > 0) stats += $"  ·  counter-hit +{w.CounterHitBonus * 100:0}%";
            return stats;
        }

        /// <summary>Status line for a weapon: owned level, price, or where it unlocks.</summary>
        public static string Status(SaveData data, string id)
        {
            if (ProgressionService.OwnsWeapon(data, id))
            {
                int lvl = ProgressionService.WeaponLevel(data, id);
                return $"<color=#5AC76B>OWNED · LEVEL {lvl}/{Shop.MaxWeaponLevel}</color>  (+{(PowerRating.WeaponLevelScale(lvl) - 1f) * 100f:0}% strike damage)";
            }
            if (ProgressionService.IsWeaponInShop(data, id)) return $"<color=#E8A33D>For sale: {Coins(Shop.WeaponPrice(id))}</color>";
            var l = ProgressionService.WeaponShopLevel(id);
            return $"<color=#9EA8B5>Sold after you win chapter {l.Chapter}-{l.Number}: {l.Title}</color>";
        }

        /// <summary>Adds the BUY / UPGRADE button for a weapon (nothing when maxed or not yet sold).</summary>
        public static Button WeaponButton(Transform parent, Navigator nav, string id)
        {
            var services = nav.App.Services;
            var data = services.Data;
            bool owned = ProgressionService.OwnsWeapon(data, id);
            int cost = owned ? ProgressionService.WeaponUpgradeCost(data, id) : Shop.WeaponPrice(id);
            string label = owned ? (cost > 0 ? $"UPGRADE\n<size=20>{Coins(cost)}</size>" : "MAX LEVEL")
                : ProgressionService.IsWeaponInShop(data, id) ? $"BUY\n<size=20>{Coins(cost)}</size>" : "LOCKED";
            var b = UIFactory.Button(parent, label, () =>
            {
                var name = WeaponCatalog.Get(id).DisplayName;
                var result = owned ? ProgressionService.TryUpgradeWeapon(data, id) : ProgressionService.TryBuyWeapon(data, id);
                Report(nav, result, owned ? $"{name} upgraded to level {ProgressionService.WeaponLevel(data, id)}." : $"{name} bought. Equip it in Fighters > Armory.");
            }, ButtonStyle.Primary, 24);
            b.GetComponentInChildren<Text>().supportRichText = true;
            b.interactable = cost > 0 && data.Currency >= cost && (owned || ProgressionService.IsWeaponInShop(data, id));
            return b;
        }

        public static void Report(Navigator nav, PurchaseResult result, string success)
        {
            var services = nav.App.Services;
            switch (result)
            {
                case PurchaseResult.Success:
                    services.Persist();
                    services.Audio.PlayUI("ui_confirm");
                    nav.Toast(success);
                    break;
                case PurchaseResult.InsufficientFunds:
                    nav.Toast($"Not enough {Shop.CurrencyName} — win or replay fights, play Versus, or get a coin pack.");
                    break;
                case PurchaseResult.Locked: nav.Toast("Not for sale yet — keep going in the campaign."); break;
                case PurchaseResult.Maxed: nav.Toast("Already at the highest level."); break;
            }
            nav.Refresh();
        }
    }

    /// <summary>Buy, upgrade and choose the weapon a fighter brings to weapon-mode fights.</summary>
    public sealed class ArmoryScreen : ScreenBase
    {
        private readonly string _fighterId;

        public ArmoryScreen(string fighterId) => _fighterId = fighterId;

        public override void Build(RectTransform root, Navigator nav)
        {
            var data = nav.App.Services.Data;
            var def = FighterRoster.Get(_fighterId);
            var content = ScreenFrame.Build(root, nav, $"ARMORY — {def.DisplayName.ToUpperInvariant()}");
            var fp = data.GetOrCreateFighter(_fighterId);
            var list = UIFactory.ScrollList(content, "Weapons", 14, 6);
            UIFactory.Stretch((RectTransform)list.parent);
            UIFactory.Size(UIFactory.Label(list, $"Weapons are bought and upgraded with {Shop.CurrencyName} and work in Weapons and Gauntlet levels and in Versus. " +
                $"Power with this fighter: unarmed {ProgressionService.PlayerPower(data, _fighterId, false)}, armed {ProgressionService.PlayerPower(data, _fighterId, true)}.",
                24, Theme.TextDim, TextAnchor.MiddleLeft), 80);

            AddRow(list, nav, fp, "", "Unarmed", "Your fighter's own punches and special. Fastest recovery; no weapon to lose.", true);
            foreach (var id in WeaponCatalog.All)
            {
                var w = WeaponCatalog.Get(id);
                AddRow(list, nav, fp, id, w.DisplayName, GearUI.Stats(w) + "\n" + GearUI.Status(data, id), ProgressionService.OwnsWeapon(data, id));
            }
        }

        private void AddRow(RectTransform list, Navigator nav, FighterProgress fp, string id, string name, string text, bool owned)
        {
            bool equipped = (fp.EquippedWeapon ?? "") == id;
            var row = UIFactory.Panel(list, "Weapon_" + name, owned ? Theme.Panel : Theme.Disabled);
            UIFactory.Size(row, 170);
            var t = UIFactory.Label(row.transform, $"<b>{name.ToUpperInvariant()}</b>{(equipped ? "   <color=#5AC76B>EQUIPPED</color>" : "")}\n{text}", 22, Theme.Text, TextAnchor.MiddleLeft);
            t.supportRichText = true;
            UIFactory.Stretch(t.rectTransform, 24, 500, 10, 10);
            if (id.Length > 0)
            {
                var shop = GearUI.WeaponButton(row.transform, nav, id);
                UIFactory.Fixed((RectTransform)shop.transform, new Vector2(1f, 0.5f), new Vector2(1f, 0.5f), new Vector2(-260, 0), new Vector2(220, 100));
            }
            var b = UIFactory.Button(row.transform, equipped ? "EQUIPPED" : owned ? "EQUIP" : "NOT OWNED", () =>
            {
                if (ProgressionService.TryEquipWeapon(nav.App.Services.Data, _fighterId, id))
                {
                    nav.App.Services.Persist();
                    nav.App.Services.Audio.PlayUI("ui_confirm");
                }
                nav.Refresh();
            }, equipped ? ButtonStyle.Primary : ButtonStyle.Secondary, 24);
            b.interactable = owned && !equipped;
            UIFactory.Fixed((RectTransform)b.transform, new Vector2(1f, 0.5f), new Vector2(1f, 0.5f), new Vector2(-20, 0), new Vector2(220, 100));
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
                var t = UIFactory.Label(row.transform, $"<b>{a.DisplayName}</b>  {(done ? "<color=#5AC76B>UNLOCKED</color>" : "")}\n{a.Description}   (+{a.RewardCurrency} {Shop.CurrencyName})", 26, done ? Theme.Text : Theme.TextDim, TextAnchor.MiddleLeft);
                t.supportRichText = true;
                UIFactory.Stretch(t.rectTransform, 24, 24, 8, 8);
            }
        }
    }

    public sealed class ShopScreen : ScreenBase
    {
        private enum Tab { Gear, Style, Coins }
        private static Tab _tab = Tab.Gear;

        public override void Build(RectTransform root, Navigator nav)
        {
            var content = ScreenFrame.Build(root, nav, "SHOP");
            var tabs = UIFactory.HStack(content, "Tabs", 14);
            UIFactory.Anchor((RectTransform)tabs.transform, 0f, 0.9f, 0.6f, 1f);
            foreach (Tab tab in System.Enum.GetValues(typeof(Tab)))
            {
                var tb = tab;
                UIFactory.Size(UIFactory.Button(tabs.transform, tab.ToString().ToUpperInvariant(), () => { _tab = tb; nav.Refresh(); },
                    _tab == tab ? ButtonStyle.Primary : ButtonStyle.Secondary, 26), -1, -1, 1);
            }
            var list = UIFactory.ScrollList(content, "Items", 10, 6);
            UIFactory.Anchor((RectTransform)list.parent, 0f, 0f, 1f, 0.88f);
            if (_tab == Tab.Gear) BuildGear(list, nav);
            else if (_tab == Tab.Style) BuildStyle(list, nav);
            else BuildCoins(list, nav);
        }

        private static void Header(RectTransform list, string text) =>
            UIFactory.Size(UIFactory.Label(list, text, 28, Theme.Accent, TextAnchor.MiddleLeft, FontStyle.Bold), 56);

        private static RectTransform Row(RectTransform list, string name, string text, bool active)
        {
            var row = UIFactory.Panel(list, name, active ? Theme.Panel : Theme.Disabled);
            UIFactory.Size(row, 130);
            var t = UIFactory.Label(row.transform, text, 24, Theme.Text, TextAnchor.MiddleLeft);
            t.supportRichText = true;
            UIFactory.Stretch(t.rectTransform, 24, 280, 8, 8);
            return row.rectTransform;
        }

        private static void Place(Button b) =>
            UIFactory.Fixed((RectTransform)b.transform, new Vector2(1f, 0.5f), new Vector2(1f, 0.5f), new Vector2(-20, 0), new Vector2(240, 100));

        private static void BuildGear(RectTransform list, Navigator nav)
        {
            var data = nav.App.Services.Data;
            var me = data.SelectedFighterId;
            UIFactory.Size(UIFactory.Label(list, $"Your power ({FighterRoster.Get(me).DisplayName}): unarmed {ProgressionService.PlayerPower(data, me, false)}, " +
                $"armed {ProgressionService.PlayerPower(data, me, true)}.  Each level shows a recommended power — gear closes the gap on Hard and Extreme fights.",
                24, Theme.TextDim, TextAnchor.MiddleLeft), 70);

            Header(list, "ARMOUR  (worn by every fighter)");
            var current = Shop.GetArmor(data.ArmorTier);
            string now = current == null ? "No armour" : $"{current.DisplayName}: {(1f - current.DamageTaken) * 100f:0}% less damage taken";
            var next = Shop.GetArmor(data.ArmorTier + 1);
            var armourRow = Row(list, "Armour", $"<b>TIER {data.ArmorTier}/{Shop.MaxArmorTier}</b>  {now}" +
                (next != null ? $"\nNext: <b>{next.DisplayName}</b> — {(1f - next.DamageTaken) * 100f:0}% less damage. {next.Description}" : "\nFully armoured."), true);
            var ab = UIFactory.Button(armourRow, next != null ? $"UPGRADE\n<size=20>{GearUI.Coins(next.Price)}</size>" : "MAX TIER",
                () => GearUI.Report(nav, ProgressionService.TryUpgradeArmor(data), "Armour upgraded."), ButtonStyle.Primary, 24);
            ab.GetComponentInChildren<Text>().supportRichText = true;
            ab.interactable = next != null && data.Currency >= next.Price;
            Place(ab);

            Header(list, "WEAPONS  (Weapons and Gauntlet levels, Versus)");
            foreach (var id in WeaponCatalog.All)
            {
                var w = WeaponCatalog.Get(id);
                var row = Row(list, "Weapon_" + id, $"<b>{w.DisplayName.ToUpperInvariant()}</b>  {GearUI.Status(data, id)}\n{w.Description}", ProgressionService.IsWeaponInShop(data, id));
                Place(GearUI.WeaponButton(row, nav, id));
            }

            Header(list, "FIGHTER LEVEL");
            var fp = data.GetOrCreateFighter(me);
            var lrow = Row(list, "FighterLevel", $"<b>{FighterRoster.Get(me).DisplayName.ToUpperInvariant()} LV {fp.Level}/{Catalog.MaxFighterLevel}</b>\nEach level: +2% health, +2% damage, more stamina; skills unlock at levels 3, 5, 7 and 9.", true);
            bool maxed = fp.Level >= Catalog.MaxFighterLevel;
            var lb = UIFactory.Button(lrow, maxed ? "MAX LEVEL" : $"UPGRADE\n<size=20>{GearUI.Coins(Catalog.UpgradeCost(fp.Level))}</size>", () =>
                GearUI.Report(nav, ProgressionService.TryUpgradeFighter(data, me) ? PurchaseResult.Success : PurchaseResult.InsufficientFunds, "Fighter upgraded."), ButtonStyle.Primary, 24);
            lb.GetComponentInChildren<Text>().supportRichText = true;
            lb.interactable = ProgressionService.CanUpgradeFighter(data, me, out _);
            Place(lb);
        }

        private static void BuildStyle(RectTransform list, Navigator nav)
        {
            var services = nav.App.Services;
            var data = services.Data;
            foreach (var c in Catalog.Cosmetics)
            {
                bool owned = data.OwnedCosmetics.Contains(c.Id);
                if (c.RewardOnly && !owned) continue;
                var row = UIFactory.HStack(list, c.Id, 16);
                UIFactory.Size(row, 76);
                var sw = UIFactory.Panel(row.transform, "Swatch", VisualAssets.Hex(c.PrimaryColor));
                UIFactory.Size(sw, 60, 60);
                UIFactory.Size(UIFactory.Label(row.transform, $"{c.DisplayName}   <color=#9EA8B5>{c.Slot}</color>", 28), -1, -1, 1).GetComponent<Text>().supportRichText = true;
                var id = c.Id;
                var b = UIFactory.Button(row.transform, owned ? "OWNED" : GearUI.Coins(c.Price), () =>
                {
                    var result = ProgressionService.TryPurchaseCosmetic(data, id);
                    GearUI.Report(nav, result, $"{c.DisplayName} added. Equip it from Fighters > Style.");
                }, ButtonStyle.Primary, 24);
                b.interactable = !owned && data.Currency >= c.Price;
                UIFactory.Size(b, 64, 240);
            }
        }

        private static void BuildCoins(RectTransform list, Navigator nav)
        {
            var services = nav.App.Services;
            var billing = services.Billing;
            string status = billing.Status == IntegrationStatus.Live ? "Coin packs are charged through the store." :
                billing.Status == IntegrationStatus.DevelopmentMock ? "DEVELOPMENT BUILD: purchases here are test purchases and charge nothing." :
                "Coin packs are not available in this build (store billing is not configured).";
            UIFactory.Size(UIFactory.Label(list, status + $" Every item can also be earned by playing: wins, replays (half reward), Versus wins (more against harder AI) and a small purse when you lose after winning a round.",
                24, Theme.TextDim, TextAnchor.MiddleLeft), 90);
            foreach (var pack in Shop.CoinPacks)
            {
                string price = null;
                foreach (var p in billing.Products) if (p.ProductId == pack.ProductId) price = p.LocalizedPrice;
                var row = Row(list, pack.ProductId, $"<b>{pack.DisplayName.ToUpperInvariant()}</b>\n+{GearUI.Coins(pack.Coins)}", price != null);
                var id = pack.ProductId;
                var b = UIFactory.Button(row, price ?? "UNAVAILABLE", () => nav.App.BuyCoinPack(id, msg => { nav.Toast(msg); nav.Refresh(); }), ButtonStyle.Primary, 24);
                b.interactable = price != null;
                Place(b);
            }
            var restore = UIFactory.Button(list, "RESTORE PURCHASES", () =>
                billing.RestorePurchases(items => nav.Toast(billing.Status == IntegrationStatus.Live
                    ? $"Restored {items.Count} purchases." : "Purchase restoration is unavailable: billing is not configured in this build.")), ButtonStyle.Secondary, 22);
            UIFactory.Size(restore, 80);
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
                nav.Confirm("Erase all progress, unlocks, gear and Coins? Settings are kept. This cannot be undone.", "ERASE", () =>
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
