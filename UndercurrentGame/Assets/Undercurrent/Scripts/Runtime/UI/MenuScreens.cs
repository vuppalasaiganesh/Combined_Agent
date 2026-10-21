using System.Collections.Generic;
using System.Text;
using Undercurrent.Core.AI;
using Undercurrent.Core.Combat;
using Undercurrent.Core.Content;
using Undercurrent.Core.Progression;
using Undercurrent.Runtime.App;
using Undercurrent.Runtime.Fight;
using UnityEngine;
using UnityEngine.UI;

namespace Undercurrent.Runtime.UI
{
    /// <summary>Shared screen chrome: header with back button, title and the player's level/currency.</summary>
    public static class ScreenFrame
    {
        public static RectTransform Build(RectTransform root, Navigator nav, string title, bool back = true, bool solidBackground = true)
        {
            if (solidBackground)
            {
                var bg = UIFactory.Panel(root, "Background", Theme.Background, rounded: false);
                UIFactory.Stretch(bg.rectTransform);
            }
            var header = UIFactory.Rect(root, "Header");
            UIFactory.Anchor(header, 0f, 0.86f, 1f, 1f);
            if (back)
            {
                var b = UIFactory.Button(header, "< BACK", nav.Back, ButtonStyle.Ghost);
                UIFactory.Fixed((RectTransform)b.transform, new Vector2(0f, 0.5f), new Vector2(0f, 0.5f), new Vector2(30, 0), new Vector2(220, 86));
            }
            var t = UIFactory.Label(header, title, Theme.HeaderSize, Theme.Text, TextAnchor.MiddleLeft, FontStyle.Bold);
            UIFactory.Anchor(t.rectTransform, back ? 0.15f : 0.03f, 0f, 0.7f, 1f);
            var data = nav.App.Services.Data;
            var wallet = UIFactory.Label(header, $"LV {data.PlayerLevel}    <color=#E8A33D>{GearUI.Coins(data.Currency)}</color>", Theme.BodySize, Theme.Text, TextAnchor.MiddleRight, FontStyle.Bold);
            wallet.supportRichText = true;
            UIFactory.Anchor(wallet.rectTransform, 0.6f, 0f, 0.97f, 1f);

            var content = UIFactory.Rect(root, "Content");
            UIFactory.Anchor(content, 0.03f, 0.04f, 0.97f, 0.85f);
            return content;
        }

        public static string DifficultyName(AIDifficulty d) => d.ToString().ToUpperInvariant();

        public static string TierTag(LevelTier tier) => $"<color=#{TierRules.ColorHex(tier)}>{TierRules.Name(tier)}</color>";

        /// <summary>"POWER 140 / 160" coloured green when the player meets the recommendation, red when well below.</summary>
        public static string PowerCompare(int mine, int recommended)
        {
            string color = mine >= recommended ? "5AC76B" : mine >= recommended * 0.9f ? "E8A33D" : "DC4038";
            return $"<color=#{color}>YOUR POWER {mine}</color> / RECOMMENDED {recommended}";
        }
    }

    public sealed class MainMenuScreen : ScreenBase
    {
        public override void Build(RectTransform root, Navigator nav)
        {
            var app = nav.App;
            var data = app.Services.Data;

            // Left: title over the 3D backdrop
            var title = UIFactory.Label(root, "UNDERCURRENT", 104, Theme.Text, TextAnchor.UpperLeft, FontStyle.Bold);
            UIFactory.Anchor(title.rectTransform, 0.03f, 0.78f, 0.6f, 0.97f);
            var tag = UIFactory.Label(root, "Underground fighting championship of Varrow Bay", 28, Theme.Accent, TextAnchor.UpperLeft);
            UIFactory.Anchor(tag.rectTransform, 0.035f, 0.72f, 0.6f, 0.79f);
            var fighter = FighterRoster.Get(data.SelectedFighterId);
            var who = UIFactory.Label(root, $"{fighter.DisplayName}  \"{fighter.Nickname}\"   ·   {fighter.Style}   ·   LV {data.GetOrCreateFighter(fighter.Id).Level}", 26, Theme.TextDim, TextAnchor.LowerLeft);
            UIFactory.Anchor(who.rectTransform, 0.035f, 0.03f, 0.6f, 0.1f);

            // Right: menu column
            var panel = UIFactory.Panel(root, "MenuPanel", new Color(0.05f, 0.06f, 0.08f, 0.82f));
            UIFactory.Anchor(panel.rectTransform, 0.62f, 0.03f, 0.98f, 0.97f);
            var col = UIFactory.VStack(panel.transform, "Buttons", 14, 26);
            UIFactory.Stretch((RectTransform)col.transform);

            var wallet = UIFactory.Label(col.transform, $"PLAYER LV {data.PlayerLevel}   ·   {GearUI.Coins(data.Currency)}", 26, Theme.Accent, TextAnchor.MiddleCenter, FontStyle.Bold);
            UIFactory.Size(wallet, 50);

            var next = ProgressionService.NextLevel(data);
            string continueLabel = data.MatchesPlayed == 0 ? "START CAMPAIGN" : "CONTINUE";
            UIFactory.Size(UIFactory.Button(col.transform, $"{continueLabel}", () => nav.Push(new BriefingScreen(next))), 104);
            var sub = UIFactory.Label(col.transform, $"Ch.{next.Chapter}-{next.Number}  {next.Title}", 22, Theme.TextDim, TextAnchor.MiddleCenter);
            UIFactory.Size(sub, 30);
            UIFactory.Size(UIFactory.Button(col.transform, "CAMPAIGN", () => nav.Push(new CampaignScreen()), ButtonStyle.Secondary), 84);
            UIFactory.Size(UIFactory.Button(col.transform, "VERSUS AI", () => nav.Push(new VersusScreen()), ButtonStyle.Secondary), 84);
            UIFactory.Size(UIFactory.Button(col.transform, "TRAINING", () => app.StartTraining(), ButtonStyle.Secondary), 84);
            var row = UIFactory.HStack(col.transform, "Row", 14);
            UIFactory.Size(row, 84);
            UIFactory.Size(UIFactory.Button(row.transform, "FIGHTERS", () => nav.Push(new FighterSelectScreen(null)), ButtonStyle.Secondary, 26), -1, -1, 1);
            UIFactory.Size(UIFactory.Button(row.transform, "SHOP", () => nav.Push(new ShopScreen()), ButtonStyle.Secondary, 26), -1, -1, 1);
            var row2 = UIFactory.HStack(col.transform, "Row2", 14);
            UIFactory.Size(row2, 84);
            UIFactory.Size(UIFactory.Button(row2.transform, "AWARDS", () => nav.Push(new AchievementsScreen()), ButtonStyle.Secondary, 26), -1, -1, 1);
            UIFactory.Size(UIFactory.Button(row2.transform, "SETTINGS", () => nav.Push(new SettingsScreen()), ButtonStyle.Secondary, 26), -1, -1, 1);
            var build = UIFactory.Label(col.transform, GameApp.BuildLabel, 18, Theme.TextDim, TextAnchor.MiddleCenter);
            UIFactory.Size(build, 30);
        }
    }

    public sealed class CampaignScreen : ScreenBase
    {
        private static int _chapter = 1;

        public override void Build(RectTransform root, Navigator nav)
        {
            var content = ScreenFrame.Build(root, nav, "CAMPAIGN");
            var data = nav.App.Services.Data;

            var chapters = UIFactory.ScrollList(content, "Chapters", 12, 4);
            UIFactory.Anchor((RectTransform)chapters.parent, 0f, 0f, 0.3f, 1f);
            foreach (var ch in CampaignDatabase.Chapters)
            {
                int number = ch.Number;
                int stars = 0, maxStars = 0;
                foreach (var l in CampaignDatabase.LevelsInChapter(ch.Number)) { stars += data.GetLevel(l.Id)?.Stars ?? 0; maxStars += 3; }
                var b = UIFactory.Button(chapters, $"CH. {ch.Number}  {ch.Title.ToUpperInvariant()}\n<size=20>{stars}/{maxStars} STARS</size>", () => { _chapter = number; nav.Refresh(); },
                    number == _chapter ? ButtonStyle.Primary : ButtonStyle.Secondary, 26);
                b.GetComponentInChildren<Text>().supportRichText = true;
                UIFactory.Size(b, 96);
            }

            var detail = UIFactory.Rect(content, "Levels");
            UIFactory.Anchor(detail, 0.32f, 0f, 1f, 1f);
            var chapter = CampaignDatabase.Chapters[Mathf.Clamp(_chapter - 1, 0, CampaignDatabase.Chapters.Count - 1)];
            var summary = UIFactory.Label(detail, chapter.Summary, 28, Theme.TextDim, TextAnchor.UpperLeft);
            UIFactory.Anchor(summary.rectTransform, 0f, 0.9f, 1f, 1f);

            if (!chapter.ContentAvailable)
            {
                var soon = UIFactory.Label(detail, "This chapter has not been built yet.\nChapter 1 is the current playable content; later chapters are planned (see PROGRESS.md).",
                    30, Theme.Text, TextAnchor.MiddleCenter);
                UIFactory.Anchor(soon.rectTransform, 0.05f, 0.3f, 0.95f, 0.8f);
                return;
            }

            var list = UIFactory.ScrollList(detail, "LevelList", 14, 4);
            UIFactory.Anchor((RectTransform)list.parent, 0f, 0f, 1f, 0.88f);
            foreach (var level in CampaignDatabase.LevelsInChapter(chapter.Number))
            {
                bool unlocked = ProgressionService.IsLevelUnlocked(data, level);
                var progress = data.GetLevel(level.Id);
                var card = UIFactory.Panel(list, "Level", unlocked ? Theme.Panel : Theme.Disabled);
                UIFactory.Size(card, 170);
                var opponent = FighterRoster.Get(level.OpponentId);
                var arena = CampaignDatabase.GetArena(level.ArenaId);
                var name = UIFactory.Label(card.transform, $"{level.Chapter}-{level.Number}   {level.Title.ToUpperInvariant()}   {ScreenFrame.TierTag(level.Tier)}{(level.IsBoss ? "   [BOSS]" : "")}", 32, unlocked ? Theme.Text : Theme.TextDim, TextAnchor.UpperLeft, FontStyle.Bold);
                name.supportRichText = true;
                UIFactory.Stretch(name.rectTransform, 24, 330, 16, 90);
                string format = level.IsGauntlet ? $"GAUNTLET x{level.Gauntlet.Count}" : level.RoundsToWin == 1 ? "1 round" : $"Best of {level.RoundsToWin * 2 - 1}";
                string vs = level.IsGauntlet ? "vs " + string.Join(", ", level.Gauntlet.ConvertAll(g => FighterRoster.Get(g.OpponentId).Nickname))
                    : $"vs {opponent.DisplayName} \"{opponent.Nickname}\"" + (level.Mode == LevelMode.Weapons && level.OpponentWeaponId != null ? $" with {WeaponCatalog.Get(level.OpponentWeaponId).DisplayName.ToLowerInvariant()}" : "");
                string mode = level.Mode == LevelMode.Standard ? "" : level.Mode == LevelMode.Gauntlet ? "" : "  ·  WEAPONS";
                var info = UIFactory.Label(card.transform,
                    $"{vs}  ·  {arena.DisplayName}  ·  {format}{mode}  ·  {ScreenFrame.PowerCompare(ProgressionService.PlayerPower(data, level), level.RecommendedPower)}",
                    24, Theme.TextDim, TextAnchor.UpperLeft);
                info.supportRichText = true;
                UIFactory.Stretch(info.rectTransform, 24, 330, 62, 50);
                string status = !unlocked ? "LOCKED — win the previous fight"
                    : progress != null && progress.Completed ? $"CLEARED  ·  {StarText.Of(progress.Stars)}  ·  best {progress.BestTimeSeconds}s"
                    : $"Reward: {GearUI.Coins(level.RewardCurrency + level.FirstClearBonusCurrency)}, {level.RewardXp} XP" + (level.UnlocksFighterId != null ? $", unlocks {FighterRoster.Get(level.UnlocksFighterId).DisplayName}" : "")
                      + (level.UnlocksWeaponId != null ? $", {WeaponCatalog.Get(level.UnlocksWeaponId).DisplayName.ToLowerInvariant()} goes on sale" : "");
                var st = UIFactory.Label(card.transform, status, 22, unlocked ? Theme.Accent : Theme.TextDim, TextAnchor.UpperLeft);
                UIFactory.Stretch(st.rectTransform, 24, 330, 110, 10);
                var lv = level;
                var go = UIFactory.Button(card.transform, unlocked ? "FIGHT" : "LOCKED", () => nav.Push(new BriefingScreen(lv)), unlocked ? ButtonStyle.Primary : ButtonStyle.Secondary);
                go.interactable = unlocked;
                UIFactory.Fixed((RectTransform)go.transform, new Vector2(1f, 0.5f), new Vector2(1f, 0.5f), new Vector2(-24, 0), new Vector2(260, 100));
            }
        }
    }

    public static class StarText
    {
        public static string Of(int stars) => $"STARS {stars}/3";
    }

    public sealed class BriefingScreen : ScreenBase
    {
        private readonly LevelDefinition _level;

        public BriefingScreen(LevelDefinition level) => _level = level;

        public override void Build(RectTransform root, Navigator nav)
        {
            var content = ScreenFrame.Build(root, nav, $"CHAPTER {_level.Chapter}-{_level.Number}: {_level.Title.ToUpperInvariant()}");
            var data = nav.App.Services.Data;
            var opponent = FighterRoster.Get(_level.OpponentId);
            var me = FighterRoster.Get(data.SelectedFighterId);
            var progress = data.GetLevel(_level.Id);

            var story = UIFactory.Panel(content, "Story", Theme.Panel);
            UIFactory.Anchor(story.rectTransform, 0f, 0.42f, 0.62f, 1f);
            var storyText = UIFactory.Label(story.transform, _level.StoryIntro, 32, Theme.Text, TextAnchor.UpperLeft, FontStyle.Italic);
            UIFactory.Stretch(storyText.rectTransform, 30, 30, 26, 26);

            var opp = UIFactory.Panel(content, "Opponent", Theme.Panel);
            UIFactory.Anchor(opp.rectTransform, 0.64f, 0.42f, 1f, 1f);
            var oppText = UIFactory.Label(opp.transform,
                $"<b>{opponent.DisplayName.ToUpperInvariant()}</b>\n\"{opponent.Nickname}\" · {opponent.Style}\n\n{opponent.Bio}\n\n<color=#E8A33D>Watch for:</color> {opponent.Strengths}",
                24, Theme.Text, TextAnchor.UpperLeft);
            oppText.supportRichText = true;
            UIFactory.Stretch(oppText.rectTransform, 24, 24, 20, 20);

            var obj = UIFactory.Panel(content, "Objectives", Theme.Panel);
            UIFactory.Anchor(obj.rectTransform, 0f, 0f, 0.62f, 0.4f);
            var sb = new StringBuilder();
            if (_level.IsGauntlet)
            {
                var names = _level.Gauntlet.ConvertAll(g => FighterRoster.Get(g.OpponentId).Nickname + (g.WeaponId != null ? $" ({WeaponCatalog.Get(g.WeaponId).DisplayName.ToLowerInvariant()})" : ""));
                sb.Append($"<b>Victory:</b> win {_level.Gauntlet.Count} bouts in a row: {string.Join(", ", names)}\n");
            }
            else
                sb.Append($"<b>Victory:</b> win {(_level.RoundsToWin == 1 ? "the round" : $"{_level.RoundsToWin} rounds")}  ·  {_level.RoundSeconds}s timer\n");
            int myPower = ProgressionService.PlayerPower(data, _level);
            sb.Append($"<b>Challenge:</b> {ScreenFrame.TierTag(_level.Tier)}  ·  AI {ScreenFrame.DifficultyName(_level.Difficulty)}  ·  {ScreenFrame.PowerCompare(myPower, _level.RecommendedPower)}\n");
            if (_level.Mode == LevelMode.Weapons && !string.IsNullOrEmpty(_level.OpponentWeaponId))
                sb.Append($"<b>Opponent weapon:</b> {WeaponCatalog.Get(_level.OpponentWeaponId).DisplayName}\n");
            if (!string.IsNullOrEmpty(_level.RuleText)) sb.Append($"<color=#E8A33D><b>Rules:</b> {_level.RuleText}</color>\n");
            sb.Append($"<b>Rewards:</b> {GearUI.Coins(_level.RewardCurrency)} + {_level.FirstClearBonusCurrency} first-clear bonus, {_level.RewardXp} XP\n");
            sb.Append("<b>Optional objectives:</b>\n");
            foreach (var o in _level.Objectives)
            {
                bool done = progress != null && progress.CompletedObjectives.Contains(o.Id);
                sb.Append(done ? "  <color=#5AC76B>[DONE]</color> " : "  [  ]  ").Append(o.Describe()).Append($"  (+{o.BonusCurrency})\n");
            }
            if (myPower < _level.RecommendedPower)
                sb.Append("<color=#DC4038>You are under-powered for this fight. Upgrade armour, weapon or fighter level in the Shop — or win on skill.</color>\n");
            if (_level.StartsTutorial && !data.TutorialCompleted) sb.Append("\n<color=#33B3B8>Guided training runs first: every action is explained before you need it.</color>");
            var objText = UIFactory.Label(obj.transform, sb.ToString(), 26, Theme.Text, TextAnchor.UpperLeft);
            objText.supportRichText = true;
            UIFactory.Stretch(objText.rectTransform, 24, 24, 18, 18);

            var actions = UIFactory.VStack(content, "Actions", 16);
            UIFactory.Anchor((RectTransform)actions.transform, 0.64f, 0f, 1f, 0.4f);
            var meLabel = UIFactory.Label(actions.transform, $"Your fighter: {me.DisplayName} ({me.Style})", 26, Theme.TextDim, TextAnchor.MiddleCenter);
            UIFactory.Size(meLabel, 50);
            UIFactory.Size(UIFactory.Button(actions.transform, "CHANGE FIGHTER", () => nav.Push(new FighterSelectScreen(null)), ButtonStyle.Secondary), 76);
            if (_level.PlayerWeaponsAllowed)
            {
                var w = WeaponCatalog.Get(data.GetOrCreateFighter(me.Id).EquippedWeapon);
                UIFactory.Size(UIFactory.Button(actions.transform, $"WEAPON: {(w != null ? w.DisplayName.ToUpperInvariant() : "UNARMED")}", () => nav.Push(new ArmoryScreen(me.Id)), ButtonStyle.Secondary, 26), 76);
            }
            UIFactory.Size(UIFactory.Button(actions.transform, "SHOP / UPGRADES", () => nav.Push(new ShopScreen()), ButtonStyle.Secondary, 26), 76);
            bool unlocked = ProgressionService.IsLevelUnlocked(data, _level);
            var fight = UIFactory.Button(actions.transform, unlocked ? "FIGHT" : "LOCKED", () => nav.App.StartLevel(_level));
            fight.interactable = unlocked;
            UIFactory.Size(fight, 120);
        }
    }

    public sealed class VersusScreen : ScreenBase
    {
        private static int _opponent = 1;
        private static int _difficulty = (int)AIDifficulty.Normal;
        private static int _arena;
        private static int _rounds = 1;
        private static int _myWeapon;
        private static int _oppWeapon;

        public override void Build(RectTransform root, Navigator nav)
        {
            var content = ScreenFrame.Build(root, nav, "VERSUS AI");
            var data = nav.App.Services.Data;
            var ids = new List<string>(FighterRoster.All.Keys);
            var names = new string[ids.Count];
            for (int i = 0; i < ids.Count; i++)
            {
                var f = FighterRoster.Get(ids[i]);
                names[i] = f.DisplayName;
            }
            var diffNames = System.Enum.GetNames(typeof(AIDifficulty));
            var arenaNames = new string[CampaignDatabase.Arenas.Count];
            for (int i = 0; i < arenaNames.Length; i++) arenaNames[i] = CampaignDatabase.Arenas[i].DisplayName;

            var colT = UIFactory.ScrollList(content, "Options", 16, 10);
            UIFactory.Anchor((RectTransform)colT.parent, 0.08f, 0f, 0.92f, 1f);
            var me = FighterRoster.Get(data.SelectedFighterId);
            var meRow = UIFactory.HStack(colT, "Me", 16);
            UIFactory.Size(meRow, 80);
            UIFactory.Size(UIFactory.Label(meRow.transform, $"You: {me.DisplayName} ({me.Style})", Theme.BodySize), -1, -1, 1);
            UIFactory.Size(UIFactory.Button(meRow.transform, "CHANGE", () => nav.Push(new FighterSelectScreen(null)), ButtonStyle.Secondary), 72, 220);
            UIFactory.Selector(colT, "Opponent", names, Mathf.Clamp(_opponent, 0, names.Length - 1), i => _opponent = i);
            UIFactory.Selector(colT, "Difficulty", diffNames, _difficulty, i => { _difficulty = i; nav.Refresh(); });
            UIFactory.Selector(colT, "Arena", arenaNames, _arena, i => _arena = i);
            UIFactory.Selector(colT, "Format", new[] { "1 Round", "Best of 3", "Best of 5" }, _rounds - 1, i => _rounds = i + 1);
            var myWeapons = new List<string> { "" };
            foreach (var w in WeaponCatalog.All) if (ProgressionService.OwnsWeapon(data, w)) myWeapons.Add(w);
            var allWeapons = new List<string> { "" };
            allWeapons.AddRange(WeaponCatalog.All);
            string Label(string w) => w == "" ? "Unarmed" : WeaponCatalog.Get(w).DisplayName;
            _myWeapon = Mathf.Clamp(_myWeapon, 0, myWeapons.Count - 1);
            UIFactory.Selector(colT, "Your weapon", myWeapons.ConvertAll(Label).ToArray(), _myWeapon, i => _myWeapon = i);
            UIFactory.Selector(colT, "Opponent weapon", allWeapons.ConvertAll(Label).ToArray(), _oppWeapon, i => _oppWeapon = i);
            var note = UIFactory.Label(colT, $"A win pays {GearUI.Coins(Shop.QuickFightCoins((AIDifficulty)_difficulty))} per round won at this difficulty (harder AI pays more). Your armour, levels and weapon upgrades apply.", 22, Theme.TextDim, TextAnchor.MiddleCenter);
            UIFactory.Size(note, 50);
            UIFactory.Size(UIFactory.Button(colT, "FIGHT", () =>
                nav.App.StartVersus(ids[Mathf.Clamp(_opponent, 0, ids.Count - 1)], (AIDifficulty)_difficulty, CampaignDatabase.Arenas[_arena].Id, _rounds,
                    myWeapons[_myWeapon], allWeapons[Mathf.Clamp(_oppWeapon, 0, allWeapons.Count - 1)])), 110);
        }
    }

    public sealed class ResultsScreen : ScreenBase
    {
        private readonly FightOutcome _outcome;
        private readonly RewardResult _reward;

        public ResultsScreen(FightOutcome outcome, RewardResult reward)
        {
            _outcome = outcome;
            _reward = reward;
        }

        public override void Build(RectTransform root, Navigator nav)
        {
            var bg = UIFactory.Panel(root, "Background", new Color(0.03f, 0.035f, 0.045f, 0.9f), rounded: false);
            UIFactory.Stretch(bg.rectTransform);
            var s = _outcome.Summary;
            bool won = s.Result == MatchResultKind.Victory;
            var title = UIFactory.Label(root, won ? "VICTORY" : s.Result == MatchResultKind.Draw ? "DRAW" : "DEFEAT", 110, won ? Theme.Success : Theme.Danger, TextAnchor.MiddleCenter, FontStyle.Bold);
            UIFactory.Anchor(title.rectTransform, 0.1f, 0.8f, 0.9f, 0.97f);

            var stats = UIFactory.Panel(root, "Stats", Theme.Panel);
            UIFactory.Anchor(stats.rectTransform, 0.04f, 0.2f, 0.48f, 0.78f);
            var st = UIFactory.Label(stats.transform,
                $"<b>MATCH</b>\nRounds   {s.RoundsWon} – {s.RoundsLost}\nFight time   {s.FightSeconds}s\nDamage dealt   {s.DamageDealt}\nDamage taken   {s.DamageTaken}\n" +
                $"Best combo   {s.MaxCombo} hits\nParries   {s.Parries}\nThrows   {s.Throws}\nPerfect rounds   {s.PerfectRounds}",
                28, Theme.Text, TextAnchor.UpperLeft);
            st.supportRichText = true;
            UIFactory.Stretch(st.rectTransform, 30, 30, 24, 24);

            var rewards = UIFactory.Panel(root, "Rewards", Theme.Panel);
            UIFactory.Anchor(rewards.rectTransform, 0.52f, 0.2f, 0.96f, 0.78f);
            var sb = new StringBuilder("<b>REWARDS</b>\n");
            if (_reward != null)
            {
                sb.Append($"+{GearUI.Coins(_reward.Currency)}    +{_reward.Xp} XP\n");
                if (_reward.LevelsGained > 0) sb.Append($"<color=#E8A33D>Player level up! Now LV {_reward.NewPlayerLevel}</color>\n");
                var level = _outcome.Setup.Level;
                foreach (var id in _reward.CompletedObjectives)
                    foreach (var o in level.Objectives) if (o.Id == id) sb.Append($"Objective: {o.Describe()} (+{o.BonusCurrency})\n");
                foreach (var f in _reward.UnlockedFighters) sb.Append($"<color=#33B3B8>New fighter unlocked: {FighterRoster.Get(f).DisplayName}!</color>\n");
                foreach (var c in _reward.UnlockedCosmetics) sb.Append($"<color=#33B3B8>New item: {Catalog.GetCosmetic(c)?.DisplayName}</color>\n");
                foreach (var w in _reward.WeaponsInShop) sb.Append($"<color=#33B3B8>The {WeaponCatalog.Get(w)?.DisplayName.ToLowerInvariant()} is now on sale in the Shop ({GearUI.Coins(Shop.WeaponPrice(w))})</color>\n");
                if (won && level != null) sb.Append($"<color=#E8A33D>{StarText.Of(_reward.Stars)}{(_reward.NewBestStars ? "  (new best)" : "")}</color>\n");
                foreach (var a in _reward.UnlockedAchievements) sb.Append($"<color=#E8A33D>Award: {Catalog.GetAchievement(a)?.DisplayName}</color>\n");
                if (_reward.FirstClear && level != null && !string.IsNullOrEmpty(level.StoryOutro)) sb.Append($"\n<i>{level.StoryOutro}</i>");
                if (!won)
                {
                    int power = level != null ? ProgressionService.PlayerPower(nav.App.Services.Data, level) : 0;
                    if (level != null && power < level.RecommendedPower)
                        sb.Append($"\n<color=#E8A33D>You were under-powered ({power} vs {level.RecommendedPower}). Upgrade in the Shop, replay earlier levels or play Versus for more {Shop.CurrencyName}.</color>");
                    else sb.Append("\nTip: watch your stamina, and punish heavy attacks after you block them.");
                }
            }
            var rt = UIFactory.Label(rewards.transform, sb.ToString(), 26, Theme.Text, TextAnchor.UpperLeft);
            rt.supportRichText = true;
            UIFactory.Stretch(rt.rectTransform, 30, 30, 24, 24);

            var row = UIFactory.HStack(root, "Actions", 24);
            UIFactory.Anchor((RectTransform)row.transform, 0.1f, 0.04f, 0.9f, 0.16f);
            var app = nav.App;
            var setup = _outcome.Setup;
            if (setup.Level != null)
            {
                var next = CampaignDatabase.NextAfter(setup.Level);
                if (won && next != null)
                    UIFactory.Size(UIFactory.Button(row.transform, "NEXT FIGHT", () => nav.Reset(new BriefingScreen(next))), -1, 360, 1);
                UIFactory.Size(UIFactory.Button(row.transform, won ? "REPLAY" : "RETRY", () => app.StartLevel(setup.Level), won ? ButtonStyle.Secondary : ButtonStyle.Primary), -1, 300, 1);
                if (!won) UIFactory.Size(UIFactory.Button(row.transform, "SHOP", () => nav.Push(new ShopScreen()), ButtonStyle.Secondary), -1, 240, 1);
            }
            else
            {
                UIFactory.Size(UIFactory.Button(row.transform, "REMATCH", () => app.StartVersus(setup.Opponent.Id, setup.Difficulty, setup.Arena.Id, setup.Config.RoundsToWin, setup.Player.Weapon?.Id, setup.Opponent.Weapon?.Id)), -1, 300, 1);
            }
            UIFactory.Size(UIFactory.Button(row.transform, "MAIN MENU", app.ShowMainMenu, ButtonStyle.Secondary), -1, 300, 1);
        }
    }
}
