using System.Collections.Generic;
using System.Text;
using Undercurrent.Core.AI;
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
            var wallet = UIFactory.Label(header, $"LV {data.PlayerLevel}    <color=#E8A33D>{data.Currency} MARKS</color>", Theme.BodySize, Theme.Text, TextAnchor.MiddleRight, FontStyle.Bold);
            wallet.supportRichText = true;
            UIFactory.Anchor(wallet.rectTransform, 0.6f, 0f, 0.97f, 1f);

            var content = UIFactory.Rect(root, "Content");
            UIFactory.Anchor(content, 0.03f, 0.04f, 0.97f, 0.85f);
            return content;
        }

        public static string DifficultyName(AIDifficulty d) => d.ToString().ToUpperInvariant();
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

            var wallet = UIFactory.Label(col.transform, $"PLAYER LV {data.PlayerLevel}   ·   {data.Currency} MARKS", 26, Theme.Accent, TextAnchor.MiddleCenter, FontStyle.Bold);
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
            UIFactory.Size(UIFactory.Button(row.transform, "STORE", () => nav.Push(new StoreScreen()), ButtonStyle.Secondary, 26), -1, -1, 1);
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
                var b = UIFactory.Button(chapters, $"CH. {ch.Number}  {ch.Title.ToUpperInvariant()}", () => { _chapter = number; nav.Refresh(); },
                    number == _chapter ? ButtonStyle.Primary : ButtonStyle.Secondary, 26);
                UIFactory.Size(b, 84);
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
                var name = UIFactory.Label(card.transform, $"{level.Chapter}-{level.Number}   {level.Title.ToUpperInvariant()}{(level.IsBoss ? "   [BOSS]" : "")}", 32, unlocked ? Theme.Text : Theme.TextDim, TextAnchor.UpperLeft, FontStyle.Bold);
                UIFactory.Stretch(name.rectTransform, 24, 330, 16, 90);
                var info = UIFactory.Label(card.transform,
                    $"vs {opponent.DisplayName} \"{opponent.Nickname}\" ({opponent.Style})  ·  {arena.DisplayName}  ·  {ScreenFrame.DifficultyName(level.Difficulty)}  ·  {(level.RoundsToWin == 1 ? "1 round" : $"Best of {level.RoundsToWin * 2 - 1}")}",
                    24, Theme.TextDim, TextAnchor.UpperLeft);
                UIFactory.Stretch(info.rectTransform, 24, 330, 62, 50);
                string status = !unlocked ? "LOCKED — win the previous fight"
                    : progress != null && progress.Completed ? $"CLEARED  ·  objectives {progress.CompletedObjectives.Count}/{level.Objectives.Count}  ·  best {progress.BestTimeSeconds}s"
                    : $"Reward: {level.RewardCurrency + level.FirstClearBonusCurrency} Marks, {level.RewardXp} XP" + (level.UnlocksFighterId != null ? $", unlocks {FighterRoster.Get(level.UnlocksFighterId).DisplayName}" : "");
                var st = UIFactory.Label(card.transform, status, 22, unlocked ? Theme.Accent : Theme.TextDim, TextAnchor.UpperLeft);
                UIFactory.Stretch(st.rectTransform, 24, 330, 110, 10);
                var lv = level;
                var go = UIFactory.Button(card.transform, unlocked ? "FIGHT" : "LOCKED", () => nav.Push(new BriefingScreen(lv)), unlocked ? ButtonStyle.Primary : ButtonStyle.Secondary);
                go.interactable = unlocked;
                UIFactory.Fixed((RectTransform)go.transform, new Vector2(1f, 0.5f), new Vector2(1f, 0.5f), new Vector2(-24, 0), new Vector2(260, 100));
            }
        }
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
            sb.Append($"<b>Victory:</b> win {(_level.RoundsToWin == 1 ? "the round" : $"{_level.RoundsToWin} rounds")}  ·  {_level.RoundSeconds}s timer  ·  {ScreenFrame.DifficultyName(_level.Difficulty)}\n");
            sb.Append($"<b>Rewards:</b> {_level.RewardCurrency} Marks + {_level.FirstClearBonusCurrency} first-clear bonus, {_level.RewardXp} XP\n");
            sb.Append("<b>Optional objectives:</b>\n");
            foreach (var o in _level.Objectives)
            {
                bool done = progress != null && progress.CompletedObjectives.Contains(o.Id);
                sb.Append(done ? "  <color=#5AC76B>[DONE]</color> " : "  [  ]  ").Append(o.Describe()).Append($"  (+{o.BonusCurrency})\n");
            }
            if (_level.StartsTutorial && !data.TutorialCompleted) sb.Append("\n<color=#33B3B8>Guided training runs first: every action is explained before you need it.</color>");
            var objText = UIFactory.Label(obj.transform, sb.ToString(), 26, Theme.Text, TextAnchor.UpperLeft);
            objText.supportRichText = true;
            UIFactory.Stretch(objText.rectTransform, 24, 24, 18, 18);

            var actions = UIFactory.VStack(content, "Actions", 16);
            UIFactory.Anchor((RectTransform)actions.transform, 0.64f, 0f, 1f, 0.4f);
            var meLabel = UIFactory.Label(actions.transform, $"Your fighter: {me.DisplayName} ({me.Style})", 26, Theme.TextDim, TextAnchor.MiddleCenter);
            UIFactory.Size(meLabel, 50);
            UIFactory.Size(UIFactory.Button(actions.transform, "CHANGE FIGHTER", () => nav.Push(new FighterSelectScreen(null)), ButtonStyle.Secondary), 86);
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

            var col = UIFactory.VStack(content, "Options", 22, 10);
            UIFactory.Anchor((RectTransform)col.transform, 0.1f, 0.1f, 0.9f, 1f);
            var me = FighterRoster.Get(data.SelectedFighterId);
            var meRow = UIFactory.HStack(col.transform, "Me", 16);
            UIFactory.Size(meRow, 80);
            UIFactory.Size(UIFactory.Label(meRow.transform, $"You: {me.DisplayName} ({me.Style})", Theme.BodySize), -1, -1, 1);
            UIFactory.Size(UIFactory.Button(meRow.transform, "CHANGE", () => nav.Push(new FighterSelectScreen(null)), ButtonStyle.Secondary), 72, 220);
            UIFactory.Selector(col.transform, "Opponent", names, Mathf.Clamp(_opponent, 0, names.Length - 1), i => _opponent = i);
            UIFactory.Selector(col.transform, "Difficulty", diffNames, _difficulty, i => _difficulty = i);
            UIFactory.Selector(col.transform, "Arena", arenaNames, _arena, i => _arena = i);
            UIFactory.Selector(col.transform, "Format", new[] { "1 Round", "Best of 3", "Best of 5" }, _rounds - 1, i => _rounds = i + 1);
            var note = UIFactory.Label(col.transform, "Versus matches award a little XP and Marks. The campaign is the main path for unlocks.", 22, Theme.TextDim, TextAnchor.MiddleCenter);
            UIFactory.Size(note, 50);
            UIFactory.Size(UIFactory.Button(col.transform, "FIGHT", () =>
                nav.App.StartVersus(ids[Mathf.Clamp(_opponent, 0, ids.Count - 1)], (AIDifficulty)_difficulty, CampaignDatabase.Arenas[_arena].Id, _rounds)), 110);
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
                sb.Append($"+{_reward.Currency} Marks    +{_reward.Xp} XP\n");
                if (_reward.LevelsGained > 0) sb.Append($"<color=#E8A33D>Player level up! Now LV {_reward.NewPlayerLevel}</color>\n");
                var level = _outcome.Setup.Level;
                foreach (var id in _reward.CompletedObjectives)
                    foreach (var o in level.Objectives) if (o.Id == id) sb.Append($"Objective: {o.Describe()} (+{o.BonusCurrency})\n");
                foreach (var f in _reward.UnlockedFighters) sb.Append($"<color=#33B3B8>New fighter unlocked: {FighterRoster.Get(f).DisplayName}!</color>\n");
                foreach (var c in _reward.UnlockedCosmetics) sb.Append($"<color=#33B3B8>New item: {Catalog.GetCosmetic(c)?.DisplayName}</color>\n");
                foreach (var a in _reward.UnlockedAchievements) sb.Append($"<color=#E8A33D>Award: {Catalog.GetAchievement(a)?.DisplayName}</color>\n");
                if (_reward.FirstClear && level != null && !string.IsNullOrEmpty(level.StoryOutro)) sb.Append($"\n<i>{level.StoryOutro}</i>");
                if (!won) sb.Append("\nTip: watch your stamina, and punish heavy attacks after you block them.");
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
                LevelDefinition next = null;
                foreach (var l in CampaignDatabase.Levels) if (l.RequiresLevelId == setup.Level.Id) next = l;
                if (won && next != null)
                    UIFactory.Size(UIFactory.Button(row.transform, "NEXT FIGHT", () => nav.Reset(new BriefingScreen(next))), -1, 360, 1);
                UIFactory.Size(UIFactory.Button(row.transform, won ? "REPLAY" : "RETRY", () => app.StartLevel(setup.Level), won ? ButtonStyle.Secondary : ButtonStyle.Primary), -1, 300, 1);
            }
            else
            {
                UIFactory.Size(UIFactory.Button(row.transform, "REMATCH", () => app.StartVersus(setup.Opponent.Id, setup.Difficulty, setup.Arena.Id, setup.Config.RoundsToWin)), -1, 300, 1);
            }
            UIFactory.Size(UIFactory.Button(row.transform, "MAIN MENU", app.ShowMainMenu, ButtonStyle.Secondary), -1, 300, 1);
        }
    }
}
