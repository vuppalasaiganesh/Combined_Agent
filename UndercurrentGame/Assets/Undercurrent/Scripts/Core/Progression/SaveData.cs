using System;
using System.Collections.Generic;

namespace Undercurrent.Core.Progression
{
    // All persistent types use public fields and Lists only, so they serialize with Unity's JsonUtility
    // and with System.Text.Json (IncludeFields) in the standalone test runner.

    [Serializable]
    public sealed class SaveData
    {
        public int Version;
        public int Currency;
        public int PlayerXp;
        public int PlayerLevel = 1;
        public string SelectedFighterId;
        public string LastPlayedLevelId;
        public bool TutorialCompleted;

        public List<string> UnlockedFighters = new List<string>();
        public List<FighterProgress> Fighters = new List<FighterProgress>();
        public List<string> OwnedCosmetics = new List<string>();
        public List<LevelProgress> Levels = new List<LevelProgress>();
        public List<string> UnlockedAchievements = new List<string>();
        /// <summary>Weapons the player owns (bought in the Shop; saves before v3 kept their free campaign weapons).</summary>
        public List<string> UnlockedWeapons = new List<string>();
        /// <summary>Upgrade level of each owned weapon (1..5).</summary>
        public List<WeaponProgress> WeaponLevels = new List<WeaponProgress>();
        /// <summary>Armour tier 0..5, shared by every fighter.</summary>
        public int ArmorTier;
        /// <summary>Total Coins received from coin packs (support/refund bookkeeping).</summary>
        public int PurchasedCoins;

        public int MatchesPlayed;
        public int TotalWins;
        public int TotalParries;
        public int TotalThrows;
        public int TotalPerfectRounds;
        public int BestCombo;
        public int TotalDisarms;
        public int GauntletsCleared;

        public SettingsData Settings = new SettingsData();

        public FighterProgress GetFighter(string id)
        {
            foreach (var f in Fighters) if (f.FighterId == id) return f;
            return null;
        }

        public FighterProgress GetOrCreateFighter(string id)
        {
            var f = GetFighter(id);
            if (f != null) return f;
            f = new FighterProgress { FighterId = id };
            Fighters.Add(f);
            return f;
        }

        public LevelProgress GetLevel(string id)
        {
            foreach (var l in Levels) if (l.LevelId == id) return l;
            return null;
        }

        public LevelProgress GetOrCreateLevel(string id)
        {
            var l = GetLevel(id);
            if (l != null) return l;
            l = new LevelProgress { LevelId = id };
            Levels.Add(l);
            return l;
        }

        public bool IsLevelCompleted(string id) => GetLevel(id)?.Completed ?? false;

        public WeaponProgress GetWeapon(string id)
        {
            foreach (var w in WeaponLevels) if (w.WeaponId == id) return w;
            return null;
        }
    }

    [Serializable]
    public sealed class WeaponProgress
    {
        public string WeaponId;
        public int Level = 1;
    }

    [Serializable]
    public sealed class FighterProgress
    {
        public string FighterId;
        public int Level = 1;
        public string EquippedOutfit = "outfit_default";
        public string EquippedWraps = "wraps_white";
        public int SkinToneIndex = -1;
        /// <summary>Weapon brought to weapon-mode fights ("" = unarmed).</summary>
        public string EquippedWeapon = "";
    }

    [Serializable]
    public sealed class LevelProgress
    {
        public string LevelId;
        public bool Completed;
        public int Attempts;
        public int Wins;
        public int BestTimeSeconds;
        /// <summary>0-3: one for the win, one per completed optional objective.</summary>
        public int Stars;
        public List<string> CompletedObjectives = new List<string>();
    }

    [Serializable]
    public sealed class ControlOffset
    {
        public string ControlId;
        public float X;
        public float Y;
    }

    public enum GraphicsTier
    {
        Low = 0,
        Medium = 1,
        High = 2,
        Ultra = 3,
    }

    [Serializable]
    public sealed class SettingsData
    {
        public float MusicVolume = 0.7f;
        public float SfxVolume = 0.9f;
        public float VoiceVolume = 0.9f;
        public bool HapticsEnabled = true;
        public float HapticIntensity = 1f;
        public float ScreenShakeScale = 1f;
        public bool ReduceFlashes;
        public int GraphicsTier = 1;
        public int TargetFps = 60;
        public float ControlScale = 1f;
        public float ControlOpacity = 0.75f;
        public bool ShowInputHints = true;
        public List<ControlOffset> ControlOffsets = new List<ControlOffset>();

        public ControlOffset GetOffset(string id)
        {
            foreach (var o in ControlOffsets) if (o.ControlId == id) return o;
            return null;
        }

        public void SetOffset(string id, float x, float y)
        {
            var o = GetOffset(id);
            if (o == null) { o = new ControlOffset { ControlId = id }; ControlOffsets.Add(o); }
            o.X = x;
            o.Y = y;
        }
    }
}
