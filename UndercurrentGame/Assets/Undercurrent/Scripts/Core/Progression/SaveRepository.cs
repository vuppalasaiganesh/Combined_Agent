using System;
using System.Collections.Generic;
using Undercurrent.Core.Content;

namespace Undercurrent.Core.Progression
{
    /// <summary>Turns save data into text and back (JsonUtility in Unity, System.Text.Json in tests).</summary>
    public interface ISaveCodec
    {
        string Encode(SaveData data);
        SaveData Decode(string text);
    }

    /// <summary>Where save text lives (file on device, memory in tests, cloud in the future).</summary>
    public interface ISaveStore
    {
        bool TryRead(out string text);
        void Write(string text);
        /// <summary>Keeps a copy of unreadable data so it is never silently destroyed.</summary>
        void WriteBackup(string text, string reason);
    }

    public enum SaveLoadStatus
    {
        NewSave,
        Loaded,
        Migrated,
        RecoveredFromCorruption,
        NewerVersionReadOnly,
    }

    public static class SaveMigrator
    {
        /// <summary>Bump when the save layout changes, and add a step to <see cref="Steps"/>.</summary>
        public const int CurrentVersion = 3;

        /// <summary>Migration steps keyed by the version they upgrade FROM. Version 0 = pre-release/unversioned saves.</summary>
        private static readonly Dictionary<int, Action<SaveData>> Steps = new Dictionary<int, Action<SaveData>>
        {
            { 0, s => { /* Unversioned development saves: structure matches v1, normalisation fills gaps. */ } },
            { 1, s =>
                {
                    // v2 adds weapons and star ratings: compute stars for levels cleared under v1.
                    if (s.Levels == null) return;
                    foreach (var lp in s.Levels)
                    {
                        if (lp.CompletedObjectives == null) lp.CompletedObjectives = new List<string>();
                        var def = CampaignDatabase.GetLevel(lp.LevelId);
                        if (def != null) lp.Stars = ProgressionService.StarsFor(def, lp);
                    }
                }
            },
            // v3 sells weapons for Coins and adds armour and weapon levels. Weapons a v2 save already won are kept
            // (as level 1); normalisation creates their level entries.
            { 2, s => { } },
        };

        /// <summary>Upgrades in place. Returns false when the save comes from a newer build and must not be modified.</summary>
        public static bool Migrate(SaveData save, out bool changed)
        {
            changed = false;
            if (save.Version > CurrentVersion) return false;
            while (save.Version < CurrentVersion)
            {
                if (Steps.TryGetValue(save.Version, out var step)) step(save);
                save.Version++;
                changed = true;
            }
            changed |= Normalize(save);
            return true;
        }

        /// <summary>Repairs missing lists, invalid values and references to content that no longer exists.</summary>
        public static bool Normalize(SaveData s)
        {
            bool changed = false;
            if (s.UnlockedFighters == null) { s.UnlockedFighters = new List<string>(); changed = true; }
            if (s.Fighters == null) { s.Fighters = new List<FighterProgress>(); changed = true; }
            if (s.OwnedCosmetics == null) { s.OwnedCosmetics = new List<string>(); changed = true; }
            if (s.Levels == null) { s.Levels = new List<LevelProgress>(); changed = true; }
            if (s.UnlockedAchievements == null) { s.UnlockedAchievements = new List<string>(); changed = true; }
            if (s.UnlockedWeapons == null) { s.UnlockedWeapons = new List<string>(); changed = true; }
            changed |= s.UnlockedWeapons.RemoveAll(id => WeaponCatalog.Get(id) == null) > 0;
            if (s.WeaponLevels == null) { s.WeaponLevels = new List<WeaponProgress>(); changed = true; }
            changed |= s.WeaponLevels.RemoveAll(w => w == null || !s.UnlockedWeapons.Contains(w.WeaponId)) > 0;
            foreach (var id in s.UnlockedWeapons)
                if (s.GetWeapon(id) == null) { s.WeaponLevels.Add(new WeaponProgress { WeaponId = id, Level = 1 }); changed = true; }
            foreach (var w in s.WeaponLevels)
            {
                int lvl = Math.Max(1, Math.Min(Shop.MaxWeaponLevel, w.Level));
                if (lvl != w.Level) { w.Level = lvl; changed = true; }
            }
            int armor = Math.Max(0, Math.Min(Shop.MaxArmorTier, s.ArmorTier));
            if (armor != s.ArmorTier) { s.ArmorTier = armor; changed = true; }
            if (s.PurchasedCoins < 0) { s.PurchasedCoins = 0; changed = true; }
            if (s.Settings == null) { s.Settings = new SettingsData(); changed = true; }
            if (s.Settings.ControlOffsets == null) { s.Settings.ControlOffsets = new List<ControlOffset>(); changed = true; }

            foreach (var id in ProgressionService.StartingFighters)
            {
                if (!s.UnlockedFighters.Contains(id)) { s.UnlockedFighters.Add(id); changed = true; }
            }
            changed |= s.UnlockedFighters.RemoveAll(id => !FighterRoster.All.ContainsKey(id)) > 0;
            changed |= s.OwnedCosmetics.RemoveAll(id => Catalog.GetCosmetic(id) == null) > 0;
            foreach (var basic in new[] { "outfit_default", "wraps_white" })
                if (!s.OwnedCosmetics.Contains(basic)) { s.OwnedCosmetics.Add(basic); changed = true; }

            foreach (var id in s.UnlockedFighters)
                if (s.GetFighter(id) == null) { s.GetOrCreateFighter(id); changed = true; }
            foreach (var f in s.Fighters)
            {
                int lvl = Math.Max(1, Math.Min(Catalog.MaxFighterLevel, f.Level));
                if (lvl != f.Level) { f.Level = lvl; changed = true; }
                if (!s.OwnedCosmetics.Contains(f.EquippedOutfit)) { f.EquippedOutfit = "outfit_default"; changed = true; }
                if (!s.OwnedCosmetics.Contains(f.EquippedWraps)) { f.EquippedWraps = "wraps_white"; changed = true; }
                if (f.EquippedWeapon == null || (f.EquippedWeapon.Length > 0 && !s.UnlockedWeapons.Contains(f.EquippedWeapon))) { f.EquippedWeapon = ""; changed = true; }
            }
            foreach (var l in s.Levels)
                if (l.CompletedObjectives == null) { l.CompletedObjectives = new List<string>(); changed = true; }

            if (s.Currency < 0) { s.Currency = 0; changed = true; }
            if (s.PlayerLevel < 1) { s.PlayerLevel = 1; changed = true; }
            if (s.PlayerXp < 0) { s.PlayerXp = 0; changed = true; }
            if (string.IsNullOrEmpty(s.SelectedFighterId) || !s.UnlockedFighters.Contains(s.SelectedFighterId))
            {
                s.SelectedFighterId = s.UnlockedFighters.Count > 0 ? s.UnlockedFighters[0] : FighterRoster.Mara;
                changed = true;
            }

            var st = s.Settings;
            float Clamp01(float v) => v < 0f ? 0f : v > 1f ? 1f : v;
            st.MusicVolume = Clamp01(st.MusicVolume);
            st.SfxVolume = Clamp01(st.SfxVolume);
            st.VoiceVolume = Clamp01(st.VoiceVolume);
            st.HapticIntensity = Clamp01(st.HapticIntensity);
            st.ScreenShakeScale = Clamp01(st.ScreenShakeScale);
            st.ControlOpacity = Math.Max(0.2f, Clamp01(st.ControlOpacity));
            st.ControlScale = Math.Max(0.7f, Math.Min(1.4f, st.ControlScale));
            st.GraphicsTier = Math.Max(0, Math.Min(3, st.GraphicsTier));
            if (st.TargetFps != 30 && st.TargetFps != 60) st.TargetFps = 60;
            return changed;
        }
    }

    /// <summary>Loads and saves progress safely: corrupt data is backed up, newer-version data is never overwritten.</summary>
    public sealed class SaveRepository
    {
        private readonly ISaveStore _store;
        private readonly ISaveCodec _codec;

        public SaveData Data { get; private set; }
        public SaveLoadStatus Status { get; private set; }
        public bool ReadOnly => Status == SaveLoadStatus.NewerVersionReadOnly;

        public SaveRepository(ISaveStore store, ISaveCodec codec)
        {
            _store = store ?? throw new ArgumentNullException(nameof(store));
            _codec = codec ?? throw new ArgumentNullException(nameof(codec));
        }

        public SaveData Load()
        {
            if (!_store.TryRead(out var text) || string.IsNullOrWhiteSpace(text))
            {
                Data = ProgressionService.CreateNew();
                Status = SaveLoadStatus.NewSave;
                return Data;
            }

            SaveData decoded = null;
            try { decoded = _codec.Decode(text); }
            catch (Exception) { decoded = null; }

            if (decoded == null)
            {
                _store.WriteBackup(text, "corrupt");
                Data = ProgressionService.CreateNew();
                Status = SaveLoadStatus.RecoveredFromCorruption;
                Save();
                return Data;
            }

            int originalVersion = decoded.Version;
            if (!SaveMigrator.Migrate(decoded, out bool changed))
            {
                Data = decoded;
                Status = SaveLoadStatus.NewerVersionReadOnly;
                return Data;
            }

            Data = decoded;
            Status = originalVersion < SaveMigrator.CurrentVersion ? SaveLoadStatus.Migrated : SaveLoadStatus.Loaded;
            if (Status == SaveLoadStatus.Migrated) _store.WriteBackup(text, "pre-migration-v" + originalVersion);
            if (changed) Save();
            return Data;
        }

        /// <summary>Writes the current data. Returns false (and writes nothing) for read-only newer-version saves.</summary>
        public bool Save()
        {
            if (Data == null || ReadOnly) return false;
            _store.Write(_codec.Encode(Data));
            return true;
        }

        public void ResetProgress()
        {
            var settings = Data?.Settings;
            Data = ProgressionService.CreateNew();
            if (settings != null) Data.Settings = settings;
            Status = SaveLoadStatus.NewSave;
            Save();
        }
    }
}
