using Undercurrent.Core.Progression;

namespace Undercurrent.Runtime.Services
{
    /// <summary>Single place that owns the game's long-lived services.</summary>
    public sealed class ServiceHub
    {
        public SaveRepository Save;
        public AudioService Audio;
        public HapticsService Haptics;

        // Optional online services: offline implementations until real backends are configured (see KNOWN_ISSUES.md).
        public IBillingService Billing = new UnconfiguredBillingService();
        public ICloudSaveService CloudSave = new UnconfiguredCloudSaveService();
        public ILeaderboardService Leaderboards = new UnconfiguredLeaderboardService();
        public IAccountService Account = new OfflineAccountService();

        public SaveData Data => Save.Data;
        public SettingsData Settings => Save.Data.Settings;

        public void Persist() => Save.Save();
    }
}
