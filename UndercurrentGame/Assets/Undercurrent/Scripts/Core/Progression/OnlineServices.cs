using System;
using System.Collections.Generic;

namespace Undercurrent.Core.Progression
{
    /// <summary>Integration state reported honestly to the UI and logs.</summary>
    public enum IntegrationStatus
    {
        /// <summary>No store/backend credentials or SDK configured in this build.</summary>
        NotConfigured,
        /// <summary>Development mock: behaves predictably, performs no real transaction or network call.</summary>
        DevelopmentMock,
        /// <summary>Real backend connected and verified (not available in this build).</summary>
        Live,
    }

    public sealed class StoreProduct
    {
        public string ProductId;
        public string Title;
        public string LocalizedPrice;
        public bool IsSubscription;
    }

    public enum BillingOutcome
    {
        Purchased,
        Cancelled,
        Unavailable,
        Failed,
    }

    /// <summary>
    /// Real-money purchases (coin packs). Everything coins buy can also be earned by playing, so the game is fully
    /// playable when this is unavailable.
    /// </summary>
    public interface IBillingService
    {
        IntegrationStatus Status { get; }
        IReadOnlyList<StoreProduct> Products { get; }
        void Purchase(string productId, Action<BillingOutcome> callback);
        void RestorePurchases(Action<IReadOnlyList<string>> callback);
    }

    public interface ICloudSaveService
    {
        IntegrationStatus Status { get; }
        void Upload(string saveText, Action<bool> callback);
        void Download(Action<bool, string> callback);
    }

    public interface ILeaderboardService
    {
        IntegrationStatus Status { get; }
        void SubmitScore(string boardId, long score, Action<bool> callback);
    }

    public interface IAccountService
    {
        IntegrationStatus Status { get; }
        bool IsSignedIn { get; }
        string DisplayName { get; }
        void SignIn(Action<bool> callback);
    }

    /// <summary>Billing placeholder: reports NotConfigured and refuses every purchase. No fake success paths.</summary>
    public sealed class UnconfiguredBillingService : IBillingService
    {
        public IntegrationStatus Status => IntegrationStatus.NotConfigured;
        public IReadOnlyList<StoreProduct> Products { get; } = new List<StoreProduct>();
        public void Purchase(string productId, Action<BillingOutcome> callback) => callback?.Invoke(BillingOutcome.Unavailable);
        public void RestorePurchases(Action<IReadOnlyList<string>> callback) => callback?.Invoke(new List<string>());
    }

    /// <summary>
    /// Editor/development-build stand-in so the coin-pack flow can be exercised. Reports
    /// <see cref="IntegrationStatus.DevelopmentMock"/>; the UI labels its purchases as test purchases. Never ship it.
    /// </summary>
    public sealed class DevelopmentMockBillingService : IBillingService
    {
        private readonly List<StoreProduct> _products = new List<StoreProduct>();

        public DevelopmentMockBillingService()
        {
            foreach (var p in Shop.CoinPacks)
                _products.Add(new StoreProduct { ProductId = p.ProductId, Title = p.DisplayName, LocalizedPrice = "TEST" });
        }

        public IntegrationStatus Status => IntegrationStatus.DevelopmentMock;
        public IReadOnlyList<StoreProduct> Products => _products;
        public void Purchase(string productId, Action<BillingOutcome> callback) =>
            callback?.Invoke(Shop.GetCoinPack(productId) != null ? BillingOutcome.Purchased : BillingOutcome.Failed);
        public void RestorePurchases(Action<IReadOnlyList<string>> callback) => callback?.Invoke(new List<string>());
    }

    public sealed class UnconfiguredCloudSaveService : ICloudSaveService
    {
        public IntegrationStatus Status => IntegrationStatus.NotConfigured;
        public void Upload(string saveText, Action<bool> callback) => callback?.Invoke(false);
        public void Download(Action<bool, string> callback) => callback?.Invoke(false, null);
    }

    public sealed class UnconfiguredLeaderboardService : ILeaderboardService
    {
        public IntegrationStatus Status => IntegrationStatus.NotConfigured;
        public void SubmitScore(string boardId, long score, Action<bool> callback) => callback?.Invoke(false);
    }

    /// <summary>Offline guest profile. The game never requires sign-in.</summary>
    public sealed class OfflineAccountService : IAccountService
    {
        public IntegrationStatus Status => IntegrationStatus.NotConfigured;
        public bool IsSignedIn => false;
        public string DisplayName => "Guest";
        public void SignIn(Action<bool> callback) => callback?.Invoke(false);
    }
}
