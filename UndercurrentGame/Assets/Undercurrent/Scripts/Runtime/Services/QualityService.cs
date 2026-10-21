using Undercurrent.Core.Progression;
using UnityEngine;
#if UC_URP
using UnityEngine.Rendering;
using UnityEngine.Rendering.Universal;
#endif

namespace Undercurrent.Runtime.Services
{
    /// <summary>
    /// Applies Low/Medium/High/Ultra presets. Values are starting points; real device profiling is still required
    /// (no device measurements exist for this project yet — see KNOWN_ISSUES.md).
    /// </summary>
    public static class QualityService
    {
        public struct TierValues
        {
            public float RenderScale;
            public float ShadowDistance;
            public int Msaa;
            public float LodBias;
            public int TextureMipLimit;
            public float VfxDensity;
            public bool Shadows;
        }

        public static TierValues Values(GraphicsTier tier)
        {
            switch (tier)
            {
                case GraphicsTier.Low: return new TierValues { RenderScale = 0.7f, ShadowDistance = 0f, Msaa = 1, LodBias = 0.6f, TextureMipLimit = 1, VfxDensity = 0.4f, Shadows = false };
                case GraphicsTier.Medium: return new TierValues { RenderScale = 0.85f, ShadowDistance = 12f, Msaa = 1, LodBias = 1f, TextureMipLimit = 0, VfxDensity = 0.7f, Shadows = true };
                case GraphicsTier.High: return new TierValues { RenderScale = 1f, ShadowDistance = 18f, Msaa = 2, LodBias = 1.5f, TextureMipLimit = 0, VfxDensity = 1f, Shadows = true };
                default: return new TierValues { RenderScale = 1f, ShadowDistance = 25f, Msaa = 4, LodBias = 2f, TextureMipLimit = 0, VfxDensity = 1f, Shadows = true };
            }
        }

        public static float VfxDensity { get; private set; } = 1f;
        /// <summary>Tessellation multiplier for procedural characters.</summary>
        public static float CharacterDetail { get; private set; } = 1f;
        public static bool ShadowsEnabled { get; private set; } = true;

        /// <summary>Picks a conservative default tier from device memory the first time the game runs.</summary>
        public static GraphicsTier SuggestDefault()
        {
            int mb = SystemInfo.systemMemorySize;
            if (mb > 0 && mb < 3000) return GraphicsTier.Low;
            if (mb > 0 && mb < 6000) return GraphicsTier.Medium;
            return GraphicsTier.High;
        }

        public static void Apply(SettingsData settings)
        {
            var tier = (GraphicsTier)Mathf.Clamp(settings.GraphicsTier, 0, 3);
            var v = Values(tier);
            QualitySettings.vSyncCount = 0;
            Application.targetFrameRate = settings.TargetFps;
            QualitySettings.lodBias = v.LodBias;
            QualitySettings.globalTextureMipmapLimit = v.TextureMipLimit;
            QualitySettings.shadowDistance = v.ShadowDistance;
            VfxDensity = v.VfxDensity;
            CharacterDetail = tier == GraphicsTier.Low ? 0.6f : tier == GraphicsTier.Medium ? 0.85f : tier == GraphicsTier.High ? 1f : 1.2f;
            ShadowsEnabled = v.Shadows;

#if UC_URP
            var urp = GraphicsSettings.currentRenderPipeline as UniversalRenderPipelineAsset;
            if (urp != null)
            {
                if (Application.isEditor)
                {
                    // Avoid permanently editing the project's pipeline asset from Play Mode.
                    Debug.Log($"[Undercurrent] Quality {tier}: renderScale={v.RenderScale} shadowDistance={v.ShadowDistance} msaa={v.Msaa} (URP asset not modified in Editor)");
                }
                else
                {
                    urp.renderScale = v.RenderScale;
                    urp.shadowDistance = v.ShadowDistance;
                    urp.msaaSampleCount = v.Msaa;
                }
            }
#endif
        }
    }
}
