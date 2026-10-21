using Undercurrent.Core.Progression;
using UnityEngine;

namespace Undercurrent.Runtime.Services
{
    public enum HapticPattern
    {
        LightHit,
        HeavyHit,
        Blocked,
        Parry,
        Knockdown,
        KO,
        UiTap,
    }

    /// <summary>
    /// Distinct vibration patterns through Android's Vibrator/VibrationEffect APIs (API 26+ amplitude control),
    /// falling back to Handheld.Vibrate where fine control is unavailable. Respects the haptics settings.
    /// Not verified on hardware in this environment.
    /// </summary>
    public sealed class HapticsService
    {
        private readonly SettingsData _settings;
#if UNITY_ANDROID && !UNITY_EDITOR
        private AndroidJavaObject _vibrator;
        private AndroidJavaClass _effectClass;
        private bool _hasAmplitude;
        private int _sdk;
#endif
        private bool _initialized;

        public HapticsService(SettingsData settings)
        {
            _settings = settings;
        }

        private void EnsureInit()
        {
            if (_initialized) return;
            _initialized = true;
#if UNITY_ANDROID && !UNITY_EDITOR
            try
            {
                using (var version = new AndroidJavaClass("android.os.Build$VERSION")) _sdk = version.GetStatic<int>("SDK_INT");
                using (var player = new AndroidJavaClass("com.unity3d.player.UnityPlayer"))
                {
                    var activity = player.GetStatic<AndroidJavaObject>("currentActivity");
                    _vibrator = activity.Call<AndroidJavaObject>("getSystemService", "vibrator");
                }
                if (_sdk >= 26 && _vibrator != null)
                {
                    _effectClass = new AndroidJavaClass("android.os.VibrationEffect");
                    _hasAmplitude = _vibrator.Call<bool>("hasAmplitudeControl");
                }
            }
            catch (System.Exception e)
            {
                Debug.LogWarning("[Undercurrent] Haptics unavailable: " + e.Message);
                _vibrator = null;
            }
#endif
        }

        /// <summary>Duration (ms) and amplitude (1-255) of each pattern before the intensity setting is applied.</summary>
        public static (long ms, int amplitude) Shape(HapticPattern pattern)
        {
            switch (pattern)
            {
                case HapticPattern.LightHit: return (14, 70);
                case HapticPattern.HeavyHit: return (32, 170);
                case HapticPattern.Blocked: return (18, 50);
                case HapticPattern.Parry: return (45, 120);
                case HapticPattern.Knockdown: return (60, 200);
                case HapticPattern.KO: return (140, 255);
                default: return (8, 40);
            }
        }

        public void Play(HapticPattern pattern)
        {
            if (_settings == null || !_settings.HapticsEnabled || _settings.HapticIntensity <= 0.01f) return;
            EnsureInit();

#if UNITY_ANDROID && !UNITY_EDITOR
            if (_vibrator == null) return;
            var (ms, baseAmplitude) = Shape(pattern);
            int amplitude = Mathf.Clamp(Mathf.RoundToInt(baseAmplitude * _settings.HapticIntensity), 1, 255);
            try
            {
                if (_effectClass != null)
                {
                    using (var effect = _effectClass.CallStatic<AndroidJavaObject>("createOneShot", ms, _hasAmplitude ? amplitude : -1))
                        _vibrator.Call("vibrate", effect);
                }
                else
                {
                    _vibrator.Call("vibrate", ms);
                }
            }
            catch (System.Exception e)
            {
                Debug.LogWarning("[Undercurrent] Vibrate failed: " + e.Message);
            }
#elif UNITY_IOS && !UNITY_EDITOR
            // iOS without a native plugin only exposes a single fixed vibration; reserve it for heavy moments.
            if (pattern == HapticPattern.KO || pattern == HapticPattern.Knockdown) Handheld.Vibrate();
#endif
        }

#if UNITY_ANDROID
        // Referencing Handheld.Vibrate makes Unity add the VIBRATE permission to the Android manifest.
        private static void EnsureVibratePermissionIsDeclared() => Handheld.Vibrate();
#endif
    }
}
