using System;
using System.IO;
using UnityEditor;
using UnityEditor.Build.Reporting;
using UnityEngine;

namespace Undercurrent.EditorTools
{
    /// <summary>
    /// Android build entry points, usable from the menu or batch mode, e.g.:
    ///   Unity -batchmode -quit -projectPath &lt;path&gt; -executeMethod Undercurrent.EditorTools.BuildScript.BuildAndroidDebugApk -logFile build.log
    /// Release bundles require signing credentials supplied through environment variables (never committed):
    ///   UC_KEYSTORE_PATH, UC_KEYSTORE_PASS, UC_KEY_ALIAS, UC_KEY_PASS
    /// A build is only reported as successful when Unity's BuildReport says so and the output file exists.
    /// </summary>
    public static class BuildScript
    {
        private const string OutputDir = "Builds/Android";

        [MenuItem("Undercurrent/Build/Android Debug APK")]
        public static void BuildAndroidDebugApk()
        {
            EditorUserBuildSettings.buildAppBundle = false;
            PlayerSettings.Android.useCustomKeystore = false; // debug keystore
            Build(Path.Combine(OutputDir, "Undercurrent-debug.apk"), BuildOptions.Development);
        }

        [MenuItem("Undercurrent/Build/Android Release AAB")]
        public static void BuildAndroidReleaseAab()
        {
            string keystore = Environment.GetEnvironmentVariable("UC_KEYSTORE_PATH");
            string storePass = Environment.GetEnvironmentVariable("UC_KEYSTORE_PASS");
            string alias = Environment.GetEnvironmentVariable("UC_KEY_ALIAS");
            string aliasPass = Environment.GetEnvironmentVariable("UC_KEY_PASS");
            if (string.IsNullOrEmpty(keystore) || !File.Exists(keystore) || string.IsNullOrEmpty(storePass) ||
                string.IsNullOrEmpty(alias) || string.IsNullOrEmpty(aliasPass))
            {
                Fail("Release signing is not configured. Set UC_KEYSTORE_PATH, UC_KEYSTORE_PASS, UC_KEY_ALIAS and UC_KEY_PASS.");
                return;
            }
            PlayerSettings.Android.useCustomKeystore = true;
            PlayerSettings.Android.keystoreName = keystore;
            PlayerSettings.Android.keystorePass = storePass;
            PlayerSettings.Android.keyaliasName = alias;
            PlayerSettings.Android.keyaliasPass = aliasPass;
            EditorUserBuildSettings.buildAppBundle = true;
            try
            {
                Build(Path.Combine(OutputDir, "Undercurrent-release.aab"), BuildOptions.None);
            }
            finally
            {
                // Do not leave credentials in ProjectSettings.
                PlayerSettings.Android.keystorePass = string.Empty;
                PlayerSettings.Android.keyaliasPass = string.Empty;
                EditorUserBuildSettings.buildAppBundle = false;
            }
        }

        private static void Build(string path, BuildOptions options)
        {
            ProjectSetup.CreateBaseMaterials();
            ProjectSetup.CreateBootScene();
            Directory.CreateDirectory(Path.GetDirectoryName(path) ?? OutputDir);

            var scenes = Array.ConvertAll(EditorBuildSettings.scenes, s => s.path);
            var report = BuildPipeline.BuildPlayer(new BuildPlayerOptions
            {
                scenes = scenes,
                locationPathName = path,
                target = BuildTarget.Android,
                targetGroup = BuildTargetGroup.Android,
                options = options,
            });

            var summary = report.summary;
            bool ok = summary.result == BuildResult.Succeeded && File.Exists(path);
            string message = $"[Undercurrent] Build {summary.result}: {path} ({summary.totalSize / (1024 * 1024)} MB, {summary.totalErrors} errors, " +
                             $"{summary.totalWarnings} warnings, {summary.totalTime.TotalSeconds:0}s)";
            if (ok) Debug.Log(message);
            else Fail(message);
            if (Application.isBatchMode) EditorApplication.Exit(ok ? 0 : 1);
        }

        private static void Fail(string message)
        {
            Debug.LogError(message);
            if (Application.isBatchMode) EditorApplication.Exit(1);
        }
    }
}
