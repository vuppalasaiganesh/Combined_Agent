using System.IO;
using UnityEditor;
using UnityEditor.Build;
using UnityEditor.SceneManagement;
using UnityEngine;
using UnityEngine.Rendering;
using UnityEngine.SceneManagement;

namespace Undercurrent.EditorTools
{
    /// <summary>
    /// One-click project configuration. Safe to run repeatedly. Deliberately depends only on UnityEditor APIs (not on
    /// the runtime assembly) so it still compiles and runs if a runtime package is missing.
    /// </summary>
    public static class ProjectSetup
    {
        public const string Root = "Assets/Undercurrent";
        public const string MaterialsFolder = Root + "/Resources/Undercurrent/Materials";
        public const string ScenesFolder = Root + "/Scenes";
        public const string BootScene = ScenesFolder + "/Boot.unity";
        public const string ApplicationId = "com.undercurrentdev.placeholder"; // replace before any store submission

        [MenuItem("Undercurrent/Setup Project", priority = 0)]
        public static void SetupAll()
        {
            CheckRenderPipeline();
            CreateBaseMaterials();
            CreateBootScene();
            ConfigurePlayerSettings();
            AssetDatabase.SaveAssets();
            Debug.Log("[Undercurrent] Project setup complete. Open Assets/Undercurrent/Scenes/Boot.unity and press Play.");
        }

        private static void CheckRenderPipeline()
        {
            if (GraphicsSettings.defaultRenderPipeline == null)
                Debug.LogWarning("[Undercurrent] No render pipeline asset is assigned. Create the project from Unity's URP 3D template " +
                                 "(or assign a Universal Render Pipeline asset in Project Settings > Graphics). The game falls back to built-in shaders.");
        }

        private static void EnsureFolder(string path)
        {
            if (AssetDatabase.IsValidFolder(path)) return;
            var parent = Path.GetDirectoryName(path)?.Replace('\\', '/');
            if (!string.IsNullOrEmpty(parent)) EnsureFolder(parent);
            AssetDatabase.CreateFolder(parent, Path.GetFileName(path));
        }

        private static Shader FirstShader(params string[] names)
        {
            foreach (var n in names)
            {
                var s = Shader.Find(n);
                if (s != null) return s;
            }
            return null;
        }

        /// <summary>
        /// Materials in a Resources folder are always included in player builds, which also guarantees their shaders
        /// are included — runtime-created materials then clone these instead of relying on Shader.Find.
        /// </summary>
        [MenuItem("Undercurrent/Setup/Create Base Materials")]
        public static void CreateBaseMaterials()
        {
            EnsureFolder(MaterialsFolder);
            CreateMaterial("BaseLit", FirstShader("Universal Render Pipeline/Lit", "Standard"), null);
            CreateMaterial("BaseUnlit", FirstShader("Universal Render Pipeline/Unlit", "Unlit/Color"), null);
            CreateMaterial("BaseParticle", FirstShader("Universal Render Pipeline/Particles/Unlit", "Particles/Standard Unlit", "Sprites/Default"), m =>
            {
                // Additive blending for sparks.
                if (m.HasProperty("_Surface")) m.SetFloat("_Surface", 1f);
                if (m.HasProperty("_Blend")) m.SetFloat("_Blend", 2f);
                if (m.HasProperty("_SrcBlend")) m.SetFloat("_SrcBlend", (float)BlendMode.SrcAlpha);
                if (m.HasProperty("_DstBlend")) m.SetFloat("_DstBlend", (float)BlendMode.One);
                if (m.HasProperty("_ZWrite")) m.SetFloat("_ZWrite", 0f);
                m.EnableKeyword("_SURFACE_TYPE_TRANSPARENT");
                m.renderQueue = (int)RenderQueue.Transparent;
            });
        }

        private static void CreateMaterial(string name, Shader shader, System.Action<Material> configure)
        {
            var path = $"{MaterialsFolder}/{name}.mat";
            if (shader == null)
            {
                Debug.LogError($"[Undercurrent] No suitable shader found for {name}.");
                return;
            }
            var existing = AssetDatabase.LoadAssetAtPath<Material>(path);
            if (existing != null && existing.shader == shader) return;
            if (existing != null) AssetDatabase.DeleteAsset(path);
            var mat = new Material(shader) { name = name };
            configure?.Invoke(mat);
            AssetDatabase.CreateAsset(mat, path);
            Debug.Log($"[Undercurrent] Created {path} using '{shader.name}'.");
        }

        [MenuItem("Undercurrent/Setup/Create Boot Scene")]
        public static void CreateBootScene()
        {
            EnsureFolder(ScenesFolder);
            if (!File.Exists(BootScene))
            {
                if (!EditorSceneManager.SaveCurrentModifiedScenesIfUserWantsTo()) return;
                // The game builds itself at runtime (GameApp boots after scene load), so the scene stays empty.
                var scene = EditorSceneManager.NewScene(NewSceneSetup.EmptyScene, NewSceneMode.Single);
                EditorSceneManager.SaveScene(scene, BootScene);
                Debug.Log("[Undercurrent] Created " + BootScene);
            }
            EditorBuildSettings.scenes = new[] { new EditorBuildSettingsScene(BootScene, true) };
        }

        [MenuItem("Undercurrent/Setup/Configure Player Settings")]
        public static void ConfigurePlayerSettings()
        {
            PlayerSettings.companyName = "Undercurrent Dev (placeholder)";
            PlayerSettings.productName = "Undercurrent (working title)";
            PlayerSettings.bundleVersion = "0.1.0";
            PlayerSettings.defaultInterfaceOrientation = UIOrientation.AutoRotation;
            PlayerSettings.allowedAutorotateToPortrait = false;
            PlayerSettings.allowedAutorotateToPortraitUpsideDown = false;
            PlayerSettings.allowedAutorotateToLandscapeLeft = true;
            PlayerSettings.allowedAutorotateToLandscapeRight = true;
            PlayerSettings.colorSpace = ColorSpace.Linear;

            PlayerSettings.SetApplicationIdentifier(NamedBuildTarget.Android, ApplicationId);
            PlayerSettings.SetApplicationIdentifier(NamedBuildTarget.iOS, ApplicationId);
            PlayerSettings.SetScriptingBackend(NamedBuildTarget.Android, ScriptingImplementation.IL2CPP);
            PlayerSettings.Android.targetArchitectures = AndroidArchitecture.ARM64;
            PlayerSettings.Android.bundleVersionCode = Mathf.Max(1, PlayerSettings.Android.bundleVersionCode);
            EditorUserBuildSettings.androidBuildSubtarget = MobileTextureSubtarget.ASTC;
            Debug.Log("[Undercurrent] Player settings configured (landscape, IL2CPP, ARM64, ASTC). Application id is a placeholder: " + ApplicationId);
        }

        [MenuItem("Undercurrent/Open Boot Scene", priority = 1)]
        public static void OpenBoot()
        {
            if (!File.Exists(BootScene)) CreateBootScene();
            if (EditorSceneManager.SaveCurrentModifiedScenesIfUserWantsTo()) EditorSceneManager.OpenScene(BootScene);
        }

        [MenuItem("Undercurrent/Open Save Folder")]
        public static void OpenSaveFolder() => EditorUtility.RevealInFinder(Application.persistentDataPath);
    }
}
