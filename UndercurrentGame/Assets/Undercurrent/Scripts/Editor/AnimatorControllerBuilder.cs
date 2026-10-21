using System.Collections.Generic;
using System.IO;
using System.Linq;
using Undercurrent.Core.Combat;
using Undercurrent.Core.Content;
using UnityEditor;
using UnityEditor.Animations;
using UnityEngine;

namespace Undercurrent.EditorTools
{
    /// <summary>
    /// Builds an Animator Controller for a licensed humanoid character from a folder of clips named after the states
    /// the runtime expects (e.g. "Idle", "WalkForward", "Attack_jab"). Reports every required state that has no clip.
    /// Runtime contract: see AnimatorFighterPresenter (attack clips are scrubbed to simulation frames).
    /// </summary>
    public static class AnimatorControllerBuilder
    {
        /// <summary>Every Animator state name the runtime can request for the given fighter.</summary>
        public static List<string> RequiredStates(string fighterId)
        {
            var names = new List<string>();
            foreach (FighterStateId s in System.Enum.GetValues(typeof(FighterStateId)))
                if (s != FighterStateId.Attack) names.Add(s.ToString());
            names.Add("Attack_Generic");
            var def = FighterRoster.Get(fighterId);
            if (def != null)
                foreach (var m in def.Moves)
                    if (!m.IsThrow) names.Add("Attack_" + m.AnimationId);
            return names.Distinct().ToList();
        }

        [MenuItem("Undercurrent/Characters/Build Animator Controller From Selected Clip Folder")]
        public static void BuildFromSelection()
        {
            var folder = AssetDatabase.GetAssetPath(Selection.activeObject);
            if (string.IsNullOrEmpty(folder) || !AssetDatabase.IsValidFolder(folder))
            {
                EditorUtility.DisplayDialog("Undercurrent", "Select a folder containing the fighter's animation clips first.\n" +
                                                            "Name the folder after the fighter id (mara, dima, haruto, ines, rook, sable).", "OK");
                return;
            }
            string fighterId = Path.GetFileName(folder).ToLowerInvariant();
            Build(folder, fighterId);
        }

        public static AnimatorController Build(string clipFolder, string fighterId)
        {
            var clips = new Dictionary<string, AnimationClip>();
            foreach (var guid in AssetDatabase.FindAssets("t:AnimationClip", new[] { clipFolder }))
            {
                var path = AssetDatabase.GUIDToAssetPath(guid);
                foreach (var asset in AssetDatabase.LoadAllAssetsAtPath(path))
                    if (asset is AnimationClip clip && !clip.name.StartsWith("__preview__")) clips[clip.name] = clip;
            }

            var outPath = $"{clipFolder}/{fighterId}_Fighter.controller";
            var controller = AnimatorController.CreateAnimatorControllerAtPath(outPath);
            controller.AddParameter("HitWeight", AnimatorControllerParameterType.Float);
            var sm = controller.layers[0].stateMachine;

            var missing = new List<string>();
            foreach (var state in RequiredStates(fighterId))
            {
                var s = sm.AddState(state);
                if (clips.TryGetValue(state, out var clip)) s.motion = clip;
                else missing.Add(state);
                if (state == "Idle") sm.defaultState = s;
            }
            AssetDatabase.SaveAssets();

            if (missing.Count > 0)
                Debug.LogWarning($"[Undercurrent] {outPath}: {missing.Count} states have no clip and will hold the bind pose:\n  " + string.Join("\n  ", missing));
            else
                Debug.Log($"[Undercurrent] {outPath}: all states mapped.");
            Debug.Log("[Undercurrent] Next: put the character prefab (Humanoid avatar + this controller) at " +
                      $"Assets/Undercurrent/Resources/Undercurrent/Characters/{fighterId}.prefab");
            return controller;
        }
    }
}
