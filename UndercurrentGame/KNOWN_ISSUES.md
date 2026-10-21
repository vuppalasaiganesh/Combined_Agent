# Known Issues, Blockers and Risks

## Blockers (cannot proceed in the current environment)

| ID | Blocker | Impact | Needed to unblock |
|---|---|---|---|
| B1 | **No Unity Editor.** Network policy rejects `download.unity3d.com`, `unity.com`, `packages.unity.com`. | Unity layer never compiled for real; no Play Mode, UI, rendering or Unity Test Runner verification. | A machine with Unity 6 LTS; follow README setup and report console errors. |
| B2 | **No Android build toolchain or signing keys.** | No APK/AAB exists. Nothing in this repo should be described as a build. | Unity Android module (SDK/NDK/JDK); release keystore supplied via env vars. |
| B3 | **No realistic character/environment/audio assets.** | Game uses code-generated placeholder mannequins, greybox arenas and synthesised audio. Not commercial quality. | Licensed or commissioned assets (see `ASSET_LICENSES.md` § MISSING). |
| B4 | **No test devices.** | No FPS, frame-pacing, thermal, battery, haptics or touch-ergonomics data. | Low/mid/high-tier Android devices + Unity Profiler / Android GPU Inspector. |
| B5 | **Title not cleared;** a conflicting indie title "Undercurrent" exists. | Cannot publish under this name. | Trademark clearance; new title. |
| B6 | **No store/back-end configuration.** | Billing, cloud save, leaderboards, accounts are explicit NotConfigured stubs. | Play Console products, Play Billing Library integration, server-side receipt verification, backend. |

## Unverified assumptions in the Unity layer (check first when opening in Unity)

1. `InputSystemUIInputModule.AssignDefaultActions()` gives touch/mouse UI input when the module is added at runtime.
2. Referencing a missing optional assembly (`Unity.InputSystem`, URP) by name in `Undercurrent.Runtime.asmdef` is tolerated; `versionDefines` guard all usages.
3. URP Lit/Unlit/Particles material property names (`_BaseColor`, `_Surface`, `_SrcBlend`, `_EmissionColor`…) and manual keyword setup produce the intended look (transparent contact shadow, additive sparks, hit-flash emission).
4. Legacy `UnityEngine.UI.Text` with `LegacyRuntime.ttf` is available in the URP template's uGUI version.
5. Unity adds the Android `VIBRATE` permission because `Handheld.Vibrate` is referenced; `VibrationEffect.createOneShot` JNI call signature (long, int).
6. Mirroring fighters with negative X scale renders correctly (Unity flips winding for negative scale), including the SkinnedMeshRenderer.
8. The generated mesh's bind poses (`joint.worldToLocalMatrix * root.localToWorldMatrix`) and 2-bone weights render as in the offline preview.
9. The held weapon's grip offset/rotation on a **licensed** humanoid rig (`HumanBodyBones.RightHand`) matches the procedural rig; it is tuned only for the procedural skeleton.
7. Particle `EmitParams` with `applyShapeToPosition` emits around the given position with shape velocities.

## Gameplay / design issues

- Characters are procedurally generated humans (continuous skinned meshes) — clearly human but stylised: flat-coloured
  materials, no skin/cloth textures, no facial animation, no finger animation (fists are always wrapped), hair is a
  sculpted shell rather than strands. Photoreal fighters still need licensed/commissioned assets (B3).
- Animation is procedural key poses (not motion capture). Foot-planting IK now keeps feet on the floor in grounded states;
  some strikes (hooks, roundhouse) are approximations. No cloth simulation.
- Foot-planting IK costs ≈74 µs per character per frame on desktop .NET (measured); mobile cost unmeasured (B4).
- Fighters never cross sides (no cross-ups). Deliberate simplification for mobile readability.
- Only one parry window per fighter (no directional parries). Parry is mostly a read on slow moves at AI reaction speeds.
- AI-vs-AI throws are frequent (≈2 per round) because heavy blocking makes throws the correct answer — re-tune after human playtests.
- Campaign difficulty curve (30 levels) is tuned against AI-vs-AI runs only; real players may find gauntlets / Iron Body / Exhaustion levels too hard or too easy.
- Weapons are balanced from AI-vs-AI data only (`TEST_RESULTS.md` §2). Unarmed loses ≈75–80 % to any weapon by design; levels that arm only the opponent compensate with rules.
- Weapon strike poses are procedural approximations (no real sword/staff technique animation); weapons have no physics or trails.
- A disarmed fighter fights with fists for 5 s; the dropped weapon is visual only (cannot be picked up by either fighter).
- Arenas are greybox; level 1 and Training share one arena.
- "Voice" sounds are synthesised breaths, not recorded effort vocals.
- Optional objectives are evaluated only on victory (by design).
- Mirror-match colour swap applies to Versus only.

## Technical debt / follow-ups

- `.meta` files are not committed (no Editor to generate them); commit them after the first import.
- No Play Mode tests yet (need Unity). Add tests for: boot → main menu, start level, pause/quit, results flow.
- Menu screens rebuild their whole hierarchy on refresh — fine for menus, avoid in gameplay.
- `FightHud`/`TouchControls` update every frame without allocation except text assignment of the timer (one string per frame); consider caching digit strings.
- Crash handling is local logging only (`undercurrent_errors.log`); no crash-reporting service.
- No localisation system; all text is English.
- iOS haptics fall back to `Handheld.Vibrate` for heavy events only (needs a native Core Haptics plugin for parity).
