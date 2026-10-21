# Test Results

**Run date:** 2026-10-21 · **Environment:** Linux container, .NET SDK 8.0.132, NUnit 3.14 · **Unity Editor:** not available.

## 1. Core gameplay + character tests — `dotnet test Tools/CoreTests` → **76 passed, 0 failed**

These are the same source files Unity's Test Runner executes (`Assets/Undercurrent/Tests/EditMode`), compiled against the
same Core and CharacterGen sources. They have **not** yet been run inside Unity. Latest run: 2026-10-21 (character update).

```
Passed AI_DoesNotReadOpponentInput
Passed AI_UsesDefense_AttacksAndCombos
Passed AllDifficulties_AndFighters_CompleteMatches
Passed AntiAir_KnocksDownAirborneOpponent
Passed Attack_OutOfRange_Whiffs_WithNoDamage
Passed Attacks_RequireStamina_PreventingSpam
Passed Blocking_ReducesDamageToChip_AndDrainsStamina
Passed BodiesAreLeftRightSymmetric
Passed BossVictory_UnlocksHaruto_AndCompletesChapter
Passed Boss_EntersSecondPhase_BelowHalfHealth
Passed Campaign_Levels_AreCompleteAndChained
Passed CancelChain_ProducesComboWithDamageScaling
Passed Cancel_IsRejectedWhenMoveWhiffs
Passed ChipDamage_NeverKnocksOut
Passed ComboCap_ForcesKnockdown
Passed CorruptSave_IsBackedUp_AndReplaced
Passed Cosmetics_PurchaseAndEquip
Passed CounterHit_DealsBonusDamage
Passed Defeat_GivesPartialXp_NoCurrency_NoCompletion
Passed Dodge_InvulnerabilityEvadesAttack
Passed EulerConvention_MatchesUnity
Passed EveryLevel_PlaysToACompleteMatch
Passed EveryMove_HasAPose
Passed EveryTutorialStep_IsCompletableInTheLiveSimulation
Passed FighterUpgrade_CostsCurrency_AndIsCapped
Passed FrameAdvantage_MatchesSimulation
Passed GeneratedBodies_AreValidSkinnedMeshes
Passed Grounding_LeavesKickingLegLifted
Passed Grounding_PlantsFeetInEveryGroundedPose
Passed GuardBreak_WhenBlockingWithoutStamina
Passed HighAttack_WhiffsOverCrouchingOpponent
Passed HigherDifficulty_WinsMoreOften
Passed HitStop_FreezesBothFighters
Passed InputsDuringHitStop_AreBufferedForCancels
Passed Inputs_AreIgnoredDuringIntro
Passed Jab_HitsOnFirstActiveFrame_AndNotBefore
Passed Jump_LeavesGround_AndLands
Passed JumpingAttack_IsOverhead_AndBeatsCrouchBlock
Passed KO_EndsRound_AndBestOfThreeEndsMatch
Passed LightAttacks_ArePlusOnHit_HeavyKicksAreUnsafeOnBlock
Passed Loadout_AppliesLevelAndSkillBonuses_Modestly
Passed LowAttack_BeatsStandingBlock_ButNotCrouchingBlock
Passed MatchSummary_IsBuiltFromSimulation
Passed Move_HitsOnlyOncePerExecution
Passed NewSave_HasStartingRosterAndFirstLevelOnly
Passed NewerVersionSave_IsNeverOverwritten
Passed ObjectiveRules_AreEvaluatedCorrectly
Passed Objectives_AreAwardedOnce
Passed Observation_IsDelayedByReactionTime
Passed OnlineServices_ReportNotConfigured_AndNeverFakeSuccess
Passed Parry_NegatesDamage_AndStunsAttacker
Passed PlayableFighters_HaveDistinctIdentities
Passed QuickFight_GivesSmallRewards_AndCountsStats
Passed ReactionTimes_StayWithinHumanRange
Passed Replay_GivesReducedCurrency_NoDuplicateUnlocks
Passed ResetProgress_KeepsSettings
Passed RestPoseSkinning_IsIdentity
Passed Roster_FrameData_IsValid
Passed Round_StartsAfterIntro_WithBothFightersIdle
Passed SaveAndLoad_RoundTrips
Passed Separation_IsCappedByCameraLimit
Passed Simulation_IsDeterministic
Passed Special_RequiresFullFocus_AndIsInvulnerableOnStartup
Passed StageBoundaries_AndPushboxes_AreRespected
Passed Stamina_RegeneratesAfterDelay
Passed SurfaceNormals_PointOutwards
Passed Sweep_KnocksDown_AndDownedFighterIsInvulnerable
Passed Throw_BeatsBlock_AndCausesKnockdown
Passed Throw_CanBeBrokenDuringTechWindow
Passed Throw_CannotGrabOpponentInHitStun
Passed Timeout_AwardsRoundToHealthier
Passed UnversionedSave_IsMigrated_WithBackup
Passed Victory_FirstClear_GrantsRewards_AndUnlocksNextLevel
Passed Walking_MovesForwardAndBack
Passed WhiffedParry_HasRecovery_ThatCanBeCounterHit
Passed Xp_LevelsUpPlayer
Total tests: 76 — Passed: 76
```

The explicit diagnostic `BalanceReport.PrintAIMatchReport` is excluded from the gate and run on demand.

### What the tests cover (acceptance-criteria mapping)

| # | Criterion | Evidence |
|---|---|---|
| 6 | Movement works | `Walking_MovesForwardAndBack`, `Jump_LeavesGround_AndLands`, `StageBoundaries_AndPushboxes_AreRespected`, `Separation_IsCappedByCameraLimit` |
| 7 | Attacks trigger correct hit detection | `Jab_HitsOnFirstActiveFrame_AndNotBefore`, `Attack_OutOfRange_Whiffs_WithNoDamage`, `Move_HitsOnlyOncePerExecution`, `HighAttack_WhiffsOverCrouchingOpponent`, `FrameAdvantage_MatchesSimulation` |
| 8 | Blocking reduces/prevents damage | `Blocking_ReducesDamageToChip_AndDrainsStamina`, `LowAttack_BeatsStandingBlock_ButNotCrouchingBlock`, `JumpingAttack_IsOverhead_AndBeatsCrouchBlock`, `ChipDamage_NeverKnocksOut`, `GuardBreak_WhenBlockingWithoutStamina` |
| 9 | Dodging and parrying work | `Dodge_InvulnerabilityEvadesAttack`, `Parry_NegatesDamage_AndStunsAttacker`, `WhiffedParry_HasRecovery_ThatCanBeCounterHit` |
| 10 | Enemy AI behaves correctly | `AllDifficulties_AndFighters_CompleteMatches`, `HigherDifficulty_WinsMoreOften` (Expert 16 – Beginner 0), `AI_UsesDefense_AttacksAndCombos`, `AI_DoesNotReadOpponentInput`, `Observation_IsDelayedByReactionTime`, `Boss_EntersSecondPhase_BelowHalfHealth`, `ReactionTimes_StayWithinHumanRange` |
| 11 | Health and stamina update correctly | `Attacks_RequireStamina_PreventingSpam`, `Stamina_RegeneratesAfterDelay`, damage assertions above |
| 12–13 | Rounds end; match victory/defeat | `KO_EndsRound_AndBestOfThreeEndsMatch`, `Timeout_AwardsRoundToHealthier`, `EveryLevel_PlaysToACompleteMatch`, `MatchSummary_IsBuiltFromSimulation` |
| 14 | Rewards calculated correctly | `Victory_FirstClear_*`, `Replay_*`, `Defeat_*`, `Objectives_AreAwardedOnce`, `ObjectiveRules_*`, `QuickFight_*`, `Xp_LevelsUpPlayer` |
| 15 | Progress saved and restored | `SaveAndLoad_RoundTrips`, `CorruptSave_IsBackedUp_AndReplaced`, `NewerVersionSave_IsNeverOverwritten`, `UnversionedSave_IsMigrated_WithBackup`, `ResetProgress_KeepsSettings` |
| 16 | Character unlocks work | `BossVictory_UnlocksHaruto_AndCompletesChapter`, `NewSave_HasStartingRosterAndFirstLevelOnly` |
| — | Combat system rules | throws/techs, cancels & combo scaling, counter-hits, hit-stop freeze & buffering, combo cap, special invulnerability, determinism (`Simulation_IsDeterministic`, 3 seeds × 4000 ticks) |
| — | Tutorial | `EveryTutorialStep_IsCompletableInTheLiveSimulation` (scripted player completes all 15 steps) |
| — | Procedural humans | `GeneratedBodies_AreValidSkinnedMeshes` (all 6 fighters: indices, weights, normals, height, feet on floor, triangle budget), `SurfaceNormals_PointOutwards`, `BodiesAreLeftRightSymmetric`, `RestPoseSkinning_IsIdentity`, `EulerConvention_MatchesUnity`, `Grounding_PlantsFeetInEveryGroundedPose`, `Grounding_LeavesKickingLegLifted`, `EveryMove_HasAPose` |
| — | Content integrity | `Roster_FrameData_IsValid`, `Campaign_Levels_AreCompleteAndChained`, `PlayableFighters_HaveDistinctIdentities`, `LightAttacks_ArePlusOnHit_HeavyKicksAreUnsafeOnBlock` |

**Not covered by any executed test:** criteria 1–5, 17–20 (application launch, menus, fighter animation, settings UI,
crash-free device runs, device performance, asset licences in a store build) — they need a Unity Editor / device.

## 2. AI-vs-AI balance diagnostic (simulation only)

`BalanceReport.PrintAIMatchReport`: every playable pairing × 3 seeds per difficulty, same difficulty on both sides.
These numbers describe simulated AI play, **not** human play, performance, or fun.

```
Easy    matches=27 rounds=62 timeouts=0 avgRoundSec=31.1 hits=1639 blocks=287 parries=6 evades=2 throws=100 specials=56 knockdowns=229 maxCombo=3
Normal  matches=27 rounds=69 timeouts=0 avgRoundSec=28.4 hits=1724 blocks=490 parries=9 evades=5 throws=144 specials=69 knockdowns=330 maxCombo=4
Hard    matches=27 rounds=68 timeouts=0 avgRoundSec=28.5 hits=1748 blocks=630 parries=6 evades=7 throws=132 specials=64 knockdowns=361 maxCombo=4
Expert  matches=27 rounds=63 timeouts=0 avgRoundSec=27.0 hits=1536 blocks=569 parries=3 evades=4 throws=131 specials=70 knockdowns=383 maxCombo=3
```
Average round length ≈ 27–31 s of fight time; no round reached the 60 s timeout.

## 3. Unity-layer type-check (non-authoritative) — `Tools/UnityCompileCheck` → **0 errors, 0 warnings** in 4 configurations

The Runtime, Editor and Test scripts were compiled against hand-written stubs of the Unity APIs they use
(`Tools/UnityCompileCheck/UnityApiStubs.cs`) with `-warnaserror`, for define sets:
`UC_INPUT_SYSTEM;ENABLE_INPUT_SYSTEM;UC_URP` · `ENABLE_LEGACY_INPUT_MANAGER` · `UNITY_ANDROID;ENABLE_LEGACY_INPUT_MANAGER;UC_URP` ·
`UNITY_IOS;UC_INPUT_SYSTEM;ENABLE_INPUT_SYSTEM;ENABLE_LEGACY_INPUT_MANAGER`.

This proves our scripts are internally consistent. It does **not** prove they compile against real Unity 6: the stubs
were written from API knowledge, not from Unity's assemblies. First real compilation must happen in the Editor.

## 4. Offline character renders

`Docs/preview/*.png` were rendered with `Tools/CharacterPreview` (+ `render.py`, a numpy software rasteriser) from the
same generator, pose library and IK code the game uses. They show geometry and poses only — not Unity/URP lighting,
shadows or materials. Measured foot-planting IK cost: ≈74 µs per character per frame (desktop .NET 8, Release).

## 5. Not executed

| Item | Reason |
|---|---|
| Unity EditMode tests inside Unity | No Unity Editor (download hosts blocked by network policy) |
| Play Mode / UI / visual verification | No Unity Editor |
| Android APK / AAB build | No Unity Editor, Android SDK/NDK or signing keys |
| Device performance, FPS, thermals, battery | No devices; **no performance numbers are claimed** |
| Haptics on hardware | No devices |
