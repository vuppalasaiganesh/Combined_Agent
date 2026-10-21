# Test Results

**Run date:** 2026-10-21 · **Environment:** Linux container, .NET SDK 8.0.132, NUnit 3.14 · **Unity Editor:** not available.

## 1. Core gameplay + character tests — `dotnet test Tools/CoreTests` → **112 passed, 0 failed**

These are the same source files Unity's Test Runner executes (`Assets/Undercurrent/Tests/EditMode`), compiled against the
same Core and CharacterGen sources. They have **not** yet been run inside Unity. Latest run: 2026-10-21 (shop, tiers and power update).

```
Passed AISkill_FollowsTier_AndBossesUseBossAI
Passed AI_DoesNotReadOpponentInput
Passed AI_UsesDefense_AttacksAndCombos
Passed AllDifficulties_AndFighters_CompleteMatches
Passed AllPlayableFighters_AndWeapons_AreUnlockable
Passed AntiAir_KnocksDownAirborneOpponent
Passed ArmedAI_MatchesComplete_AndUseWeaponMechanics
Passed Armour_ReducesDamageTaken_InTheSimulation
Passed Attack_OutOfRange_Whiffs_WithNoDamage
Passed Attacks_RequireStamina_PreventingSpam
Passed Blades_ChipThroughBareHandedBlock_MoreThanFists
Passed Blocking_ReducesDamageToChip_AndDrainsStamina
Passed BodiesAreLeftRightSymmetric
Passed BossVictory_UnlocksHaruto_AndCompletesChapter
Passed Boss_EntersSecondPhase_BelowHalfHealth
Passed BuildOpponent_AppliesStatScale_AndWeapon
Passed Campaign_Has30Levels_InSixPlayableChapters
Passed Campaign_Levels_AreCompleteAndChained
Passed CancelChain_ProducesComboWithDamageScaling
Passed Cancel_IsRejectedWhenMoveWhiffs
Passed ChipDamage_NeverKnocksOut
Passed CoinPacks_CreditCoins_OnlyForKnownProducts
Passed ComboCap_ForcesKnockdown
Passed CorruptSave_IsBackedUp_AndReplaced
Passed Cosmetics_PurchaseAndEquip
Passed CounterHit_DealsBonusDamage
Passed Defeat_GivesPartialXp_SmallConsolation_NoCompletion
Passed Difficulty_RisesAcrossChapters
Passed Dodge_InvulnerabilityEvadesAttack
Passed EulerConvention_MatchesUnity
Passed EveryLevel_PlaysThroughAllBouts_WithWeapons
Passed EveryLevel_PlaysToACompleteMatch
Passed EveryLevel_ReferencesValidContent
Passed EveryMove_HasAPose
Passed EveryTutorialStep_IsCompletableInTheLiveSimulation
Passed FighterLevels_AddHealthAndDamage
Passed FighterUpgrade_CostsCurrency_AndIsCapped
Passed FrameAdvantage_MatchesSimulation
Passed FreePlayer_CanReachRecommendedPower_ByPlaying
Passed GauntletRun_CarriesHealth_AndRequiresEveryBout
Passed GeneratedBodies_AreValidSkinnedMeshes
Passed Grounding_LeavesKickingLegLifted
Passed Grounding_PlantsFeetInEveryGroundedPose
Passed GuardBreak_DisarmsWeaponUser_ThenWeaponReturns
Passed GuardBreak_WhenBlockingWithoutStamina
Passed HarderTiers_PayMore_AndScaleOpponentsMore
Passed HighAttack_WhiffsOverCrouchingOpponent
Passed HigherDifficulty_WinsMoreOften
Passed HigherPower_WinsMoreOften_InTheSimulation
Passed HitStop_FreezesBothFighters
Passed InputsDuringHitStop_AreBufferedForCancels
Passed Inputs_AreIgnoredDuringIntro
Passed Jab_HitsOnFirstActiveFrame_AndNotBefore
Passed Jump_LeavesGround_AndLands
Passed JumpingAttack_IsOverhead_AndBeatsCrouchBlock
Passed KO_EndsRound_AndBestOfThreeEndsMatch
Passed KnifeCounterHit_DoesExtraDamage
Passed LightAttacks_ArePlusOnHit_HeavyKicksAreUnsafeOnBlock
Passed Loadout_AppliesLevelAndSkillBonuses_Modestly
Passed Loadout_ReplacesPunches_KeepsKicksAndThrow
Passed LowAttack_BeatsStandingBlock_ButNotCrouchingBlock
Passed MatchSummary_IsBuiltFromSimulation
Passed Modifiers_ScaleDamage_HealthAndStamina
Passed Move_HitsOnlyOncePerExecution
Passed NewObjectives_ClashAndDisarm
Passed NewSave_HasStartingRosterAndFirstLevelOnly
Passed NewerVersionSave_IsNeverOverwritten
Passed ObjectiveRules_AreEvaluatedCorrectly
Passed Objectives_AreAwardedOnce
Passed Observation_IsDelayedByReactionTime
Passed OnlineServices_ReportNotConfigured_AndNeverFakeSuccess
Passed Parry_NegatesDamage_AndStunsAttacker
Passed PlayableFighters_HaveDistinctIdentities
Passed Prices_RiseWithWeaponReach
Passed QuickFight_GivesSmallRewards_AndCountsStats
Passed ReactionTimes_StayWithinHumanRange
Passed RecommendedPower_UsesTheSameFormulaAsPlayerPower
Passed Replay_GivesReducedCurrency_NoDuplicateUnlocks
Passed ResetProgress_KeepsSettings
Passed RestPoseSkinning_IsIdentity
Passed Roster_FrameData_IsValid
Passed Round_StartsAfterIntro_WithBothFightersIdle
Passed SaveAndLoad_RoundTrips
Passed Separation_IsCappedByCameraLimit
Passed Simulation_IsDeterministic
Passed SimultaneousWeaponStrikes_Clash
Passed Special_RequiresFullFocus_AndIsInvulnerableOnStartup
Passed StaffPoke_HitsFromRangeWhereJabWhiffs
Passed StageBoundaries_AndPushboxes_AreRespected
Passed Stamina_RegeneratesAfterDelay
Passed Stars_CountWinPlusObjectives
Passed SurfaceNormals_PointOutwards
Passed Sweep_KnocksDown_AndDownedFighterIsInvulnerable
Passed Throw_BeatsBlock_AndCausesKnockdown
Passed Throw_CanBeBrokenDuringTechWindow
Passed Throw_CannotGrabOpponentInHitStun
Passed Throw_DisarmsWeaponUser
Passed Tiers_AreMixed_InEveryChapter
Passed Timeout_AwardsRoundToHealthier
Passed UnversionedSave_IsMigrated_WithBackup
Passed V1Save_MigratesToV2_WithStars
Passed V2Save_KeepsWonWeapons_AsLevelOne
Passed Versus_PaysMore_AgainstHarderAI
Passed Victory_FirstClear_GrantsRewards_AndUnlocksNextLevel
Passed Walking_MovesForwardAndBack
Passed WeaponIsRestoredAtNextRound
Passed WeaponPurchase_Equip_AndLoadoutRespectsRules
Passed WeaponTiers_TradeReachForSpeed
Passed WeaponUpgrades_RaiseStrikeDamage_UpToMax
Passed Weapons_AreSold_OnlyAfterTheirLevel_AndCostCoins
Passed WhiffedParry_HasRecovery_ThatCanBeCounterHit
Passed Xp_LevelsUpPlayer
Total tests: 112 — Passed: 112
```

The explicit diagnostics `BalanceReport.PrintAIMatchReport`, `PrintWeaponMatchupReport` and `EconomyTests.PrintEconomyReport` are excluded from the gate and run on demand.

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
| — | Weapons | `WeaponTiers_TradeReachForSpeed`, `Throw_DisarmsWeaponUser`, `WeaponIsRestoredAtNextRound`, clash/disarm/rearm, knife counter-hit bonus, loadout validity for every fighter × weapon (see `WeaponAndCampaignTests.cs`) |
| — | 30-level campaign | `EveryLevel_PlaysThroughAllBouts_WithWeapons` (every level incl. gauntlets and modifiers played AI-vs-AI to completion), unlock chain, ids, rewards, stars, gauntlet health carry-over, save v1→v2 migration |
| — | Economy, tiers, power | `FreePlayer_CanReachRecommendedPower_ByPlaying` (full-campaign economy simulation, below), `HigherPower_WinsMoreOften_InTheSimulation`, `Armour_ReducesDamageTaken_InTheSimulation`, `WeaponUpgrades_RaiseStrikeDamage_UpToMax`, `Weapons_AreSold_OnlyAfterTheirLevel_AndCostCoins`, `CoinPacks_CreditCoins_OnlyForKnownProducts`, `Tiers_AreMixed_InEveryChapter`, `AISkill_FollowsTier_AndBossesUseBossAI`, `V2Save_KeepsWonWeapons_AsLevelOne` |
| — | Content integrity | `Roster_FrameData_IsValid`, `Campaign_Levels_AreCompleteAndChained`, `PlayableFighters_HaveDistinctIdentities`, `LightAttacks_ArePlusOnHit_HeavyKicksAreUnsafeOnBlock` |

**Not covered by any executed test:** criteria 1–5, 17–20 (application launch, menus, fighter animation, settings UI,
crash-free device runs, device performance, asset licences in a store build) — they need a Unity Editor / device.

## 2. AI-vs-AI balance diagnostic (simulation only)

`BalanceReport.PrintAIMatchReport`: every playable pairing (7 × 7) × 3 seeds per difficulty, same difficulty on both sides.
These numbers describe simulated AI play, **not** human play, performance, or fun.

```
Easy    matches=147 rounds=361 timeouts=0 avgRoundSec=34.4 hits=10004 blocks=1942 parries=28 evades=25 throws=732 specials=224 knockdowns=1464 maxCombo=4
Normal  matches=147 rounds=359 timeouts=0 avgRoundSec=30.3 hits=9391 blocks=2699 parries=44 evades=40 throws=860 specials=282 knockdowns=1894 maxCombo=4
Hard    matches=147 rounds=372 timeouts=0 avgRoundSec=28.3 hits=9513 blocks=3253 parries=34 evades=38 throws=867 specials=336 knockdowns=2152 maxCombo=4
Expert  matches=147 rounds=362 timeouts=0 avgRoundSec=26.9 hits=8904 blocks=3386 parries=29 evades=34 throws=743 specials=373 knockdowns=2147 maxCombo=5
```
Average round length ≈ 27–34 s of fight time; no round reached the 60 s timeout.

`BalanceReport.PrintWeaponMatchupReport`: identical bodies (Mara), Hard AI both sides, 20 seeds × both sides = 40 matches per cell.
Row weapon's win % against the column weapon:

```
            unarmed    knife    sword    staff
unarmed         40%      25%      23%      20%
knife           65%      45%      45%      55%
sword           83%      48%      53%      40%
staff           63%      38%      48%      48%
```
Mirror cells are not exactly 50 % (side/seed asymmetry and drawn matches count as non-wins). Reading: any weapon beats
unarmed (intended — armed levels compensate unarmed players with rules), and the three weapons form a loose cycle
(knife > staff, staff > sword, sword ≈ knife) rather than one dominant pick. AI-only data; re-tune after human playtests.

## 2b. Economy and power diagnostics (simulation only)

`EconomyTests.PrintEconomyReport` — an idealised **free** player (no coin packs) who plays the campaign in order with one
fighter, wins every level first try, completes one objective per level and spends greedily on the best power-per-coin
upgrade. `rec` = level's recommended power, `arrive` = player's power on arrival, `power` = after any replays needed.

```
level   tier     rec  arrive  power  replays  coins-after
c1_l1   Easy       85     100    100        0          350
c1_l2   Medium    101     104    104        0          580
c1_l3   Medium    103     108    108        0          950
c2_l1   Easy       88     112    112        0          800
c2_l2   Easy       97     112    112        0         1380
c2_l3   Hard      120     140    140        0         1600
c2_l4   Medium    144     148    148        0         1680
c2_l5   Hard      155     157    157        0         1820
c3_l1   Easy      118     163    163        0         1810
c3_l2   Medium    113     127    127        0         1760
c3_l3   Hard      129     132    132        0         1970
c3_l4   Medium    145     181    181        0         1620
c3_l5   Hard      165     181    181        0         3560
c4_l1   Easy      126     192    192        0         2360
c4_l2   Hard      169     198    198        0         2850
c4_l3   Medium    162     209    209        0         2210
c4_l4   Hard      139     150    150        0         4630
c4_l5   Medium    125     217    217        0         4420
c4_l6   Extreme   228     228    228        0         4700
c5_l1   Medium    160     242    242        0         2930
c5_l2   Hard      228     242    242        0         5810
c5_l3   Extreme   258     254    270        9         4260
c5_l4   Easy      141     270    270        0         6370
c5_l5   Hard      218     270    270        0         9620
c5_l6   Extreme   246     288    288        0         5180
c6_l1   Medium    173     288    288        0         7840
c6_l2   Hard      196     288    288        0        11450
c6_l3   Extreme   284     288    288        0        17230
c6_l4   Hard      201     288    288        0        21070
c6_l5   Extreme   260     288    288        0        27080
```
Reading: Easy levels leave a wide margin; Medium/Hard are close; Extreme levels sit at or above even this ideal player's
power (one needs 9 replays; c6_l3 needs 284 of the 288 maximum). Real players lose fights and miss objectives, so they will
need more replays, Versus wins or coin packs — that is the intended role of coins. The gating test asserts: every level is
reachable without paying, Easy/Medium never need grinding, no level needs more than 12 replays, ≤ 60 replays in total.

`EconomyTests.HigherPower_WinsMoreOften_InTheSimulation` (same body and Hard AI on both sides; opponent at late-Extreme
stats ×1.41): fresh player won **0/16**, fully-geared player (level 10 + top armour, unarmed) won **9/16**. Power is real in the
simulation, not just a label.

## 3. Unity-layer type-check (non-authoritative) — `Tools/UnityCompileCheck` → **0 errors, 0 warnings** in 4 configurations

The Runtime, Editor and Test scripts were compiled against hand-written stubs of the Unity APIs they use
(`Tools/UnityCompileCheck/UnityApiStubs.cs`) with `-warnaserror`, for define sets:
`UC_INPUT_SYSTEM;ENABLE_INPUT_SYSTEM;UC_URP` · `ENABLE_LEGACY_INPUT_MANAGER` · `UNITY_ANDROID;ENABLE_LEGACY_INPUT_MANAGER;UC_URP` ·
`UNITY_IOS;UC_INPUT_SYSTEM;ENABLE_INPUT_SYSTEM;ENABLE_LEGACY_INPUT_MANAGER`.

This proves our scripts are internally consistent. It does **not** prove they compile against real Unity 6: the stubs
were written from API knowledge, not from Unity's assemblies. First real compilation must happen in the Editor.

## 4. Offline character renders

`Docs/preview/*.png` (incl. `weapons.png`: Wren/knife, Brandt/sword, Lin/staff) were rendered with `Tools/CharacterPreview` (+ `render.py`, a numpy software rasteriser) from the
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
