# Progress

Status legend — **VERIFIED**: implemented and tested successfully · **IMPLEMENTED_UNVERIFIED**: code exists, testing remains ·
**NOT_IMPLEMENTED** · **BLOCKED**: needs a specific dependency (see `KNOWN_ISSUES.md`).
No percentage estimates are given. "VERIFIED" below means verified by the automated .NET test run in `TEST_RESULTS.md`,
not inside Unity or on a device.

## Acceptance criteria (master prompt §18)

| # | Criterion | Status |
|---|---|---|
| 1 | Application launches | BLOCKED (B1) — bootstrap code IMPLEMENTED_UNVERIFIED |
| 2 | Main menu loads | BLOCKED (B1) — IMPLEMENTED_UNVERIFIED |
| 3 | Player selects a fighter | Data/logic VERIFIED; UI IMPLEMENTED_UNVERIFIED |
| 4 | Player enters an arena | IMPLEMENTED_UNVERIFIED |
| 5 | Both fighters animate correctly | IMPLEMENTED_UNVERIFIED (placeholder procedural animation) |
| 6 | Movement works | VERIFIED (simulation) |
| 7 | Attacks trigger correct hit detection | VERIFIED (simulation) |
| 8 | Blocking reduces or prevents damage | VERIFIED |
| 9 | Dodging and parrying work | VERIFIED |
| 10 | Enemy AI behaves correctly | VERIFIED (simulation-level behaviour and fairness) |
| 11 | Health and stamina update correctly | VERIFIED |
| 12 | Rounds end correctly | VERIFIED |
| 13 | Match victory and defeat | VERIFIED (logic); results screen IMPLEMENTED_UNVERIFIED |
| 14 | Rewards calculated correctly | VERIFIED |
| 15 | Progress saved and restored | VERIFIED with test codec; Unity `JsonUtility`/file store IMPLEMENTED_UNVERIFIED |
| 16 | Character unlocks work | VERIFIED |
| 17 | Menus and settings work | IMPLEMENTED_UNVERIFIED |
| 18 | No critical crashes in supported tests | Core: no exceptions across all test matches. Unity: BLOCKED (B1) |
| 19 | Performance evaluated on real devices | BLOCKED (B4) |
| 20 | Required assets have verified licences | Shipped assets are original/engine (see manifest); final assets BLOCKED (B3) |

## Features by phase

### Phase A — Foundation
| Feature | Status |
|---|---|
| Project structure, asmdefs (Core engine-free / Runtime / Editor / Tests) | IMPLEMENTED_UNVERIFIED in Unity; Core VERIFIED via .NET |
| Automated test foundation (NUnit shared by Unity + .NET runner, `run_checks.sh`) | VERIFIED |
| Boot scene + build settings + player settings generator | IMPLEMENTED_UNVERIFIED |
| Input: touch, keyboard, gamepad; both input backends | IMPLEMENTED_UNVERIFIED (type-checked) |

### Phase B — Vertical slice
| Feature | Status |
|---|---|
| Frame-aware combat engine (startup/active/recovery, hit/hurtboxes, damage, stamina, knockback, cancels, block, counter, interruptibility via armor) | VERIFIED |
| Hit-stun, block-stun, guard break, knockdown, get-up, throws + breaks, parry, dodge i-frames, combo scaling & cap, hit-stop | VERIFIED |
| Character collision, stage bounds, camera separation limit | VERIFIED |
| Round timer, KO/timeout, best-of-N, match result, restart | Logic VERIFIED; restart UI IMPLEMENTED_UNVERIFIED |
| Enemy AI (6 difficulties, reaction-delayed perception, combos, punishes, anti-air, throw tech, stamina mgmt, boss phase 2) | VERIFIED (simulation) |
| HUD (health+trail, stamina, focus, timer, round pips, announcer, combo, callouts) | IMPLEMENTED_UNVERIFIED |

### Phase C — Combat quality
| Feature | Status |
|---|---|
| Final realistic character models & mocap animation | BLOCKED (B3) — integration pipeline (Animator presenter, controller generator) IMPLEMENTED_UNVERIFIED |
| Procedural human characters (skinned mesh, face, hair, beard, clothing) | Generator VERIFIED (mesh validity tests + offline renders in `Docs/preview`); Unity rendering IMPLEMENTED_UNVERIFIED |
| Frame-synced procedural poses for all states/moves + foot-planting IK | Pose/IK maths VERIFIED (tests); in-engine IMPLEMENTED_UNVERIFIED |
| Hit reactions, slow-mo final KO, special-move camera push-in, screen shake (settings-scaled) | IMPLEMENTED_UNVERIFIED |
| Audio pipeline (music crossfade, ambience, pooled SFX, voice channel, licensed-override lookup) + procedural placeholder library | IMPLEMENTED_UNVERIFIED |
| Haptics (Android VibrationEffect with amplitude, fallbacks, intensity setting) | IMPLEMENTED_UNVERIFIED (B4) |
| Pooled VFX (sparks, block, parry, dust) scaled by graphics tier | IMPLEMENTED_UNVERIFIED |
| Character selection with stats/strengths/weaknesses | IMPLEMENTED_UNVERIFIED |
| 3 playable fighters with distinct styles, stats, timing, combos; 3 additional opponents | VERIFIED (data & simulation) |
| 8+ fighter roster | NOT_IMPLEMENTED (6 defined) |

### Phase D — Campaign
| Feature | Status |
|---|---|
| Chapter 1: 3 complete levels (opponent, arena, difficulty, rounds, rewards, objectives, unlocks, story) | VERIFIED (data, unlock chain, each level plays to completion AI-vs-AI) |
| Campaign / briefing / results screens | IMPLEMENTED_UNVERIFIED |
| Chapters 2–6 (27 more levels) | NOT_IMPLEMENTED (chapters listed as "not built") |
| Guided tutorial covering every core action, auto-run before level 1 | Logic VERIFIED; overlay IMPLEMENTED_UNVERIFIED |

### Phase E — Product readiness
| Feature | Status |
|---|---|
| Upgrades (fighter levels), skills, cosmetics, store (earned currency), inventory, achievements | Logic VERIFIED; screens IMPLEMENTED_UNVERIFIED |
| Settings (audio ×3, haptics, shake, reduce flashes, quality tier, 30/60 FPS, control size/opacity/layout) | IMPLEMENTED_UNVERIFIED |
| Save versioning, migration, corruption backup, newer-version protection, atomic file writes | Logic VERIFIED; file store IMPLEMENTED_UNVERIFIED |
| Crash handling | Local error log IMPLEMENTED_UNVERIFIED; crash reporting service NOT_IMPLEMENTED |
| Optional services (billing, cloud save, leaderboards, accounts) | Interfaces + NotConfigured implementations VERIFIED; real integrations BLOCKED (B6) |
| Graphics tiers Low/Medium/High/Ultra | IMPLEMENTED_UNVERIFIED |
| Accessibility (shake/haptics/flash settings, remappable layout) | IMPLEMENTED_UNVERIFIED; colour-blind & text-size options NOT_IMPLEMENTED |
| Localisation | NOT_IMPLEMENTED |
| PvP | NOT_IMPLEMENTED (deliberately, per prompt) |

### Phase F — Release
| Feature | Status |
|---|---|
| Debug APK / Release AAB build scripts | IMPLEMENTED_UNVERIFIED |
| Actual APK / AAB artefacts, build logs, performance reports | BLOCKED (B1, B2, B4) |

## Next implementation activities
1. Open in Unity 6 (README), fix any real compile errors, run EditMode tests in the Test Runner.
2. Play-test the slice on device: touch ergonomics, readability of placeholder animation, hit feedback timing; profile on a low-end device.
3. Add Play Mode smoke tests (boot → menu → fight → results).
4. Acquire/commission licensed character models + animations; integrate through the Animator presenter.
5. Build Chapter 2 (levels 4–8) with two new opponents and arena, after slice validation.
6. Title clearance and store configuration.
