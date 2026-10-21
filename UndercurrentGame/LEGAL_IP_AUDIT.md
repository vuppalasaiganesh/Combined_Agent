# Preliminary IP Risk Review

**Date:** 2026-10-21 · **Prepared by:** development team (AI-assisted) · **Status:** PRELIMINARY — not legal advice.
Nothing in this document asserts that any name, character or asset is legally cleared. All items marked
**HUMAN REVIEW** require a qualified IP attorney before commercial release.

## 1. Scope
Covers the game concept, working title, characters, story, combat design, code, and all assets currently in the repository.

## 2. External checks performed

| Check | Method | Result | Status |
|---|---|---|---|
| Working title "Undercurrent" vs existing games | Two general web searches (2026-10-21) | Found an indie VR underwater adventure titled **"Undercurrent"** listed on IndieDB/ModDB (https://www.indiedb.com/games/undercurrent). No fighting game of that name surfaced. | **CONFLICT RISK — rename before release** |
| Trademark registers (USPTO, EUIPO, WIPO Global Brand DB, UKIPO, JPO, CNIPA) for Class 9 / 41 | Not performed (no direct register access) | — | **PENDING — HUMAN REVIEW** |
| App store name availability (Google Play, App Store) | Not performed | — | **PENDING** |
| Domain / social handle availability | Not performed | — | **PENDING** |
| Character names (Mara Okonkwo, Dmitri Vasko, Haruto Sen, Ines Dunmore, Bram "Rook" Calloway, Teodora "Sable" Vance) | Not searched | Names are generic personal names; nicknames are common words. Potential coincidental overlap with real people/athletes not checked. | **PENDING — HUMAN REVIEW** |
| Place/organisation names (Varrow Bay, Pier 9 Net-Loft, Calloway Freight Yard, The Old Cistern, "The Ledger") | Not searched | Believed fictional; "Pier 9" is a common real-world place name (generic). | **PENDING** |
| Patents on fighting-game mechanics | Not performed | See §4 | **PENDING — HUMAN REVIEW** |

The working title remains internal (`PlayerSettings.productName = "Undercurrent (working title)"`, placeholder application
id `com.undercurrentdev.placeholder`). Development of original gameplay is not blocked by branding.

## 3. Generic mechanics vs. original expression

| Element | Classification | Notes |
|---|---|---|
| 1v1 side-view fighting, health bars, rounds, best-of-three, timer | Generic genre convention (ideas, not protectable expression) | Implemented independently from first principles. |
| Light/heavy punch/kick buttons, block, dodge, parry, throw & throw-break, special meter | Generic mechanics | Button layout, frame values and rules are our own. |
| Frame data (startup/active/recovery), hit-stun, block-stun, hit-stop, counter-hit, combo scaling, cancels | Generic design vocabulary | All numeric values authored for this project (`Core/Content/MoveFactory.cs`, `FighterRoster.cs`). |
| Stamina-gated attacks and guard-break | Generic | Own tuning. |
| Martial-arts disciplines (Muay Thai, boxing, karate/taekwondo, kickboxing, wrestling, MMA) | Real-world sports, public domain | Moves are named with generic technique names (jab, teep, roundhouse, side kick…). |
| Underground tournament premise | Common trope | Expressed through original setting (Varrow Bay), organisation ("The Ledger"), characters and dialogue. |
| Special move names ("Monsoon Rush", "Metronome Barrage", "Kestrel Descent", "Collection Day", "Eclipse Combination", "Old Lesson") | Original expression | Not derived from any franchise; **verify no conflict** (PENDING). |
| No supernatural powers, shadow forms, fatalities, transformations, factions | Deliberate avoidance | Avoids distinctive elements of Shadow Fight, Mortal Kombat, Tekken, Street Fighter. |
| Gore | Avoided | Hit sparks and dust only; no blood. |

## 4. Patent watch-list (HUMAN REVIEW)
Some fighting-game and mobile-input techniques have been patented historically. Items to clear:
input buffering / cancel windows, on-screen virtual pad with floating knob, slow-motion final-hit presentation,
AI difficulty that scales reaction delay. These are widely used industry techniques, but no freedom-to-operate
search has been done.

## 5. Asset provenance
See `ASSET_LICENSES.md`. Summary: **no third-party art, audio, fonts (other than Unity's built-in runtime font),
animation or code libraries are included**. All meshes, materials, textures, animations, sounds and music are
generated at runtime by this project's code. No proprietary game files were extracted, decompiled or referenced.

## 6. AI-assisted creation record
| Artefact | Tool | Human modification | Licence terms note |
|---|---|---|---|
| All C# source, data (roster, campaign, story text), procedural audio/visual generators, documentation | Claude (Anthropic) via Claude Code, session 2026-10-21 | Pending human review & edits | Users should confirm ownership/usage terms under their Anthropic agreement; AI-generated material may have limited copyright protection in some jurisdictions — **HUMAN REVIEW**. |

## 7. Open issues for human review
1. Choose a final title; run full trademark clearance (title conflict already identified).
2. Clear character/special-move names.
3. Patent freedom-to-operate review (§4).
4. Verify the Unity built-in font licence terms for commercial distribution (bundled with Unity; expected to be covered by the Unity licence — confirm).
5. Before integrating any purchased models/animations/audio: verify commercial, mobile, modification and redistribution rights and record them in `ASSET_LICENSES.md`.
6. Store policy review for in-app purchases and loot-box rules (none implemented; no randomized paid items planned).
