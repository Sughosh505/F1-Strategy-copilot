# Live F1 Strategy Co-Pilot — Project Roadmap

**Type:** Multi-agent strategy synthesis system
**Data source:** FastF1 (historical race replay, no live dependency)
**Core principle:** cheap rule-based math runs field-wide every lap; expensive LLM reasoning only fires on-demand for a selected driver. Every number the LLM sees is traceable to a specific upstream agent — no invented reasoning, no unattributed numbers.

This roadmap is deliberately phased. **Do not let an agent (or yourself) jump ahead to a later phase before the current one is checked off.** Each phase produces something independently testable — that's the point. A semester-long project graded well is one where every stage of the pipeline can be demoed in isolation, not just the final integrated system.

See `AGENTS.md` for the standing rules that govern how any agent (or you) should work through this roadmap in this repo.

## Current status

**Active phase:** Phase 1
**Last completed phase:** Phase 0

Update these two lines as you progress — this is the fastest way for anyone (including a fresh Antigravity session) to know where the project actually stands.

## Repo structure (target — build out as phases progress)

```
/AGENTS.md              — standing rules for any agent working in this repo
/ROADMAP.md              — this file
/data_notes.md           — Phase 0 output: documented FastF1 schema quirks
/agents/
  tire_degradation.py     — Phase 1
  sc_probability.py       — Phase 2
  rival_inference.py      — Phase 3
  weather_nowcast.py      — Phase 4
  qualifying_strategy.py  — Phase 8.5
/pipeline/
  precompute.py            — Phase 5, wires agents 1-4 into field_state
  snapshot.py               — Phase 6, assembles the orchestrator's JSON input
  orchestrator.py            — Phase 6 + 6.5, LLM synthesis + lookahead/comparative reasoning
  report_generator.py         — Phase 8.75
/backtest/
  harness.py                — Phase 8, scored validation across many races
  results.md                 — Phase 8 output: quantified accuracy metric + limitations
/demo/
  cli.py                     — Phase 7
  backend/
    main.py                    — Phase 7.5, thin FastAPI wrapper (optional, only if building web UI)
  demo_script.md              — Phase 9
/cache/                      — FastF1 + field_state cache (gitignored)
```

Not all of this exists yet — it's the target shape. Create directories as the phase that needs them starts, not before.

---

## Phase 0 — Environment & Data Familiarization
**Goal:** Understand FastF1's actual data shape before designing anything around assumptions.

- [x] Set up Python venv, install `fastf1`, `pandas`, `numpy`
- [x] Enable FastF1 local cache
- [x] Pull one full race session (laps, tire data, pit stops, track status, weather)
- [x] Manually inspect: how tire age resets across a pit stop, how track status timestamps align to lap numbers, how weather sample rate compares to lap rate
- [x] Write a short `data_notes.md` documenting quirks found (e.g. "weather is per-timestamp not per-lap, needs alignment")
- [x] Pick 3-5 candidate demo races (see Phase 6 criteria) and confirm FastF1 has clean data for each

**Deliverable:** a scratch notebook/script that loads a race and prints sane, correct-looking lap/tire/pit/weather/status data. No agents yet.

**Why this phase exists:** every downstream agent depends on assumptions about this data. Get burned by a schema surprise now, not in week 8.

---

## Phase 1 — Tire Degradation Engine (rule-based)
**Goal:** For one driver, one stint: turn raw lap times into a `degradation_score`, `pit_urgency`, and `pace_delta_vs_fresh`.

- [ ] Implement fuel-correction adjustment on raw lap times
- [ ] Fit a degradation curve per stint (start with linear, evaluate if quadratic/piecewise fits observed falloff better)
- [ ] Normalize into `degradation_score` (0-1 scale)
- [ ] Derive `pit_urgency` (low/medium/high) from the score + trend
- [ ] Test on a single driver's single stint — plot it, sanity check against what you know actually happened in that race
- [ ] Generalize to loop over all stints for one driver, then all drivers for one race
- [ ] Write a small unit test or assertion set (e.g. degradation should generally increase within a stint, reset after a pit stop)

**Deliverable:** given a race + lap number, output `degradation_score` for every car on track. Testable standalone via a CLI script or notebook — no other agents needed yet.

---

## Phase 2 — Safety Car / VSC Probability Agent (rule-based)
**Goal:** Field-wide (not per-car) probability of SC/VSC in the next 3 and 5 laps.

- [ ] Pull track_status history across multiple historical races per circuit
- [ ] Compute historical base rates of SC/VSC incidents by race-phase (e.g. early/mid/late race) per circuit
- [ ] Implement `next_3_laps` / `next_5_laps` probability output + a short templated `basis` string
- [ ] Handle small-sample circuits explicitly — flag low-confidence output rather than pretending precision
- [ ] It's fine to stub this with a rough constant for circuits with insufficient data, documented as a known limitation

**Deliverable:** given a race + lap number, output SC/VSC probabilities. Standalone testable.

---

## Phase 3 — Rival Strategy Inference Engine (rule-based)
**Goal:** For each car, pit-window likelihood for nearby competitors + `undercut_threat` flags.

- [ ] Reuse Phase 1's degradation logic, applied relationally across the field
- [ ] Compute pit-probability per car based on stint length vs historical typical pit lap for that compound/track
- [ ] Compute `undercut_threat` for cars within realistic pitting range of the evaluated driver
- [ ] Build the relevance filter — only nearby/threatening cars make it into a driver's `field_context`, not all 20
- [ ] Test: for a known real undercut moment in your chosen demo race, confirm the engine flags the threat *before* it happened in reality

**Deliverable:** given a race + lap number + driver, output a filtered list of relevant rival cars with pit probabilities and undercut flags.

---

## Phase 4 — Weather Nowcast Agent (rule-based, can start as a stub)
**Goal:** Short-horizon rain/drying trend signal.

- [ ] For dry races (your initial demo set), this can be a near-stub: report stable/no-rain confidently
- [ ] Implement a real rolling-trend version using weather channel data (rain intensity, track temp trend) for when you add a wet-race demo
- [ ] Align weather sample timestamps to lap numbers (this is the Phase 0 quirk you documented)

**Deliverable:** given a race + lap number, output a weather trend signal. Standalone testable.

---

## Phase 5 — Precompute Pipeline (wiring 1-4 together)
**Goal:** Run all four agents for every car, every lap, for a full race — once — and cache it.

- [ ] Implement the `for lap in 1..N: compute all agents; store field_state[lap]` loop from the architecture doc
- [ ] Confirm output size is small (a handful of scalars × ~20 cars × ~60 laps) and fast (well under a second per race on a laptop)
- [ ] Serialize `field_state` to disk (JSON or similar) per race, so you don't recompute on every run
- [ ] Sanity-check a few specific laps by hand against what actually happened in the real race

**Deliverable:** run a script against a race, get a cached `field_state` file covering the whole race. This is the last purely rule-based phase — no LLM involved yet.

---

## Phase 6 — Snapshot Assembly + Orchestrator (LLM synthesis)
**Goal:** Given a selected driver + lap, assemble the strict JSON snapshot and get a grounded recommendation.

- [ ] Implement the snapshot-assembly function per the schema in the architecture doc (`triggered_car`, filtered `field_context`, `sc_probability`, `historical_reference`, etc.)
- [ ] Write the orchestrator prompt: strictly grounded, must cite which snapshot fields drove the call, structured output (`pit_now` / `stay_out` / `prep_for_sc_window` + rationale)
- [ ] Use structured/JSON-constrained output from the LLM API so responses are reliably parseable
- [ ] Test manually against 3-5 chosen historical moments — compare the system's call to what actually happened
- [ ] Explicitly test a "boring" lap (nothing strategically interesting) to confirm the orchestrator doesn't manufacture false urgency

**Deliverable:** given any (driver, lap) pair from a cached race, get back a ranked, cited recommendation. This is the first point where the full pipeline is end-to-end functional.

---

## Phase 6.5 — Recommendation Depth: Lookahead + Comparative Reasoning
**Goal:** Make a single recommendation read as a considered judgment instead of a single-instant verdict. Do this only after Phase 6's core snapshot → recommendation path is working end-to-end — these are refinements to an already-correct pipeline, not new agents.

- [ ] **Multi-lap lookahead:** instead of evaluating `degradation_score` / `pit_urgency` at only the current lap, evaluate the existing Phase 1 degradation curve fits forward 3-5 laps and include the trend in the snapshot (e.g. `degradation_trend_next_3_laps`)
- [ ] Update the orchestrator prompt/schema so the recommendation can express a *window* ("pit within the next 2-3 laps") rather than only a single-lap verdict, when the data supports it
- [ ] **"Why not the alternative" reasoning:** extend the orchestrator's structured output to include a brief rejection rationale for the runner-up call (e.g. why `stay_out` was rejected in favor of `pit_now`), grounded in the same snapshot fields — no new numbers invented to justify the rejection
- [ ] Test both together on a known real decision point: confirm the window framing and the rejected-alternative reasoning both stay traceable to snapshot fields, not free-floating LLM justification
- [ ] Re-run the "boring lap" test from Phase 6 — confirm lookahead doesn't manufacture false urgency out of a flat trend, and that alternative-rejection reasoning doesn't invent a rivalry that isn't there

**Deliverable:** a recommendation that states a decision window (not just an instant) and explains why the top alternative was rejected — both still fully traceable to snapshot data. This is what turns the demo from "the system says pit now" into "the system reasoned through it," which reads much stronger live.

---

## Phase 7 — Interactive Demo Layer
**Goal:** Let a user pick driver + lap and instantly see the call, switchable mid-replay.

- [ ] CLI first: type a driver code + lap number, print snapshot + recommendation
- [ ] Once CLI works cleanly, decide if a simple web UI (driver/lap selectors, a call display) is worth the time investment vs. staying CLI — this is a judgment call based on how much semester time is left
- [ ] Confirm switching between drivers/laps is instant (cached field_state, no recompute) — this was a core design promise, verify it holds

**Deliverable:** a working, demoable interface — even a polished CLI is legitimate if time is tight.

### Phase 7.5 — Minimal Backend (only if building the web UI)
**Goal:** A thin FastAPI layer that exposes the already-working pipeline to a frontend. This is explicitly scoped small — no auth, no multi-user handling, no deployment/production concerns. Those are real future-work items, not semester scope (see Phase 9's "future work" note).

- [ ] Single FastAPI app, run locally, no auth — this is a demo tool for one user on one machine, not a hosted product
- [ ] Endpoint to list available cached races (from the `field_state` cache built in Phase 5)
- [ ] Endpoint to fetch a snapshot + recommendation for a given (race, driver, lap) — wraps the existing snapshot assembly (Phase 6) and orchestrator call (Phase 6 + 6.5), doesn't reimplement any of that logic
- [ ] Endpoint to trigger the precompute pass for a new race (wraps Phase 5's pipeline) — fine for this to be slow/blocking for a single local demo user
- [ ] Confirm the frontend can call these endpoints and get the same output the CLI produces — the backend is a thin wrapper, not a second implementation of the logic
- [ ] Explicitly do NOT add: user accounts, API keys/auth, request queuing, rate limiting, or a real database. If any of these start to feel necessary, that's a signal you've drifted into "production" scope — stop and note it as future work instead

**Deliverable:** a local FastAPI server that the web UI can call, exposing exactly what the CLI already does. Skip this phase entirely if staying CLI-only.

---

## Phase 8 — Multi-Race Validation + Backtest Harness + Known Limitations Writeup
**Goal:** Prove this isn't overfit to one race, with an actual scored metric, and write the honest report section.

- [ ] Run the full pipeline against all 3-5 chosen demo races
- [ ] For each, identify at least one moment where the system's call matches or plausibly could have matched what a real strategist would call
- [ ] **Backtest harness:** run the pipeline against a larger set of historical races (aim for dozens, not just your 3-5 demo races) and score each recommendation against what actually proved optimal in hindsight — e.g. "within N laps of the real pit lap" or "matched the compound choice that led to the better outcome"
- [ ] Produce a single summary metric (e.g. "matched or came within 2 laps of the real optimal window in X% of tested cases") — this is the number you'll actually quote if anyone asks "does it work"
- [ ] Break the metric down by circuit/race type if possible, to honestly surface where it does/doesn't perform well (ties directly into the small-sample-circuit limitation below)
- [ ] Document known limitations explicitly: degradation modeling is an estimate not ground truth; SC probability is base-rate not true prediction; small-sample circuits have shaky priors
- [ ] (Optional stretch) simple counterfactual: would the system's alternative call have beaten the real outcome, using simulated lap-time deltas

**Deliverable:** a short results/limitations section with a real quantified accuracy metric across many races — this is what makes the project look rigorous rather than lucky-on-one-demo, and it's the single strongest artifact for both grading and any future pitch.

---

## Phase 8.5 — Qualifying / Grid Strategy Agent (architecture validation module)
**Goal:** A second, smaller application of the same rule-based-agent → structured-snapshot → LLM-synthesis pattern, proving the architecture generalizes beyond pit strategy. Build this only after Phase 8's core system is validated — it's a demonstration of the pattern's reusability, not a dependency for anything else.

- [ ] Rule-based agent: given quali results + historical overtaking difficulty at a circuit (derivable from FastF1 across seasons — how often do on-track passes actually happen there vs. positions holding station), estimate how much track position matters this specific race
- [ ] Structured snapshot: grid position, circuit overtaking-difficulty score, historical reference note — same grounding discipline as the main snapshot schema
- [ ] LLM synthesis: a short grounded read on what the grid position implies for the race (e.g. "track position will be critical — passing is rare here, so an early strategic gamble may be the main lever")
- [ ] Test against 2-3 real races: does the overtaking-difficulty score match what you know actually happened (a high score at a circuit known for being hard to pass at, like Monaco, vs. a low score somewhere like Bahrain)

**Deliverable:** a standalone secondary agent using the identical architectural pattern as the main system. Worth explicitly calling out in your report as evidence the architecture is a reusable pattern, not a one-off pipeline built to fit a single question.

---

## Phase 8.75 — Post-Race Strategy Report Generator
**Goal:** Turn the system from an interactive query tool into something that produces a standalone shareable artifact — a written retrospective of a completed race.

- [ ] Run the full precomputed `field_state` for a completed race through a new synthesis pass that scans across all laps (not just one queried lap) to identify the most strategically significant moments — largest degradation swings, actual SC/VSC events, biggest undercut/overcut successes
- [ ] Feed these identified moments, in order, to an LLM synthesis call that writes a short narrative report ("the critical decision point was lap 34, when...") — still grounded in the same structured data as the interactive orchestrator, just synthesizing across the whole race instead of one snapshot
- [ ] Output as a clean, readable document (markdown or similar) — this is meant to be handed to someone to read standalone, not just displayed in a demo
- [ ] Test on 1-2 of your demo races: does the generated report correctly identify the moments you already know were the real turning points

**Deliverable:** an auto-generated race retrospective document. This is a strong portfolio artifact independent of the interactive tool — something you can literally hand someone to read, which is a different (and often more persuasive) proof of the system's value than a live demo.

---

## Phase 9 — Polish for Presentation
**Goal:** Make the demo itself land well, independent of the underlying engineering quality.

- [ ] Prepare the live demo script: which races, which (driver, lap) moments, in what order, building to the most impressive one last
- [ ] Prepare the "run precompute pass live" opening (per the architecture doc's demo plan) — it's a good way to show the pipeline is real, not canned
- [ ] Clean up output formatting — this is often underweighted but matters a lot for how "finished" the project looks
- [ ] Rehearse the fallback story for known limitations — own them confidently rather than getting caught off guard by a question
- [ ] Prepare a brief, explicit "future work" note for production/commercial questions (auth, multi-user hosting, a real database, live-race data instead of historical replay) — having this ready shows you understand the gap between a demo and a product, without having spent semester time building it prematurely

**Deliverable:** a rehearsed, confident demo.

---

## How to use this with Antigravity IDE

- Treat each phase as a **separate task/goal** you hand to the agent — don't paste the whole roadmap and say "build all of this." Give it one phase at a time, with the phase's checklist as the spec.
- Review each phase's Artifacts (implementation plan, task list, test results) before approving and moving to the next phase — this is exactly the checkpoint discipline the roadmap is designed to enforce.
- If an agent's plan for a phase tries to reach into a later phase's concerns (e.g. Phase 1 agent starts writing orchestrator prompt code), redirect it — scope creep across phases is the main way "let AI build it" projects end up shallow and hard to explain in a report.
- Keep this file itself updated as you complete checkboxes — it doubles as your progress-tracking artifact for the semester and a built-in status report if anyone (professor, teammates) asks where the project stands.