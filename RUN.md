# RUN — deviation record & run registry

Record of how corpora are generated, what deviates from the PRD, and every run
that produced a shipped artifact. The PRD (`trace-bench`, frozen 2026-09-10) is
the specification; this file records the implementation's departures from it.

## Environment

- Conda env `trace-bench` (python 3.11, conda-forge), created 2026-09-10 on
  macOS (Apple silicon). Runtime deps pinned to a minor range in
  `requirements.txt`; the manifest of every corpus records the exact numpy and
  pyarrow versions in force, and `numpy` minor is asserted at load because
  Generator bit streams are only stable within a minor version.
- Every shipped rung generates in the cloud under the lab's private execution
  configuration, which is not part of this repository: one job per (rung, seed)
  for s/m/l/xl and one job for all five xs seeds, each running
  `python -m tracebench.pipeline` (generate the latent instance and its twin,
  verify both, upload both). Rungs xs–l also generate locally on CPU, which is
  how the xs checksum fixture is frozen and how a change is checked before a
  cloud run.

## Deviations vs the PRD (D-TB-n)

- **D-TB-1 — SLOW outcome class.** Outcomes are `OK, 4XX, 5XX, ERR, SLOW`
  (`SLOW` = duration above a per-operation quantile threshold shipped in
  `slow-thresholds.json`; precedence `ERR > 5XX > 4XX > SLOW > OK`). The PRD
  Definitions name four classes. *Reason:* latency-only faults are otherwise
  invisible in the token stream. Owner decision 2026-09-10.
- **D-TB-2 — scoring-target nodes are `(operation, outcome)` tokens.** The
  mechanism is built over operation-outcome state variables plus latent states;
  token-level edges are derived by exact expansion of the conditional tables,
  then latent-projected, then floored. Operation- and service-level graphs are
  coarsenings. Owner decision 2026-09-10.
- **D-TB-3 — strength context.** The strength of a mechanism edge is the
  maximum over parent-state pairs of the total-variation distance between the
  child's next-state distributions, with the child's other parents held at
  their *nominal* configuration. Edges additionally record a context-max
  strength where the parent set is small. *Reason:* computable by construction
  from the tables; reproducible by forcing the other parents.
- **D-TB-4 — bidirected strength = min of the two arms.** For a latent common
  ancestor of two observables, the projected bidirected edge carries the
  smaller of the two projected latent→observable strengths.
- **D-TB-5 — twin = observed latents.** The fully-observable twin is the same
  simulation with the same random draws; every latent variable's value is
  written onto the records it influences and surfaces as extra per-span
  columns. Its scoring target is the full mechanism DAG over tokens and state
  tokens. Owner decision 2026-09-10.
- **D-TB-6 — both span orderings and both grains.** The correlated view ships
  as four trees: `{end,start}-{request,session}`. Owner decision 2026-09-10.
- **D-TB-7 — byte-identity excludes the `run/` directory** (`run_meta.json`
  carries wall clock and host; `arguments.json` carries the output path).
- **D-TB-8 — client replay records are not emitted in v1.** The raw feed
  carries `sentry.error` (tagged and auto-captured) and sampled
  `sentry.transaction` records; `sentry.replay` metadata is deferred.
- **D-TB-9 — a residual duration channel is not a recorded dependence.** A
  callee's total duration reaches its caller's SLOW class; the mechanism
  records that dependence through the callee's final class (its duration
  band) and through its first-attempt class (retry time), but the callee's
  duration also varies *within* its class band, and a categorical graph cannot
  carry that residual. `check_mechanism` measures it: non-edges are held to
  a 0.05 tolerance and the largest residual is reported per corpus
  (`reports/mechanism-check.json:max_non_edge_residual`; 0.001 on the
  2026-09-11 xs corpus, 0.04 on the v0.1 one — the value depends on which
  non-edges the run sampled).
- **D-TB-10 — the scoring target is restricted to co-occurrence support.**
  Token pairs whose operations can never share a request tree (request grain)
  or a journey (session grain) are outside the scoring universe; without this
  the daily-intensity latent alone would make every pair of slow tokens a
  bidirected edge. The request-grain target is acyclic by construction; the
  session-grain target adds journey edges and may be cyclic at the type
  level, in which case the causal-validity axis is not applicable for the
  charter's cyclicity reason. Within-operation token pairs (retries) are
  recorded as `retry_pairs` and never scored.
- **D-TB-11 — degraded and exhausted states are explicit outcome shares.**
  `error_shift.degraded` / `error_shift.pool_exhausted` are probability maps
  (the share of requests a degraded endpoint or an exhausted pool fails),
  mixed on top of the nominal class rates; multipliers on rates of a few
  tenths of a percent could not double a hop's error rate as PRD scenario 8
  requires once propagated errors dominate its baseline.

- **D-TB-12 — rung rates are sized to the cap, not to the plan's budget
  table.** The named instances keep the planned service × endpoint counts at
  lower rates than the plan's budget table assumed: 5 / 4 / 3 / 3 rps for
  s / m / l / xl, against 5 / 10 / 20 / 30. In v0.1 the windows were also cut
  to 24 h (m, l, xl ≈ 16, 26 and 35 GB), because every operation must be
  reachable from a journey (below) and full request trees pushed hops per
  session far above the fitted mean. Per-edge call probabilities (D-TB-13)
  bring a request to about 4.5 hops, so on 2026-09-11 the windows returned to
  the plan's day counts. Sizes are estimates from `tracebench.estimate` at
  seed 0 (the estimator matched the realised xs corpus within 10 %):

  | rung | services × endpoints + BFF | scenarios | window | rate | est. GB | est. CPU-h | expected realized tokens |
  |---|---|---|---|---|---|---|---|
  | xs | 3 × 3 + 4 | 2 | 1 h | 1 rps | 0.03 | 0.00 | 44 |
  | s | 10 × 5 + 8 | 4 | 1 d | 5 rps | 4.2 | 0.4 | 188 |
  | m | 50 × 10 + 20 | 8 | 2 d | 4 rps | 6.7 | 0.6 | 1,573 |
  | l | 150 × 10 + 30 | 12 | 3 d | 3 rps | 8.1 | 0.7 | 4,325 |
  | xl | 350 × 12 + 60 | 20 | 5 d | 3 rps | 16.8 | 1.6 | 10,750 |

  CPU-hours are single-core; `--workers` parallelises across shards. The xl
  rung exceeds the PRD's 8,000-token alphabet target by expectation (10,750 of
  12,849 potential tokens; 8,875 at a 24 h window). The manifest's
  `alphabet_size_realized_train` is the measured value. The estimator
  counts tokens at the nominal latent context only, so incidents and load
  states realise more classes than it predicts (v0.1 s: 322 realised against
  185 estimated). Single-shard timing on Apple silicon (noon shard, D-TB-13
  code): s 28 s (engine 0.3, emission 21, write 6.5) and xl 24 s (engine 5,
  emission 14, write 4.5). Emission dominates and scales with hops.

- **D-TB-13 — per-edge call probability; request depth calibrated
  (2026-09-11).** Every call edge carries `p_call`, the share of its caller's
  requests that invoke the callee. The call is drawn per request from its own
  counter-keyed coordinate (`H_CALL`); the earlier coordinates keep their
  numbers, so every other draw is unchanged. A callee is invoked when a caller
  is, the call draw falls under `p_call` and the call misses its cache. No
  latent and no mechanism node is added. The invoke tables gain the factor:
  an invoke edge's strength is `p_call × (1 − p_hit)` on a WARM cached call
  and `p_call` otherwise. The latency model, the size and alphabet estimators
  and the twin read the same probabilities; in the latency model a callee
  that is not invoked adds no time, retries included. The topology sampler's
  structural depth gate is removed: every endpoint above the deepest layer
  samples callees, and the probabilities set how deep a request goes.
  - *Depth semantics (pinned).* A request's depth is the deepest
    backend-service layer its invoked hops reach; the BFF is layer 0 and
    externals set no depth. `depth_pmf[d-1]` is the share of requests at depth
    d among requests that reach layer 1, and requests answered without any
    backend call are reported as `root_only_share`. The fitted constant is on
    the same axis (SERVER ancestors below the root, root-only traces
    excluded).
  - *Calibration.* The parameters are per-layer stop probabilities `s_d`: an
    invoked layer-d endpoint with k backend callees gives each edge
    `1 − s_d^(1/k)`, and its external calls take the same probability. The
    closed form `s_d = stop_d^(1/f^d)` seeds a topology-only Monte Carlo:
    20,000 requests over BFF endpoints in proportion to scenario weight ×
    visiting steps, WARM caches at the nominal stationary share, and common
    random numbers across steps. It refits the effective exponent for at most
    6 steps, stops at TV ≤ 0.02 and keeps the best step. Probabilities are
    floored at 0.02. `instantiation.json:topology.calibration` records the
    steps, TV, realised pmf, root-only share and stop probabilities. Seed 0:
    xs TV 0.004, s 0.004, m 0.012, l 0.016, xl 0.003.
  - *BFF edges (owner selection 2026-09-11).* A BFF endpoint makes its sampled
    layer-1 calls and its external calls on every request, so every request
    reaches layer 1. The `m` layer-1 endpoints attached to it only by the
    reachability rule carry `1/m` each (floor 0.02), and a deeper endpoint
    attached to a BFF is called as often as one layer-1 endpoint continues.
    Edges record `attached`. Under the plan's literal rule (every BFF →
    layer-1 edge at 1.0) an xl BFF endpoint called about 40 backends on every
    request, half of all xl edges sat at the floor and the xl calibration
    stalled at TV 0.17 (l: 0.032).
  - *Effect.* About 4.5 hops per request at every rung (full request trees on
    the gate-free topology: 12 non-BFF hops at s, 51 at m, 105 at l, 161 at
    xl), so D-TB-12's windows were re-tuned. The realism report scores fan-out
    on the callees a calling hop actually invokes and reports the topology's
    edge count beside it.

- **D-TB-14 — publishing splits into an unversioned `upload` and a `release`
  that tags (2026-09-12).** `publish` was one command that uploaded N corpora
  and created the version tag in a single shot. It could not accumulate the 50
  corpora of a release from independent jobs, it uploaded with a non-resumable
  whole-folder call, and it would have shipped `artifacts.json` (which names
  the private object store). It is now two subcommands:
  - `publish upload` puts complete corpora on the dataset host's default
    branch at `instances/<instance>/<variant>/seed=<k>`, with no release
    metadata and no tag. It refuses a corpus that is not complete or does not
    verify against its manifest, refuses a remote path that already exists
    unless `--replace` is given, and post-checks that every uploaded corpus has
    a remote `manifest.json`. This is what an unattended generation job runs,
    so 21 independent jobs can fill one release.
  - `publish release --version vX.Y.Z` is the single-writer step: it refuses an
    existing tag (PRD scenario 14, unchanged), downloads every remote
    `manifest.json`, checks that each corpus is `COMPLETE` and that every file
    it lists is present on the host with the same size and — where the host
    exposes a per-file hash — the same sha256, then writes the card, the
    licence and `release.json` and tags the repository. A host that stores
    files without exposing a hash degrades the check to size only, and
    `release.json` records how many files were hash-verified.
  - *Uploads stage hard links.* The host library's resumable upload call takes
    no in-repository path, so each corpus is hard-linked into
    `<stage>/instances/<instance>/<variant>/seed=<k>` and the stage is what is
    uploaded: the repository layout without copying a byte. `run/` and
    `artifacts.json` are never linked in, and the stage is removed only after
    the post-check passes (a failed upload keeps it, because it carries the
    library's own resume state). The stage must sit outside every corpus —
    a stage inside one would be hashed into it — and outside the directory an
    execution environment copies afterwards.
  - *The dataset is public from creation* and its card carries
    `viewer: false`: a corpus is a tree of gzipped JSON lines and parquet files
    with several schemas, not one table.
  - *Corpora are versioned by the tool version, releases are dataset tags.*
    Every `manifest.json` names the tool version that produced it, so a
    corpus's identity does not depend on which release names it; a release is a
    tag over corpora already uploaded.

- **D-TB-15 — Monte-Carlo-derived floats are quantised, because they are not
  bit-stable across CPU architectures (2026-09-12).** The cross-machine half of
  PRD scenario 12 was exercised for the first time against a cloud run of the
  `xs` rung (five seeds, both variants, generated on x86-64 Linux) and **it
  failed**: 6 of the 70 files of `xs/latent/seed=0` differed from the same
  corpus generated on arm64 macOS — `instantiation.json`,
  `slow-thresholds.json` and the four copies of that file under `views/`.
  - *Cause.* Exactly one float: the fitted SLOW threshold of one operation came
    out `0.02184745943679561` on x86-64 and `0.021847459436795613` on arm64 —
    **1 unit in the last place**, relative 1.6e-16. Python (3.11.16), numpy
    (2.4.6) and pyarrow (25.0.1) were identical on both sides, so this is the
    architecture, not version drift: `fit_thresholds` takes a quantile over
    20,000 lognormal draws, and the ordering of the transcendental transform
    inside libm/vectorised code differs between the two. The extra digit also
    made the file one byte longer, so the divergence was visible to a size
    check as well as to a checksum.
  - *What was unaffected.* Every `raw/`, `oracle/`, `views/` and `graphs/`
    artifact was byte-identical, so no outcome class flipped: the perturbation
    is ~14 orders of magnitude below the millisecond granularity of a record.
    The D-TB-13 `topology.calibration` block was bit-identical — **because it
    was already rounded** (to 6 decimals), which is what located the gap: the
    fitted latency thresholds were the one Monte-Carlo quantity that reached an
    artifact unrounded.
  - *Fix.* `MC_DECIMALS = 9` (`constants.py`). The fitted SLOW threshold is
    rounded **where it is computed**, not where it is written, so the value
    that ships is the value the engine classifies against; rounding only at
    serialisation would have left an architecture-dependent threshold in force
    and could have flipped a class. Call probabilities are rounded at
    `_clamp_call_p`, the single point every computed `p_call` passes through,
    and therefore inside the calibration loop — so `calibration.realised_pmf`
    records the depth of exactly the probabilities that ship. Nine decimals is
    one nanosecond on a seconds-valued threshold and 1e-9 on a probability:
    ~7 orders of magnitude above the ULP noise, and far below the calibration's
    0.02 tolerance and the 0.05 strength floor.
  - *Consequence — not a pure serialisation change.* Because the threshold
    feeds the Monte-Carlo estimate of the slow-related mechanism strengths,
    quantisation moves those estimates slightly: on `xs/latent/seed=0`, 8 of 70
    files differ from 0.2.0 (the 6 above plus both scoring targets), with 30 of
    8,781 scoring-target leaves changing by at most **2.3e-11** absolute
    (5.8e-10 relative). Edge sets and floor counts are unchanged — 520
    directed, 915 bidirected, 467 and 108 at the floor. The emitted data is
    unchanged.
  - *Guard.* `tests/test_byte_identity.py::test_every_monte_carlo_derived_float_is_quantised`
    asserts that every fitted threshold and every call probability, in memory
    and as serialised, equals its own rounding — so a future unrounded fitted
    quantity fails a test instead of a rung.
  - Corpora generated at 0.2.0 are superseded: the ten `xs` corpora published
    on 2026-09-12 must be regenerated at 0.2.1 (their manifests name 0.2.0, so
    they are distinguishable, and the versioned object-store prefix keeps them
    apart).

- **D-TB-16 — the spill allowance is a heuristic, so overspill is recovered by
  re-running the shard (2026-09-12).** Two of the five `s` jobs of the v0.2.1
  ladder died in `generate` with `IndexError: index 2872 is out of bounds for
  axis 0 with size 2839` (seed 1) and `index 2845 ... size 2839` (seed 4).
  - *Cause.* `2839 = run.shard_ticks (900) + spill_allowance (1939)`, the height
    of the latent slice a shard holds. `spill_allowance` is a tail heuristic —
    `max_steps × (3·p99_step_gap + max_attempts × (3·p99_bff_latency × 4 + 2)) + 5`,
    of which the step gap is 1,885 ticks at `s` — and **not a bound**. A long
    enough journey outruns it and `engine._state_at` indexed the slice out of
    range.
  - *Where it bites systematically.* The `.done` markers of the failed runs name
    the culprit: every shard up to 31 completed **except shard 15** in both
    seeds (seed 1 also lost shard 28). Shard 15 ends at tick 14,400 — exactly
    where the configured `degrade` fault on `service:2` begins. Journeys
    arriving late in that shard spill into the degraded window, where retries
    and inflated latencies stretch them far past the heuristic. So **the shard
    immediately preceding each scheduled fault is the structural risk**, at
    every rung, and it is not seed-specific. Seed 1's shard 28 sits near no
    fault, so ordinary tail overshoot is a second, independent cause. Rungs with
    more sessions are likelier to hit both.
  - *Fix.* The engine raises `SpillExceeded(needed_ticks, held_ticks, t0)` from
    the latent access instead of letting numpy raise, and `run_one_shard`
    re-simulates the slice at `needed − (t1 − t0) + SPILL_RETRY_MARGIN` (600
    ticks) and re-runs the shard, up to `SPILL_RETRY_MAX` (4) attempts, after
    which it refuses and names the knob to raise. The bounds check is
    vectorised: `tick` is a vector of ticks, one per request in the group.
  - *Why re-running is sound.* `simulate_latents` draws its uniforms keyed by
    the **absolute tick** and advances the chain from a recorded checkpoint, so a
    taller slice holds the same values at every tick the shorter one held. A
    retried shard is therefore byte-identical to one generated with a large
    enough allowance from the start — asserted by
    `tests/test_engine.py::test_a_retried_shard_is_byte_identical_to_one_with_enough_allowance`.
    The retry costs one shard's work (about 30 s at `s`, 24 s at `xl`).
  - *Measured on the real failures.* Re-running the exact failing shards, each
    recovered in **one** retry: seed 1 shard 15 needed 2,873 ticks against 2,839
    held (34 over), seed 1 shard 28 needed 3,375 (536 over — the 600-tick margin
    covered it, but not by much), seed 4 shard 15 needed 2,846 (7 over). At the
    shipped allowance the `xs` rung needs no retry at all, which a test pins.
  - *Effect on a corpus.* None beyond the recorded version: against 0.2.1 the
    only file that moves is `instantiation.json`, which names `tool_version`.
    Corpora generated at 0.2.1 that never overspilled are otherwise
    byte-identical to their 0.2.2 regeneration, and `config_hash` is unchanged.
  - *Diagnosability.* `tracebench.pipeline` previously logged only
    `"IndexError: ..."` for a failed step, so the cloud log carried no stack and
    the location had to be re-derived from the array size. It now emits the
    traceback line by line as `pipeline_traceback` events before the step
    record.

- **D-TB-17 — held distributions are memoised; the latent projection was the
  ladder's long pole (2026-09-13).** The v0.2.2 ladder lost four of the five
  `m` jobs and all five `l` jobs to the 12-hour runtime cap without a single
  shard emitted. The `m` job that finished (seed 3, 37,142 s) spent
  **6 h 19 min between the size estimate and the `graphs` event** — inside
  `write_graph_artifacts`, on one core — before its 192 shards took 1.5 h;
  the other four `m` boxes were still in that phase after 11 h 55 min and the
  five `l` boxes after 9.7 h.
  - *Cause.* `Projector.effect` marginalises the latent intermediates by
    frontier-merged enumeration and asks `Node.dist_held` for the child's
    distribution under every particle's held context. For a lag-1 state
    variable that is the stationary distribution of its tick chain, found by a
    power iteration to 1e-13 — and it was recomputed on every request: 593,683
    iterations in the first fifteen minutes of a local profile, 94 % of the
    time. The requests repeat a few hundred thousand distinct contexts (m seed
    3: 11.1 M requests for 255 k contexts; seed 0: 24.8 M for 299 k — the
    per-seed spread that let seed 3 finish and not the others).
    `bidirected_groups` carries almost all of it; the directed pass and the
    twin are cheap.
  - *Fix.* `dist_held` memoises per `(node, held context)` and hands out a
    read-only array. The same function on the same inputs. Measured locally
    (Apple silicon): `s` latent 37.5 s → 4.2 s; `m` latent seed 3 181.7 s and
    seed 0 275.2 s against 6.3 h and > 11.9 h on the cloud box; `m` twin seed 3
    31.3 s. `stationary` runs 5,938 times per `m` variant instead of millions.
  - *Byte identity.* `s` seed 0, latent and twin: `graphs/` identical to the
    uncached code (`diff -r`). All 8 + 8 `graphs/` files of the published
    `m/latent/seed=3` and `m/twin/seed=3` reproduce with the sha256 their
    manifests record. `xs` seed 0 at 0.2.3 differs from the 0.2.2 fixture in
    `instantiation.json` (the tool version) and in `topology/callgraph.json`,
    the latter for an unrelated reason (D-TB-18). `config_hash` is unchanged;
    corpora generated at 0.2.1 and 0.2.2 stay valid and are not re-run.
  - *Cost of the miss.* About 108 box-hours. The per-rung wall-clock estimates
    were extrapolated from emission timing at `s`, where the projection takes
    two minutes; it was never timed above `s`. Same lesson as D-TB-13 and
    D-TB-16: prototype every phase on every rung.
  - *Work count by rung* (`effect` calls, by graph traversal): xs 2.5 k, s
    9.7 k, m 90–99 k across seeds, l 273 k, xl 715 k. Projected for the cloud
    box with the cache: m ≈ 10 min, l ≈ 0.5–1 h, xl ≈ 1.5–3 h in this phase.
  - *Guard.* `tests/test_mechanism.py::test_held_distributions_are_memoised_and_identical_to_a_cold_evaluation`
    — repeats are served without a `stationary` call and equal a cold
    evaluation exactly.

- **D-TB-18 — the call-graph artifact is dated by the simulated window, not
  the wall clock (2026-09-13).** `topology/callgraph.json` follows the lab's
  CallGraphTruth shape, whose `derived` field is a date; `generate` filled it
  with the day the corpus was generated. A corpus therefore regenerated on a
  different day differs in that one file, which contradicts PRD scenario 12 —
  the xs checksum fixture only passed because it was frozen and checked on the
  day it was made — and the published latent/twin pairs of `s` seeds 1 and 4
  and `m` seed 3 already carry different dates (their latent instance was
  generated before midnight UTC, the twin after). Found while checking the
  0.2.3 regeneration against the 0.2.2 fixture. The consumer of the artifact
  (`CallGraphTruth.from_json`) never reads the field. From 0.2.3 `derived` is
  the start date of the configured simulation window (the date the deployment
  topology is "as of"), a function of the configuration alone. Owner decision
  2026-09-13. The 22 corpora already published keep their calendar dates and
  are not re-run; a regeneration of any of them at 0.2.3 differs from the
  published bytes in `instantiation.json` and this field only.

## Implementation notes (not deviations)

- **Every operation lies on a journey.** The topology sampler attaches an
  operation no sampled call reaches to a caller that is itself reachable, and
  the scenario sampler draws each scenario's BFF endpoints first from those no
  earlier scenario visits, so with enough steps every BFF endpoint (and hence
  every operation) carries traffic. Without both rules a third of the
  operations at l/xl were dead and the realized alphabet fell short.
- **The correlator streams one shard at a time.** Every join key lives inside
  one session and a session's records all lie in the shard of its arrival, so
  key resolution, attribution, request trees and session-level sequences are
  shard-local. Device-/user-/ip-level sequences are stitched across shards: a
  run stays open until no record that could still join it can arrive (after
  shard k every record stamped before the end of shard k has been read, less a
  5 s skew margin). Sequence ids hash (level, key, first-minute bucket), so
  they do not depend on the sharding. On xs the streamed report is identical
  to the earlier whole-corpus pass; resident memory ≈ 0.5 GB for xs and one
  shard's records at any rung (xl shards are 30 simulated minutes).
- **View orderings break millisecond ties by containment, then by inferred
  depth.** A caller starts no later than its callee and ends no earlier; when
  both stamps tie (a caller whose own time rounds to zero) the inferred link
  is the only evidence. Genuine inversions (a rounding step, browser skew)
  are left in place and counted in `export-stats.json:orientation_violations`
  (xs: 0.05 % under `start`, 1.5 % under `end`, the latter from browser skew on
  client → edge links).
- **Fault-phase labels** (`labels/sequence-labels.parquet`) compare row times
  and fault intervals on the same clock (window-relative seconds).
- **The manifest is written before `COMPLETE`** by `generate`, and rewritten
  by a standalone `correlate` (the views change the file set).
- **Tests run on the shipped constants** (`realism-v1.json`); the labelled
  placeholder `realism-dev.json` is used only by the loader's fail-closed
  test. The placeholder's 20 ms server clock skew is not a property of the
  fitted feed (0 ms), and tests on it validated the placeholder, not the
  benchmark.
- **Realism at the s rung (2026-09-11, D-TB-13 code, 12 sampled shards of 96,
  fitted constants, `error_rate_multiplier: 10`).** 18 of 20 scored items
  pass. **Request depth now matches the configuration:** realised
  `[0.497, 0.350, 0.153]` against the configured `[0.5, 0.35, 0.15]`, total
  variation 0.003, with 7.1 % of requests answered without any backend call
  (cache hits) and an instantiation calibration TV of 0.004. The v0.1 corpus
  realised `[0.070, 0.033, 0.897]` — 90 % of requests at the deepest layer —
  and failed the item. Invoked fan-out is 2.14 against the configured 2.0
  (the topology's static edge count per calling endpoint is 2.40). Latency
  quantiles match at every tier except the BFF p95 (+16 %): the BFF's own
  time absorbs the callee retry back-off, the same unresolved channel that
  showed as p99 +11 % in v0.1 (the p99 now passes). Leaf error rates match
  the declared ×10 target, external p50–p99 match, session step-gap p90–p99
  match while the p50 is +29 % (audited-response gaps include the next
  request's duration, and the external tier's heavy tail inflates the median;
  unresolved). The xs corpus on the same code: 18/20, depth TV 0.004,
  invoked fan-out 1.95. Depth and fan-out are scored against the
  configuration targets, with the fitted values reported beside them.
- **The realism report samples 12 evenly spaced shards** (`--max-shards`,
  0 = all): the quantities are per-hop and per-session statistics, and the
  whole s corpus as Python objects took hours. `verify` re-hashes every
  file (5 s on s) and scans 200k records for the name grammar.
- **Denylist verification (2026-09-11).** `verify --denylist <private list>`
  is part of a corpus being called verified. The first list also held
  word-level tokens of real paths ("error", "https", "entry") and a
  hex-plausible short identifier, which matched the synthetic feed's ordinary
  vocabulary and ids; it now holds full service names, hosts, paths and
  operation names only. It also caught one emitted `log_source` value that
  coincided with a real module name; the value was renamed. The scan compiles
  the list into a prefix-trie regex (about 10 s per corpus).
- **One unattended job = one rung's `pipeline` invocation.** `tracebench.pipeline`
  runs generate (latent, then twin) → verify → verify → one upload per job,
  prints one `pipeline_step` line per step (step, corpus, status, seconds), and
  stops at the first failure with later steps skipped, so whatever was already
  generated stays on disk for an execution environment that copies the output
  directory afterwards. Its exit status is 0 only when every step succeeded; a
  malformed configuration or a refused size estimate exits 3, a failed step
  exits 1. The job's own record is written to `<out>/pipeline-<rung>/run/`,
  outside every corpus. `--skip-existing` skips *generating* a corpus already
  marked `COMPLETE` (a relaunch on a box that still holds the output); such a
  corpus is still verified before it is uploaded, because verification against
  the private denylist is part of a corpus being called verified. Nothing
  private is compiled into the module: the denylist path, the staging directory
  and the dataset repository are arguments.
- **The xs checksum fixture** (`tests/fixtures/xs-checksums.json`, re-frozen
  2026-09-13 at tool version 0.2.3 for D-TB-17/18) pins the per-file sha256 of
  `xs/{latent,twin}/seed=0` — 70 and 74 files — generated from the shipped
  `configs/instances/xs.yaml`. The configuration path matters: `config_hash`
  covers the `constants` field as written in the file, so a rewritten copy of
  the config hashes differently, and the fixture is frozen from the shipped one
  with `python -m tracebench.manifest freeze`. `tests/test_byte_identity.py`
  regenerates the pair and compares; with
  `TRACEBENCH_XS_MANIFEST_DIR=<dir of pulled manifest.json files>` it compares
  manifests produced on another machine instead, which is the cross-machine
  half of PRD scenario 12. The freeze ran with `--workers 3` and the test
  regenerates with one worker, so the pair also pins sharding independence. A
  tool-version bump changes `instantiation.json` (it records the version) and
  therefore requires a re-freeze.
- **Local corpora belong outside synced folders**: a 26 GB rung × 5 seeds ×
  2 variants is not something to put under Dropbox; pass `--out` accordingly.

## Constants provenance

- **`constants/realism-v1.json`** (2026-09-10): 63 leaves fitted by the private
  fitter (`publications/trace-bench-calibration/`, workspace-private) from two
  sources — A: one 90-minute normalised window of the application-log and
  browser error-tracking feed of a consumer web platform (test environment;
  37,255 spans, 69 sequences); B: three hourly span files of a production
  data-centre trace archive (7.0 M spans), its operation vocabulary (1,781
  entries, 549 SERVER-kind) and 207 hourly build-stat rows — plus 21 `spec`
  leaves (structural design choices declared in the fitter's
  `mechanism-spec.json`: state-machine hazards, state multipliers, incident
  failure shares, client latent shares, and two platform facts the merged feed
  cannot expose). Every fitted leaf carries its sample count. Small samples:
  client-side error rates (10 events), health-check period (2 pods), browser
  skew (5 welded errors).
- **`constants/realism-dev.json`** is the labelled development placeholder that
  preceded the fit; tests and named instances now reference `realism-v1.json`.
- Known calibration limits: the fitted depth and fan-out describe a star-shaped
  reference system (96 % of root spans call nothing); the named instances keep
  depth and fan-out as configuration *targets* and the realism report states
  the deviation. The fitted error rates are own-class rates; a hop's realised
  rate also carries failures propagated from its callees, so the report scores
  leaf operations and reports the propagated totals beside them.

## Verification

- M0 (2026-09-10): config validation, RNG/identifier determinism, naming
  grammar, repository hygiene — see `tests/`.
- M1–M4 (2026-09-10): instantiation, mechanism graph and regimes, latent
  projection and scoring, twin target, byte-identical regeneration, resume,
  size cap, raw-feed defects, fault labels and the paired fault effect, and
  the forced-rerun ground-truth check (sampled).
- M5–M7 (2026-09-10): bundled correlator (four views in the consumer's
  column contract, correlation-loss report — xs on fitted constants:
  parent-link F1 0.999, unattributed 1 %, session Jaccard 0.96), realism
  report against the fitted constants (latency quantiles and leaf error rates
  within tolerance on xs), manifest write/verify, publish refusal on an
  existing version (stubbed host), family sampling with a disjoint split.
  `pytest tests/ -m "not slow"` is the gate; the slow xl alphabet check runs on
  demand.
- Publishing (2026-09-12): the upload path (hard-linked staging tree without
  `run/` or `artifacts.json`, duplicate refusal, `--replace`, the remote
  manifest post-check), the release path (remote verification against every
  manifest, a tampered size or hash refused, an existing tag refused, the card
  and index) and the size-only fallback when the host exposes no per-file hash
  are gated by `tests/test_manifest.py` against an in-memory dataset host;
  the per-rung job by `tests/test_pipeline.py` (step events, a failed
  verification stopping the job before any upload, exit statuses); byte
  identity against the frozen fixture by `tests/test_byte_identity.py`.
- M5 streaming (2026-09-10): the per-shard correlator reproduces the
  whole-corpus report on xs field for field (parent-link P/R/F1, attribution
  histograms, session recovery, orphans, normaliser counts).

## Run registry

| date | instance | variant | seed | command | tool | constants | where | verdict |
|---|---|---|---|---|---|---|---|---|
| 2026-09-10 | s | latent | 0 | `python -m tracebench.generate --config configs/instances/s.yaml --seed 0 --out <scratch> --workers 3` | 0.1.0 (uncommitted tree) | realism-v1 | local, macOS Apple silicon, 18 GB | complete: 96 shards, 6.0 GB (raw 3.5 / oracle 2.1 / views 0.4), 47 min wall / 81 CPU-min, driver RSS 6.05 GB; correlator parent-link F1 0.9994, unattributed 0.56 %, session Jaccard 1.00, 434,181 sessions, 1.49 M request rows; alphabet realized (train) 322 of 323 potential, vocab 326; orientation violations start 0.05 % / end 1.4 %; `verify` manifest ok, names ok; realism 17/20 scored items pass on 12 sampled shards (fails: BFF own-time p99 +11 %, `depth_pmf` 90 % deepest layer vs configured 15 %, step-gap p50 +28 %; see Implementation notes). Scratch run, superseded 2026-09-11 by the row below (one `log_source` value renamed). |
| 2026-09-11 | s | latent | 0 | `python -m tracebench.generate --config configs/instances/s.yaml --seed 0 --out ~/tracebench-corpora --workers 3` | 0.1.0 (uncommitted tree) | realism-v1 | local, macOS Apple silicon, 18 GB | complete: 96 shards, 6.0 GB, 50 min wall / 86 CPU-min, driver RSS 5.5 GB; parent-link F1 0.9994, unattributed 0.56 %; alphabet realized (train) 322, vocab 326, 3.86 M view rows; orientation violations start 0.05 % / end 1.4 %; `verify` manifest ok, names ok **including the private denylist**; realism 17/20 (same three misses as above). Pushed to the lab's private object store under `corpora/s/latent/seed=0`; also xs latent + twin (seed 0, same date, verify + denylist clean). Not frozen, not published. **Superseded 2026-09-11 by the D-TB-13 rows below**: removed from the object store, local copies kept aside as `seed=0.v01-superseded-26-09-11`. |
| 2026-09-11 | xs | latent + twin | 0 | `python -m tracebench.generate --config configs/instances/xs.yaml --seed 0 --out ~/tracebench-corpora [--twin]` | 0.1.0 (uncommitted, D-TB-13) | realism-v1 | local, macOS Apple silicon, 18 GB | complete: 4 shards each, 29 MB each, 55 s for both variants; realism 18/20 (request depth passes, TV 0.004, invoked fan-out 1.95; misses: external p95 −10 %, step-gap p50 +38 %; the external p95 is sampling noise in a steep tail — 1.1 resampling SE on 4,578 spans, the constant inside the item's 90 % bootstrap interval [3.77, 4.92] s, and external p50–p99 all pass at s); `check_mechanism --max-edges 40 --non-edges 8` on the latent variant: 39/40 sampled edges within 0.03 (the miss is the SLOW-mediated `F:9 → A:6:0` at 0.031, the D-TB-9 residual channel), non-edges 8/8 with a largest residual of 0.0012; `verify` manifest, names and private denylist clean. |
| 2026-09-11 | s | latent | 0 | `python -m tracebench.generate --config configs/instances/s.yaml --seed 0 --out ~/tracebench-corpora --workers 3` | 0.1.0 (uncommitted, D-TB-13) | realism-v1 | local, macOS Apple silicon, 18 GB | complete: 96 shards, 3.5 GB (estimate 4.2), 27.6 min wall with 3 workers; correlator parent-link F1 0.9991, unattributed 0.88 %, session Jaccard 1.00, 434,181 sessions, 3.83 M view rows; alphabet realized (train) 322 of 323 potential, vocab 323; realism 18/20 (request depth passes, TV 0.003; misses: BFF p95 +16 %, step-gap p50 +29 %); `verify` manifest, names and private denylist clean. Not frozen, not published. |
| 2026-09-12 | xs | latent + twin | 0 | `python -m tracebench.pipeline --config configs/instances/xs.yaml --seeds 0 --out <scratch> --workers 3 --denylist <private list>` | 0.2.0 | realism-v1 | local, macOS Apple silicon, 18 GB | complete: 4 shards per variant, 49 s for the whole job (generate 20.6 s + 17.0 s, verify 5.4 s + 5.7 s); `verify` manifest, names and private denylist clean on both variants (200k records scanned each). **Freeze source of `tests/fixtures/xs-checksums.json`** (70 files latent, 74 twin, config_hash `1656f43b4b6e5deb690a31574bd98e55`); regeneration with one worker reproduces every checksum. Not uploaded: the published xs corpora come from the cloud job over all five seeds. |
| 2026-09-12 | xs | latent + twin | 0..4 | `python -m tracebench.pipeline --config configs/instances/xs.yaml --seeds 0 1 2 3 4 --out data/corpora --workers 8 --denylist data/denylist.json --upload --stage-dir data/hf-stage` | 0.2.0 | realism-v1 | cloud, x86-64 Linux, 8 vCPU | complete: 10 corpora, 21 steps ok, 581 s total (40-55 s generate and ~11 s verify per corpus; the single upload of all ten, 292 MB, took 21.2 s); `verify` manifest, names and private denylist clean on all ten; published to the dataset host as `instances/xs/{latent,twin}/seed=0..4` (72 / 76 files each, no `run/`, no `artifacts.json`) and written back to the versioned object-store prefix (783 objects). **Cross-machine byte identity FAILED on 6 of 70 metadata files — see D-TB-15.** Superseded by 0.2.1; to be regenerated. |
| 2026-09-12 | xs | latent + twin | 0 | `python -m tracebench.pipeline --config configs/instances/xs.yaml --seeds 0 --out <scratch> --workers 3 --denylist <private list>` | 0.2.1 | realism-v1 | local, macOS Apple silicon, 18 GB | complete: 49 s for the job; `verify` manifest, names and private denylist clean on both variants. **Re-freeze source of `tests/fixtures/xs-checksums.json` after D-TB-15** (70 / 74 files, config_hash `1656f43b4b6e5deb690a31574bd98e55` — unchanged, the configuration did not move). |
| 2026-09-12 | xs | latent + twin | 0..4 | `python -m tracebench.pipeline --config configs/instances/xs.yaml --seeds 0 1 2 3 4 --out data/corpora --workers 8 --denylist data/denylist.json --upload --stage-dir data/hf-stage` | 0.2.1 | realism-v1 | cloud, x86-64 Linux, 8 vCPU | complete: 10 corpora, 21 steps ok, 577 s, upload 27.2 s; `verify` manifest, names and denylist clean on all ten; realised alphabet by seed 78 / 77 / 76 / 79 / 79. **Cross-machine byte identity PASSES** — all 70 sha256 of `latent/seed=0` and 74 of `twin/seed=0` match the fixture frozen on arm64 macOS, confirming D-TB-15. Published to the dataset host and the versioned object-store prefix. |
| 2026-09-12 | s | latent + twin | 0, 2, 3 | `python -m tracebench.pipeline --config configs/instances/s.yaml --seeds <k> --out data/corpora --workers 8 --denylist data/denylist.json --upload --stage-dir data/hf-stage` | 0.2.1 | realism-v1 | cloud, x86-64 Linux, 8 vCPU | complete: 2 corpora per job, 5 steps ok each. Wall clock per job 7,656–8,622 s (2.1–2.4 h): generate ≈ 3,778–4,218 s per corpus, verify only ~21 s, upload 64–297 s. 96 shards, realised alphabet 322 of 323 potential, identical `config_hash` across seeds. `verify` manifest, names and private denylist clean. **2.3× the plan's ≈ 1 h estimate** — EC2 is slower than the local Apple-silicon timing the estimate came from, and emission is single-core-bound; re-estimates m ≈ 4.6 h, l ≈ 6.9 h, xl ≈ 14–23 h against caps of 12 h and 36 h. |
| 2026-09-12 | s | latent | 1, 4 | same as above, `--seeds 1` / `--seeds 4` | 0.2.1 | realism-v1 | cloud, x86-64 Linux, 8 vCPU | **FAILED at shard 15 (both) and shard 28 (seed 1)** after 852 s and 906 s: `IndexError ... size 2839` — the spill allowance, see **D-TB-16**. Nothing was uploaded; the partial corpora reached the object store without a `COMPLETE` marker (196 and 202 objects), so they cannot be mistaken for whole ones. To be re-run at 0.2.2. |
| 2026-09-12 | xs | latent + twin | 0 | `python -m tracebench.pipeline --config configs/instances/xs.yaml --seeds 0 --out <scratch> --workers 3 --denylist <private list>` | 0.2.2 | realism-v1 | local, macOS Apple silicon, 18 GB | complete: 47 s; `verify` manifest, names and denylist clean. **Re-freeze source of `tests/fixtures/xs-checksums.json` after D-TB-16**; only `instantiation.json` differs from the 0.2.1 corpus (it records `tool_version`), `config_hash` unchanged. |
| 2026-09-12/13 | s | latent + twin | 1, 4 | `python -m tracebench.pipeline --config configs/instances/s.yaml --seeds <k> --out data/corpora --workers 8 --denylist data/denylist.json --upload --stage-dir data/hf-stage` | 0.2.2 | realism-v1 | cloud, x86-64 Linux, 8 vCPU | complete (the D-TB-16 re-runs): seed 1 9,528 s (generate 4,749 / 4,670 s, verify 23 / 24 s, upload 62 s); seed 4 10,424 s (5,223 / 5,076 s, 23 / 25 s, 77 s). Spill retries, one attempt each and identical in both variants: shard 15 needed 2,873 (seed 1) / 2,846 (seed 4) ticks against 2,839 held, shard 28 needed 3,375 (seed 1), and shard 87 needed 3,036 (seed 4 — a shard the 0.2.1 run never reached). 96 shards, realised alphabet 322 of 323, `config_hash` identical to seeds 0, 2, 3; `verify` manifest, names and private denylist clean; published to the dataset host and written back under the versioned prefix. **The `s` rung is complete: 10 of 10.** |
| 2026-09-12/13 | m | latent + twin | 3 | same shape, `--config configs/instances/m.yaml --seeds 3` | 0.2.2 | realism-v1 | cloud, x86-64 Linux, 8 vCPU | complete: **37,142 s (10.3 h)** — latent generate 29,656 s, of which 6 h 19 min passed before the first shard (**D-TB-17**), twin generate 7,241 s (its projection 2.5 min), verify 28 / 29 s, upload 188 s (13.4 GB, 2,856 files). 192 shards, realised alphabet 2,650 of 2,661 (the estimator expected 1,585), `config_hash d72cb6a3…`; request target 26,995 directed / 775,455 bidirected (24,633 / 13,470 at the floor), session 27,393 / 1,560,908; correlator unattributed 2.4 %, session Jaccard 1.00 over 694,444 sessions; `verify` clean; published and written back. |
| 2026-09-12/13 | m | latent | 0, 1, 2, 4 | same shape, `--seeds <k>` | 0.2.2 | realism-v1 | cloud, x86-64 Linux, 8 vCPU | **TERMINATED by the 12-hour runtime cap** (2026-09-13 11:18–11:20 UTC) still inside the latent projection: the log ends at the size estimate, no `graphs` event in 11 h 55 min, one of eight vCPUs busy throughout. Nothing synced to the object store, nothing uploaded. See D-TB-17; to be re-run at 0.2.3. |
| 2026-09-13 | l | latent | 0–4 | same shape, `--config configs/instances/l.yaml --seeds <k>` | 0.2.2 | realism-v1 | cloud, x86-64 Linux, 8 vCPU | **DESTROYED by the owner after 9.7 h** (11:31 UTC), all five still inside the latent projection at 2.9× the `m` work — they could not have finished inside the cap. Nothing synced, nothing uploaded. See D-TB-17; to be re-run at 0.2.3. |
| 2026-09-13 | xs | latent + twin | 0 | `python -m tracebench.pipeline --config configs/instances/xs.yaml --seeds 0 --out <scratch> --workers 3 --denylist <private list>` | 0.2.3 | realism-v1 | local, macOS Apple silicon, 18 GB | complete: 43 s; `verify` manifest, names and denylist clean. Against the 0.2.2 fixture, `instantiation.json` (tool version) and `topology/callgraph.json` (`derived` now `2026-01-05`, D-TB-18) differ; every other file, `config_hash` included, is identical. **Re-freeze source of `tests/fixtures/xs-checksums.json` at 0.2.3** (70 / 74 files). |
