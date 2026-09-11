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
- Rungs xs–l generate locally (CPU only). The top rung (xl) generates in the
  cloud under the lab's private execution configuration, which is not part of
  this repository.

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
  (`reports/mechanism-check.json:max_non_edge_residual`, 0.04 on xs).
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

- **D-TB-12 — rung windows are sized to the cap, not to the plan's budget
  table.** The named instances keep the planned service × endpoint counts but
  simulate shorter windows at lower rates than the plan's budget table
  assumed, because every operation must be reachable from a journey (below)
  and that pushes the effective fan-out — hence hops per session — well above
  the fitted mean. Sizes are estimates from `tracebench.estimate` at seed 0
  (the estimator matched the realised xs corpus within 10 %):

  | rung | services × endpoints + BFF | scenarios | window | rate | est. GB | est. CPU-h | expected realized tokens |
  |---|---|---|---|---|---|---|---|
  | xs | 3 × 3 + 4 | 2 | 1 h | 1 rps | 0.03 | 0.00 | 45 |
  | s | 10 × 5 + 8 | 4 | 24 h | 5 rps | 7.0 | 0.6 | 185 |
  | m | 50 × 10 + 20 | 8 | 24 h | 4 rps | 15.9 | 1.4 | 1,548 |
  | l | 150 × 10 + 30 | 12 | 24 h | 3 rps | 26.4 | 2.3 | 4,531 |
  | xl | 350 × 12 + 60 | 20 | 24 h | 3 rps | 35.1 | 3.1 | 12,554 |

  CPU-hours are single-core; `--workers` parallelises across shards. The xl
  rung exceeds the PRD's 8,000-token alphabet target by expectation; the
  manifest's `alphabet_size_realized_train` is the measured value. Calibration
  of the estimator on the realised s run (below): size within 15 % (6.0 GB
  realised vs 7.0 estimated), CPU time under-estimated 2.2× (1.36 CPU-h
  realised on Apple silicon), realized alphabet under-estimated (322 vs 185:
  the estimator counts tokens at the nominal latent context only, and
  incidents and load states realise more classes). Per-shard correlation ran
  at ~14 s per 15-minute shard single-threaded (22 min for s); peak resident
  memory of the driver was 6 GB.

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
- **Realism at the s rung (2026-09-10, 12 sampled shards of 96, fitted
  constants, `error_rate_multiplier: 10`).** Latency quantiles match at every
  tier except the BFF p99 (+11 %, p95 +9 %: the BFF's own time absorbs the
  callee retry back-off; unresolved), leaf error rates match the declared
  ×10 target, external p50–p99 match, session step-gap p90–p99 match while
  the p50 is +28 % (audited-response gaps include the next request's
  duration, and the external tier's heavy tail inflates the median;
  unresolved). **Request depth is the open realism gap:** 90 % of s requests
  reach the deepest layer against a configured pmf of `[0.5, 0.35, 0.15]`,
  because a request traverses every reachable callee and the reachability
  rule makes trees full. Restoring the configured depth needs a per-edge
  call probability (a callee invoked on a share of its caller's requests,
  with `p = 1 - stop^(1/(f^d·k))` from the pmf), which changes the mechanism's
  invoke tables, the engine's draws, the estimator and the alphabet — an
  owner decision, recorded as open in the PRD. Until then `depth_pmf` and
  `fanout_mean` are scored against the configuration targets (fitted values
  reported beside them) and `depth_pmf` fails at s.
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
- M5 streaming (2026-09-10): the per-shard correlator reproduces the
  whole-corpus report on xs field for field (parent-link P/R/F1, attribution
  histograms, session recovery, orphans, normaliser counts).

## Run registry

| date | instance | variant | seed | command | tool | constants | where | verdict |
|---|---|---|---|---|---|---|---|---|
| 2026-09-10 | s | latent | 0 | `python -m tracebench.generate --config configs/instances/s.yaml --seed 0 --out <scratch> --workers 3` | 0.1.0 (uncommitted tree) | realism-v1 | local, macOS Apple silicon, 18 GB | complete: 96 shards, 6.0 GB (raw 3.5 / oracle 2.1 / views 0.4), 47 min wall / 81 CPU-min, driver RSS 6.05 GB; correlator parent-link F1 0.9994, unattributed 0.56 %, session Jaccard 1.00, 434,181 sessions, 1.49 M request rows; alphabet realized (train) 322 of 323 potential, vocab 326; orientation violations start 0.05 % / end 1.4 %; `verify` manifest ok, names ok; realism 17/20 scored items pass on 12 sampled shards (fails: BFF own-time p99 +11 %, `depth_pmf` 90 % deepest layer vs configured 15 %, step-gap p50 +28 %; see Implementation notes). Scratch run, superseded 2026-09-11 by the row below (one `log_source` value renamed). |
| 2026-09-11 | s | latent | 0 | `python -m tracebench.generate --config configs/instances/s.yaml --seed 0 --out ~/tracebench-corpora --workers 3` | 0.1.0 (uncommitted tree) | realism-v1 | local, macOS Apple silicon, 18 GB | complete: 96 shards, 6.0 GB, 50 min wall / 86 CPU-min, driver RSS 5.5 GB; parent-link F1 0.9994, unattributed 0.56 %; alphabet realized (train) 322, vocab 326, 3.86 M view rows; orientation violations start 0.05 % / end 1.4 %; `verify` manifest ok, names ok **including the private denylist**; realism 17/20 (same three misses as above). Pushed to the lab's private object store under `corpora/s/latent/seed=0`; also xs latent + twin (seed 0, same date, verify + denylist clean). Not frozen, not published. |
