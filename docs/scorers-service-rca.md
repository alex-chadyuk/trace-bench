# Score-side scorers for the root-cause arms (pre-registered 2026-10-09, before the first root-cause run)

Two scorers beside `tracebench.score`, for the dataset paper's root-cause arms (its plan §3; the
harness that runs those arms lives in the method author's repository). Neither touches a corpus,
the token-level target or `tracebench.score`; both read shipped files only. Changes are dated
addenda at the end of this file, never edits in place.

## 1. `tracebench.score_service` — the service-level graph axis

- **Target.** `graphs/views/service.json`: the request-grain token target coarsened to
  `svc:<index>` nodes (the largest member strength per ordered pair; `projection.coarsen`).
- **Universe.** Every ordered pair of distinct **deployed** services — the BFF, backend and
  external services of `instantiation.json`; the synthetic client is not a deployed service and no
  request-grain target edge touches it — passed explicitly to `score_at_floor` (`ordered`,
  `unordered`), so a prediction over services is scored like a prediction over tokens: a listed
  edge is present, `score` ranks edges for AUROC / AP, pairs outside the universe are counted
  (`universe.predictions_outside_universe`) and ignored, the floor is the view's `default_floor`
  (0.05). SID / AID are not computed on this axis (the view is a function of the ADMG target).
- **Reference.** The deployment topology coarsened to services: callee → caller per call edge,
  score = the largest `p_call` among the op-level edges of the pair, within-service calls dropped
  (`--prediction topology`). It is the service-level counterpart of the token-level topology
  floor and is reported in every service-level table.
- **Prediction shape.** `{"directed": [{"src": "svc:<i>", "dst": "svc:<j>", "score": s}],
  "bidirected": [{"a": …, "b": …, "score": s}]}`; a method's output over service names is mapped
  to indices through `instantiation.json`'s `services` by the caller.

## 2. `tracebench.score_rca` — the ranking axis

- **Labels.** `labels/cases.json`: one case per injected fault, `root_cause_component` = a
  service name.
- **Input.** One ranking per case, best first (`{"cases": {"<case_id>": ["<component>", …]}}`).
  A case without a ranking, or whose root cause is absent from it, is a miss at every k and is
  listed (`n_missing`, `cases[].missing`).
- **Metrics.** `AC@k` for k ∈ {1, 3, 5} = the share of cases whose root cause is among the first
  k entries; `Avg@5` = the mean of `AC@1 … AC@5`. Per fault kind as well.
- **Uncertainty.** A percentile bootstrap over cases (resampled with replacement, `--bootstrap-b`
  draws from `numpy.random.default_rng(--bootstrap-seed)`, 95 % interval) on every metric; the
  draw count and seed are recorded with the result. Registered values for the paper: B = 2000,
  seed = 0.
- **Random expectation.** A uniformly random ranking of the n candidate components places the
  root cause among the first k with probability min(k, n) / n; n = the deployed services of the
  corpus (clients excluded), so the expectation is analytic and reported beside every table.
- **Pooling.** `--pool` concatenates the cases of several per-corpus results (the five seeds of a
  rung, or every rung) into one table with its own bootstrap; the random expectation is pooled
  when every result shares one n and listed per result otherwise.

## 3. Pins and identity

The two modules are additive. `tracebench.score` is not changed by them; the version a consumer
pins is the consumer's record (the reference harness pins commit `8e91aa2`, tag `v0.3.0-score2`;
the paper's root-cause tables name the tag they were scored under).

## 4. `tracebench.score_rca_trace` — the per-trace axis (addendum 2026-10-31, before the first per-trace read)

A third score-side scorer for the dataset paper's root-cause arms (its plan of 2026-10-31, approved
by the owner the same day): the root-cause question asked per **failed request** rather than per
fault case. Additive like §1–§2; `tracebench.score`, `score_rca`, `score_service` and `export_rca`
are not changed by it.

- **Labels.** `labels/sequence-labels.parquet` (`correlate.views.LABELS_SCHEMA`; one row per request
  trace and injected fault): `phase` (PRE 0 / POST 1 / STRADDLE 2 / NONE −1 against the fault
  window), `rc_positions_end` (the end-ordered request row's positions whose operation belongs to the
  fault's service and endpoints), `rc_took` (one of them has any non-OK outcome, 4xx included).
  Beside them `labels/cases.json` (the root-cause service) and the end-ordered request view's rows
  (`ops`, `outcomes`, `offsets`, `durations`, `parent_pos`, `start_ns`) and `model-vocab.json`.
- **Selection (rule R, label-free, recomputed here and refused on a mismatch with the method side's
  record).** Per case, every request trace of the view, all splits, wholly inside the fault window
  (`inject · 1e9 ≤ start_ns` and `start_ns + max(offsets + durations) < end · 1e9`, integer ns) with
  at least one failed non-client span (`outcomes` in {5xx, err, slow}); ordered by `start_ns`, then
  `trace_id`; cap 0 = all. **Scored set** = the selected traces in which the fault took: a
  root-cause position with an outcome in {5xx, err, slow}, recomputed from the view row (the label's
  `rc_took` also counts 4xx; took ⇒ `rc_took` is asserted). A selected trace the method skipped is a
  miss.
- **Input.** Mode `arm`: per case and probed trace the method's ranked **reached** services (indices
  into the name-ordered deployed services) and ranked window positions — the harness's
  `ranking-trace.json`, whose header carries the cell's context `c`, length cap and the selected and
  skipped ids. Mode `broadcast`: a case-level ranking in `score_rca`'s shape (or a list of per-run
  rankings), filtered to the trace's own services with the order kept, absent services appended in
  name order, an unranked case a miss, a failed run 0. Modes `topology` (the rule on the true call
  tree: the descendants of every failed non-client span by `parent_pos`, ranked by the number of
  failed spans whose subtree holds the service, then the smallest hop, then name order), `random`
  (analytic) and `callees` (analytic, over the failed spans' descendant services) take no input.
  `hop0` adds a failed span itself at hop 0 (`arm` reads the method's `hop0` lists).
- **Metrics.** Per trace: hit@1 and hit@3 of the root-cause service among the trace's distinct
  non-client services; of any root-cause position among the positions (an arm's window position is
  `i − c + 1`, reachable for `c − 1 ≤ i < min(len(ops), 64)`; unreachable positions are counted and
  are misses); the **exact set** = 1 iff the method's reached-service set equals {root-cause
  service}, with its size (trace-reading modes only). Per case the means over its scored traces;
  **the reported point is the mean over cases** (the ranking axis's unit). Per fault kind; per split
  (train / val+test) trace-weighted, as the training-overlap audit.
- **Uncertainty.** A percentile bootstrap over cases of the case-mean rates (`numpy.random
  .default_rng(seed).integers(0, n, (b, n))`, 95 %), the same draws as §2's; registered values B =
  2000, seed 0. `pool` concatenates the case rows of several results (the five seeds of a rung, or
  every rung), so the resampling unit stays the (corpus, case) pair.
- **Random expectation per trace.** Services: min(k, n_t) / n_t over the trace's n_t distinct
  non-client services; positions: 1 − C(m − r, k′) / C(m, k′) with k′ = min(k, m), m the candidate
  positions and r the reachable root-cause positions (0 when r = 0); reported as the mean over cases
  of the per-case means.
- **Pins and identity.** The reference harness's root-cause sibling moves its pin to the `clear-2027`
  commit carrying this module (its D-TBR-14); the three earlier scorers are byte-identical at that
  commit.
