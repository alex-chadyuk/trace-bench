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
