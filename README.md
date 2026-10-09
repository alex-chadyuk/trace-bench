# trace-bench

A simulated microservice trace benchmark that ships the causal mechanism that
generated it. One configuration produces a corpus of realistically-shaped
request traces at pretraining scale, the mechanism graph over every state
variable the simulation carries (latent ones flagged), its latent projection
over observable `(operation, outcome)` event types as the scoring target, a
deployment call topology usable as a structural prior, injected faults with
root-cause labels, and a bundled correlator whose loss is measured against the
simulator's own linkage.

Ground truth is emitted by construction, never inferred from the output.

## Status

Milestones M0–M7 of the implementation plan are built and gated by the test
suite (`pytest tests/ -m "not slow"`). Byte-identical regeneration of the `xs`
rung is pinned by a committed checksum fixture
(`tests/fixtures/xs-checksums.json`); the corpora of the other rungs are
generated per (rung, seed) and published to the dataset host with the two
commands below. Scoring-target strengths are exact marginalisations where the
projection's frontier fits the tool's particle cap and counter-keyed
Monte-Carlo estimates above it, flagged per edge (`RUN.md`, D-TB-19). The package name is `tracebench`; every command runs as
`python -m tracebench.<command>`:

| command | what it does |
|---|---|
| `generate --config configs/instances/<rung>.yaml --seed <k> --out <dir> [--variant latent\|twin] [--workers N]` | estimate (refuses above the 50 GB cap), instantiate, write the graphs, simulate the shards, emit the raw feed and the oracle, run the correlator, write the manifest |
| `derive --from <latent corpus> --variant metrics --out <dir> [--resume]` | the metrics variant from a complete latent corpus (D-TB-21): hard-link its files, recompute the latent chain, sample it onto the 30 s scrape grid into `metrics/`, keep the tick-level change log in `oracle/state/`, re-derive the ground truth under the partial exposure, write the manifest |
| `graphs` / `check_mechanism` / `realism` | mechanism graphs and scoring targets; forced-rerun ground-truth check; realism report against the fitted constants |
| `correlate --corpus <dir>` | the bundled correlator alone (four views, vocabulary, correlation-loss report) |
| `score --corpus <dir> --prediction <json>` | structural axes at every floor of the sweep; SID/AID on the twin (not applicable, with the hidden groups named, elsewhere) |
| `verify --corpus <dir> [--denylist <private file>]` | manifest, name grammar and hygiene checks |
| `manifest write \| verify \| freeze` | write or re-check a corpus manifest; freeze the committed per-file checksum fixture |
| `pipeline --config <rung> --seeds 0 [1 2 3 4] --out <dir> [--variants latent twin metrics] [--denylist <f>] [--upload] [--fetch-from <repo> --fetch-revision <tag>]` | one unattended job for one rung: per seed generate the latent instance and its twin, derive the metrics variant, verify each, then upload them together — one `pipeline_step` line per step, and it stops at the first failure; a metrics-only job fetches the seed's latent corpus from the host first |
| `publish upload --corpus <dir> ... \| --corpus-root <dir>` | put complete corpora on the dataset host at `instances/<instance>/<variant>/seed=<k>`, through a hard-linked staging tree (resumable; refuses a path that is already there unless `--replace`) |
| `publish release --version vX.Y.Z` | verify every remote corpus against its own manifest, then write the card, the licence and `release.json` and tag the dataset (refuses an existing tag) |
| `publish fetch --corpus-key <instance>/<variant>/seed=<k> --out <dir> [--revision <tag>]` | pull one corpus from the host, verified against its own manifest before it lands |
| `family`, `artifacts` | family sampling with a held-out split; object-store push/pull |

Corpora are versioned by the **tool version** that generated them — every
`manifest.json` names it, and the checksum fixture is frozen against it — while
a **release is a tag** over corpora already uploaded. So many independent jobs
can accumulate the corpora of one release, and the release step is the single
writer that freezes and verifies what the host holds.

Named instances (`configs/instances/`): `xs` (test fixture, seconds),
`s`, `m`, `l` (local, up to ~8 GB per corpus) and `xl` (cloud, ~17 GB,
> 10,000 expected realized tokens). Sizes and the deviation record are in
`RUN.md`.

Three variants per instance and seed, one simulation: `latent` (no state
variable observable), `twin` (every latent observed; the causal-validity axis
is defined), and `metrics` (the latent corpus plus a `metrics/` channel of
30-second categorical time series of traffic intensity, per-service load and
pool state and per-endpoint health, with cache contents and per-session network
and auth state hidden — what a real cluster exports; derived, not re-simulated;
D-TB-21). A method may read `raw/`, `views/` and `metrics/`.

## Install

```bash
conda env create -f environment.yml
conda activate trace-bench
pip install -e ".[dev,sid,publish]"   # sid: SID/AID on the observable twin; publish: HuggingFace releases
pytest tests/
```

## Layout

- `src/tracebench/` — the generator, correlator, scorer and publisher
- `configs/instances/{xs,s,m,l,xl}.yaml` — the named, citable instances
- `constants/realism-v<N>.json` — fitted realism constants (aggregates only; the fitter reads private sources and is not part of this repository)
- `tests/` — one test per acceptance scenario of the PRD
- a corpus: `raw/` (the feed), `views/` (correlated sequences), `metrics/` (the metrics variant's channel), `graphs/` (ground truth), `labels/`, `oracle/` (linkage, true spans and sessions; `oracle/state/` the tick-level latent change log of the metrics variant), `topology/`, `manifest.json`, `COMPLETE`
- `RUN.md` — deviation record and run registry

## Licence

Code is MIT (see `LICENSE`). Published corpora are CC BY 4.0.
