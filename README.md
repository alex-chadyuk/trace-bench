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
suite (`pytest tests/ -m "not slow"`); no corpus has been frozen or published
yet. The package name is `tracebench`; every command runs as
`python -m tracebench.<command>`:

| command | what it does |
|---|---|
| `generate --config configs/instances/<rung>.yaml --seed <k> --out <dir> [--twin] [--workers N]` | estimate (refuses above the 50 GB cap), instantiate, write the graphs, simulate the shards, emit the raw feed and the oracle, run the correlator, write the manifest |
| `graphs` / `check_mechanism` / `realism` | mechanism graphs and scoring targets; forced-rerun ground-truth check; realism report against the fitted constants |
| `correlate --corpus <dir>` | the bundled correlator alone (four views, vocabulary, correlation-loss report) |
| `score --corpus <dir> --prediction <json>` | structural axes at every floor of the sweep; SID/AID on the twin |
| `verify --corpus <dir> [--denylist <private file>]` | manifest, name grammar and hygiene checks |
| `family`, `artifacts`, `publish` | family sampling with a held-out split; object-store push/pull; HuggingFace release |

Named instances (`configs/instances/`): `xs` (test fixture, seconds),
`s`, `m`, `l` (local, up to ~26 GB per corpus) and `xl` (cloud, ~35 GB,
> 10,000 expected realized tokens). Sizes and the deviation record are in
`RUN.md`.

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
- `RUN.md` — deviation record and run registry

## Licence

Code is MIT (see `LICENSE`). Published corpora are CC BY 4.0.
