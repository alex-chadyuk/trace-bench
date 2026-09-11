"""Graph artifacts derived from an instantiation (read-only derivation).

Per variant (latent instance, observable twin) the derivation writes:

    graphs/mechanism-graph[.r<k>].json   one per regime (regimes.py)
    graphs/changepoints.json
    graphs/alphabet.json                  the token universe
    graphs/scoring-target.json            request-grain latent projection
    graphs/scoring-target-session.json    session-grain projection (may be cyclic)
    graphs/floor-sensitivity.json         the floor sweep over the request-grain target
    graphs/views/endpoint.json            coarsened views (functions of the target alone)
    graphs/views/service.json

Nothing here reads the raw feed; `python -m tracebench.graphs <corpus>` re-derives
every file from `instantiation.json`, `config.yaml` and `constants.json`.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .constants import (
    ALPHABET_JSON, CHANGEPOINTS_JSON, FLOOR_SENSITIVITY_JSON, GRAPHS_DIR, MECHANISM_GRAPH_JSON,
    SCORING_TARGET_JSON, VARIANT_TWIN, VIEW_ENDPOINT_JSON, VIEW_SERVICE_JSON,
)
from .instantiate import Instantiation, load_instantiation
from .log import log
from .mechanism import Mechanism
from .projection import build_alphabet, build_targets, coarsen, floor_sensitivity
from .record import RunRecord, write_json
from .regimes import apply_overlay, changed_edges, regime_intervals, validate_overlay

SCORING_TARGET_SESSION_JSON = "scoring-target-session.json"


def regime_mechanisms(inst: Instantiation):
    """[(regime_name, start_s, end_s, Mechanism)] — the base regime reuses the
    instantiation's mechanism; later regimes rebuild it under the overlay
    with the same topology, scenarios and latency model."""
    cfg = inst.cfg
    out = []
    intervals = regime_intervals(cfg)
    mech_by_name = {"base": inst.mechanism}
    for i, r in enumerate(cfg.schedules.regimes):
        validate_overlay(r.overlay, i)
        cfg_r, const_r = apply_overlay(cfg, inst.constants, r.overlay)
        mech_by_name[r.name] = Mechanism(cfg_r, const_r, inst.topo, inst.sset, inst.latency, inst.seed)
    for name, start, end in intervals:
        out.append((name, start, end, mech_by_name[name]))
    return out


def write_mechanism_graphs(inst: Instantiation, corpus_dir):
    """Writes graphs/mechanism-graph.json (base regime), one
    graphs/mechanism-graph.r<k>.json per later regime, and
    graphs/changepoints.json. Returns the list of (name, graph) pairs."""
    corpus_dir = Path(corpus_dir)
    gdir = corpus_dir / GRAPHS_DIR
    gdir.mkdir(parents=True, exist_ok=True)
    regimes = regime_mechanisms(inst)
    graphs = []
    cps = []
    for k, (name, start, end, mech) in enumerate(regimes):
        g = mech.graph_json()
        g["regime"] = {"name": name, "index": k, "start_s": start, "end_s": end}
        fname = MECHANISM_GRAPH_JSON if k == 0 else MECHANISM_GRAPH_JSON.replace(".json", f".r{k}.json")
        write_json(gdir / fname, g)
        graphs.append((name, g))
        if k > 0:
            prev = graphs[k - 1][1]["edges"]
            cps.append({
                "index": k, "name": name, "at_s": start,
                "changed_edges": changed_edges(prev, g["edges"], floor=inst.cfg.mechanism.floor),
            })
    write_json(gdir / CHANGEPOINTS_JSON, {
        "n_regimes": len(regimes),
        "regimes": [{"index": k, "name": n, "start_s": s, "end_s": e, "file": MECHANISM_GRAPH_JSON if k == 0 else MECHANISM_GRAPH_JSON.replace(".json", f".r{k}.json")}
                    for k, (n, s, e, _) in enumerate(regimes)],
        "changepoints": cps,
    })
    return graphs


def write_target_artifacts(inst: Instantiation, corpus_dir, variant):
    """Alphabet, scoring targets (both grains), floor sweep and views for one variant."""
    corpus_dir = Path(corpus_dir)
    gdir = corpus_dir / GRAPHS_DIR
    gdir.mkdir(parents=True, exist_ok=True)
    twin = variant == VARIANT_TWIN
    alphabet = build_alphabet(inst, twin=twin)
    write_json(gdir / ALPHABET_JSON, alphabet)
    req, ses, _ = build_targets(inst, twin=twin)
    write_json(gdir / SCORING_TARGET_JSON, req)
    write_json(gdir / SCORING_TARGET_SESSION_JSON, ses)
    write_json(gdir / FLOOR_SENSITIVITY_JSON, floor_sensitivity(req, inst.cfg.mechanism.sweep))
    write_json(gdir / VIEW_ENDPOINT_JSON, coarsen(req, "endpoint", inst.topo))
    write_json(gdir / VIEW_SERVICE_JSON, coarsen(req, "service", inst.topo))
    return {"alphabet_size": alphabet["n_tokens"],
            "request": {k: req[k] for k in ("n_directed", "n_bidirected", "n_directed_at_floor", "n_bidirected_at_floor", "directed_acyclic_at_floor")},
            "session": {k: ses[k] for k in ("n_directed", "n_bidirected", "n_directed_at_floor", "n_bidirected_at_floor", "directed_acyclic_at_floor")}}


def write_graph_artifacts(inst: Instantiation, corpus_dir, variant):
    graphs = write_mechanism_graphs(inst, corpus_dir)
    summary = write_target_artifacts(inst, corpus_dir, variant)
    summary["n_regimes"] = len(graphs)
    summary["mechanism_nodes"] = graphs[0][1]["n_nodes"]
    summary["mechanism_edges"] = graphs[0][1]["n_edges"]
    return summary


def build_parser():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--corpus", required=True, help="corpus directory holding instantiation.json, config.yaml, constants.json")
    p.add_argument("--variant", required=True, choices=["latent", "twin"], help="which variant's targets to derive")
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    inst = load_instantiation(args.corpus)
    rec = RunRecord(args.corpus, "graphs", vars(args))
    summary = write_graph_artifacts(inst, args.corpus, args.variant)
    log({"event": "graphs", **summary})
    rec.finish(summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
