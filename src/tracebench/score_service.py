"""Service-level graph axis (docs/scorers-service-rca.md, 2026-10-09).

Scores a prediction over *services* against the shipped service-level view of the request-grain
target (`graphs/views/service.json`: the token target coarsened to `svc:<index>` nodes, the
largest member strength per pair) with an explicit universe: every ordered pair of distinct
deployed services (the BFF, backend and external services of `instantiation.json`; the synthetic
client is not a deployed service and no request-grain target edge touches it). Metrics, floor and
conventions are `tracebench.score`'s: a listed edge is present, `score` ranks edges for the
threshold-free axes, pairs outside the universe are counted and ignored, SID / AID are not
computed on this axis. The built-in reference is the deployment topology coarsened to services —
callee → caller, score = the largest `p_call` among the op-level call edges of the pair — the
service-level counterpart of the token-level topology floor.

    python -m tracebench.score_service --corpus <dir> --prediction <json | topology> --out <json>

Prediction shape: `{"directed": [{"src": "svc:<i>", "dst": "svc:<j>", "score": s}, ...],
"bidirected": [{"a": ..., "b": ..., "score": s}, ...]}`.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .constants import ALPHABET_JSON, GRAPHS_DIR, INSTANTIATION_JSON, KIND_CLIENT, VIEW_SERVICE_JSON
from .log import log
from .record import read_json, write_json
from .score import score_at_floor

LEVEL = "service"
TOPOLOGY_REFERENCE = "topology"


def service_key(index):
    return f"svc:{int(index)}"


def load_view(corpus_dir):
    view = read_json(Path(corpus_dir) / GRAPHS_DIR / VIEW_SERVICE_JSON)
    if view.get("level") != LEVEL:
        raise ValueError(f"{VIEW_SERVICE_JSON} is a {view.get('level')!r} view, not {LEVEL!r}")
    return view


def deployed_services(corpus_dir):
    """`[(index, name, kind)]` of the deployed services: every kind but the client."""
    inst = read_json(Path(corpus_dir) / INSTANTIATION_JSON)["topology"]
    return [(int(s["index"]), str(s["name"]), int(s["kind"])) for s in inst["services"] if int(s["kind"]) != KIND_CLIENT]


def service_universe(services):
    """Every ordered pair of distinct deployed services, sorted as the sparse scorer needs."""
    keys = sorted(service_key(i) for i, _, _ in services)
    ordered = sorted((a, b) for a in keys for b in keys if a != b)
    unordered = sorted({(a, b) if a < b else (b, a) for a, b in ordered})
    return ordered, unordered


def topology_reference(corpus_dir):
    """The deployment topology as a service-level prediction: callee → caller per call edge, score =
    the largest `p_call` of the pair's op-level edges; within-service calls dropped."""
    topo = read_json(Path(corpus_dir) / INSTANTIATION_JSON)["topology"]
    svc_of = {int(o["id"]): int(o["service"]) for o in topo["ops"]}
    best = {}
    for e in topo["edges"]:
        a, b = svc_of[int(e["callee"])], svc_of[int(e["caller"])]
        if a == b:
            continue
        key = (service_key(a), service_key(b))
        best[key] = max(best.get(key, 0.0), float(e.get("p_call", 1.0)))
    return {"directed": [{"src": a, "dst": b, "score": p} for (a, b), p in sorted(best.items())],
            "bidirected": [], "reference": TOPOLOGY_REFERENCE}


def score_service(corpus_dir, prediction, floor=None):
    corpus_dir = Path(corpus_dir)
    view = load_view(corpus_dir)
    services = deployed_services(corpus_dir)
    ordered, unordered = service_universe(services)
    alphabet = read_json(corpus_dir / GRAPHS_DIR / ALPHABET_JSON)
    floor = view["default_floor"] if floor is None else float(floor)
    result = score_at_floor(view, alphabet, prediction, floor, ordered=ordered, unordered=unordered)
    result["level"] = LEVEL
    result["grain"] = view.get("grain")
    result["universe"]["services"] = [{"key": service_key(i), "name": n, "kind": k} for i, n, k in services]
    result["universe"]["n_services"] = len(services)
    result["reference"] = prediction.get("reference")
    return result


def build_parser():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--corpus", required=True)
    p.add_argument("--prediction", required=True,
                   help=f"a service-level prediction JSON, or '{TOPOLOGY_REFERENCE}' for the built-in reference")
    p.add_argument("--floor", type=float, default=None, help="override the view's default floor")
    p.add_argument("--out", required=True)
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    if args.prediction == TOPOLOGY_REFERENCE:
        prediction = topology_reference(args.corpus)
    else:
        prediction = json.loads(Path(args.prediction).read_text(encoding="utf-8"))
    result = score_service(args.corpus, prediction, floor=args.floor)
    write_json(args.out, result)
    log({"event": "score_service", "level": LEVEL, "f1_directed": result["directed"]["f1"],
         "n_services": result["universe"]["n_services"],
         "predictions_outside_universe": result["universe"]["predictions_outside_universe"]})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
