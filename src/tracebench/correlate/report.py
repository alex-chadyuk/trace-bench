"""Correlation loss, measured against the oracle linkage, accumulated shard by shard."""
from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path

import pyarrow.parquet as pq

from ..constants import ATTR_NAMES, ATTR_NONE
from ..shards import shard_name


def _prf(c):
    tp, fp, fn = c["tp"], c["fp"], c["fn"]
    p = tp / (tp + fp) if tp + fp else None
    r = tp / (tp + fn) if tp + fn else None
    f1 = (2 * p * r / (p + r)) if p and r else (0.0 if p is not None and r is not None else None)
    return {"precision": p, "recall": r, "f1": f1, "tp": tp, "fp": fp, "fn": fn, "tn": c["tn"]}


class ReportAccumulator:
    """Parent-link confusion (per hop kind), attribution histograms and session
    recovery. The oracle shard k holds exactly the spans of the raw shard k, so
    the parent-link comparison is shard-local; session recovery is scored when
    a reconstructed sequence closes, against the true session id each of its
    hop members carries."""

    def __init__(self):
        self.counts = defaultdict(Counter)
        self.levels = Counter()
        self.seq_levels = Counter()
        self.n_spans = 0
        self.n_hops_of = Counter()      # true session -> oracle hops
        self.best = {}                  # true session -> (hops in its best sequence, hops in that sequence)
        self.n_seqs_of = Counter()      # true session -> sequences holding at least one of its hops
        self.n_sequences = 0
        self.merged_sequences = 0
        self.client_links = 0
        self.client_errors = 0
        self.orphans = 0
        self.norm = Counter()

    def add_shard(self, corpus_dir, shard, spans, trees, attribution, link, norm_stats):
        """Returns span_id -> true session id for the shard's oracle spans."""
        t = pq.read_table(Path(corpus_dir) / "oracle" / shard_name(shard) / "spans.parquet",
                          columns=["span_id", "parent_span_id", "session_gid", "is_edge"]).to_pydict()
        inferred = {h["span_id"]: h["parent"] for tree in trees.values() for h in tree["hops"]}
        gid_of = {}
        for sid, tp, g, e in zip(t["span_id"], t["parent_span_id"], t["session_gid"], t["is_edge"]):
            kind = "edge" if e else "internal"
            ip = inferred.get(sid)
            c = self.counts[kind]
            if tp is None and ip is None:
                c["tn"] += 1
            elif tp is None:
                c["fp"] += 1
            elif ip is None:
                c["fn"] += 1
            elif tp == ip:
                c["tp"] += 1
            else:
                c["fp"] += 1
                c["fn"] += 1
            if g is not None and g >= 0:
                gid_of[sid] = g
                self.n_hops_of[g] += 1
        self.levels.update(attribution.get(s["span_id"], (ATTR_NONE, None))[0] for s in spans)
        self.n_spans += len(spans)
        self.client_errors += sum(1 for s in spans if s["kind"] == "sentry.error")
        self.client_links += sum(1 for s in spans if s["kind"] == "sentry.error" and s["span_id"] in link)
        self.orphans += sum(1 for tree in trees.values() for h in tree["hops"] if not h["is_edge"] and h["parent"] is None)
        for k, v in norm_stats.items():
            if isinstance(v, (int, float)):
                self.norm[k] += v
        return gid_of

    def add_sequence(self, seq):
        self.n_sequences += 1
        self.seq_levels[seq["level"]] += 1
        per_gid = Counter(m[3] for m in seq["members"] if m[3] is not None)
        seq_hops = sum(per_gid.values())
        if len(per_gid) > 1:
            self.merged_sequences += 1
        for g, c in per_gid.items():
            self.n_seqs_of[g] += 1
            if c > self.best.get(g, (0, 0))[0]:
                self.best[g] = (c, seq_hops)

    def finalize(self):
        parent_link = {k: _prf(v) for k, v in sorted(self.counts.items())}
        allc = Counter()
        for v in self.counts.values():
            allc.update(v)
        parent_link["all"] = _prf(allc)
        jacc = []
        for g, n in self.n_hops_of.items():
            c, seq_hops = self.best.get(g, (0, 0))
            jacc.append(c / (n + seq_hops - c) if c else 0.0)
        n_sess = len(self.n_hops_of)
        split_cnt = sum(1 for g in self.n_hops_of if self.n_seqs_of[g] > 1)
        return {
            "description": "correlation loss of the bundled correlator measured against the oracle linkage (reported quantities, not pass thresholds)",
            "parent_link": parent_link,
            "unattributed_fraction": self.levels.get(ATTR_NONE, 0) / max(self.n_spans, 1),
            "identity_level_histogram_spans": {ATTR_NAMES[k]: v for k, v in sorted(self.levels.items())},
            "identity_level_histogram_sequences": {ATTR_NAMES[k]: v for k, v in sorted(self.seq_levels.items())},
            "session_recovery": {"n_true_sessions": n_sess, "mean_jaccard": (sum(jacc) / len(jacc)) if jacc else None,
                                 "split_rate": split_cnt / max(n_sess, 1),
                                 "merge_rate": self.merged_sequences / max(self.n_sequences, 1)},
            "client_error_linked_fraction": self.client_links / max(self.client_errors, 1),
            "orphan_internal_hops": self.orphans,
            **dict(self.norm),
        }
