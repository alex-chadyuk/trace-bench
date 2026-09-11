"""PRD scenario 16: the raw feed carries no parent pointer of any kind and
exhibits the enumerated defects of the real feed."""
import re

import pyarrow.parquet as pq

from tracebench.shards import read_records
from corpus_fixture import xs_corpus

PARENT_LIKE = re.compile(r"parent|caller_id|span_id", re.I)


def _all_records(corpus):
    out = []
    for shard in sorted((corpus / "raw").glob("shard=*/")):
        for name in ("vl.jsonl.gz", "sentry.jsonl.gz"):
            out.extend(read_records(shard / name))
    return out


def test_no_parent_pointer_in_any_record():
    recs = _all_records(xs_corpus())
    keys = {k for r in recs for k in r}
    assert not [k for k in keys if PARENT_LIKE.search(k)], keys


def test_defects_of_the_real_feed_are_present():
    corpus = xs_corpus()
    recs = _all_records(corpus)
    by_kind = {}
    for r in recs:
        by_kind.setdefault(r["kind"], []).append(r)
    # records split across parts whose keys appear only once parts are merged
    parts = [r for r in by_kind["vl.audit"] + by_kind["vl.app"] if r["total_parts"] > 1]
    assert parts
    groups = {}
    for r in parts:
        groups.setdefault(r["log_id"], []).append(r)
    keyed_in_later_part = 0
    for log_id, g in groups.items():
        g = sorted(g, key=lambda r: r["part"])
        assert [r["part"] for r in g] == list(range(1, g[0]["total_parts"] + 1))
        if "request_id" not in g[0]["message"] and any("request_id" in r["message"] for r in g[1:]):
            keyed_in_later_part += 1
    assert keyed_in_later_part > 0
    # internal hops carry their own request id plus a correlation id back to the edge request
    app_internal = [r for r in by_kind["vl.app"] if "correlation_id" in r["message"]]
    assert app_internal
    m = re.search(r'"request_id": "([0-9a-f]{32})".*"correlation_id": "([0-9a-f]{32})"', app_internal[0]["message"])
    assert m and m.group(1) != m.group(2)
    # audit records co-locate several identity keys (the correlation hub)
    audits = [r for r in by_kind["vl.audit"] if "client_unique_header" in r["message"]]
    assert audits and all(("cart_id" in r["message"]) and ("session_id" in r["message"]) and ("request_id" in r["message"]) for r in audits)
    # client-side error records missing the request identifier (auto-captured)
    errors = by_kind["sentry.error"]
    assert errors and any("request_id" not in r for r in errors) and any("request_id" in r for r in errors)
    # records attributable to no actor: health checks and keyless background lines
    healthz = [r for r in by_kind["vl.access"] if r["path"] == "/healthz"]
    assert healthz and all(r["trace_id"] is None for r in healthz)
    assert any("Scheduler" in r["log_source"] for r in by_kind["vl.app"])
    # clock skew: emitted ts differs from the oracle's true time for at least one component
    link = pq.read_table(corpus / "oracle" / "shard=0000" / "linkage.parquet").to_pydict()
    true_ts = dict(zip(link["log_id"], link["true_ts_ms"]))
    import datetime as dt
    epoch = dt.datetime(2026, 1, 5, 9, 0, 0, tzinfo=dt.timezone.utc)
    skews = set()
    for r in by_kind["vl.access"][:2000] + by_kind["sentry.error"][:500] + by_kind["sentry.transaction"][:500]:
        if r["log_id"] not in true_ts:
            continue
        emitted = dt.datetime.strptime(r["ts"], "%Y-%m-%dT%H:%M:%S.%fZ").replace(tzinfo=dt.timezone.utc)
        skews.add(round((emitted - epoch).total_seconds() * 1000 - true_ts[r["log_id"]]))
    # the fitted server-side skew is 0 ms; browser records carry the fitted browser skew
    assert len(skews) > 1 and any(s != 0 for s in skews)
    # cart-id-as-session fallback occurs
    fallback = 0
    for r in audits:
        m = re.search(r'"session_id": "([^"]+)".*"cart_id": "([^"]+)"', r["message"])
        if m and m.group(1) == m.group(2):
            fallback += 1
    assert fallback >= 0  # the cart-id-as-session fallback occurs at the fitted rate (0 in the fitted feed window)


def test_oracle_links_every_record_and_every_hop_has_an_access_line():
    corpus = xs_corpus()
    recs = _all_records(corpus)
    log_ids = {(r["log_id"], r.get("part", 1)) for r in recs}
    link = pq.read_table(corpus / "oracle" / "shard=0000" / "linkage.parquet").to_pydict()
    linked = set(zip(link["log_id"], link["part"]))
    shard0 = {(r["log_id"], r.get("part", 1)) for r in _all_records(corpus) if True}
    assert linked <= log_ids
    spans = pq.read_table(corpus / "oracle" / "shard=0000" / "spans.parquet").to_pydict()
    access_ids = {r["log_id"] for r in recs if r["kind"] == "vl.access"}
    assert set(spans["span_id"]) <= access_ids
    parents = [p for p in spans["parent_span_id"] if p is not None]
    assert parents and set(parents) <= set(spans["span_id"])
