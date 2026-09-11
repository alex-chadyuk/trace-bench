"""Expected-count estimates from an instantiation (no simulation).

Used for the alphabet-size statement in the run record (PRD scenario 17) and
by the size estimator that guards the corpus cap (scenario 22). Everything
here is an expectation under the nominal mechanism; the realised counts are
recorded in the manifest after generation.
"""
from __future__ import annotations

from .constants import KIND_BFF, KIND_CLIENT, KIND_EXTERNAL, KIND_SERVICE, T_VALUES
from .intensity import expected_sessions


def expected_invocations(inst):
    """Expected number of invocations per op over the window and the expected
    attempts per invocation, from journey structure and the nominal mechanism.
    Returns (inv: {op_id: float}, attempts_per_invocation: {op_id: float},
    sessions: float)."""
    cfg, topo, sset, mech = inst.cfg, inst.topo, inst.sset, inst.mechanism
    sessions = expected_sessions(cfg, inst.constants)
    inv = {op.id: 0.0 for op in topo.ops}
    attempts = {op.id: 1.0 for op in topo.ops}
    # BFF steps: P(reach step j) = prod_{i<j} P(step i ends ok)
    for sc in sset.scenarios:
        p_reach = 1.0
        for st in sc.steps:
            tnode = mech.nodes[f"T:bff:{sc.index}:{st.index}:0"]
            d = tnode.dist(tnode.nominal_context(mech))
            p_ok = float(d[0] + d[4])
            n_att = 1.0
            p_retry_eligible = sum(float(d[T_VALUES.index(o)]) for o in sc.retry_on)
            for k in range(1, sc.max_retries + 1):
                n_att += (p_retry_eligible * mech.p_retry) ** k
            inv[st.bff_op] += sessions * sc.weight * p_reach * n_att
            inv[st.client_op] += sessions * sc.weight * p_reach * n_att
            attempts[st.bff_op] = n_att
            p_reach *= p_ok
    # backend ops: forward propagation of reach probability below each BFF op
    for b in topo.bff_ops:
        p = {b: 1.0}
        for v in topo.reachable_from(b)[1:]:
            p_absent = 1.0
            for e in topo.caller_edges(v):
                pu = p.get(e.caller, 0.0)
                miss = 1.0 - (e.p_hit if e.cached else 0.0)
                p_absent *= 1.0 - pu * miss
            p[v] = 1.0 - p_absent
        for v, pv in p.items():
            if v != b:
                inv[v] += inv[b] * pv
    retry_on = set(mech.backend_retry_on)
    for op in topo.ops:
        if op.kind in (KIND_SERVICE, KIND_EXTERNAL):
            a0 = mech.nodes[f"A:{op.id}:0"]
            d = a0.dist(a0.nominal_context(mech))
            p_el = sum(float(d[T_VALUES.index(o)]) for o in retry_on)
            n_att = 1.0
            for k in range(1, mech.backend_retries + 1):
                n_att += (p_el * mech.p_retry) ** k
            attempts[op.id] = n_att
    return inv, attempts, sessions


def expected_token_counts(inst):
    """{(op_id, outcome_name): expected occurrences} under the nominal mechanism."""
    topo, mech = inst.topo, inst.mechanism
    inv, attempts, _ = expected_invocations(inst)
    counts = {}
    for op in topo.ops:
        n = inv[op.id] * attempts[op.id]
        if n <= 0:
            continue
        if op.kind == KIND_CLIENT:
            sc, st = _client_step(inst, op.id)
            node = mech.nodes[f"C:{sc.index}:{st.index}:0"]
            d = node.dist(node.nominal_context(mech))
            counts[(op.id, "ok")] = n * float(d[0])
            counts[(op.id, "err")] = n * float(d[1])
            continue
        node = mech.nodes[f"A:{op.id}:0"] if op.kind != KIND_BFF else None
        if node is None:
            # a BFF op: average its context nodes' nominal distributions
            ctxs = mech.bff_contexts.get(op.id, [])
            d = None
            for (s, j, k) in ctxs:
                if k != 0:
                    continue
                t = mech.nodes[f"T:bff:{s}:{j}:0"]
                dd = t.dist(t.nominal_context(mech))
                d = dd if d is None else d + dd
            if d is None:
                continue
            d = d / d.sum()
        else:
            d = node.dist(node.nominal_context(mech))
        for i, name in enumerate(T_VALUES[:5]):
            counts[(op.id, name)] = n * float(d[i])
    return counts


def _client_step(inst, client_op):
    for sc in inst.sset.scenarios:
        for st in sc.steps:
            if st.client_op == client_op:
                return sc, st
    raise KeyError(client_op)


def alphabet_estimate(inst, min_count=1):
    """Expected realized alphabet: tokens with expected count >= min_count,
    plus the potential alphabet (every token with positive probability)."""
    counts = expected_token_counts(inst)
    realized = sum(1 for c in counts.values() if c >= min_count)
    potential = sum(1 for c in counts.values() if c > 0)
    vocab_min = inst.cfg.vocab.min_count
    variants = sum(1 for (op, o), c in counts.items() if o != "ok" and c >= vocab_min)
    base_ops = sum(1 for op in inst.topo.ops if op.kind != KIND_CLIENT) + len(inst.topo.client_ops)
    return {
        "expected_realized_alphabet": realized,
        "potential_alphabet": potential,
        "expected_vocab_size": 4 + base_ops + variants,
        "n_ops": len(inst.topo.ops),
        "expected_sessions": expected_invocations(inst)[2],
    }
