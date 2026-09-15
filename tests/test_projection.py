"""PRD scenarios 3, 3a (target half), 5 and 6: the latent projection, the
twin's acyclic bidirected-free target, the floor sweep and the coarsened views."""
import numpy as np

from tracebench.constants import FLOOR_SWEEP
from tracebench.graphs import write_graph_artifacts
from tracebench.mechanism import FunctionNode, StateVar
from tracebench.projection import Projector, build_targets, coarsen, floor_sensitivity, is_acyclic
from tracebench.record import read_json
from xs_fixture import xs_instantiation


class TinyMechanism:
    """A hand-built mechanism exposing the interface the projector needs."""

    def __init__(self):
        self.nodes = {}
        self.order = []

    def add(self, node):
        self.nodes[node.var.id] = node
        self.order.append(node.var.id)
        return node

    def latent_ids(self):
        return [i for i in self.order if self.nodes[i].var.latent]


def _tok(vid, group, op, values, latent=False, derived=False):
    return StateVar(vid, group, "event", values, latent=latent, derived=derived, token_op=op)


def _binary_child(parent, p_if_a, p_if_b):
    def fn(ctx):
        p = p_if_a if ctx[parent] == "ok" else p_if_b
        return (p, 1 - p)
    return fn


def test_latent_mediator_gives_directed_edge_and_latent_confounder_gives_bidirected():
    m = TinyMechanism()
    m.add(FunctionNode(_tok("X", "attempt", 0, ("ok", "err")), [], lambda ctx: (0.9, 0.1)))
    # L is a latent mediator X -> L -> Y
    m.add(FunctionNode(_tok("L", "health", None, ("ok", "err"), latent=True), ["X"], _binary_child("X", 0.95, 0.2)))
    m.add(FunctionNode(_tok("Y", "attempt", 1, ("ok", "err")), ["L"], _binary_child("L", 0.9, 0.3)))
    # H is a latent common cause of U and V, and a derived node D mediates U -> D -> W
    m.add(FunctionNode(_tok("H", "pool", None, ("ok", "err"), latent=True), [], lambda ctx: (0.8, 0.2)))
    m.add(FunctionNode(_tok("U", "attempt", 2, ("ok", "err")), ["H"], _binary_child("H", 0.95, 0.4)))
    m.add(FunctionNode(_tok("V", "attempt", 3, ("ok", "err")), ["H"], _binary_child("H", 0.9, 0.5)))
    m.add(FunctionNode(_tok("D", "final", 2, ("ok", "err"), derived=True), ["U"], _binary_child("U", 1.0, 0.0)))
    m.add(FunctionNode(_tok("W", "attempt", 4, ("ok", "err")), ["D"], _binary_child("D", 0.9, 0.2)))
    # Z has no path to anything
    m.add(FunctionNode(_tok("Z", "attempt", 5, ("ok", "err")), [], lambda ctx: (0.5, 0.5)))
    proj = Projector(m, twin=False)
    directed, within = proj.directed_token_edges()
    groups = proj.bidirected_groups()
    d = {k: v["strength"] for k, v in directed.items()}
    # mediator collapses: X -> Y with the composed strength |P(Y=err|X=err) - P(Y=err|X=ok)|
    y_err_x_ok = 0.95 * 0.1 + 0.05 * 0.7
    y_err_x_err = 0.2 * 0.1 + 0.8 * 0.7
    assert abs(d[("0:err", "1:err")] - abs(y_err_x_err - y_err_x_ok)) < 1e-9
    assert ("1:err", "0:err") not in d                       # direction preserved
    assert d[("2:err", "4:err")] > 0.5                          # derived mediator collapses too
    assert not any(k[0].startswith("5:") or k[1].startswith("5:") for k in d)   # no path, no edge
    assert not any(k[0].startswith("0:") and k[1].startswith("2:") for k in d)
    assert not within
    # confounder: exactly one group, via H, over U's and V's tokens
    assert len(groups) == 1 and groups[0]["latent"] == "H"
    members = groups[0]["members"]
    assert set(members) == {"2:ok", "2:err", "3:ok", "3:err"}
    delta_u = abs((1 - 0.4) - (1 - 0.95))      # P(U=err | H=err) - P(U=err | H=ok)
    delta_v = abs((1 - 0.5) - (1 - 0.9))
    assert abs(max(members["2:err"]) - delta_u) < 1e-9 and abs(max(members["3:err"]) - delta_v) < 1e-9
    # nothing bidirected via the mediator L (it has a single token endpoint)
    assert all(g["latent"] != "L" for g in groups)


def test_xs_targets_have_expected_shape():
    inst = xs_instantiation()
    req, ses, proj = build_targets(inst)
    assert req["n_directed"] > 0 and req["n_bidirected"] > 0
    assert req["directed_acyclic_at_floor"]
    for e in req["directed"]:
        assert 0 < e["strength"] <= 1 and e["src"].split(":")[0] != e["dst"].split(":")[0]
    for e in req["bidirected"]:
        assert 0 < e["strength"] <= 1 and e["a"] < e["b"]
    # no latent node appears in the target; every via is a latent for bidirected edges
    latent = set(inst.mechanism.latent_ids())
    assert all(e["via"] in latent for e in req["bidirected"])
    assert all(not e["src"].startswith("state:") for e in req["directed"])
    # session grain is a superset in support
    assert ses["support_pairs"] >= req["support_pairs"] and ses["n_directed"] >= req["n_directed"]


def test_twin_target_is_acyclic_without_bidirected_edges():
    inst = xs_instantiation()
    req, ses, proj = build_targets(inst, twin=True)
    assert req["n_bidirected"] == 0 and req["bidirected_groups"] == []
    assert req["directed_acyclic_at_floor"]
    assert is_acyclic([(e["src"], e["dst"]) for e in req["directed"]])
    assert any(e["src"].startswith("state:") for e in req["directed"])


def test_floor_sweep_and_default_floor(tmp_path):
    inst = xs_instantiation()
    summary = write_graph_artifacts(inst, tmp_path, "latent")
    fs = read_json(tmp_path / "graphs" / "floor-sensitivity.json")
    assert fs["default_floor"] == inst.cfg.mechanism.floor
    floors = [r["floor"] for r in fs["sweep"]]
    assert floors == list(FLOOR_SWEEP)
    counts = [r["n_directed"] + r["n_bidirected"] for r in fs["sweep"]]
    assert counts == sorted(counts, reverse=True)
    assert summary["request"]["n_directed"] > 0


def test_views_are_functions_of_the_target_alone(tmp_path):
    inst = xs_instantiation()
    write_graph_artifacts(inst, tmp_path, "latent")
    target = read_json(tmp_path / "graphs" / "scoring-target.json")
    endpoint = read_json(tmp_path / "graphs" / "views" / "endpoint.json")
    service = read_json(tmp_path / "graphs" / "views" / "service.json")
    assert coarsen(target, "endpoint", inst.topo) == endpoint
    assert coarsen(target, "service", inst.topo) == service
    assert len(service["directed"]) <= len(endpoint["directed"]) <= len(target["directed"])


# --- D-TB-19: per-chain order, per-grain through-set, capped exact / Monte-Carlo fallback --------
import time

import pytest

from tracebench.constants import KIND_BFF, KIND_CLIENT, KIND_EXTERNAL, KIND_SERVICE, MC_DECIMALS, PROJECTION_CAP_PARTICLES, PROJECTION_MC_N
from tracebench.hashing import D_PROJECTION, uniforms
from tracebench.projection import CROSS_REQUEST_PREFIX, TOKEN_GROUPS, build_alphabet
from tracebench.rng import stable_id

FIXTURE_DIR = __import__("pathlib").Path(__file__).resolve().parent / "fixtures"
S_CONFIG = __import__("xs_fixture").REPO / "configs" / "instances" / "s.yaml"


def _all_chains(proj):
    """Every (src, dst) projection pair of a projector with its ordered chain."""
    mech = proj.mech
    for src in list(proj.sources()) + sorted(proj.latent, key=proj.pos.__getitem__):
        endpoints, _ = proj.forward(src)
        for dst in sorted(endpoints, key=proj.pos.__getitem__):
            if dst == src or mech.nodes[dst].var.group not in TOKEN_GROUPS:
                continue
            yield src, dst, proj.intermediates(src, dst)


def _longest_chains(proj, k):
    return sorted(_all_chains(proj), key=lambda r: -len(r[2]))[:k]


def effect_mc_reference(proj, src, src_value, dst, chain, n):
    """Per-sample reference of `Projector._effect_mc`: the same counter-keyed
    draws consumed one sample at a time through the ordinary context builder."""
    mech = proj.mech
    pair_key = int(stable_id(src, dst)[:15], 16)
    full = list(chain) + [dst]
    rows = np.zeros((n, len(mech.nodes[dst].var.values)))
    for i in range(n):
        a = {src: src_value}
        for j, m in enumerate(full):
            node = mech.nodes[m]
            d = node.dist_held(mech, Projector._context(mech, node, a))
            if m == dst:
                rows[i] = d
                break
            u = float(uniforms(mech.seed, D_PROJECTION, pair_key, i, j))
            k = int(np.searchsorted(np.cumsum(d), u, side="right"))
            a[m] = node.var.values[min(k, len(d) - 1)]
    return rows


@pytest.mark.parametrize("config", ["xs", "s"])
def test_chain_order_is_topological(config):
    """The enumeration assumes every chain node sees its assigned parents: in
    every chain of every pair, at both grains, a parent precedes its child. (The
    0.2.x order was the construction order, which put I:callee before I:caller
    and silently dropped every multi-hop invocation dependence — D-TB-19.)"""
    inst = xs_instantiation() if config == "xs" else __import__("tracebench.instantiate", fromlist=["x"]).instantiate_from_paths(S_CONFIG, seed=0)
    mech = inst.mechanism
    n_chains, n_multi_hop = 0, 0
    for grain in ("request", "session"):
        proj = Projector(mech, grain=grain)
        for src, dst, chain in _all_chains(proj):
            n_chains += 1
            pos = {m: i for i, m in enumerate(chain)}
            for i, m in enumerate(chain):
                for p in mech.nodes[m].parents:
                    if p == m:
                        continue          # the lag-1 self-loop of a tick latent
                    assert p not in pos or pos[p] < i, f"{grain} {src}->{dst}: {p} after its child {m}"
            if sum(1 for m in chain if m.startswith("I:")) >= 2:
                n_multi_hop += 1
    assert n_chains > 0 and n_multi_hop > 0, "no multi-hop chain to exercise the order"


def test_request_grain_excludes_cross_request_paths():
    """Request grain: `I:bff:*` are not through nodes, so the directed token
    set equals the one the 0.2.3 tool computed (its order bug never touched a
    within-request path) and the target is acyclic at the floor by the
    through-set, not by accident; the session grain includes them."""
    inst = xs_instantiation()
    req, ses, projs = build_targets(inst)
    frozen = read_json(FIXTURE_DIR / "xs-request-directed-0.2.3.json")
    expected = {tuple(p) for p in frozen["directed_pairs"]} - {tuple(p) for p in frozen["removed_by_d_tb_20"]}
    assert {(e["src"], e["dst"]) for e in req["directed"]} == expected
    assert req["directed_acyclic_at_floor"]
    assert not any(n.startswith(CROSS_REQUEST_PREFIX) for n in projs["request"].through)
    assert any(n.startswith(CROSS_REQUEST_PREFIX) for n in projs["session"].through)
    assert any(n.startswith(CROSS_REQUEST_PREFIX) for n in inst.mechanism.nodes)
    # the floor sweep of the request grain stays acyclic at every floor
    fs = floor_sensitivity(req, FLOOR_SWEEP)
    assert all(r["acyclic"] for r in fs["sweep"])


def test_session_grain_has_journey_edges():
    """D-TB-10 promises journey edges at the session grain (a step's client
    outcome gating the next step's invocation); the shipped 0.2.x targets had
    none because the chain order killed them. Now client -> backend and
    BFF -> backend edges exist above the floor, and the target may be cyclic."""
    inst = xs_instantiation()
    req, ses, _ = build_targets(inst)
    kind = {t["token"]: t["kind"] for t in build_alphabet(inst)["tokens"]}
    floor = inst.cfg.mechanism.floor

    def kinds(target):
        return {(kind[e["src"]], kind[e["dst"]]) for e in target["directed"] if e["strength"] >= floor}
    assert (KIND_CLIENT, KIND_SERVICE) in kinds(ses) and (KIND_BFF, KIND_SERVICE) in kinds(ses)
    assert (KIND_CLIENT, KIND_SERVICE) not in kinds(req) and (KIND_BFF, KIND_SERVICE) not in kinds(req)
    assert ses["n_directed"] > req["n_directed"]
    assert {(e["src"], e["dst"]) for e in req["directed"]} <= {(e["src"], e["dst"]) for e in ses["directed"]}
    assert ses["n_directed"] > read_json(FIXTURE_DIR / "xs-request-directed-0.2.3.json")["n_session_directed_0_2_3"]


def test_mc_matches_exact_within_four_standard_errors():
    inst = xs_instantiation()
    proj = Projector(inst.mechanism, grain="session")
    n, checked = 4000, 0
    for src, dst, chain in _longest_chains(proj, 20):
        for v in inst.mechanism.nodes[src].var.values:
            exact = proj.effect(src, v, dst)
            assert proj.effect_meta(src, v, dst) is None
            mc, se = proj._effect_mc(src, v, dst, chain, n)
            assert np.abs(exact - mc).max() <= 4 * max(se, 0.5 / np.sqrt(n)) + 1e-9, (src, v, dst)
            checked += 1
    assert checked >= 40


def test_mc_vectorised_equals_the_per_sample_reference():
    inst = xs_instantiation()
    proj = Projector(inst.mechanism, grain="session")
    for src, dst, chain in _longest_chains(proj, 4):
        assert any(m.startswith("I:") for m in chain)      # exercises the closed-form noisy-OR
        for v in inst.mechanism.nodes[src].var.values[:2]:
            mc, se = proj._effect_mc(src, v, dst, chain, 300)
            ref = effect_mc_reference(proj, src, v, dst, chain, 300)
            assert np.array_equal(mc, np.round(ref.mean(axis=0), MC_DECIMALS)), (src, v, dst)
            assert se == float(np.round((ref.std(axis=0, ddof=1) / np.sqrt(300)).max(), MC_DECIMALS))


def test_mc_is_deterministic_and_quantised():
    """The estimate is a function of (seed, pair, sample, step) only: a second
    projector, a different call order and a different cap give the same bytes,
    and every value is its own rounding at MC_DECIMALS."""
    inst = xs_instantiation()
    a = Projector(inst.mechanism, grain="request", cap=0, mc_n=1500)
    b = Projector(inst.mechanism, grain="request", cap=0, mc_n=1500)
    pairs = _longest_chains(a, 6)
    got_a = [(a.effect(s, v, d), a.effect_meta(s, v, d)) for s, d, _ in pairs for v in inst.mechanism.nodes[s].var.values]
    got_b = [(b.effect(s, v, d), b.effect_meta(s, v, d)) for s, d, _ in reversed(pairs) for v in reversed(inst.mechanism.nodes[s].var.values)]
    got_b.reverse()
    for (pa, ma), (pb, mb) in zip(got_a, got_b):
        assert np.array_equal(pa, pb) and ma == mb and ma["n"] == 1500
        assert np.array_equal(pa, np.round(pa, MC_DECIMALS)) and ma["se"] == round(ma["se"], MC_DECIMALS)
        assert abs(pa.sum() - 1.0) < 1e-6
    c = Projector(inst.mechanism, grain="request", cap=10 ** 9, mc_n=1500)     # cap irrelevant to the estimate
    s, d, chain = pairs[0]
    v = inst.mechanism.nodes[s].var.values[0]
    assert np.array_equal(c._effect_mc(s, v, d, chain, 1500)[0], got_a[0][0])


def test_cap_forces_and_forbids_fallback():
    inst = xs_instantiation()
    forced = Projector(inst.mechanism, grain="request", cap=0, mc_n=500)
    forbidden = Projector(inst.mechanism, grain="request", cap=None)
    default = Projector(inst.mechanism, grain="request")
    for proj in (forced, forbidden, default):
        proj.directed_token_edges()
        proj.bidirected_groups()
    n_direct = sum(1 for _s, _d, chain in _all_chains(default) if not chain)
    assert forced.n_effects_mc > 0 and forced.n_effects_exact > 0
    assert all(not chain for s, d, chain in _all_chains(forced) if forced.effect_meta(s, inst.mechanism.nodes[s].var.values[0], d) is None)
    assert n_direct > 0
    assert forbidden.n_effects_mc == 0 and forbidden.mc_se_max == 0.0 and forbidden.peak_particles > 1
    # xs never reaches the shipped cap: the default projector is exact everywhere and equals the uncapped one
    assert default.n_effects_mc == 0 and default.peak_particles < PROJECTION_CAP_PARTICLES
    for key, (pmf, meta) in default._effect_cache.items():
        assert np.array_equal(pmf, forbidden._effect_cache[key][0]) and meta is None


def test_mc_flag_present_only_when_used():
    inst = xs_instantiation()
    req, ses, _ = build_targets(inst)
    for t in (req, ses):
        assert t["n_effects_mc"] == 0 and t["mc_se_max"] == 0.0 and t["n_effects_exact"] > 0
        assert t["projection_cap"] == PROJECTION_CAP_PARTICLES and t["mc_n"] == PROJECTION_MC_N
        assert t["n_directed_mc"] == 0 and t["n_bidirected_mc"] == 0
        assert not any("mc" in e for e in t["directed"]) and not any("mc" in e for e in t["bidirected"])
        assert all(g["mc"] == {} for g in t["bidirected_groups"])
    req0, ses0, _ = build_targets(inst, cap=0, mc_n=800)
    assert req0["n_effects_mc"] > 0 and req0["mc_n"] == 800 and req0["projection_cap"] == 0
    flagged = [e for e in req0["directed"] if "mc" in e]
    assert flagged and req0["n_directed_mc"] == len(flagged)
    for e in flagged:
        assert e["mc"]["n"] == 800 and 0.0 <= e["mc"]["se"] <= req0["mc_se_max"] <= 0.51 / np.sqrt(800)
    assert req0["n_bidirected_mc"] == sum(1 for e in req0["bidirected"] if "mc" in e) > 0
    assert any(g["mc"] for g in req0["bidirected_groups"])
    # the flag never touches what the scorer and the views read
    assert coarsen(req0, "endpoint", inst.topo)["directed"] and all(set(e) == {"src", "dst", "strength"} for e in coarsen(req0, "endpoint", inst.topo)["directed"])


@pytest.mark.slow
def test_m_seed4_worst_pair_completes_under_budget():
    """The pair that killed the 0.2.3 m/l boxes: intensity -> the external
    operation's first attempt at m seed 4 (exact frontier bound 10^19). Under
    the shipped cap it must fall back to Monte Carlo inside a modest budget and
    memory envelope."""
    import resource

    from tracebench.instantiate import instantiate_from_paths

    inst = instantiate_from_paths(__import__("xs_fixture").REPO / "configs" / "instances" / "m.yaml", seed=4)
    topo = inst.topo
    ext = [o.id for o in topo.ops if topo.services[o.service].kind == KIND_EXTERNAL][-1]
    proj = Projector(inst.mechanism, grain="request")
    t0 = time.time()
    for v in inst.mechanism.nodes["intensity"].var.values:
        proj.effect("intensity", v, f"A:{ext}:0")
    elapsed = time.time() - t0
    assert proj.n_effects_mc == 3 and proj.peak_particles > PROJECTION_CAP_PARTICLES
    assert elapsed < 900, elapsed
    maxrss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    assert maxrss / 1e9 < 6.0 if __import__("sys").platform == "darwin" else maxrss / 1e6 < 6.0


def effect_exact_reference(proj, src, src_value, dst, chain, later_sets):
    """Per-particle reference of `Projector._effect_exact`: the 0.2.x dictionary
    enumeration (one frozenset of assignments per particle), kept to check the
    matrix form. Same prune, same context builder; only the merge order differs."""
    from collections import defaultdict

    mech = proj.mech
    full = list(chain) + [dst]
    particles = {frozenset({(src, src_value)}): 1.0}
    for idx, m in enumerate(full):
        node = mech.nodes[m]
        later_parents = later_sets[idx]
        new = defaultdict(float)
        for assign, p in particles.items():
            a = dict(assign)
            d = node.dist_held(mech, Projector._context(mech, node, a))
            if m == dst:
                for i, pv in enumerate(d):
                    new[("__out__", i)] += p * pv
                continue
            for i, pv in enumerate(d):
                q = p * pv
                if q <= proj.prune:
                    continue
                a2 = {k: v for k, v in a.items() if k in later_parents}
                if m in later_parents:
                    a2[m] = node.var.values[i]
                new[frozenset(a2.items())] += q
        particles = new
    out = np.zeros(len(mech.nodes[dst].var.values))
    for k, p in particles.items():
        out[k[1]] = p
    return out / out.sum()


def test_exact_matrix_enumeration_equals_the_per_particle_reference():
    inst = xs_instantiation()
    for grain in ("request", "session"):
        proj = Projector(inst.mechanism, grain=grain, cap=None)
        checked = 0
        for src, dst, chain in _longest_chains(proj, 30) + [r for r in _all_chains(proj) if not r[2]][:5]:
            order, later = proj.chain(src, dst)
            for v in inst.mechanism.nodes[src].var.values:
                got, peak = proj._effect_exact(src, v, dst, order, later)
                ref = effect_exact_reference(proj, src, v, dst, order, later)
                assert np.allclose(got, ref, atol=1e-12, rtol=0), (grain, src, v, dst)
                checked += 1
        assert checked > 60
    # the hand-built mechanism of the first test, through build-time API
    m = TinyMechanism()
    m.add(FunctionNode(_tok("X", "attempt", 0, ("ok", "err")), [], lambda ctx: (0.9, 0.1)))
    m.add(FunctionNode(_tok("L", "health", None, ("ok", "err"), latent=True), ["X"], _binary_child("X", 0.95, 0.2)))
    m.add(FunctionNode(_tok("Y", "attempt", 1, ("ok", "err")), ["L"], _binary_child("L", 0.9, 0.3)))
    proj = Projector(m, grain="session")
    assert abs(proj.effect("X", "err", "Y")[1] - (0.2 * 0.1 + 0.8 * 0.7)) < 1e-12
