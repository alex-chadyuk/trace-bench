"""Tick-level latent trajectory and the forcing hook.

The latent chain (intensity -> load -> pool / cache -> health) is advanced
one tick at a time with the mechanism's own transition tables and
counter-keyed uniforms (hashing.py), so it depends on nothing the requests
do. A `Forcing` overrides a node's value over a tick interval — the one
mechanism behind fault injection (PRD scenario 7), the ground-truth check
(scenario 21) and the twin's exposure of latents (which forces nothing but
records everything).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .config import parse_component_ref
from .constants import KIND_CLIENT, KIND_EXTERNAL, KIND_SERVICE
from .hashing import D_LATENT, categorical, uniforms
from .intensity import intensity_index_vector
from .mechanism import CACHE_VALUES, HEALTH_VALUES, LOAD_VALUES, POOL_VALUES

ARRAYS = ("load", "pool", "cache", "health")
K_LOAD, K_POOL, K_CACHE, K_HEALTH = 0, 1, 2, 3


@dataclass
class Forcing:
    array: str                # load | pool | cache | health
    slots: np.ndarray         # slot indices in that array
    value: int                # value index
    t_lo: int                 # tick, inclusive
    t_hi: int                 # tick, exclusive
    label: str = ""


@dataclass
class Slots:
    """Index maps between topology entities and latent-array columns."""
    svc_of_slot: list[int]            # state-service slot -> Service.index
    slot_of_svc: dict                 # Service.index -> slot
    op_of_slot: list[int]             # health slot -> op id
    slot_of_op: dict                  # op id -> health slot
    svc_slot_of_op: np.ndarray        # op id -> service slot (or -1 for client ops)

    @classmethod
    def build(cls, topo):
        svc_of_slot, slot_of_svc = [], {}
        for s in topo.services:
            if s.kind == KIND_CLIENT:
                continue
            slot_of_svc[s.index] = len(svc_of_slot)
            svc_of_slot.append(s.index)
        op_of_slot, slot_of_op = [], {}
        for op in topo.ops:
            if op.kind == KIND_CLIENT:
                continue
            slot_of_op[op.id] = len(op_of_slot)
            op_of_slot.append(op.id)
        svc_slot_of_op = np.full(len(topo.ops), -1, dtype=np.int64)
        for op in topo.ops:
            if op.kind != KIND_CLIENT:
                svc_slot_of_op[op.id] = slot_of_svc[op.service]
        return cls(svc_of_slot, slot_of_svc, op_of_slot, slot_of_op, svc_slot_of_op)


@dataclass
class LatentSlice:
    t0: int
    intensity: np.ndarray     # [T] int8
    load: np.ndarray          # [T, S] int8
    pool: np.ndarray          # [T, S] int8
    cache: np.ndarray         # [T, S] int8
    health: np.ndarray        # [T, O] int8

    def at(self, tick):
        i = tick - self.t0
        return {"intensity": self.intensity[i], "load": self.load[i], "pool": self.pool[i],
                "cache": self.cache[i], "health": self.health[i]}

    def end_state(self):
        return self.at(self.t0 + len(self.intensity) - 1)


def initial_state(slots: Slots):
    S, O = len(slots.svc_of_slot), len(slots.op_of_slot)
    return {"load": np.zeros(S, np.int8), "pool": np.zeros(S, np.int8), "cache": np.zeros(S, np.int8),
            "health": np.zeros(O, np.int8)}


def _cdf(table):
    c = np.cumsum(table, axis=-1)
    c[..., -1] = 1.0
    return c


def simulate_latents(inst, seed, slots: Slots, state, t0, t1, forcings=()):
    """Advance the latent chain from `state` (values at tick t0 - 1, i.e. the
    previous tick's state) over ticks [t0, t1). Returns (LatentSlice, end_state)."""
    mech = inst.mechanism
    cfg, constants = inst.cfg, inst.constants
    T = t1 - t0
    S, O = len(slots.svc_of_slot), len(slots.op_of_slot)
    intensity = intensity_index_vector(cfg, constants, t0, T)
    load_cdf, pool_cdf, cache_cdf, health_cdf = _cdf(mech.load_table), _cdf(mech.pool_table), _cdf(mech.cache_table), _cdf(mech.health_table)
    svc_slot_of_health = np.array([slots.slot_of_svc[inst.topo.ops[op].service] for op in slots.op_of_slot], dtype=np.int64)
    out = {k: np.zeros((T, S if k != "health" else O), np.int8) for k in ARRAYS}
    load, pool, cache, health = (state["load"].copy(), state["pool"].copy(), state["cache"].copy(), state["health"].copy())
    svc_idx = np.arange(S)
    op_idx = np.arange(O)
    active = [f for f in forcings if f.t_hi > t0 and f.t_lo < t1]
    for f in active:
        if f.array == "intensity":
            lo, hi = max(f.t_lo, t0) - t0, min(f.t_hi, t1) - t0
            intensity[lo:hi] = f.value
    active = [f for f in active if f.array != "intensity"]
    for i in range(T):
        t = t0 + i
        inten = int(intensity[i])
        u_load = uniforms(seed, D_LATENT, t, svc_idx, K_LOAD)
        load = categorical(u_load, load_cdf[inten, load]).astype(np.int8)
        u_pool = uniforms(seed, D_LATENT, t, svc_idx, K_POOL)
        pool = categorical(u_pool, pool_cdf[load, pool]).astype(np.int8)
        u_cache = uniforms(seed, D_LATENT, t, svc_idx, K_CACHE)
        cache = categorical(u_cache, cache_cdf[load, cache]).astype(np.int8)
        u_health = uniforms(seed, D_LATENT, t, op_idx, K_HEALTH)
        health = categorical(u_health, health_cdf[pool[svc_slot_of_health], health]).astype(np.int8)
        for f in active:
            if f.t_lo <= t < f.t_hi:
                arr = {"load": load, "pool": pool, "cache": cache, "health": health}[f.array]
                arr[f.slots] = f.value
        out["load"][i], out["pool"][i], out["cache"][i], out["health"][i] = load, pool, cache, health
    sl = LatentSlice(t0, intensity, out["load"], out["pool"], out["cache"], out["health"])
    return sl, {"load": load, "pool": pool, "cache": cache, "health": health}


def shard_checkpoints(inst, seed, slots, shard_ticks, n_ticks, forcings=()):
    """State at the start of every shard (the state after the previous shard's
    last tick), computed sequentially once; shard 0 starts from all-nominal."""
    state = initial_state(slots)
    checkpoints = [state]
    t = 0
    while t + shard_ticks < n_ticks:
        _, state = simulate_latents(inst, seed, slots, state, t, t + shard_ticks, forcings)
        checkpoints.append(state)
        t += shard_ticks
    return checkpoints


FAULT_TARGET = {
    "crash": ("health", HEALTH_VALUES.index("failed")),
    "degrade": ("health", HEALTH_VALUES.index("degraded")),
    "pool_exhaust": ("pool", POOL_VALUES.index("exhausted")),
    "cache_flush": ("cache", CACHE_VALUES.index("cold")),
    "breaker_open": ("health", HEALTH_VALUES.index("failed")),
}


def compile_fault_forcings(inst, slots: Slots):
    """Every configured fault as a Forcing; returns (forcings, fault_records)."""
    cfg, topo = inst.cfg, inst.topo
    tick_s = cfg.run.tick_s
    forcings, records = [], []
    for i, f in enumerate(cfg.schedules.faults):
        ref = parse_component_ref(f.component, cfg.counts)
        array, value = FAULT_TARGET[f.kind]
        if ref.kind == "bff":
            svc = topo.services[0]
        elif ref.kind == "external":
            svc = [s for s in topo.services if s.kind == KIND_EXTERNAL][ref.service_index]
        else:
            svc = [s for s in topo.services if s.kind == KIND_SERVICE][ref.service_index]
        if array == "health":
            ops = svc.endpoints if ref.endpoint_index is None else [svc.endpoints[ref.endpoint_index]]
            slot_ids = np.array([slots.slot_of_op[o] for o in ops], dtype=np.int64)
            nodes = [f"health:{o}" for o in ops]
        else:
            slot_ids = np.array([slots.slot_of_svc[svc.index]], dtype=np.int64)
            nodes = [f"{array}:{svc.index}"]
        t_lo, t_hi = int(f.start_s // tick_s), int(np.ceil(f.end_s / tick_s))
        label = f"fault-{i}"
        forcings.append(Forcing(array, slot_ids, value, t_lo, t_hi, label))
        records.append({
            "index": i, "label": label, "component": f.component, "kind": f.kind,
            "service": svc.name, "endpoints": [topo.ops[o].name for o in (svc.endpoints if ref.endpoint_index is None or array != "health" else [svc.endpoints[ref.endpoint_index]])],
            "start_s": f.start_s, "end_s": f.end_s, "tick_lo": t_lo, "tick_hi": t_hi,
            "forced_nodes": nodes, "forced_value": {"health": HEALTH_VALUES, "pool": POOL_VALUES, "cache": CACHE_VALUES, "load": LOAD_VALUES}[array][value],
            "indicator": array,
        })
    return forcings, records


def state_events(slice_: LatentSlice, slots: Slots, topo, prev_state=None):
    """Latent value changes within a slice as records (the twin's state log)."""
    events = []
    names = {"load": LOAD_VALUES, "pool": POOL_VALUES, "cache": CACHE_VALUES, "health": HEALTH_VALUES}
    for array in ARRAYS:
        arr = getattr(slice_, array)
        prev = prev_state[array] if prev_state is not None else np.zeros(arr.shape[1], np.int8)
        prev_row = np.concatenate([prev[None, :], arr[:-1]], axis=0)
        ti, ci = np.nonzero(arr != prev_row)
        for t, c in zip(ti.tolist(), ci.tolist()):
            if array == "health":
                node = f"health:{slots.op_of_slot[c]}"
                subject = topo.ops[slots.op_of_slot[c]].name
                service = topo.services[topo.ops[slots.op_of_slot[c]].service].name
            else:
                node = f"{array}:{slots.svc_of_slot[c]}"
                subject = topo.services[slots.svc_of_slot[c]].name
                service = subject
            events.append({"tick": slice_.t0 + t, "node": node, "service": service, "subject": subject,
                           "value": names[array][int(arr[t, c])], "previous": names[array][int(prev_row[t, c])]})
    inten = slice_.intensity
    from .mechanism import INTENSITY_VALUES
    for t in np.nonzero(np.diff(inten, prepend=inten[0]))[0].tolist():
        events.append({"tick": slice_.t0 + t, "node": "intensity", "service": None, "subject": "intensity",
                       "value": INTENSITY_VALUES[int(inten[t])], "previous": INTENSITY_VALUES[int(inten[t - 1])] if t > 0 else None})
    events.sort(key=lambda e: (e["tick"], e["node"]))
    return events
