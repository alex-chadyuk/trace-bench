"""Business scenarios: client journeys through the BFF.

A scenario is a monotone path in one global order over BFF endpoints (the BFF
endpoint index order), so journey edges are acyclic across scenarios. Each
step owns one client operation (`/page/<word>`), is served by one BFF endpoint
and carries the scenario's outcome masks and retry policy.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class Step:
    scenario: int
    index: int
    client_op: int         # op id (KIND_CLIENT)
    bff_op: int            # op id (KIND_BFF)


@dataclass
class Scenario:
    index: int
    name: str
    weight: float          # normalised arrival share
    steps: list[Step]
    client_mask: list[str]
    bff_mask: list[str]
    max_retries: int
    retry_on: list[str]
    fatal_error: bool
    terminal_failure: bool

    def to_dict(self):
        d = dict(vars(self))
        d["steps"] = [vars(s) for s in self.steps]
        return d

    @classmethod
    def from_dict(cls, d):
        d = dict(d)
        d["steps"] = [Step(**s) for s in d["steps"]]
        return cls(**d)


@dataclass
class ScenarioSet:
    scenarios: list[Scenario]
    order: list[int] = field(default_factory=list)     # BFF op ids in the global journey order

    def to_dict(self):
        return {"scenarios": [s.to_dict() for s in self.scenarios], "order": self.order}

    @classmethod
    def from_dict(cls, d):
        return cls([Scenario.from_dict(s) for s in d["scenarios"]], list(d["order"]))

    def total_steps(self):
        return sum(len(s.steps) for s in self.scenarios)


def build_scenarios(cfg, topo, rng):
    """Sample scenarios against a built topology. Client ops are consumed in
    order, one per step; BFF ops per scenario are a sorted sample without
    replacement of the global order."""
    order = list(topo.bff_ops)
    total_w = sum(s.weight for s in cfg.scenarios)
    scenarios = []
    client_cursor = 0
    # Coverage: an endpoint no journey visits would carry no traffic, so each
    # scenario draws first from the BFF ops no earlier scenario has taken and
    # only then from the rest; with total steps >= BFF ops every endpoint is on
    # at least one journey.
    uncovered = list(range(len(order)))
    for i, spec in enumerate(cfg.scenarios):
        k = min(spec.steps, len(uncovered))
        first = [int(p) for p in rng.choice(uncovered, size=k, replace=False)] if k else []
        rest_pool = [q for q in range(len(order)) if q not in first]
        rest = [int(p) for p in rng.choice(rest_pool, size=spec.steps - k, replace=False)] if spec.steps > k else []
        positions = sorted(first + rest)
        uncovered = [q for q in uncovered if q not in first]
        steps = []
        for j, pos in enumerate(positions):
            steps.append(Step(scenario=i, index=j, client_op=topo.client_ops[client_cursor], bff_op=order[pos]))
            client_cursor += 1
        scenarios.append(Scenario(
            index=i, name=spec.name, weight=spec.weight / total_w, steps=steps,
            client_mask=list(spec.repertoire.client), bff_mask=list(spec.repertoire.bff),
            max_retries=spec.retry.max_retries, retry_on=list(spec.retry.retry_on),
            fatal_error=spec.fatal_error, terminal_failure=spec.terminal_failure,
        ))
    return ScenarioSet(scenarios, order)


def scenario_weights(sset: ScenarioSet):
    return np.array([s.weight for s in sset.scenarios], dtype=np.float64)
