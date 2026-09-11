"""Declarative configuration: the primary input of every command.

A configuration is validated in full before any work begins (PRD scenario 23).
Every rejection is a `ConfigError` whose problems each name the offending
field, the value received and the constraint violated. Only PRD-pinned values
(strength floor, floor sweep, size cap, tick length) carry defaults; every
other tunable is required so the file is the complete record of a run.
"""
from __future__ import annotations

import math
import re
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from .constants import (
    CAP_GB_DEFAULT, CONFIG_SCHEMA, DEFAULT_FLOOR, FLOOR_SWEEP, OUTCOME_NAMES,
)

OutcomeName = Literal["ok", "4xx", "5xx", "err", "slow"]
ClientOutcomeName = Literal["ok", "err"]
RetryOn = Literal["4xx", "5xx", "err", "slow"]
FaultKind = Literal["crash", "degrade", "pool_exhaust", "cache_flush", "breaker_open"]

MIN_SEEDS = 5


class ConfigError(ValueError):
    """Actionable configuration rejection: `problems` is a list of
    {field, value, constraint} dicts; `str()` renders one line per problem."""

    def __init__(self, problems):
        self.problems = list(problems)
        super().__init__("\n".join(format_problem(p) for p in self.problems))


def format_problem(p):
    return f"field={p['field']} value={p['value']!r} constraint={p['constraint']}"


def _problem(field, value, constraint):
    return {"field": field, "value": _jsonable(value), "constraint": constraint}


def _jsonable(v):
    if isinstance(v, (int, float, str, bool)) or v is None:
        return v
    return repr(v)


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_assignment=True)


class RetryPolicy(StrictModel):
    max_retries: int = Field(ge=0, description="retries permitted per non-fatal error; finite")
    retry_on: list[RetryOn]

    @model_validator(mode="after")
    def _finite(self):
        # pydantic already rejects float('inf') for an int field; the check is
        # kept so the constraint text is explicit if the type ever loosens.
        if not math.isfinite(self.max_retries):
            raise ValueError("max_retries must be finite")
        return self


class Repertoire(StrictModel):
    client: list[ClientOutcomeName] = Field(min_length=1)
    bff: list[OutcomeName] = Field(min_length=1)


class ScenarioSpec(StrictModel):
    name: str = Field(pattern=r"^[a-z][a-z0-9-]{1,31}$")
    steps: int = Field(ge=1, description="number of BFF calls on the journey")
    weight: float = Field(gt=0, description="relative arrival share, normalised over scenarios")
    repertoire: Repertoire
    retry: RetryPolicy
    fatal_error: bool = Field(description="the journey can end in a fatal error")
    terminal_failure: bool = Field(description="the scenario declares a terminal-failure state")

    @model_validator(mode="after")
    def _fatal_needs_terminal(self):
        if self.fatal_error and not self.terminal_failure:
            raise ValueError("a scenario declaring a fatal error must declare a terminal-failure state")
        return self


class EndpointDefaults(StrictModel):
    repertoire: list[OutcomeName] = Field(min_length=1)
    retry: RetryPolicy


class TopologyShape(StrictModel):
    depth_pmf: list[float] = Field(min_length=1, description="P(max call depth below the BFF = 1..K)")
    fanout_mean: float = Field(gt=0, description="mean callees per calling endpoint")
    external_services: int = Field(ge=0)
    criticality_share: float = Field(ge=0, le=1, description="share of call edges whose failure is critical to the caller")
    cache_hit_share: float = Field(ge=0, le=1, description="share of call edges fronted by a cache")

    @model_validator(mode="after")
    def _pmf(self):
        if any(p < 0 for p in self.depth_pmf) or abs(sum(self.depth_pmf) - 1.0) > 1e-6:
            raise ValueError("depth_pmf must be non-negative and sum to 1")
        return self


class MechanismParams(StrictModel):
    floor: float = Field(default=DEFAULT_FLOOR, ge=0.0, le=1.0, description="strength floor of the scoring target (PRD default 0.05)")
    sweep: list[float] = Field(default_factory=lambda: list(FLOOR_SWEEP))
    error_rate_multiplier: float = Field(gt=0, description="scales the fitted nominal error rates; 1.0 = as fitted")
    ctxmax: bool = Field(description="also record the context-max strength where the parent set is small")
    mc_samples: int = Field(ge=1000, description="Monte Carlo samples behind every duration-mediated (SLOW) probability in the mechanism")

    @model_validator(mode="after")
    def _sweep(self):
        bad = [f for f in self.sweep if not (0.0 <= f <= 1.0)]
        if bad:
            raise ValueError(f"every floor in sweep must lie in [0, 1]; got {bad}")
        if self.floor not in self.sweep:
            raise ValueError("sweep must contain the default floor")
        return self


class FaultSpec(StrictModel):
    component: str = Field(description="service:<i> | service:<i>/endpoint:<j> | bff | bff/endpoint:<j> | external:<i>")
    kind: FaultKind
    start_s: float = Field(ge=0)
    end_s: float = Field(gt=0)

    @model_validator(mode="after")
    def _interval(self):
        if self.end_s <= self.start_s:
            raise ValueError("end_s must be greater than start_s")
        return self


class RegimeSpec(StrictModel):
    name: str = Field(pattern=r"^[a-z][a-z0-9-]{1,31}$")
    at_s: float = Field(gt=0)
    overlay: dict[str, Any] = Field(description="mechanism overlay applied at the changepoint")


class Schedules(StrictModel):
    faults: list[FaultSpec] = Field(default_factory=list)
    regimes: list[RegimeSpec] = Field(default_factory=list)


class Window(StrictModel):
    start: str = Field(pattern=r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z$", description="simulated UTC start")
    duration_hours: float = Field(gt=0)

    @property
    def seconds(self):
        return self.duration_hours * 3600.0


class RunControl(StrictModel):
    seed: int = Field(ge=0, description="the seed this file is generated with by default")
    seeds: list[int] = Field(min_length=MIN_SEEDS, description="shipped seeds; at least five (evaluation-integrity protocol)")
    window: Window
    tick_s: int = Field(default=1, ge=1)
    shard_ticks: int = Field(gt=0)
    base_rps: float = Field(gt=0, description="session arrivals per second when the daily profile is 1.0")
    cap_gb: float = Field(default=CAP_GB_DEFAULT, gt=0)
    twin: bool = Field(description="also generate the fully-observable twin")

    @model_validator(mode="after")
    def _seeds(self):
        if len(set(self.seeds)) != len(self.seeds):
            raise ValueError("seeds must be distinct")
        if any(s < 0 for s in self.seeds):
            raise ValueError("seeds must be non-negative")
        return self


class Counts(StrictModel):
    clients: int = Field(gt=0)
    services: int = Field(gt=0)
    endpoints_per_service: int = Field(gt=0)
    bff_endpoints: int = Field(gt=0)
    scenarios: int = Field(gt=0)


class VocabParams(StrictModel):
    min_count: int = Field(ge=1, description="TRAIN count at which a non-OK (op, outcome) pair becomes a variant token")


class SlowParams(StrictModel):
    quantile: float = Field(gt=0, lt=1, description="per-operation duration quantile above which OK becomes SLOW")


_COMPONENT_RE = re.compile(
    r"^(?:(?P<bff>bff)(?:/endpoint:(?P<bff_ep>\d+))?"
    r"|service:(?P<svc>\d+)(?:/endpoint:(?P<svc_ep>\d+))?"
    r"|external:(?P<ext>\d+))$"
)


class ComponentRef(StrictModel):
    kind: Literal["bff", "service", "external"]
    service_index: int | None = None
    endpoint_index: int | None = None

    def key(self):
        if self.kind == "bff":
            return "bff" if self.endpoint_index is None else f"bff/endpoint:{self.endpoint_index}"
        if self.kind == "external":
            return f"external:{self.service_index}"
        base = f"service:{self.service_index}"
        return base if self.endpoint_index is None else f"{base}/endpoint:{self.endpoint_index}"


def parse_component_ref(ref, counts):
    """Resolve a component reference against the configured counts; returns a
    ComponentRef or raises ValueError naming the constraint violated."""
    m = _COMPONENT_RE.match(ref or "")
    if not m:
        raise ValueError("must be bff | bff/endpoint:<j> | service:<i> | service:<i>/endpoint:<j> | external:<i>")
    if m.group("bff"):
        ep = m.group("bff_ep")
        if ep is not None and int(ep) >= counts.bff_endpoints:
            raise ValueError(f"bff endpoint index must be < counts.bff_endpoints ({counts.bff_endpoints})")
        return ComponentRef(kind="bff", endpoint_index=None if ep is None else int(ep))
    if m.group("ext") is not None:
        return ComponentRef(kind="external", service_index=int(m.group("ext")))
    svc = int(m.group("svc"))
    if svc >= counts.services:
        raise ValueError(f"service index must be < counts.services ({counts.services})")
    ep = m.group("svc_ep")
    if ep is not None and int(ep) >= counts.endpoints_per_service:
        raise ValueError(f"endpoint index must be < counts.endpoints_per_service ({counts.endpoints_per_service})")
    return ComponentRef(kind="service", service_index=svc, endpoint_index=None if ep is None else int(ep))


class InstanceConfig(StrictModel):
    schema_version: Literal["tracebench/config@1"] = Field(default=CONFIG_SCHEMA)
    name: str = Field(pattern=r"^[a-z][a-z0-9-]{0,31}$")
    rung: Literal["local", "cloud"]
    counts: Counts
    scenarios: list[ScenarioSpec] = Field(min_length=1)
    endpoints: EndpointDefaults
    topology: TopologyShape
    mechanism: MechanismParams
    schedules: Schedules
    run: RunControl
    constants: str = Field(description="path of the fitted realism constants file")
    vocab: VocabParams
    slow: SlowParams

    @model_validator(mode="after")
    def _cross_field(self):
        problems = []
        if len(self.scenarios) != self.counts.scenarios:
            problems.append(_problem("scenarios", len(self.scenarios),
                                     f"must contain exactly counts.scenarios ({self.counts.scenarios}) entries"))
        names = [s.name for s in self.scenarios]
        if len(set(names)) != len(names):
            problems.append(_problem("scenarios", names, "scenario names must be distinct"))
        for i, s in enumerate(self.scenarios):
            if s.steps > self.counts.bff_endpoints:
                problems.append(_problem(f"scenarios.{i}.steps", s.steps,
                                         f"must be <= counts.bff_endpoints ({self.counts.bff_endpoints}); a journey visits distinct BFF endpoints"))
        window_s = self.run.window.seconds
        for i, f in enumerate(self.schedules.faults):
            if f.end_s > window_s:
                problems.append(_problem(f"schedules.faults.{i}.end_s", f.end_s,
                                         f"must be <= the simulated window ({window_s:g} s)"))
            try:
                ref = parse_component_ref(f.component, self.counts)
                if ref.kind == "external" and ref.service_index >= self.topology.external_services:
                    problems.append(_problem(f"schedules.faults.{i}.component", f.component,
                                             f"external index must be < topology.external_services ({self.topology.external_services})"))
            except ValueError as e:
                problems.append(_problem(f"schedules.faults.{i}.component", f.component, str(e)))
        for i, r in enumerate(self.schedules.regimes):
            if r.at_s >= window_s:
                problems.append(_problem(f"schedules.regimes.{i}.at_s", r.at_s,
                                         f"must be < the simulated window ({window_s:g} s)"))
        ats = [r.at_s for r in self.schedules.regimes]
        if ats != sorted(ats) or len(set(ats)) != len(ats):
            problems.append(_problem("schedules.regimes", ats, "changepoints must be strictly increasing"))
        if self.run.seed not in self.run.seeds:
            problems.append(_problem("run.seed", self.run.seed, "must be one of run.seeds"))
        if self.run.shard_ticks * self.run.tick_s > window_s:
            problems.append(_problem("run.shard_ticks", self.run.shard_ticks,
                                     "a shard must not be longer than the simulated window"))
        if problems:
            raise ConfigError(problems)
        return self

    @property
    def window_seconds(self):
        return self.run.window.seconds

    def resolved(self):
        """The fully resolved configuration (defaults included) for the run record."""
        return self.model_dump(mode="json")


def _problems_from_validation_error(err: ValidationError):
    problems = []
    for e in err.errors():
        loc = ".".join(str(x) for x in e["loc"]) or "<root>"
        msg = e["msg"]
        if msg.startswith("Value error, "):
            msg = msg[len("Value error, "):]
        ctx = e.get("ctx") or {}
        if isinstance(ctx.get("error"), ConfigError):
            problems.extend(ctx["error"].problems)
            continue
        problems.append(_problem(loc, e.get("input"), msg))
    return problems


def parse_instance_config(data) -> InstanceConfig:
    if not isinstance(data, dict):
        raise ConfigError([_problem("<root>", data, "configuration must be a mapping")])
    try:
        return InstanceConfig.model_validate(data)
    except ConfigError:
        raise
    except ValidationError as err:
        raise ConfigError(_problems_from_validation_error(err)) from None


def load_instance_config(path) -> InstanceConfig:
    path = Path(path)
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as e:
        raise ConfigError([_problem("<file>", str(path), f"not valid YAML: {e}")]) from None
    return parse_instance_config(data)


def dump_config_yaml(cfg: InstanceConfig):
    return yaml.safe_dump(cfg.resolved(), sort_keys=True, default_flow_style=False)


# --- family configurations (sampler over whole systems; the sampler itself lands with M7) ---

class IntRange(StrictModel):
    lo: int = Field(gt=0)
    hi: int = Field(gt=0)

    @model_validator(mode="after")
    def _order(self):
        if self.hi < self.lo:
            raise ValueError("hi must be >= lo")
        return self


class FloatRange(StrictModel):
    lo: float = Field(gt=0)
    hi: float = Field(gt=0)

    @model_validator(mode="after")
    def _order(self):
        if self.hi < self.lo:
            raise ValueError("hi must be >= lo")
        return self


class FamilyRanges(StrictModel):
    clients: IntRange
    services: IntRange
    endpoints_per_service: IntRange
    bff_endpoints: IntRange
    scenarios: IntRange
    fanout_mean: FloatRange
    steps: IntRange


class FamilySplit(StrictModel):
    test_fraction: float = Field(gt=0, lt=1)


class FamilyConfig(StrictModel):
    schema_version: Literal["tracebench/family@1"] = "tracebench/family@1"
    name: str = Field(pattern=r"^[a-z][a-z0-9-]{0,31}$")
    n_systems: int = Field(gt=0)
    seed: int = Field(ge=0)
    ranges: FamilyRanges
    split: FamilySplit
    template: dict[str, Any] = Field(description="InstanceConfig fields shared by every sampled system (everything the ranges do not cover)")


def load_family_config(path) -> FamilyConfig:
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    # the template's constants path is relative to this file, and the sampled
    # system configs are written elsewhere: resolve it here
    template = data.get("template") if isinstance(data, dict) else None
    if isinstance(template, dict) and isinstance(template.get("constants"), str):
        from .realism import resolve_constants_path  # same rule as instance configs (repository root, else config dir)
        template["constants"] = str(resolve_constants_path(path, template["constants"]))
    try:
        return FamilyConfig.model_validate(data)
    except ValidationError as err:
        raise ConfigError(_problems_from_validation_error(err)) from None
