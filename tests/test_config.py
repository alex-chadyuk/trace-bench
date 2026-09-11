"""PRD scenario 23: a malformed configuration is rejected before any work
begins with a message naming the field, the value received and the constraint
violated."""
import copy
from pathlib import Path

import pytest
import yaml

from tracebench.config import ConfigError, load_instance_config, parse_instance_config

REPO = Path(__file__).resolve().parents[1]
XS = REPO / "configs" / "instances" / "xs.yaml"


@pytest.fixture
def xs_dict():
    return yaml.safe_load(XS.read_text())


def test_xs_loads():
    cfg = load_instance_config(XS)
    assert cfg.name == "xs"
    assert cfg.mechanism.floor == 0.05 and 0.05 in cfg.mechanism.sweep
    assert len(cfg.run.seeds) >= 5
    assert cfg.resolved()["run"]["cap_gb"] == 50


def _reject(data, field_substr, value_substr=None, constraint_substr=None):
    with pytest.raises(ConfigError) as ei:
        parse_instance_config(data)
    msg = str(ei.value)
    assert field_substr in msg, msg
    if value_substr is not None:
        assert value_substr in msg, msg
    if constraint_substr is not None:
        assert constraint_substr in msg, msg
    return ei.value


def test_non_positive_count(xs_dict):
    d = copy.deepcopy(xs_dict)
    d["counts"]["services"] = 0
    _reject(d, "counts.services", "value=0", "greater than 0")


def test_infinite_retry_limit(xs_dict):
    d = copy.deepcopy(xs_dict)
    d["scenarios"][0]["retry"]["max_retries"] = float("inf")
    _reject(d, "scenarios.0.retry.max_retries", "inf")


def test_strength_floor_out_of_range(xs_dict):
    d = copy.deepcopy(xs_dict)
    d["mechanism"]["floor"] = 1.5
    d["mechanism"]["sweep"] = [1.5]
    _reject(d, "mechanism.floor", "1.5", "less than or equal to 1")


def test_fatal_error_without_terminal_failure(xs_dict):
    d = copy.deepcopy(xs_dict)
    d["scenarios"][1]["terminal_failure"] = False
    _reject(d, "scenarios.1", None, "terminal-failure state")


def test_fault_interval_outside_window(xs_dict):
    d = copy.deepcopy(xs_dict)
    d["schedules"]["faults"][0]["end_s"] = 99999
    _reject(d, "schedules.faults.0.end_s", "99999", "simulated window")


def test_reference_to_missing_component(xs_dict):
    d = copy.deepcopy(xs_dict)
    d["schedules"]["faults"][0]["component"] = "service:99"
    _reject(d, "schedules.faults.0.component", "service:99", "counts.services")


def test_reference_to_missing_endpoint(xs_dict):
    d = copy.deepcopy(xs_dict)
    d["schedules"]["faults"][1]["component"] = "service:2/endpoint:7"
    _reject(d, "schedules.faults.1.component", "endpoint:7", "endpoints_per_service")


def test_unknown_field_is_rejected(xs_dict):
    d = copy.deepcopy(xs_dict)
    d["run"]["gpu"] = True
    _reject(d, "run.gpu", None, "Extra inputs")


def test_scenario_count_mismatch(xs_dict):
    d = copy.deepcopy(xs_dict)
    d["counts"]["scenarios"] = 3
    _reject(d, "scenarios", "2", "counts.scenarios")


def test_steps_exceed_bff_endpoints(xs_dict):
    d = copy.deepcopy(xs_dict)
    d["scenarios"][1]["steps"] = 9
    _reject(d, "scenarios.1.steps", "9", "bff_endpoints")


def test_fewer_than_five_seeds(xs_dict):
    d = copy.deepcopy(xs_dict)
    d["run"]["seeds"] = [0, 1]
    _reject(d, "run.seeds", None, "at least 5")


def test_problem_records_are_structured(xs_dict):
    d = copy.deepcopy(xs_dict)
    d["counts"]["clients"] = -3
    err = _reject(d, "counts.clients")
    p = err.problems[0]
    assert set(p) == {"field", "value", "constraint"}
    assert p["field"] == "counts.clients" and p["value"] == -3


def test_bad_yaml_is_rejected(tmp_path):
    bad = tmp_path / "bad.yaml"
    bad.write_text("name: [unclosed\n")
    with pytest.raises(ConfigError) as ei:
        load_instance_config(bad)
    assert "not valid YAML" in str(ei.value)
