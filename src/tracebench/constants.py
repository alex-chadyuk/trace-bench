"""Structural constants — data semantics only, never tunables.

Every tunable value enters through the declarative configuration (a checked-in
named-instance file for citable corpora); the only defaults in this package are
the values the PRD itself pins (strength floor, floor sweep, size cap).
"""
from enum import IntEnum

TOOL_NAME = "trace-bench"
CORPUS_SCHEMA = "tracebench/corpus@1"
MANIFEST_SCHEMA = "tracebench/manifest@1"
REALISM_SCHEMA = "tracebench/realism@1"
CONFIG_SCHEMA = "tracebench/config@1"

# --- outcome classes ---------------------------------------------------------
# Identical to the lab's consumer (trace-rca constants.py): the int8 encoding of
# the `outcomes` column, OK must stay 0. Precedence when several apply:
# ERR > 5XX > 4XX > SLOW > OK.
OUTCOME_OK = 0
OUTCOME_4XX = 1
OUTCOME_5XX = 2
OUTCOME_ERR = 3
OUTCOME_SLOW = 4
OUTCOME_NAMES = {
    OUTCOME_OK: "ok",
    OUTCOME_4XX: "4xx",
    OUTCOME_5XX: "5xx",
    OUTCOME_ERR: "err",
    OUTCOME_SLOW: "slow",
}
OUTCOME_IDS = {name: oid for oid, name in OUTCOME_NAMES.items()}
OUTCOME_PRECEDENCE = (OUTCOME_ERR, OUTCOME_5XX, OUTCOME_4XX, OUTCOME_SLOW, OUTCOME_OK)
ERROR_OUTCOMES = (OUTCOME_4XX, OUTCOME_5XX, OUTCOME_ERR)

# Mechanism-side value set of an operation-outcome state variable T[e]. The first
# five indices coincide with the OUTCOME_* ids; ABSENT (the operation was not
# invoked) is a mechanism value but never a token. "ok" is the nominal value.
T_VALUES = ("ok", "4xx", "5xx", "err", "slow", "absent")
T_ABSENT = 5
CLIENT_T_VALUES = ("ok", "err", "absent")
CLIENT_T_ABSENT = 2

# --- model-vocab specials (mirror trace-cmi constants.py) ---------------------
PAD = 0
BOS = 1
EOS = 2
UNK = 3
N_SPECIALS = 4

# `kind` field of a base op in model-vocab.json.
KIND_BFF = 0
KIND_SERVICE = 1
KIND_CLIENT = 2
KIND_EXTERNAL = 3
KIND_NAMES = {KIND_BFF: "bff", KIND_SERVICE: "service", KIND_CLIENT: "client", KIND_EXTERNAL: "external"}


# --- RNG streams ------------------------------------------------------------------
class Stream(IntEnum):
    INSTANTIATE = 0
    LATENT = 1
    ARRIVALS = 2
    SESSION = 3
    OUTCOME = 4
    LATENCY = 5
    EMIT = 6
    SKEW = 7
    DEFECT = 8


SCOPE_INSTANTIATE = 0
SCOPE_GENERATE = 1
INSTANTIATE_SHARD = -1

# --- reproducibility ------------------------------------------------------------------
# Monte-Carlo-derived quantities are rounded to this many decimals before they
# are stored or used. A float computed through transcendentals is NOT bit-stable
# across CPU architectures even at identical library versions: on 2026-09-12 one
# fitted SLOW threshold came out 1 ULP apart on arm64 macOS and x86-64 Linux
# (0.021847459436795613 vs 0.02184745943679561), which dirtied `instantiation.json`
# and `slow-thresholds.json` (+ its four view copies) and broke the cross-machine
# half of PRD scenario 12 while every data file stayed byte-identical. Nine
# decimals is ~7 orders of magnitude above that noise, one nanosecond on a
# seconds-valued threshold (records carry milliseconds) and 1e-9 on a
# probability — below anything the calibration or the outcome classes can see.
# The `calibration` diagnostics were already rounded (to 6) and were identical
# across the two architectures, which is what located the gap.
MC_DECIMALS = 9

# --- PRD-pinned values ---------------------------------------------------------------
DEFAULT_FLOOR = 0.05
FLOOR_SWEEP = (0.01, 0.02, 0.05, 0.10, 0.20, 0.50)
CAP_GB_DEFAULT = 50.0
CPU_HOURS_BUDGET = 12.0
SESSION_GAP_MIN_DEFAULT = 30

ORDERINGS = ("end", "start")
GRAINS = ("request", "session")
SPLITS = ("train", "val", "test")
VARIANT_LATENT = "latent"
VARIANT_TWIN = "twin"
VARIANTS = (VARIANT_LATENT, VARIANT_TWIN)

# --- corpus layout ------------------------------------------------------------------
RUN_DIR = "run"
ARGUMENTS_JSON = "arguments.json"
RESULTS_JSON = "results.json"
RUN_META_JSON = "run_meta.json"
CONFIG_YAML = "config.yaml"
CONSTANTS_JSON = "constants.json"
INSTANTIATION_JSON = "instantiation.json"
MANIFEST_JSON = "manifest.json"
COMPLETE_MARKER = "COMPLETE"

TOPOLOGY_DIR = "topology"
CALLGRAPH_JSON = "callgraph.json"
PRIOR_JSON = "prior.json"

GRAPHS_DIR = "graphs"
MECHANISM_GRAPH_JSON = "mechanism-graph.json"
SCORING_TARGET_JSON = "scoring-target.json"
ALPHABET_JSON = "alphabet.json"
FLOOR_SENSITIVITY_JSON = "floor-sensitivity.json"
CHANGEPOINTS_JSON = "changepoints.json"
VIEW_ENDPOINT_JSON = "views/endpoint.json"
VIEW_SERVICE_JSON = "views/service.json"

RAW_DIR = "raw"
ORACLE_DIR = "oracle"
ORACLE_STATE_DIR = "oracle/state"
LABELS_DIR = "labels"
FAULTS_JSON = "faults.json"
CASES_JSON = "cases.json"
VIEWS_DIR = "views"
REPORTS_DIR = "reports"

SEQUENCES_DIR = "sequences"
MODEL_VOCAB_JSON = "model-vocab.json"
SCENARIO_PREVALENCE_JSON = "scenario-prevalence.json"
EXPORT_STATS_JSON = "export-stats.json"
SLOW_THRESHOLDS_JSON = "slow-thresholds.json"

CORRELATION_REPORT_JSON = "correlation-report.json"
REALISM_REPORT_JSON = "realism-report.json"
MECHANISM_CHECK_JSON = "mechanism-check.json"
TOPOLOGY_VS_TARGET_JSON = "topology-vs-target.json"
ESTIMATE_JSON = "estimate.json"

# The only corpus prefixes a method under evaluation may read (PRD scenario 25).
METHOD_READABLE_PREFIXES = ("raw/", "views/")

# Mechanism node groups whose values reach an emitted record in the latent
# instance: attempts (access records), client outcomes (client-side records),
# and the derived invocation/final variables (functions of those records).
# Every other group is latent — the flag is decided by this spec, never by hand.
EMITTED_GROUPS = ("invoke", "attempt", "final", "client")

# Raw record kinds (the six the real feed exhibits) plus the twin's state log.
KIND_ACCESS = "vl.access"
KIND_APP = "vl.app"
KIND_AUDIT = "vl.audit"
KIND_SENTRY_ERROR = "sentry.error"
KIND_SENTRY_TXN = "sentry.transaction"
KIND_SENTRY_REPLAY = "sentry.replay"
KIND_STATE_EVENT = "sim.state"
RECORD_KINDS = (
    KIND_ACCESS, KIND_APP, KIND_AUDIT,
    KIND_SENTRY_ERROR, KIND_SENTRY_TXN, KIND_SENTRY_REPLAY,
)

# Attribution levels of the correlated view, strongest first (int8 in the views).
ATTR_SESSION = 0
ATTR_DEVICE = 1
ATTR_USER = 2
ATTR_IP = 3
ATTR_NONE = 4
ATTR_NAMES = {ATTR_SESSION: "session", ATTR_DEVICE: "device", ATTR_USER: "user", ATTR_IP: "ip", ATTR_NONE: "none"}

# Sequence phase relative to a fault (trace-rca D-RCA-4 encoding).
PHASE_PRE = 0
PHASE_POST = 1
PHASE_STRADDLE = 2
PHASE_NONE = -1
