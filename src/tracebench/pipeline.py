"""One unattended job: generate, verify and upload the corpora of one rung.

    python -m tracebench.pipeline --config configs/instances/<rung>.yaml --seeds 0 [1 2 3 4] \\
        --out data/corpora --workers 8 [--variants latent twin] [--denylist <private list>] \\
        [--upload [--repo chadyuk/trace-bench] [--stage-dir data/hf-stage]] [--skip-existing]

Per seed it generates the latent instance and its fully-observable twin, then
verifies each against its own manifest and the public name grammar (and the
private denylist when one is given, which is what makes a corpus "verified").
With `--upload`, one `publish upload` puts every corpus of the job on the
dataset host once they have all verified — no release metadata and no tag, so
independent jobs accumulate the corpora of one release (see `publish`).

Every step prints one `pipeline_step` line with its status and duration, and
the job stops at the first failure: later steps are skipped and whatever was
generated stays on disk, so an execution environment that copies the output
directory afterwards still keeps the finished corpora. The exit status is 0
only if every step succeeded.

`--skip-existing` skips generating a corpus already marked COMPLETE (a relaunch
on a box that still holds the output); such a corpus is still verified before
it is uploaded. Nothing private enters this module: the denylist and the
staging directory are paths on the command line.
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

from .config import ConfigError, load_instance_config
from .constants import COMPLETE_MARKER, VARIANTS, VARIANT_LATENT, VARIANT_TWIN
from .generate import CapExceeded, corpus_dir_for, generate
from .log import log
from .manifest import verify
from .publish import DEFAULT_REPO, upload
from .record import RunRecord

EXIT_OK = 0
EXIT_FAILED = 1
EXIT_REFUSED = 3


class StepFailed(RuntimeError):
    """A pipeline step ran and did not succeed (as opposed to being refused)."""


class Steps:
    """Records one `pipeline_step` line per step, in order."""

    def __init__(self):
        self.records = []

    def run(self, step, corpus, fn):
        t0 = time.monotonic()
        try:
            out = fn()
        except Exception as e:
            self._record(step, corpus, "failed", t0, error=f"{type(e).__name__}: {e}")
            raise
        self._record(step, corpus, "ok", t0)
        return out

    def skip(self, step, corpus, reason):
        self._record(step, corpus, "skipped", time.monotonic(), reason=reason)

    def _record(self, step, corpus, status, t0, **extra):
        rec = {"event": "pipeline_step", "step": step, "corpus": corpus, "status": status,
               "seconds": round(time.monotonic() - t0, 1), **extra}
        self.records.append({k: v for k, v in rec.items() if k != "event"})
        log(rec)

    def counts(self):
        out = {}
        for r in self.records:
            out[r["status"]] = out.get(r["status"], 0) + 1
        return out


def _label(instance, variant, seed):
    return f"{instance}/{variant}/seed={seed}"


def run_pipeline(config_path, seeds, out, workers=1, variants=VARIANTS, denylist=None, upload_to=None,
                 stage_dir=None, skip_existing=False, upload_fn=None, record=True):
    """Generate, verify and (optionally) upload every (seed, variant) of one rung.

    `upload_to` is the dataset repository, or None to skip the upload;
    `upload_fn` replaces `publish.upload` (the tests pass a stub).
    """
    cfg = load_instance_config(config_path)
    instance = cfg.name
    out = Path(out)
    steps = Steps()
    rec = RunRecord(out / f"pipeline-{instance}", "pipeline",
                    {"config": str(config_path), "seeds": list(seeds), "out": str(out), "workers": workers,
                     "variants": list(variants), "denylist": str(denylist) if denylist else None,
                     "upload_to": upload_to, "stage_dir": str(stage_dir) if stage_dir else None,
                     "skip_existing": skip_existing}) if record else None
    corpora = []
    results = {"instance": instance, "seeds": list(seeds), "variants": list(variants), "corpora": [],
               "uploaded": None, "steps": steps.records}

    def finish(status):
        results["step_counts"] = steps.counts()
        if rec is not None:
            rec.finish(results, status=status)
        log({"event": "pipeline", "instance": instance, "status": status, **steps.counts(),
             "corpora": len(results["corpora"])})
        return results

    try:
        for seed in seeds:
            for variant in variants:
                label = _label(instance, variant, seed)
                corpus_dir = corpus_dir_for(out, instance, variant, seed)
                if skip_existing and (corpus_dir / COMPLETE_MARKER).exists():
                    steps.skip("generate", label, "COMPLETE exists")
                else:
                    res = steps.run("generate", label, lambda: generate(
                        config_path, seed, out, twin=(variant == VARIANT_TWIN), workers=workers))
                    if not res["complete"]:
                        raise StepFailed(f"{label}: generation did not complete")
                v = steps.run("verify", label, lambda: verify(corpus_dir, denylist))
                if not (v["manifest_ok"] and v["names_ok"]):
                    for problem in (v["manifest_problems"] + v["name_problems"])[:20]:
                        log({"event": "verify_problem", "corpus": label, "problem": problem})
                    raise StepFailed(f"{label}: verification failed "
                                     f"({len(v['manifest_problems'])} manifest, {len(v['name_problems'])} name problems)")
                corpora.append(str(corpus_dir))
                results["corpora"].append({"path": str(corpus_dir), "label": label,
                                           "records_scanned": v["records_scanned"],
                                           "denylist_used": v["denylist_used"]})
        if upload_to:
            fn = upload_fn or upload
            results["uploaded"] = steps.run("upload", f"{instance} x{len(corpora)}",
                                            lambda: fn(corpora, repo=upload_to, stage_dir=stage_dir))
    except (ConfigError, CapExceeded) as e:
        results["refused"] = str(e)
        finish("refused")
        raise
    except Exception as e:
        results["error"] = f"{type(e).__name__}: {e}"
        finish("failed")
        raise
    return finish("ok")


def build_parser():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", required=True)
    p.add_argument("--seeds", required=True, type=int, nargs="+")
    p.add_argument("--out", required=True)
    p.add_argument("--workers", type=int, default=1)
    p.add_argument("--variants", nargs="+", default=list(VARIANTS), choices=[VARIANT_LATENT, VARIANT_TWIN])
    p.add_argument("--denylist", default=None, help="private JSON {names: [...]} that must not appear in a corpus")
    p.add_argument("--upload", action="store_true", help="upload every corpus of this job after they all verify")
    p.add_argument("--repo", default=DEFAULT_REPO)
    p.add_argument("--stage-dir", default=None,
                   help="hard-link staging tree for the upload; keep it OUTSIDE --out so it is not copied with the corpora")
    p.add_argument("--skip-existing", action="store_true", help="do not regenerate a corpus already marked COMPLETE")
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    try:
        run_pipeline(args.config, args.seeds, args.out, workers=args.workers, variants=args.variants,
                     denylist=args.denylist, upload_to=args.repo if args.upload else None,
                     stage_dir=args.stage_dir, skip_existing=args.skip_existing)
    except (ConfigError, CapExceeded) as e:
        log({"event": "pipeline_refused", "reason": str(e)})
        return EXIT_REFUSED
    except Exception as e:
        log({"event": "pipeline_failed", "reason": f"{type(e).__name__}: {e}"})
        return EXIT_FAILED
    return EXIT_OK


if __name__ == "__main__":
    raise SystemExit(main())
