"""python -m tracebench.verify --corpus <dir> [--denylist <private.json>] — alias of `manifest verify`."""
from .manifest import build_parser as _bp, main as _main


def main(argv=None):
    import sys
    args = argv if argv is not None else sys.argv[1:]
    return _main(["verify", *args])


if __name__ == "__main__":
    raise SystemExit(main())
