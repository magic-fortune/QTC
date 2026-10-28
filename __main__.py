"""Small command-line entry point for the dependency-free G0 fixture."""

from __future__ import annotations

import argparse
import json

from .g0 import build_g0


def main() -> int:
    parser = argparse.ArgumentParser(prog="python -m QTC")
    parser.add_argument("--zero-domain", action="store_true", help="keep the non-zero normalization obligation conditional")
    args = parser.parse_args()
    analysis, plan = build_g0(compile_domain_proven=not args.zero_domain)
    print(json.dumps({"graph_digest": analysis.graph.digest(), "plan": plan.canonical()}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
