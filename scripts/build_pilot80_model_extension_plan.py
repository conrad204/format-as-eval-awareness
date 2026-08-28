#!/usr/bin/env python3
"""Build a pilot80 plan for the prospectively frozen model extension."""
from __future__ import annotations

from scripts import build_pilot80_factorial_causal_plan as builder

PROTOCOL_ID = "pilot80_factorial_causal_mediation_model_extension_v1"


def main() -> int:
    builder.SUPPORTED_PROTOCOL_IDS.add(PROTOCOL_ID)
    return builder.main()


if __name__ == "__main__":
    raise SystemExit(main())
