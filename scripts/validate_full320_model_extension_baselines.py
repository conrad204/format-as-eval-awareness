#!/usr/bin/env python3
"""Validate source/target reproduction for the frozen model extension."""
from __future__ import annotations

from scripts import validate_full320_transfer_baselines as validator

PROTOCOL_ID = "full320_orthogonalized_c_transfer_model_extension_v1"
SELECTION_DOMAIN = "full320_orthc_model_extension_baseline_shakedown_v1"


def main() -> int:
    validator.PROTOCOL_ID = PROTOCOL_ID
    validator.SELECTION_DOMAIN = SELECTION_DOMAIN
    return validator.main()


if __name__ == "__main__":
    raise SystemExit(main())
