#!/usr/bin/env python3
"""Bind a pilot-trained orthogonalized-C plan for the frozen model extension."""
from __future__ import annotations

from scripts import build_full320_orthc_model_extension_plan_impl as builder



def main() -> int:
    return builder.main()


if __name__ == "__main__":
    raise SystemExit(main())
