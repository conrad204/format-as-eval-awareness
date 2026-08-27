#!/usr/bin/env python3
"""Run the frozen full320 orthogonalized-C model extension."""
from __future__ import annotations

from scripts import run_full320_orthc_model_extension_impl as runner



def main() -> int:
    return runner.main()


if __name__ == "__main__":
    raise SystemExit(main())
