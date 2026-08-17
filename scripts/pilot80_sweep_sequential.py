"""Run the existing pilot80 held-fold sweep without multiprocessing copies."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

from pilot80_sweep_full import _G, sweep_layer


def main() -> None:
    items_path, npz_path, out_path = map(Path, sys.argv[1:4])
    rows = [json.loads(line) for line in items_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    npz = np.load(npz_path)
    item_ids = [str(value) for value in npz["item_ids"]]
    if item_ids != [row["item_id"] for row in rows]:
        raise ValueError("npz item_ids do not match metadata row order")
    _G["X_all"] = npz["X"]
    _G["stated"] = np.array([r["render_slot"] == "stated" for r in rows])
    _G["label"] = np.array([-1 if r["label"] is None else int(r["label"]) for r in rows])
    _G["fam"] = np.array([r["purpose_family_id"] for r in rows])
    _G["purpose"] = np.array([r["intended_purpose"] for r in rows])
    _G["fmt"] = np.array([r["format"] for r in rows])
    _G["pair_key"] = np.array([f'{r["payload_block_id"]}::{r["purpose_pair_id"]}::{r["format"]}' for r in rows])
    table = [sweep_layer(layer) for layer in range(_G["X_all"].shape[1])]
    out_path.write_text(json.dumps(table, indent=1) + "\n")
    best = max(table, key=lambda row: row["decorrelated"]["auc"])
    print(f"wrote {out_path} ({len(table)} layers); decorrelated argmax L{best['layer']} auc {best['decorrelated']['auc']:.4f}")


if __name__ == "__main__":
    main()
