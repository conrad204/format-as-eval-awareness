"""Design frame for the full320-v3 freeze.

Loads `meta.jsonl` (row order is asserted identical to the activation archive)
and exposes the strata the implicit-purpose analysis needs:

  cued    7,680 rows, evaluation/deployment fully crossed with benchmark/casual,
          partitioned into 5 vocabulary-disjoint cue folds (`purpose_family_id`)
  no_cue  3,840 rows carrying no purpose sentence at all, spanning all 320
          payload blocks and all 13 rendered templates

Every payload block appears in every condition, so all contrasts below can be
computed block-internally and bootstrapped over blocks.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

TEMPLATE_FEATURES = (
    "heading_style", "option_style", "serialization", "register",
    "terminal_field", "message_role",
)


@dataclass
class Design:
    meta: list[dict]
    item_ids: np.ndarray

    cued_idx: np.ndarray          # row indices of explicitly cued items
    cued_y: np.ndarray            # 1 = evaluation, 0 = deployment
    cued_fold: np.ndarray         # purpose_family_id as integer code
    cued_fmt: np.ndarray          # 1 = benchmark, 0 = casual
    cued_block: np.ndarray        # payload block code
    cued_tpl: np.ndarray          # template code

    none_idx: np.ndarray          # row indices of no-cue items
    none_fmt: np.ndarray
    none_block: np.ndarray
    none_tpl: np.ndarray          # template code

    fold_names: list[str]
    block_names: list[str]
    tpl_names: list[str]
    tpl_format: np.ndarray        # 1 = benchmark template, 0 = casual template
    tpl_feature_table: dict

    @property
    def n_blocks(self) -> int:
        return len(self.block_names)

    @property
    def n_folds(self) -> int:
        return len(self.fold_names)


def _codes(values, order=None):
    order = order if order is not None else sorted(set(values))
    lookup = {v: i for i, v in enumerate(order)}
    return np.array([lookup[v] for v in values], dtype=np.int32), list(order)


def load_design(meta_path: str | Path, item_ids: np.ndarray | None = None) -> Design:
    meta = [json.loads(line) for line in Path(meta_path).read_text(encoding="utf-8").splitlines() if line.strip()]
    ids = np.array([m["item_id"] for m in meta])
    if item_ids is not None:
        if len(item_ids) != len(ids) or not np.all(item_ids == ids):
            raise RuntimeError("activation item_ids do not match meta.jsonl row order")

    block_names = sorted({m["payload_block_id"] for m in meta})
    tpl_names = sorted({m["template_id"] for m in meta})
    # No-cue rows carry the literal sentinel "none" in `purpose_family_id`;
    # only the cued stratum defines real cue-vocabulary folds.
    fold_names = sorted({m["purpose_family_id"] for m in meta
                         if m["intended_purpose"] in ("evaluation", "deployment")
                         and m["purpose_family_id"] not in (None, "none")})

    cued_idx = np.array([i for i, m in enumerate(meta) if m["intended_purpose"] in ("evaluation", "deployment")])
    none_idx = np.array([i for i, m in enumerate(meta) if m["intended_purpose"] == "none"])

    cm = [meta[i] for i in cued_idx]
    nm = [meta[i] for i in none_idx]

    cued_y = np.array([1 if m["intended_purpose"] == "evaluation" else 0 for m in cm], dtype=np.int8)
    cued_fold, _ = _codes([m["purpose_family_id"] for m in cm], fold_names)
    cued_fmt = np.array([1 if m["format"] == "benchmark" else 0 for m in cm], dtype=np.int8)
    cued_block, _ = _codes([m["payload_block_id"] for m in cm], block_names)
    cued_tpl, _ = _codes([m["template_id"] for m in cm], tpl_names)

    none_fmt = np.array([1 if m["format"] == "benchmark" else 0 for m in nm], dtype=np.int8)
    none_block, _ = _codes([m["payload_block_id"] for m in nm], block_names)
    none_tpl, _ = _codes([m["template_id"] for m in nm], tpl_names)

    tpl_format = np.zeros(len(tpl_names), dtype=np.int8)
    tpl_feature_table = {}
    for m in meta:
        t = tpl_names.index(m["template_id"])
        tpl_format[t] = 1 if m["format"] == "benchmark" else 0
        tpl_feature_table.setdefault(
            m["template_id"],
            {k: m.get(k) for k in TEMPLATE_FEATURES} | {"format": m["format"]},
        )

    if cued_y.sum() != len(cued_y) - cued_y.sum():
        raise RuntimeError("cued stratum is not purpose-balanced")

    return Design(
        meta=meta, item_ids=ids,
        cued_idx=cued_idx, cued_y=cued_y, cued_fold=cued_fold,
        cued_fmt=cued_fmt, cued_block=cued_block, cued_tpl=cued_tpl,
        none_idx=none_idx, none_fmt=none_fmt, none_block=none_block, none_tpl=none_tpl,
        fold_names=fold_names, block_names=block_names, tpl_names=tpl_names,
        tpl_format=tpl_format, tpl_feature_table=tpl_feature_table,
    )
