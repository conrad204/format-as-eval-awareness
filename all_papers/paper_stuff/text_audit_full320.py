"""With-cue, held-vocabulary text-shortcut audit on the full320-v3 stated stratum.

Reuses scripts/text_redteam_v3.py's evaluate() unchanged (TF-IDF + LogisticRegression
C=1.0, leave-one-purpose-vocabulary-family-out over fg1-fg5). The repo script itself
only runs on the pilot80 bank because it requires the indirect/second-person strata;
this wrapper applies the identical fits to the 7,680 full320 stated rows only.

Scopes: cue_only (first line = the purpose statement), full_prompt, and
scaffold_without_cue (everything after the first line; eval/deploy twins are
byte-identical there, so 0.500 is expected by construction).

Output: paper_stuff/text_audit_full320_stated.json (development-only; not a
checked-in results/ artifact). Run from the repo root with the project venv.
"""
import json, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from text_redteam_v3 import evaluate, FOLDS  # noqa: E402

ITEMS = ROOT / "data" / "full320-v3" / "rendered_items_v3_full320_8-17-2026.jsonl"
OUT = ROOT / "paper_stuff" / "text_audit_full320_stated.json"

rows = [json.loads(l) for l in open(ITEMS) if l.strip()]
sel = [r for r in rows if r.get("label") in (0, 1) and r["render_slot"] == "stated"]
assert set(r["purpose_family_id"] for r in sel) == set(FOLDS)
out = {"items": str(ITEMS.relative_to(ROOT)), "n_stated_rows": len(sel), "folds": list(FOLDS), "results": {}}
for scope in ("cue_only", "full_prompt", "scaffold_without_cue"):
    for kind in ("word_1", "char_3-5", "char_2-5"):
        t = time.time()
        res = evaluate(rows, sel, scope, kind)
        out["results"][f"{scope}::{kind}"] = res
        print(f"{scope:22s} {kind:9s} mean={res['mean_auc']:.3f} max_fold={res['max_fold_auc']:.3f} ({time.time()-t:.0f}s)", flush=True)
OUT.write_text(json.dumps(out, indent=1) + "\n")
print("wrote", OUT)
