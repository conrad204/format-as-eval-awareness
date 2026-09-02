"""Upload 70B all-layer activation tensors from /scratch straight to the format-EA org.

Runs ON the cluster: the tensors are ~5.5 GB each, so pushing them from the
extraction filesystem avoids a 22 GB round trip through a workstation.

Layout matches format-EA/full320-v3: activations/<model>/{activations_all_layers.npz,
config.json, meta.jsonl}. The single-layer L18 tensor is uploaded separately from
the repo checkout under its original name, so activations.npz is skipped here to
avoid storing the same tensor twice.

Credentials: HF_TOKEN must already be in the environment, normally via
  set -a; . /home/jppatton/langea/credentials/huggingface.env; set +a
The token is never printed, logged, or written to any artifact.

Concurrent commits to one HF repo can collide, so each upload retries with
backoff.

Usage:
  python scripts/cluster_upload_all_layers_to_hf.py --dataset binding_v1
  python scripts/cluster_upload_all_layers_to_hf.py --dataset implicit_v2 [--dry-run]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

RESULTS = Path("/scratch/jppatton/langea/results")
UPLOAD = ("activations_all_layers.npz", "config.json", "meta.jsonl")
# (source directory suffix, repo model directory). The 8B run carries the
# all-layer tensor; the 70B L18 runs carry only the single-layer tensor, which
# is uploaded under its own name so it does not collide with the all-layer file.
SOURCES = (
    ("{ds}_8b_all_layers", "llama31_8b", UPLOAD),
    ("{ds}_llama31_70b_all_layers", "llama31_70b", UPLOAD),
    ("{ds}_llama33_70b_all_layers", "llama33_70b", UPLOAD),
    ("{ds}_llama31_70b_l18", "llama31_70b", ("activations.npz",)),
    ("{ds}_llama33_70b_l18", "llama33_70b", ("activations.npz",)),
)
DATASETS = {
    "binding_v1": {"repo_id": "format-EA/binding-v1"},
    "implicit_v2": {"repo_id": "format-EA/implicit-v2"},
}
# Per dataset, not one shared file: concurrent dataset jobs read-modify-write and
# would silently drop each other's entries.
PROVENANCE_TEMPLATE = "/scratch/jppatton/langea/results/hf_upload_provenance_{ds}.json"

def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def upload_with_retry(api, *, path, repo_id, path_in_repo, attempts=5):
    for i in range(attempts):
        try:
            api.upload_file(path_or_fileobj=str(path), path_in_repo=path_in_repo,
                            repo_id=repo_id, repo_type="dataset")
            return
        except Exception as exc:  # noqa: BLE001 - retry any transport/commit collision
            if i == attempts - 1:
                raise
            wait = 20 * (i + 1)
            print(f"      attempt {i + 1} failed ({type(exc).__name__}); retrying in {wait}s",
                  file=sys.stderr, flush=True)
            time.sleep(wait)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=sorted(DATASETS))
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    token = os.environ.get("HF_TOKEN")
    if not token:
        raise SystemExit("HF_TOKEN not set; source the cluster credentials file first")

    cfg = DATASETS[args.dataset]
    plan = []
    for suffix, model, names in SOURCES:
        d = RESULTS / suffix.format(ds=args.dataset)
        if not d.is_dir():
            raise SystemExit(f"missing source directory {d}")
        for name in names:
            p = d / name
            if not p.exists():
                raise SystemExit(f"missing {p} - extraction not finished")
            # Keep the L18 tensor distinguishable from the all-layer tensor.
            dst = f"activations/{model}/{'activations_l18.npz' if 'l18' in d.name else name}"
            plan.append((model, p, dst))

    print(f"=== {args.dataset} -> {cfg['repo_id']}: {len(plan)} files, "
          f"{sum(p.stat().st_size for _, p, _ in plan) / 2**30:.2f} GiB", flush=True)
    for _, p, dst in plan:
        print(f"   {p.name} ({p.stat().st_size / 2**30:.2f} GiB) -> {dst}", flush=True)
    if args.dry_run:
        return

    from huggingface_hub import HfApi
    api = HfApi(token=token)
    api.create_repo(cfg["repo_id"], repo_type="dataset", private=True, exist_ok=True)

    # Skip anything already on the Hub at the identical size, so re-runs and the
    # already-completed all-layer push are not repeated.
    try:
        info = api.repo_info(cfg["repo_id"], repo_type="dataset", files_metadata=True)
        remote = {s.rfilename: (s.size or 0) for s in info.siblings}
    except Exception:  # noqa: BLE001 - a fresh repo simply has nothing to skip
        remote = {}
    todo = []
    for model, p, dst in plan:
        if remote.get(dst) == p.stat().st_size:
            print(f"   skip (already uploaded, same size): {dst}", flush=True)
        else:
            todo.append((model, p, dst))

    record = {}
    for model, p, dst in todo:
        print(f"   hashing {p.name} ...", flush=True)
        digest = sha256(p)
        print(f"   uploading {dst} ...", flush=True)
        t0 = time.time()
        upload_with_retry(api, path=p, repo_id=cfg["repo_id"], path_in_repo=dst)
        print(f"   done in {time.time() - t0:.0f}s", flush=True)
        record[dst] = {"source": str(p), "sha256": digest, "bytes": p.stat().st_size}

    prov = Path(PROVENANCE_TEMPLATE.format(ds=args.dataset))
    existing = json.loads(prov.read_text()) if prov.exists() else {}
    existing.update(record)
    prov.write_text(json.dumps(existing, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {prov}", flush=True)
    print(f"https://huggingface.co/datasets/{cfg['repo_id']}", flush=True)


if __name__ == "__main__":
    main()
