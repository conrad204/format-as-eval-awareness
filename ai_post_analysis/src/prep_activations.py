"""Stream-decompress the frozen all-layer .npz archives into raw .npy files.

The archives are deflate-compressed, so `np.load` must materialise the whole
`X` array in RAM (6 GB for 8B, 30 GB for each 70B). Writing an uncompressed
.npy once lets every downstream analysis open it with `mmap_mode='r'` and pull
one layer at a time.

Output lands next to the source archive (``activations/`` is gitignored).
The source archive is never modified and its recorded sha256 is re-verified
against ``config.json`` before decompression.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import zipfile
from pathlib import Path

import numpy as np

MODELS = ("llama31_8b", "llama31_70b", "llama33_70b")
COPY_CHUNK = 64 << 20


def sha256_file(path: Path, chunk: int = 1 << 24) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        while block := fh.read(chunk):
            h.update(block)
    return h.hexdigest()


def decompress(model_dir: Path, out_path: Path, verify: bool) -> dict:
    archive = model_dir / "activations_all_layers.npz"
    config = json.loads((model_dir / "config.json").read_text())
    expected = config.get("activations_all_layers_sha256")

    if verify and expected:
        got = sha256_file(archive)
        if got != expected:
            raise RuntimeError(f"{archive}: sha256 {got} != recorded {expected}")

    with zipfile.ZipFile(archive) as zf:
        with zf.open("X.npy") as member:
            major, minor = np.lib.format.read_magic(member)
            reader = {(1, 0): np.lib.format.read_array_header_1_0,
                      (2, 0): np.lib.format.read_array_header_2_0}[(major, minor)]
            shape, fortran, dtype = reader(member)
            if fortran:
                raise RuntimeError("Fortran-ordered archive is not supported")
            out_path.parent.mkdir(parents=True, exist_ok=True)
            with out_path.open("wb") as out:
                np.lib.format.write_array_header_2_0(
                    out, {"descr": np.lib.format.dtype_to_descr(dtype),
                          "fortran_order": False, "shape": shape}
                )
                shutil.copyfileobj(member, out, COPY_CHUNK)

    written = np.load(out_path, mmap_mode="r")
    if written.shape != shape or written.dtype != dtype:
        raise RuntimeError(f"{out_path}: shape/dtype mismatch after write")

    return {
        "model_dir": str(model_dir),
        "archive_sha256": expected,
        "archive_sha256_verified": bool(verify and expected),
        "out_path": str(out_path),
        "shape": list(shape),
        "dtype": str(dtype),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--freeze-dir", default="activations/full320-v3")
    ap.add_argument("--models", nargs="*", default=list(MODELS))
    ap.add_argument("--no-verify", action="store_true",
                    help="skip the (slow) sha256 re-verification of the source archive")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    freeze = Path(args.freeze_dir)
    records = []
    for model in args.models:
        model_dir = freeze / model
        out_path = model_dir / "X_all_layers.npy"
        if out_path.exists() and not args.force:
            arr = np.load(out_path, mmap_mode="r")
            print(f"[skip] {out_path} exists shape={arr.shape}", flush=True)
            records.append({"model": model, "out_path": str(out_path),
                            "shape": list(arr.shape), "dtype": str(arr.dtype),
                            "skipped": True})
            continue
        print(f"[run ] {model} -> {out_path}", flush=True)
        rec = decompress(model_dir, out_path, verify=not args.no_verify)
        rec["model"] = model
        print(f"[done] {model} shape={rec['shape']}", flush=True)
        records.append(rec)

    print(json.dumps(records, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
