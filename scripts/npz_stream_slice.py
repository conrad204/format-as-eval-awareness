"""Stream an all-layers activations npz from HF and keep only chosen layers.

The 70B activations_all_layers.npz is 15.6GB compressed / 30GB raw — too big
for a laptop disk, but any single layer is only 378MB. This streams the zip
over HTTP and decompresses on the fly, copying out the [n_items, hidden] fp32
slice of each requested layer, so the full matrix never exists on disk or in
RAM. Output per layer: <out_dir>/layer_XXX.npz with {layer, item_ids, X_fp32}
— a source format scripts/devbunova_decomposition.py accepts directly.

Assumes the exact writer in src/eval_format_mvp/extract.py:_write_npz_deflate1
(CPython zipfile, ZIP_DEFLATED, force_zip64=True). zipfile has two entry
layouts: written to a seekable file (the pod's disk — the actual case) it
patches the local header in place, so sizes are known up front (0xffffffff in
the 32-bit fields + real 64-bit sizes in the zip64 extra field, streaming flag
clear); written to a pipe it sets flag bit 3 and appends a zip64 data
descriptor instead. Both are handled. Entry order is {X, item_ids, layers,
token_lengths}; we slice X, read item_ids, then stop. Every structural
assumption is asserted, and item_ids are verified against the frozen items
jsonl before anything is written.

Usage:
  npz_stream_slice.py <repo_id> <revision> <path_in_repo> <items_jsonl> <out_dir> <layer> [<layer> ...]
"""

from __future__ import annotations

import io
import json
import struct
import sys
import time
import zlib
from pathlib import Path

import numpy as np
from numpy.lib import format as npformat

READ = 1 << 22  # 4MB HTTP reads


def read_exact(f, n: int, carry: bytearray) -> bytes:
    while len(carry) < n:
        b = f.read(READ)
        if not b:
            raise EOFError(f"stream ended wanting {n} bytes")
        carry += b
    out = bytes(carry[:n])
    del carry[:n]
    return out


def parse_npy_header(buf: bytes):
    """Return (data_offset, shape, dtype) from the head of a .npy stream."""
    assert buf[:6] == b"\x93NUMPY", "not an npy entry"
    major = buf[6]
    if major == 1:
        (hlen,) = struct.unpack("<H", buf[8:10])
        off = 10 + hlen
    else:
        (hlen,) = struct.unpack("<I", buf[8:12])
        off = 12 + hlen
    header = eval(buf[10 if major == 1 else 12:off].decode("latin1"))  # noqa: S307 - numpy's own header format
    assert not header["fortran_order"], "expected C order"
    return off, header["shape"], np.dtype(header["descr"])


def main() -> None:
    repo_id, revision, path_in_repo, items_path, out_dir = sys.argv[1:6]
    layers = sorted(int(x) for x in sys.argv[6:])
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    rows = [json.loads(l) for l in open(items_path) if l.strip()]
    ids_expected = [r["item_id"] for r in rows]

    if repo_id == "LOCAL":  # self-test path: read a local file instead of HF
        f = open(path_in_repo, "rb")
    else:
        from huggingface_hub import HfFileSystem

        f = HfFileSystem().open(
            f"datasets/{repo_id}/{path_in_repo}", "rb", revision=revision)
    carry = bytearray()
    slices: dict[int, bytearray] = {}
    item_ids = None
    t0 = time.time()

    while True:
        sig = read_exact(f, 4, carry)
        if sig == b"PK\x01\x02":  # central directory: no more entries
            break
        assert sig == b"PK\x03\x04", f"unexpected zip signature {sig!r}"
        hdr = read_exact(f, 26, carry)
        flags, method = struct.unpack("<HH", hdr[2:6])
        csize32, usize32 = struct.unpack("<II", hdr[14:22])
        fnlen, exlen = struct.unpack("<HH", hdr[22:26])
        name = read_exact(f, fnlen, carry).decode()
        extra = read_exact(f, exlen, carry)
        assert method == 8, f"{name}: expected deflate, got method {method}"
        streaming = bool(flags & 0x08)
        csize, usize = csize32, usize32
        if not streaming and csize32 == 0xFFFFFFFF:
            pos = 0
            csize = None
            while pos + 4 <= len(extra):
                tag, sz = struct.unpack("<HH", extra[pos:pos + 4])
                if tag == 0x0001:  # zip64: usize8 then csize8 (both were 0xffffffff)
                    usize, csize = struct.unpack("<QQ", extra[pos + 4:pos + 20])
                    break
                pos += 4 + sz
            assert csize is not None, f"{name}: zip64 sizes not found in extra field"

        dec = zlib.decompressobj(-15)
        want_x = name == "X.npy"
        collect = bytearray()  # small entries; for X only until header parsed
        npy = None  # (data_off, shape, dtype, itemsize_bytes)
        abs_off = 0  # absolute decompressed offset

        def feed(chunk: bytes):
            nonlocal npy, abs_off
            if not chunk:
                return
            if npy is None:
                collect.extend(chunk)
                if len(collect) >= 4096 or not want_x:
                    if want_x:
                        off, shape, dt = parse_npy_header(bytes(collect))
                        n_items, n_layers, hidden = shape
                        assert dt == np.float32 and max(layers) < n_layers
                        npy = (off, shape, dt, n_layers * hidden * 4)
                        for L in layers:
                            slices[L] = bytearray(n_items * hidden * 4)
                        rest = bytes(collect[off:])
                        del collect[:]
                        abs_off = 0
                        feed(rest)
                return
            # copy the parts of this chunk that fall inside a wanted layer row
            _, (n_items, n_layers, hidden), _, item_bytes = npy
            row_bytes = hidden * 4
            lo_item = max(0, abs_off // item_bytes)
            hi_item = min(n_items - 1, (abs_off + len(chunk)) // item_bytes)
            for i in range(lo_item, hi_item + 1):
                for L in layers:
                    src0 = i * item_bytes + L * row_bytes
                    a = max(src0, abs_off)
                    b = min(src0 + row_bytes, abs_off + len(chunk))
                    if a < b:
                        dst = i * row_bytes + (a - src0)
                        slices[L][dst:dst + (b - a)] = chunk[a - abs_off:b - abs_off]
            abs_off += len(chunk)

        def consume(piece: bytes):
            if want_x:
                feed(piece)
            else:
                collect.extend(piece)
                assert len(collect) < (1 << 28), f"{name}: unexpectedly large"

        if streaming:  # pipe-written variant: deflate is self-terminating
            while not dec.eof:
                data = bytes(carry) if carry else f.read(READ)
                del carry[:]
                if not data:
                    raise EOFError(f"{name}: stream ended mid-entry")
                consume(dec.decompress(data))
            carry[:0] = dec.unused_data
            desc = read_exact(f, 24, carry)  # zip64: sig + crc4 + csize8 + usize8
            assert desc[:4] == b"PK\x07\x08", f"{name}: missing data descriptor"
            _, usize = struct.unpack("<QQ", desc[8:24])
        else:  # disk-written variant (the actual case): csize known up front
            remaining = csize
            while remaining:
                if carry:
                    take = min(len(carry), remaining)
                    data = bytes(carry[:take])
                    del carry[:take]
                else:
                    data = f.read(min(READ, remaining))
                    if not data:
                        raise EOFError(f"{name}: stream ended mid-entry")
                consume(dec.decompress(data))
                remaining -= len(data)
            consume(dec.flush())
            assert not dec.unused_data, f"{name}: trailing compressed bytes"
        if want_x:
            n_items, n_layers, hidden = npy[1]
            assert usize == npy[0] + n_items * n_layers * hidden * 4
            assert abs_off == n_items * n_layers * hidden * 4
            print(f"X sliced: shape {npy[1]}, layers {layers}, "
                  f"{time.time() - t0:.0f}s", flush=True)
        elif name == "item_ids.npy":
            item_ids = npformat.read_array(io.BytesIO(bytes(collect)),
                                           allow_pickle=False)
            break  # remaining entries (layers, token_lengths) not needed
        else:
            print(f"skipped entry {name} ({usize} bytes)", flush=True)
    f.close()

    assert item_ids is not None and slices, "npz missing X or item_ids"
    assert [str(x) for x in item_ids] == ids_expected, (
        "item_ids do not match items jsonl order")
    n_items = len(ids_expected)
    hidden = len(slices[layers[0]]) // (n_items * 4)
    for L in layers:
        X = np.frombuffer(bytes(slices[L]), dtype=np.float32).reshape(
            n_items, hidden)
        np.savez_compressed(out / f"layer_{L:03d}.npz",
                            layer=np.int16(L), item_ids=item_ids, X_fp32=X)
        print(f"wrote {out / f'layer_{L:03d}.npz'} ({X.shape})", flush=True)
    (out / "slice_meta.json").write_text(json.dumps({
        "source_repo": repo_id, "revision": revision, "path": path_in_repo,
        "layers": layers, "generated_by": "scripts/npz_stream_slice.py"},
        indent=1) + "\n")


if __name__ == "__main__":
    main()
