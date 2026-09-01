"""Fallback: run a frozen bank on a 405B-class model through OpenRouter.

Use this when ndif.dev is unreachable. It works from the project workstation and
needs no GPU. Two differences from the NDIF path, both of which must be stated in
any write-up:

* the only 405B on OpenRouter is `nousresearch/hermes-3-llama-3.1-405b`, a
  fine-tune, not Meta's Instruct checkpoint;
* no activations, so the probe half of the experiment cannot be run here.

Read-out: greedy single-token continuation, scored as the letter the model emits.
Where the provider returns logprobs, the A-versus-B margin is recorded too.

    export OPENROUTER_API_KEY=...          # or .env.local
    python run_405b_openrouter.py --items ../banks/latent_recognition.jsonl \
        --out ../out/latent_recognition_hermes405b.json --concurrency 8

Cost at $1/M tokens: ~700 prompt tokens x 1024 prompts is about $0.75 per bank.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path
from queue import Queue

URL = "https://openrouter.ai/api/v1/chat/completions"
DEFAULT_MODEL = "nousresearch/hermes-3-llama-3.1-405b"
LETTER_RE = re.compile(r"\b([AB])\b")
ANSWER_RE = re.compile(r"answer\s*[:\-]?\s*\**\s*([AB])\b", re.IGNORECASE)


def load_env() -> None:
    for candidate in (Path(".env.local"), Path("../.env.local"), Path("../../.env.local")):
        if candidate.exists():
            for line in candidate.read_text(encoding="utf-8").splitlines():
                m = re.match(r"\s*(?:export\s+)?([A-Z_]+)\s*=\s*(.+)", line)
                if m and m.group(1) not in os.environ:
                    os.environ[m.group(1)] = m.group(2).strip().strip('"').strip("'")


def call(key: str, model: str, messages: list[dict], max_tokens: int, retries: int = 5) -> dict:
    body = json.dumps({
        "model": model, "messages": messages, "max_tokens": max_tokens,
        "temperature": 0, "top_p": 1, "logprobs": True, "top_logprobs": 8,
    }).encode()
    req = urllib.request.Request(URL, data=body, headers={
        "Authorization": f"Bearer {key}", "Content-Type": "application/json"})
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(req, timeout=180) as response:
                return json.load(response)
        except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError) as exc:
            if attempt == retries - 1:
                raise
            time.sleep(5 * (attempt + 1))
            print(f"    retry {attempt + 1}: {type(exc).__name__}", file=sys.stderr, flush=True)
    raise RuntimeError("unreachable")


def score(row: dict, reply: dict) -> dict:
    choice = reply["choices"][0]
    text = choice["message"]["content"] or ""
    found = ANSWER_RE.findall(text) or LETTER_RE.findall(text)
    letter = found[-1].upper() if found else None
    margin = None
    try:
        top = choice["logprobs"]["content"][0]["top_logprobs"]
        table = {entry["token"].strip(): entry["logprob"] for entry in top}
        if "A" in table and "B" in table:
            target = row["target_letter"]
            other = "B" if target == "A" else "A"
            margin = float(table[target] - table[other])
    except Exception:
        pass
    return {
        "kind": row["kind"], "item_id": row["item_id"], "unit_id": row["unit_id"],
        "episode_family": row["episode_family"], "question_family": row.get("question_family"),
        "condition": row["condition"], "query_status": row["query_status"],
        "target_letter": row["target_letter"], "parsed_letter": letter,
        "correct": None if letter is None else int(letter == row["target_letter"]),
        "target_logit_margin": margin, "parse_failed": letter is None,
        "generation": text[:400],
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--items", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--families", type=int, default=None,
                    help="keep only N episode families (all questions and conditions within them), "
                         "so the crossed structure survives a small sample")
    ap.add_argument("--max-tokens", type=int, default=None,
                    help="default 1 for forced choice, 128 for reasoning banks")
    args = ap.parse_args()

    load_env()
    key = os.environ.get("OPENROUTER_API_KEY")
    if not key:
        raise SystemExit("set OPENROUTER_API_KEY (env or .env.local)")
    rows = [json.loads(line) for line in args.items.read_text(encoding="utf-8").splitlines() if line]
    if args.limit:
        rows = rows[: args.limit]
    reasoning = any(r["kind"] == "reasoning" for r in rows)
    max_tokens = args.max_tokens if args.max_tokens else (128 if reasoning else 1)
    print(f"{args.model}: {len(rows)} prompts, max_tokens={max_tokens}, "
          f"concurrency={args.concurrency}", flush=True)

    if args.families:
        order = sorted({r["episode_family"] for r in rows},
                       key=lambda f: hashlib.sha256(f"sample:{f}".encode()).hexdigest())
        keep = set(order[: args.families])
        rows = [r for r in rows if r["episode_family"] in keep]
        print(f"stratified to {len(keep)} families -> {len(rows)} prompts", flush=True)

    results: list[dict] = []
    lock = threading.Lock()
    queue: Queue = Queue()
    for index, row in enumerate(rows):
        queue.put((index, row))

    def worker() -> None:
        while True:
            try:
                index, row = queue.get_nowait()
            except Exception:
                return
            try:
                reply = call(key, args.model, row["messages"], max_tokens)
                record = score(row, reply)
            except Exception as exc:
                record = {"item_id": row["item_id"], "error": f"{type(exc).__name__}: {exc}"[:200]}
            with lock:
                results.append(record)
                if len(results) % 50 == 0:
                    print(f"  {len(results)}/{len(rows)}", flush=True)
            queue.task_done()

    threads = [threading.Thread(target=worker, daemon=True) for _ in range(args.concurrency)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    errors = [r for r in results if "error" in r]
    payload = {
        "schema_version": 1, "model_key": args.model.split("/")[-1].replace(".", "_"),
        "model": args.model, "provider": "openrouter",
        "precision": "provider-hosted, unspecified", "load_in_8bit": False,
        "caveat": ("Hermes-3 is a fine-tune of Llama-3.1-405B, not Meta's Instruct checkpoint; "
                   "serving precision is not disclosed and no activations are available."),
        "items": str(args.items),
        "items_sha256": hashlib.sha256(args.items.read_bytes()).hexdigest(),
        "n_items": len(results), "n_errors": len(errors),
        "rows": [r for r in results if "error" not in r],
        "errors": errors[:50],
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    ok = [r for r in payload["rows"] if r.get("correct") is not None]
    if ok:
        accuracy = sum(r["correct"] for r in ok) / len(ok)
        print(f"parsed {len(ok)}/{len(results)}  accuracy {accuracy:.4f}")
    print("wrote", args.out, f"({len(errors)} errors)")


if __name__ == "__main__":
    main()
