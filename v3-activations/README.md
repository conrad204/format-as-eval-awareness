---
license: other
viewer: false
---

# full320-v3 — purpose-by-format probe, frozen 2026-08-17

Frozen v3 dataset: 11,520 rendered items from 320 payload blocks (16 topics
x exactly 10 Grok-4.3 + 10 Gemini-3.1-Pro generated blocks each), rendered
in one pass with `render-v3 --schedule full`. Supersedes the retired,
generator-imbalanced dev/a80/b80/full160 renders.

## Files and sha256 chain

| file | sha256 |
|---|---|
| `payloads.grok43gemini31_gpt54.full320_8-17-2026.jsonl` (320 blocks) | see `render_manifest_v3_full320.json` `inputs.payloads` |
| `rendered_items_v3_full320_8-17-2026.jsonl` (11,520 items) | see `render_manifest_v3_full320.json` `output` |

The manifest binds payload -> render plus all catalog hashes; verify with
`eval-format-mvp verify-manifest-v3 --manifest render_manifest_v3_full320.json`
(paths inside are repo-relative to the plot-pipeline git repo layout:
`data/full320-v3/...`).

- Render is byte-reproducible from the payload file by pipeline code at git
  commit `bc02e10` or later (`render-v3 --schedule full`).
- Surface battery hard gates passed at freeze: purpose n-gram probes
  AUC 0.500 (report-only probes decodable as designed).
- Generation provenance (run configs, per-attempt transcripts, validator
  verdicts, request files) is tracked in the plot-pipeline git repo under
  `provenance/full320-v3/generation_runs/`.
- `activations/<model>/` will be added by the upcoming extraction runs
  (Llama-3.1-8B, Llama-3.3-70B, +1), mirroring the pilot80-v2 layout.

Pipeline code, tests, catalogs, and analysis live in the plot-pipeline git
repository; the pinned HF revision of this repo is recorded there in
`provenance/full320-v3/hf_snapshot.json`.
