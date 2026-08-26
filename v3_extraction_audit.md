# v3 activation-extraction audit

Answers below are grounded in the code as of 2026-08-19. Every claim quotes the
line(s) that establish it. Anything not determinable from the code is marked as
such.

**Chain.** There are two v3 extraction paths:

- **Local GPU** (Llama-3.1-8B, Llama-3.1-70B, Llama-3.3-70B):
  `scripts/extract_activations.py` (thin CLI) → `src/eval_format_mvp/extract.py`
  (the real code) → `activations/<freeze>/<model>/activations_all_layers.npz`,
  archived at `format-EA/full320-v3` on HF.
- **NDIF remote** (Llama-3.1-405B): `scripts/ndif_extract_405b.py` →
  `scripts/ndif_repack_405b.py` → HF `Conradf/v3-llama-3.1-405B-instruct`.

---

## Summary table

| # | Answer | Evidence |
|---|---|---|
| **1. Hook point** | **Block output** (whole decoder layer, post-residual), registered on `model.model.layers[i]` — i.e. `LlamaDecoderLayer`, **not** a sublayer. It is pre-final-norm: the forward calls `model.model(...)` (the base model), and the hook fires on the block before `model.norm` is applied. **Yes, the hook output is indexed** — `output[0]` when the block returns a tuple, else the tensor itself (transformers >=5 returns the tensor bare). No attn/mlp sublayer hooks anywhere (grep for `self_attn`/`mlp` in all four extraction files: no matches). | `extract.py:120` `model.model.layers[index].register_forward_hook(_make_hook(index))`<br>`extract.py:114` `hidden = output[0] if isinstance(output, (tuple, list)) else output`<br>`extract.py:140` `model.model(**encoded, use_cache=False, return_dict=True)`<br>`extract.py:107-108` comment: *"reads its output residual stream directly, before the final model norm"*<br>405B: `ndif_extract_405b.py:204-207` `outs = [model.model.layers[i].output for i in range(N_LAYERS)]` … `(o[0] if isinstance(o, (tuple, list)) else o)` |
| **2. Token positions** | **Yes — `[:, -1, :]` before storage. Only the last position is cached.** | `extract.py:115` `captured[index] = hidden[:, -1, :].detach()`<br>`ndif_extract_405b.py:207` `…[:, -1, :].detach().cpu()` |
| **3. Padding** | **Left-padded**, batch size **> 1** (default 8; NDIF sub-batch 16). Because padding is on the left, position `-1` **is** the last real token. The index is **not computed** from the mask — it is the literal `-1`; instead there is a hard assertion that column `-1` of the attention mask is all-1s, which fails the run if any pad ever landed there. | `extract.py:60` `tokenizer.padding_side = "left"`<br>`extract_activations.py:25` `parser.add_argument("--batch-size", type=int, default=8)`<br>`extract.py:132` `padding=True,`<br>`extract.py:135-136` `if not bool(torch.all(encoded["attention_mask"][:, -1] == 1)): raise MVPError("Last position must be a real prompt token, not padding")`<br>405B: `ndif_extract_405b.py:138`, `:177-186` (`SUB_BATCH = 16`, same assert) |
| **4. Layers** | **All transformer blocks**, `range(n_layers)`, when `--all-layers`; otherwise the single `--layer` (default 24). All three v3 local runs used all layers (8B: 32, 70B: 80). **The embedding output is NOT included** — there is no hook on `model.model.embed_tokens`; index 0 in the file is the *output of block 0*. | `extract.py:109` `target_layers = tuple(range(n_layers)) if all_layers else (layer,)`<br>`extract.py:82` `n_layers = int(model.config.num_hidden_layers)`<br>`extract.py:185` `"layer_semantics": "zero_based_transformer_block_output_before_final_model_norm"`<br>HF `activations/llama31_8b/config.json`: `"n_transformer_layers": 32, "layers": [0…31]`; `llama31_70b`: `80, [0…79]`; 405B: `provenance/full320-v3/extraction_config_405b_ndif.json` `"n_transformer_layers": 126` |
| **5. Chat template** | **Yes**, `apply_chat_template` with a single user message, `tokenize=False` (rendered to text first), and **`add_generation_prompt=True`**. Re-tokenized with `add_special_tokens=False` so BOS isn't duplicated. | `extract.py:89-96` `tokenizer.apply_chat_template([{"role": "user", "content": str(row["text"])}], tokenize=False, add_generation_prompt=True)`<br>`extract.py:97-99` `add_special_tokens=False`<br>`extract.py:86-88` comment explaining why<br>405B: `ndif_extract_405b.py:86-93` (identical) |
| **6. dtype** | **Capture: bfloat16** on CUDA (float32 on CPU — but all v3 runs were CUDA; every model config pins `torch_dtype: bfloat16`). **Save: float32** for 8B/70B (an explicit `.float()` upcast). For 405B: capture bf16, saved as **raw bf16 bits in uint16** (lossless, not upcast). | `extract.py:65` `dtype = torch.bfloat16 if device.type == "cuda" else torch.float32`<br>`extract.py:144` `batches[index].append(captured[index].float().cpu().numpy())`<br>`extract.py:192` `"dtype_stored": "float32"`<br>405B: `ndif_extract_405b.py:227-229` `assert x.dtype == torch.bfloat16` … `x.view(torch.int16).numpy().view(np.uint16)` |
| **7. Attention patterns** | **No.** `output_attentions` appears nowhere in any of the four extraction files (grep: no matches). The forward call passes only `use_cache=False, return_dict=True`. | `extract.py:140` `model.model(**encoded, use_cache=False, return_dict=True)` |
| **8. Sidecar data** | Saved: **`item_ids`**, **`layers`**, **`token_lengths`** inside the npz, plus **`meta.jsonl`** = every rendered-item row verbatim (36 fields incl. `text` = the pre-chat-template rendered prompt, `rendered_text_sha256`, `label`, `format`, `purpose_family_id`, cue spans) and a `config.json` with input/output sha256s. **NOT saved: `input_ids`, `attention_mask`, and the chat-template-applied string** (only `row["text"]`, the pre-template render, is preserved; the templated string is reconstructible from it + the pinned tokenizer revision). 405B chunks additionally save `top1_token_ids` and `row_start`. | `extract.py:163` `_write_npz_deflate1(activation_path, {"X": matrix, "item_ids": item_ids})`<br>`extract.py:166-174` `{"X": …, "item_ids": …, "layers": …, "token_lengths": …}`<br>`extract.py:175` `write_jsonl(output_dir / "meta.jsonl", rows)`<br>`extract.py:176-202` config dict<br>405B: `ndif_extract_405b.py:234-242` |
| **9. On-disk shape / bytes-per-item** | Verified by reading the actual npz central directories on HF (see below). | `extract.py:169` `np.stack([per_layer[index] for index in target_layers], axis=1)` → `[n_items, n_layers, hidden]`; `ndif_repack_405b.py:70` `np.savez(tmp, X_bf16_u16=arr, item_ids=ids_arr, layer=np.int16(L))` |

---

## 9 in detail — measured, not inferred

Read remotely from `format-EA/full320-v3` (zip central directory + `.npy`
headers only; nothing downloaded).

| file | members | raw bytes | file on disk | bytes/item (raw → on disk) |
|---|---|---|---|---|
| `llama31_8b/activations_all_layers.npz` | `X (11520, 32, 4096) float32`, `item_ids (11520,) <U56`, `layers (32,) int16`, `token_lengths (11520,) int32` | 6,039,797,888 | 2,800,419,373 | 524,288 → 243,092 |
| `llama31_8b/activations.npz` | `X (11520, 4096) float32`, `item_ids` | 188,743,808 | 87,905,030 | 16,384 → 7,630 |
| `llama31_70b/activations_all_layers.npz` | `X (11520, 80, 8192) float32`, + same three | 30,198,988,928 | 15,608,633,550 | 2,621,440 → 1,354,916 |
| 405B `layers/layer_XXX.npz` x 126 | `X_bf16_u16 (11520, 16384) uint16`, `item_ids`, `layer` | 377,487,360 / layer | 380,068,606 / layer (uncompressed `np.savez`) | 4,128,768 → 4,156,997 (all 126 layers) |

Compression is deflate level 1 (`extract.py:11-26`) — observed ratios 0.464 (8B)
and 0.517 (70B).

---

## The all-positions calculation

Token counts come from the stored `token_lengths` array inside the 8B npz (real,
unpadded prompt lengths):

```
n = 11520,  min 63,  max 461,  mean 205.197,  median 186,  sum 2,363,874
```

Hidden sizes / layer counts are the pinned model configs: 8B = 4096 x 32
(`0e9e39f249a16976918f6564b8830bc894c89659`), 70B = 8192 x 80
(`1605565b47bb9346c5515c34102e054115b4f98b`).

Per-token-per-item, fp32 (the current save dtype):

```
8B  = 32 x 4096 x 4 =   524,288 B
70B = 80 x 8192 x 4 = 2,621,440 B
```

### Per 1,000 items, caching all real token positions (ragged/packed, mean 205.2 tokens)

| | current (last token) | **all positions, fp32** | all positions, deflate-1 est. | all positions, bf16 raw |
|---|---|---|---|---|
| **8B** (32 x 4096) | 0.52 GB | **107.6 GB** (100.2 GiB) | ~49.9 GB | 53.8 GB |
| **70B** (80 x 8192) | 2.62 GB | **537.9 GB** (501.0 GiB) | ~278.0 GB | 269.0 GB |

A **205.2x** blowup over the current last-token-only cache.

Exact byte counts:

```
8B  : 205,197.4 tok x   524,288 B = 107,582,532,267 B
70B : 205,197.4 tok x 2,621,440 B = 537,912,661,333 B
```

If you padded to a rectangular `[1000, 461, L, H]` instead of packing ragged:
**241.7 GB** (8B) and **1,208.5 GB** (70B).

Scaled to the actual full320-v3 set (11,520 items): **1.24 TB** for 8B and
**6.20 TB** for 70B in fp32.

No code was changed.
