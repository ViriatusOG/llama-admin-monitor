# Gap analysis: llama-admin-monitor presets vs. Unsloth

**Date:** 2026-09-22
**Scope:** which model-tuning settings your preset editor is missing compared to what
[Unsloth](https://github.com/unslothai/unsloth) documents and exposes, with the full current
[`llama-server`](https://github.com/ggml-org/llama.cpp/tree/master/tools/server) flag surface as the reference.

## TL;DR

Unsloth does **not** have its own settings system. `unsloth run` is a thin wrapper around the
**same `llama-server` binary you launch** — their API docs state it plainly: *"Unsloth supports
most llama-server runtime flags… See the llama-server documentation for the full list of supported
runtime flags."* What they add on top:

1. **Per-model auto-recommended settings** — if no flags are set, Unsloth picks "the best/
   recommended settings for the model including context length, temperature etc." Their
   [Qwen3.8-27B page](https://unsloth.ai/docs/models/qwen3.8) publishes an exact per-mode table
   (see below) — and Qwen3.8-27B is the model you run.
2. **First-class reasoning/thinking control** (effort levels + "Preserve Thinking" toggles) —
   their headline feature for Qwen3.8.
3. **Server security model** — generated API keys, localhost-by-default, server-side tools
   switched off when bound to `0.0.0.0`.

So the real gap is: **llama-server flags that are worth exposing in your preset editor**,
prioritised by how much they matter for your models (Qwen3.8-27B: hybrid thinking + vision,
256K ctx, large Q8 quant). Everything below is available to you *today* via the `extra_args`
field — the gaps are UI fields, sane defaults, and capability gating.

---

## What your presets already expose (no gap)

From `ServerConfig` → `src/llama/server.rs`:

| Category | Your fields |
|---|---|
| Model | `model_path`, `mmproj`, port, `extra_args` |
| Offload / GPU | `gpu_layers` (`-ngl`), `ctk`, `ctv`, `flash_attn` (`-fa`), `devices` (`--device`), `tensor_split` (`-ts`), `split_mode`, `main_gpu` (`-mg`), backend/build per preset |
| Memory | `no_mmap`, `mlock` |
| Threading | `threads` (`-t`), `threads_batch` (`-tb`) |
| Context | `context_size` (`-c`), `batch_size` (`-b`), `ubatch_size` (`-ub`), rope scaling trio (**+ your auto-YaRN above 256K**, which matches Unsloth's "extend to 1M via YaRN" note) |
| Speculative | ngram-spec toggle, `spec_ngram_size`, `draft_model` (`-md`), `draft_min`, `draft_max` |
| Slots / other | `parallel_slots` (`--parallel`), `seed`, `system_prompt_file` |
| Forced on | `--jinja` (correct for Qwen), `--no-warmup`, `--metrics`, `--webui-mcp-proxy`, `--host 0.0.0.0` |

That's already the core of Unsloth's advertised surface ("context sizing, GPU layers, threading,
sampling, networking, and tool configuration" — you have all except sampling and tools).

## What Unsloth documents as first-class (the comparison target)

From their [API docs](https://unsloth.ai/docs/basics/api), [desktop docs](https://unsloth.ai/docs/desktop)
and [Qwen3.8 page](https://unsloth.ai/docs/models/qwen3.8):

- Sampling: `--temp --top-p --top-k --min-p --repeat-penalty --seed` (server defaults, overridable per request)
- Reasoning: `--reasoning on/off`, `--reasoning-effort xhigh|medium|low|none`,
  `--chat-template-kwargs '{"reasoning_effort":"medium"}'`, "Think" + "Preserved Thinking" UI toggles
- Context & threads: `-c 131072 --threads 32`
- Networking: `-H 0.0.0.0 -p 8888` (you hard-bind 0.0.0.0; they default to localhost)
- Tools: `--enable-tools` / `--disable-tools` with a security policy (on for localhost, off for
  non-loopback, process-level hard override)
- API keys: `sk-unsloth-…` per named key (maps to llama-server's `--api-key`)
- Auto RAM offload + multi-GPU detection; model hub with quant variants (≈ your HF downloads page)
- **Qwen3.8-27B recommended sampling** (their docs, verbatim):

| | thinking mode | non-thinking mode |
|---|---|---|
| `temperature` | 1.0 | 0.7 |
| `top_p` | 0.95 | 0.80 |
| `top_k` | 20 | 20 |
| `min_p` | 0.0 | 0.0 |
| `presence_penalty` | 0.0 | 1.5 |
| `repetition_penalty` | 1.0 | 1.0 |

Your presets pass **no sampling flags at all**, so llama.cpp's generic defaults
(temp 0.8, top_p 0.9, top_k 40, min_p 0.05, repeat 1.1) apply to a model whose vendor-recommended
values differ in every slot. That's the single most concrete "I'm running the wrong defaults"
item in this analysis.

They also explicitly recommend **MTP speculative decoding** for Qwen3.8 ("prepare to have 1–2 GB
extra headroom") — your app hardcodes `--spec-type ngram-mod`.

---

## Gap 1 — Sampling defaults (Tier 1, highest value)

Your server currently sets no sampling defaults. Add a **Sampling** preset section:

| Flag | Note |
|---|---|
| `--temp` | |
| `--top-p` | |
| `--top-k` | |
| `--min-p` | |
| `--typical-p`, `--top-nsigma` | optional extras |
| `--repeat-penalty`, `--presence-penalty`, `--frequency-penalty` | |
| `--repeat-last-n` | |
| `--ignore-eos` | |

These are **server startup defaults**; OpenAI-API clients (pi, SDKs) can still override per
request, so a Sampling section locks nothing in. Suggested preset defaults for Qwen-class
models: 1.0 / 0.95 / 20 / 0.0 / 1.0 (the thinking-mode row above), plus a "reasoning off"
profile of 0.7 / 0.80 / 20 / presence 1.5.

## Gap 2 — Reasoning / thinking control (Tier 1)

Your model is a *hybrid thinking* model and Unsloth's whole UI story for it is effort levels.
llama-server now supports all of these at startup:

| Flag | Note |
|---|---|
| `--rea` (`--reasoning`) | `on` / `off` / `auto` |
| `--reasoning-effort` | `xhigh` (default) / `medium` / `low` / `none` |
| `--reasoning-preserve` | "Preserve Thinking" — keeps prior thinking traces |
| `--reasoning-budget`, `--reasoning-budget-message` | budget caps |
| `--reasoning-format` | |
| `--chat-template-kwargs` | one JSON text field, e.g. `{"reasoning_effort":"medium"}` — the escape hatch for model-specific template knobs |

A **Reasoning** section (effort dropdown + preserve checkbox + kwargs text field) is the
closest single feature to "what Unsloth does and you don't."

## Gap 3 — Speculative decoding beyond ngram-mod (Tier 1)

You hardcode `--spec-type ngram-mod`. Current llama-server accepts:

```
none, draft-simple, draft-eagle3, draft-mtp, draft-dflash, draft-dspark,
ngram-simple, ngram-map-k, ngram-map-k4v, ngram-mod, ngram-cache
```

Missing knobs:

| Flag | Note |
|---|---|
| `--spec-type` menu | **`draft-mtp` is Unsloth's recommendation for Qwen3.8** |
| `--spec-default` | |
| `--spec-ngram-size-m`, `--spec-ngram-min-hits` | ngram-map knobs |
| `--spec-ngram-mod-n-match/-n-min/-n-max` | refine your existing ngram-mod setup |
| `--draft-p-min` | draft probability threshold |
| `--spec-draft-ngl` (`-ngld`), `--spec-draft-device` (`-devd`), `--spec-draft-threads` (`-td`), `--spec-draft-threads-batch` (`-tbd`), `--spec-draft-type-k/-v` (`-ctkd`/`-ctvd`), `--spec-draft-n-cpu-moe` | tune the draft model itself (you currently only pick its file) |
| `--spec-synth-len`, `--spec-synth-rates` | synthetic-rate spec |

## Gap 4 — KV cache & memory fitting (Tier 1 for a 27B Q8)

| Flag | Note |
|---|---|
| `--fit` / `--fitc` / `--fitt` | **auto-fit context to available RAM/VRAM** — pairs naturally with your existing auto-YaRN; replaces manual context guessing |
| `--cram` (`--cache-ram`) | max cache size in MiB (0 = disabled, −1 = no limit, default 8192) |
| `--kv-offload` / `--no-kv-offload` | KV to CPU when VRAM is tight |
| `--kv-unified`, `--kv-unified-per-slot` | |
| `--context-shift` | auto-shift long conversations |
| `--ctxcp` (`--ctx-checkpoints`) | sliding-window checkpoints for huge ctx |
| `--swa-full` | full attention for SWA models |
| `--cache-reuse` | |
| `--dt` (`--defrag-thold`) | |
| `--no-warmup` toggle | you currently force it off — make it a checkbox |

## Gap 5 — Multimodal projector placement (Tier 1, you just lived this)

You fixed the stale-mmproj bug; the natural sibling knobs are:

| Flag | Note |
|---|---|
| `--mmdev` (`--mmproj-device`) | pin the projector to a specific GPU |
| `--mmproj-offload` | |
| `--image-min-tokens` / `--image-max-tokens` | per-image budget |
| `--mtmd-batch-max-tokens` | |

(Qwen3.8-27B is a vision model per Unsloth's page, so mmproj tuning is live for you, not
hypothetical.)

## Gap 6 — CPU / MoE offload & placement (Tier 2)

| Flag | Note |
|---|---|
| `-ncmoe` (`--n-cpu-moe`) | keep MoE experts on CPU — the big one for big MoE models |
| `--ncffn` | CPU FFN offload |
| `--cpu-moe`, `--cpu-strict`, `--cpu-strict-batch` | |
| `--op-offload` | |
| `--ot` (`--override-tensor`) | `pattern=TYPE` tensor placement overrides — power feature |
| `--override-kv` | override GGUF KV (e.g. rope/expert settings) |
| `--numa` | |
| `--cpu-mask`, `--cpu-range` (+ `-batch` variants) | core pinning |
| `--repack`, `--lzm` (lazy mode), `--load-mode` | load-time behaviour |

## Gap 7 — LoRA adapters (Tier 2)

| Flag | Note |
|---|---|
| `--lora FILE` | load a LoRA at server start |
| `--lora-scaled FILE:SCALE` | multiple/scaled adapters |
| `--lora-init-without-apply` | |

This lets users run Unsloth (or anyone's) fine-tuned LoRAs on top of a base GGUF without
re-quantising — a natural extension of your model-management story.

## Gap 8 — Server security & exposure (Tier 2 — Unsloth's model)

You bind `0.0.0.0` unauthenticated on every preset. Unsloth's whole security story exists
because of this:

| Flag | Note |
|---|---|
| `--api-key` / `--api-key-file` | auth on the OpenAI endpoint (their `sk-unsloth-…` keys map to this) |
| host binding choice | let presets bind `127.0.0.1` instead of always `0.0.0.0` |
| `--ssl-cert-file` / `--ssl-key-file` | HTTPS |
| `--sleep-idle-seconds` | unload to memory when idle — RAM saver |
| `--to` (timeout) | |
| `--offline` | |
| `--reuse-port`, `--no-slots`, `--slots` | |
| `--threads-http`, `--sse-ping-interval` | |

## Gap 9 — Agent/chat tuning (Tier 2)

| Flag | Note |
|---|---|
| `--prefill-assistant` | prefill assistant turns — better agent TTFT (you run pi) |
| `--grammar` / `--grammar-file` | constrained decoding default |
| `--tools` + `--mcp-servers-json` / `--mcp-servers-config` | server-side tools; you already proxy MCP for the webui, so exposing tool lists is consistent |
| `--skip-chat-parsing` | |
| `--keep` | |
| `--spm-infill`, `--completion-bash`, `--agent` | specialised modes |
| `--chat-template` / `--chat-template-file` | custom template (you force `--jinja`) |

## Tier 3 — niche; `extra_args` is enough for now

`--mirostat` (+`-ent`/`-lr`), `--dry-base`/`--dry-multiplier`/`--dry-allowed-length`/
`--dry-penalty-last-n`/`--dry-sequence-breaker` (dynamic repetition penalty), `--dynatemp-range`/
`--dynatemp-exp`, `--xtc-probability`/`--xtc-threshold`, `--logit-bias`, `--sampler-seq` /
`--samplers`, `--poll`/`--poll-batch`, `--prio`/`--prio-batch`, `--tags`, `--log-file` /
`--log-jsonl` / `--log-prompts-dir`, `--props`, `--slot-save-path`, `--sps`, `--media-path`,
`--video-*`, and the embedding/rerank server modes (`--embedding`, `--rerank`, `--pooling`,
`--embd-normalize`).

## Architectural differences (not preset fields)

- **Multi-model single server** — llama-server can serve many models from one process
  (`--models-dir`, `--models-preset`, `--models-max`, `--alias`). You deliberately run one
  server per preset; keep it that way unless multi-model serving becomes a request.
- **llama.cpp built-in web UI** (`--ui`) — you have your own; no gap.
- **Unsloth product layer** — model hub with quant variants (≈ your HF page), per-model
  auto-recommended settings, API-key management UI, Anthropic `/v1/messages` dialect,
  server-side tools with "self-healing" tool calls, fine-tuning, diffusion, RAG. Only the
  first three are adjacent to your app; the rest are out of scope for a server monitor.

## Implementation notes

1. **Version-gate the fields.** The flags above are from llama.cpp `master`. Your Install page
   ships prebuilt ggml releases and older builds lack newer flags (`--rea`, `--cram`, `--fit`,
   `--swa-full`, `--ctxcp`, `--mmdev`, `--spec-draft-*` are all recent). Unknown flags crash
   `llama-server` at startup. You already run `--list-devices` on every install — run
   `llama-server --help` too, parse it, and store a per-build flag capability set; grey out
   fields the selected build doesn't support.
2. **Server defaults ≠ lock-in.** Startup sampling/reasoning flags are defaults; per-request
   overrides still work through your OpenAI-compatible endpoint.
3. **Auto-recommended defaults.** You already write per-model `.meta.json` sidecars from HF.
   A small bundled table (model → recommended temp/top_p/top_k/presence_penalty/reasoning
   effort, i.e. the Qwen3.8 table above) could pre-fill a new preset's Sampling + Reasoning
   sections — that's the Unsloth "settings auto-set" behaviour, cheap to replicate.
4. **YaRN already matches.** Your auto `--rope-scaling yarn` + computed `--rope-freq-scale`
   above 262144 does exactly what Unsloth's "extend to 1M via YaRN" note describes.

## Suggested first batch

> **Status (v2026.09.22-beta.20):** the first batch below has been implemented — the preset
> editor gained Sampling, Reasoning, Memory/KV and Projector sections plus a spec-type
> selector and draft-model tuning fields. Qwen-recommended values are surfaced as in-field
> hints in the Sampling section rather than hard defaults.

1. **Sampling section** (8 fields) + Qwen-recommended defaults
2. **Reasoning section** (`--rea`, `--reasoning-effort`, `--reasoning-preserve`, `--chat-template-kwargs`)
3. **Spec-type menu** (add `draft-mtp` for Qwen3.8) + draft-model offload fields
4. **Memory section** (`--fit`, `--kv-offload`, `--cram`, `--context-shift`, warmup checkbox)
5. **Projector section** (`--mmdev`, `--mmproj-offload`, image token limits)

---

### Sources

- llama-server full flag reference: <https://github.com/ggml-org/llama.cpp/tree/master/tools/server> (`README.md`, 275 flags)
- Unsloth `unsloth run` / API docs: <https://unsloth.ai/docs/basics/api>
- Unsloth Desktop docs: <https://unsloth.ai/docs/desktop>
- Unsloth Qwen3.8 run guide (recommended settings table, MTP, reasoning effort): <https://unsloth.ai/docs/models/qwen3.8>
- Unsloth llama.cpp fork (tracks upstream; no fork-specific server flags): <https://github.com/unslothai/llama.cpp>
