# Llama Admin Monitor

Web control panel for [llama.cpp](https://github.com/ggerganov/llama.cpp) servers: live GPU and inference monitoring, model management with Hugging Face downloads, multi-GPU VRAM visualisation, and automated tensor-split benchmarking — in a single self-contained Rust binary.

It began as a fork of [arte-fact/llama-monitor](https://github.com/arte-fact/llama-monitor) (extended into an admin dashboard as [ViriatusOG/llama-monitor](https://github.com/ViriatusOG/llama-monitor)) and now carries a UI modelled on [LLama-GUI](https://github.com/thomas9120/LLama-GUI): a grouped sidebar, card-based pages, and five WCAG-AA themes.

## Screenshots

Monitor, in the **Nebula** theme:

![Monitor](docs/images/monitor.png)

| Presets | Models |
| --- | --- |
| ![Presets](docs/images/presets.png) | ![Models](docs/images/models.png) |

| Benchmark | Chat |
| --- | --- |
| ![Benchmark](docs/images/benchmark.png) | ![Chat](docs/images/chat.png) |

Same dashboard in the **Mint** light theme:

![Mint theme](docs/images/theme-mint.png)

## Features

### Monitoring
- **Live runtime strip** — server state, loaded model, endpoint and active preset at the top of the Monitor page, plus a persistent runtime summary in the sidebar
- **Process output** — llama-server's stderr streamed into a terminal card, with Clear
- **Inference card** — prompt/generation speed, slot status and KV-cache occupancy (with a bar that turns amber at 80 % and red at 95 %), from llama-server's Prometheus endpoint
- **VRAM usage card** — one segmented bar across every GPU, coloured per vendor (AMD, NVIDIA, Intel) with an estimated context/KV segment and free space, labelled in GB
- **One card per GPU** — utilisation and VRAM bars, temperature, power draw vs. limit (flagged when capped), core and memory clocks; AMD, NVIDIA and Intel cards are shown together

### Server management
- **Launch from the sidebar** — pick a preset and port, Start/Stop, and open llama-server's own web UI while it runs; the button reflects live state
- **Presets page** — every saved configuration with its key parameters as chips; switch the active preset, edit, copy, delete, or reset to defaults
- **Preset editor** — collapsible sections covering every llama.cpp parameter; persisted to disk
- **Failure detection** — if llama-server exits on its own (a model that won't fit, a bad flag) the dashboard resets to a stopped state and surfaces the error on the Monitor page

### Model management
- **Models page** — every `.gguf` in your models directory with quantisation, size, VRAM fit, source repo, download date and the repo's last-updated date on Hugging Face; sortable columns
- **Hugging Face downloads** — search repos, browse their `.gguf` files with sizes and a VRAM-fit check *before* downloading, then download with a live progress bar that keeps going if you close the dialog
- **Delete** models from disk without leaving the dashboard

### Optimisation
- **Benchmark page** — sweeps a list of tensor-split ratios through `llama-bench`, reports prompt and generation throughput, marks the fastest and applies it to a preset in one click; cancellable, keeping results so far

### Interface
- **Five themes** — Tokyo and Nebula (dark), Graphite (mid-tone), Cappuccino and Mint (light), picked from the sidebar and remembered per browser
- **Integrated chat** — streaming chat proxied to the running server, with collapsible reasoning blocks and Markdown rendering
- **File browser** for binaries, directories and models; **persistent settings** (preset, port, paths, models directory); **responsive** layout with a navigation drawer on phones; installable as a PWA

## Supported hardware

| Vendor | Metrics | Detection |
|--------|---------|-----------|
| AMD | `rocm-smi` | `rocminfo` |
| NVIDIA | `nvidia-smi` | `nvidia-smi` |
| Intel | `xpu-smi` | — |

Multiple vendors are monitored at once — a machine with both an AMD and an NVIDIA card gets one card per GPU. Override detection with `--gpu-backend rocm|nvidia|none`.

**RDNA 4 note:** `gfx1201` (RX 9070 / 9070 XT / 9070 GRE) requires ROCm 7.2 or newer for `rocminfo` to enumerate the GPU. The versions of `rocminfo` and `rocm-smi` in Ubuntu's default repositories predate RDNA 4 and will not detect these cards; install ROCm from AMD's repository instead.

## Installation

### From source

```bash
# Install Rust if needed: https://rustup.rs
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh

git clone https://github.com/ViriatusOG/llama-admin-monitor.git && cd llama-admin-monitor
cargo build --release
```

The binary is at `target/release/llama-admin-monitor`. It's a single self-contained executable — the frontend is embedded at compile time.

### Dependencies

- **llama.cpp** — `llama-server` (with `--metrics` and `--jinja` support). `llama-bench` is required for the Benchmark page and is expected alongside `llama-server` in the same directory.
- **GPU monitoring** (optional) — `rocm-smi` / `rocminfo` (AMD) or `nvidia-smi` (NVIDIA)
- The UI loads the Inter and JetBrains Mono fonts from Google Fonts when the browser is online and falls back to system fonts otherwise.

## Quick start

```bash
# Configure paths in the web UI
./llama-admin-monitor

# Or specify them up front
./llama-admin-monitor \
  --llama-server-path /path/to/llama-server \
  --models-dir ~/models \
  --port 7778
```

Open `http://localhost:7778`. Click **Settings** in the sidebar to set your llama-server binary and models directory, create a preset on the **Presets** page, then **Start server** from the sidebar.

There is no built-in authentication. Keep it on a trusted network or behind an authenticated reverse proxy.

### Running as a service

```ini
# /etc/systemd/system/llama-admin-monitor.service
[Unit]
Description=Llama Admin Monitor
After=network.target

[Service]
Type=simple
User=youruser
WorkingDirectory=/home/youruser/llama-admin-monitor
ExecStart=/home/youruser/llama-admin-monitor/target/release/llama-admin-monitor --port 7778
Restart=always
RestartSec=3

[Install]
WantedBy=multi-user.target
```

```bash
sudo systemctl daemon-reload
sudo systemctl enable --now llama-admin-monitor
```

## CLI reference

| Flag | Short | Default | Description |
|------|-------|---------|-------------|
| `--llama-server-path` | `-s` | `llama-server` | Path to the llama-server binary (uses `$PATH` if a bare name) |
| `--llama-server-cwd` | | `.` | Working directory for llama-server |
| `--models-dir` | | — | Directory scanned for `.gguf` files; also the download target |
| `--port` | `-p` | `7778` | Monitor web UI port |
| `--presets-file` | | `~/.config/llama-admin-monitor/presets.json` | Custom presets file location |
| `--gpu-backend` | | `auto` | Force GPU backend: `auto`, `rocm`, `nvidia`, `none` |
| `--gpu-arch` | | from config | GPU architecture for ROCm (e.g. `gfx1100`, `gfx1201`, `auto`) |
| `--gpu-devices` | | from config | Visible GPU device indices (e.g. `0,1`) |

All paths can also be set from the Settings dialog. UI settings take precedence over CLI defaults and persist to `~/.config/llama-admin-monitor/ui-settings.json`.

### Upgrading from llama-monitor

Presets, GPU environment and UI settings are read from `~/.config/llama-admin-monitor/`. If that directory does not exist yet but `~/.config/llama-monitor/` (from the original fork) does, the old directory is used as-is, so nothing needs to be migrated. Theme choice lives in the browser and is chosen fresh.

## Tensor split format

llama.cpp expects tensor splits **slash-separated** — `65/35` for two GPUs, `7/8/8/8` for four. A comma-separated value is parsed as a single number, which silently places the entire model on the first GPU. The Benchmark page uses slashes throughout; if you're migrating presets from elsewhere, check this first when a split doesn't seem to be taking effect.

## Web UI

The sidebar groups the workspace the way LLama-GUI does:

- **Monitor** — runtime strip, process output, inference card, VRAM bar and one card per GPU. A **Live** badge appears while the server runs.
- **Tune → Presets** — the saved configurations; the active one is what the sidebar's Start button launches.
- **Tune → Benchmark** — tensor-split sweeps with one-click apply, plus the VRAM bar for reference.
- **Interact → Chat** — streaming chat against the running server.
- **Library → Models** — the models directory with VRAM fit and Hugging Face provenance, download and delete.

The sidebar footer holds the runtime summary, preset and port selectors, Start/Stop, the theme menu and **Settings** (paths and GPU environment).

### Themes

Every colour in the UI comes from a token in `static/tokens.css`, and each theme is one block in that file. The palettes are ported from LLama-GUI, where they were tuned to WCAG AA contrast. Adding a theme is one entry in the `THEMES` registry in `static/theme.js` plus one token block.

| Theme | Tone |
|-------|------|
| Tokyo (default) | Dark, blue accent |
| Nebula | Deep-space dark, violet→pink→amber accent |
| Graphite | Neutral mid-tone, copper accent |
| Cappuccino | Warm light |
| Mint | Pale seafoam light |

## Preset parameters

The preset editor groups llama.cpp parameters into collapsible sections:

- **Model & memory** — model path (with file browser and HF download), GPU layers, no-mmap, mlock
- **Context & KV cache** — context size, K/V quantisation (`f16`/`q8_0`), flash attention
- **Batching & slots** — batch size, micro-batch, parallel slots
- **GPU distribution** — tensor split, backend (Vulkan/CUDA), split mode, main GPU
- **Threading** — generation and batch thread counts
- **Rope scaling** — YaRN/linear scaling, frequency base/scale
- **Speculative decoding** — ngram-mod, draft model, draft min/max
- **Advanced** — seed, system prompt file, extra CLI args

## API reference

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/` | Dashboard HTML |
| `GET` | `/ws` | WebSocket (real-time metrics push) |
| `POST` | `/api/start` | Start llama-server with a `ServerConfig` body |
| `POST` | `/api/stop` | Stop the running server |
| `GET` | `/api/presets` | List presets |
| `POST` | `/api/presets` | Create a preset |
| `PUT` | `/api/presets/{id}` | Update a preset |
| `DELETE` | `/api/presets/{id}` | Delete a preset |
| `POST` | `/api/presets/reset` | Reset presets to defaults |
| `GET` | `/api/models` | List discovered models |
| `POST` | `/api/models/refresh` | Rescan the models directory |
| `POST` | `/api/models/delete` | Delete a model file |
| `GET` | `/api/hf/search?q=` | Search Hugging Face for GGUF repos |
| `GET` | `/api/hf/files?repo=` | List `.gguf` files in a repo |
| `POST` | `/api/hf/download` | Download a file to the models directory |
| `POST` | `/api/bench/run` | Start a tensor-split benchmark sweep |
| `POST` | `/api/bench/cancel` | Cancel a running sweep |
| `GET` | `/api/settings` | Get persisted UI settings |
| `PUT` | `/api/settings` | Save UI settings |
| `GET` | `/api/browse?path=&filter=` | Browse the filesystem |
| `GET` | `/api/gpu-env` | Get GPU environment config |
| `PUT` | `/api/gpu-env` | Save GPU environment config |
| `POST` | `/api/chat?port=` | Streaming proxy to `/v1/chat/completions` |

## Architecture

```
src/
  main.rs              -- Entry point: CLI parsing, wiring, tokio::main
  cli.rs               -- Clap argument definitions
  config.rs            -- AppConfig resolved from CLI args; config-dir fallback
  state.rs             -- Shared AppState, UiSettings persistence
  gpu/
    mod.rs             -- GpuMetrics, GpuBackend trait, multi-vendor detection
    rocm.rs            -- AMD via rocm-smi JSON
    nvidia.rs          -- NVIDIA via nvidia-smi CSV
    env.rs             -- GPU environment config, architecture table
    dummy.rs           -- No-op backend for headless/testing
  llama/
    metrics.rs         -- Prometheus text format parser
    server.rs          -- Subprocess management, exit detection
    poller.rs          -- Async polling loop for /health, /metrics, /slots
    bench.rs           -- llama-bench sweep runner with cancellation
  presets/
    mod.rs             -- ModelPreset, CRUD, file persistence
  models/
    mod.rs             -- GGUF discovery, filename parsing, metadata sidecars
    hf.rs              -- Hugging Face search, file listing, downloads
  web/
    mod.rs             -- Warp route composition
    api.rs             -- REST handlers, file browser, chat proxy
    ws.rs              -- WebSocket real-time metrics push
    static_assets.rs   -- Embedded frontend
static/
  index.html           -- App shell: sidebar, pages, dialogs
  tokens.css           -- Design tokens: one palette block per theme
  style.css            -- Layout and components (colours only via tokens)
  theme.js             -- Theme registry, persistence, sidebar theme menu
  app.js               -- Navigation, WebSocket rendering, pages, dialogs
  manifest.json, sw.js, icon.svg
```

### Data flow

```
GPU (rocm-smi/nvidia-smi)  -->  GPU poller (500ms)   --> AppState
llama-server /metrics      -->  Llama poller (1s)    --> AppState
HF download / benchmark    -->  Background tasks     --> AppState
                                                          |
                                                     WebSocket (500ms)
                                                          |
                                                       Browser
```

Downloaded models get a `<filename>.meta.json` sidecar recording the source repo, the repo's download count and last-modified date, and when the file was fetched. Models without a sidecar (added by hand, or downloaded before this feature) simply show no provenance.

## Development

```bash
cargo run                    # Debug mode
cargo test                   # Run tests
cargo clippy -- -D warnings  # Lint
cargo fmt                    # Format
```

Frontend files in `static/` are embedded at compile time via `include_str!` — no Node.js or build tooling required. Rebuild after changing them.

## Credits and licence

- Based on [arte-fact/llama-monitor](https://github.com/arte-fact/llama-monitor), extended in [ViriatusOG/llama-monitor](https://github.com/ViriatusOG/llama-monitor).
- The interface design — layout, components, the five theme palettes in `static/tokens.css` and the theme-menu behaviour in `static/theme.js` — is ported from [thomas9120/LLama-GUI](https://github.com/thomas9120/LLama-GUI), which is licensed under the GNU General Public License v3.0.

Because it includes material derived from LLama-GUI, this project is distributed under the [GNU General Public License v3.0](LICENSE).
