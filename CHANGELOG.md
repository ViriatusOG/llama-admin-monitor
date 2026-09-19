# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and stable releases use [Semantic Versioning](https://semver.org/) from 1.0.0 on (earlier stable releases were CalVer `YYYY.M.D`). Beta releases keep zero-padded CalVer tags, `vYYYY.MM.DD-beta.NN`.
Backwards compatibility is preserved unless explicitly noted.

## [Unreleased]
### Fixed
- The Pi page reported "pi not installed" when the monitor runs as a systemd service: the service's minimal `PATH` lacks the per-user directories a login shell adds (nvm/fnm/volta node versions, npm/pnpm/yarn/bun globals, `~/.local/bin`, pi's own `~/.pi/bin`). The monitor now searches those directories (including pi's own installer location, `~/.local/share/pi-node`), falls back to asking a login shell (`bash -lc 'command -v pi'`), and gives the spawned pi a `PATH` that includes its own `node`.

## [1.1.1] - 2026-09-19
### Changed
- Pi's model list is now one entry per **preset**, named after the preset (previously entries were per model file, so presets sharing a file collapsed into one and showed the file name). pi starts on the active preset.

## [1.1.0] - 2026-09-19
### Fixed
- The Pi terminal was capped at 24 rows regardless of the window: xterm.js gives its root element the class `terminal`, which collided with the Process Output panel's `.terminal` rule (`max-height: 360px`). The panel's class is now `log-terminal`, and the terminal follows the browser window size (re-fitting on resize, font load and page switch).

### Added
- **Pi page** (sidebar → Interact → Pi): the [pi](https://pi.dev) coding agent running in a terminal on the server, embedded in the dashboard (xterm.js over a WebSocket-attached PTY). Pick a working directory, Start, and pi is launched with a `llama-admin-monitor` provider that the monitor writes into `~/.pi/agent/models.json`, pointing at the monitor's `/v1` proxy; every preset is listed as a model and pi starts on the loaded model (or the active preset). The session survives page changes and reloads (scrollback is replayed on reattach); Stop kills it. If pi is missing, **Install pi** runs pi's own installer in the same terminal. `POST /api/pi/start|stop|install`, `GET /api/pi/status`, `/ws/pi`.

## [1.0.0] - 2026-09-19
First SemVer stable release; it carries everything from the `v2026.9.21` / `v2026.09.22` beta series below. In-app updates from `v2026.9.20` treat it as newer.

### Added
- **Install page** (sidebar → Library → Install): installs prebuilt llama.cpp releases straight from ggml-org's GitHub releases, one directory per backend and version, no compiler needed. Offers Vulkan, CUDA 12.8, CUDA 13.3 (both with the CUDA runtime libraries bundled), ROCm 10.0 and CPU on Linux x64; Vulkan/CUDA/CPU on Linux arm64; Metal on Apple Silicon. Each install is smoke-tested with `--list-devices` and lists the devices it found. Builds can be removed, or set as the binary in Settings with one click.
- Presets choose a build under GPU distribution → Build (installed builds, the configured binary, or the legacy `build-cuda` tree); the Devices picker and the Benchmark page follow that choice.

### Added (GPU cards)
- AMD GPU cards show all three temperature sensors rocm-smi reports: **Hotspot** (junction, the headline value and what the card throttles on), **Edge** and **Memory**. Hotspot and memory use looser warning thresholds (95/105 C) than edge and NVIDIA's single sensor (80/90 C), matching what those sensors normally run at.

### Added
- The sidebar's Runtime panel shows the OpenAI-compatible API URL (`http://<monitor>:<port>/v1`, proxied to the running llama-server) with a Copy button, so the endpoint is discoverable in-app rather than only in the README.
- **GPU Activity card** (nvtop-style, two card widths): per-GPU graphs of utilisation and VRAM over the last five minutes, and a table of the processes using each GPU with VRAM and GPU-busy share. NVIDIA processes come from `nvidia-smi --query-compute-apps`; AMD/Intel from each process's `/proc/<pid>/fdinfo` DRM accounting (the same source nvtop uses), refreshed every 2 s. Only processes the monitor's user can read are listed unless it runs as root. GPU metrics now carry the PCI bus id so processes match their card.
- Disk card laid out like the GPU cards: the physical drive behind the models directory (model, NVMe/SSD/HDD, capacity) with temperature (kernel hwmon, no privileges on NVMe), SMART health, wear, spare blocks, power-on time, total written, media errors and reallocated sectors. SMART comes from `smartctl -j -H -A` (plain, then `sudo -n`; README has the sudoers line) and is refreshed once a minute.
- Install page: each installed llama.cpp build shows whether it is on the newest upstream release ("up to date" / "bNNNNN available") with an **Update to bNNNNN** button. Updating installs the new tag for the same backend, moves presets and Settings that used the old build over to it, and removes the old one; a summary line above the list says how many builds are behind. (`POST /api/builds/update {id}`)

### Changed
- The Inference card matches the other cards: same size, the loaded model as its title while running, prompt/generation speed side by side, a KV-cache row and bar, and a detail block with slots, requests in flight, and prompt/generated token totals since the server started.
- Release tags are zero-padded from now on (`v2026.09.22-beta.01`), so GitHub's text-sorted release list matches numeric order. The updater already compares numerically and accepts both spellings.
- Monitor cards keep their drag grip, Hide and icon on the title line; a long device name wraps inside its own column instead of pushing the tools down. GPU card titles drop the vendor prefix the kicker already shows ("Radeon AI PRO R9700" under "GPU 0 · AMD"; the full name is in the tooltip).
- The GPU Memory Pool card is the same size as the GPU cards and lists every device's used/total VRAM under the pooled bar, in the vendor colours of the legend; the total moved from the header badge to a "Used" row.

### Fixed
- Monitor cards no longer get squeezed on wide screens: the minimum card width is 320 px (readings never wrap), so a 4K display shows fewer, wider cards per row instead of seven cramped ones.
- Crash diagnosis explains two more llama-server failures: "device VulkanN does not support split buffers" (split mode `row` is CUDA-only; use `layer`) and an invalid `--device` name (the preset's Devices belong to a different build).
- The preset editor's Build choice was dropped on save (the field was missing from the preset model on the server), so every preset silently launched the configured binary. It is now stored with the preset.
- Update checks compared release versions by GitHub's listing order, which sorts by tag name — so `beta.9` outranked `beta.12` and a fresh install was offered a downgrade. Versions are now parsed (`YYYY.M.D[-beta.N]`) and compared numerically; an update is only offered when the release is actually newer.
- The GPU environment section in Settings (rocminfo detection) resolves generic "AMD Radeon Graphics" names from the chip id the same way the monitor cards do.
- Settings → App Updates on a local cargo build now offers "Install" buttons for the latest stable and beta releases instead of a dead end; installing replaces the local binary with the GitHub build and enables normal update checks from then on.
- AMD cards that rocm-smi only knows as "AMD Radeon Graphics" (its id database lags new hardware; the Radeon AI PRO R9700 is one) are now named properly: the monitor looks the PCI device id up in a built-in table (RX 9070 XT / 9070 / 9060 XT, AI PRO R9700), then asks `lspci`, and as a last resort tags the generic name with the gfx target and device id so two unknown cards are still distinguishable.
- GPU monitoring no longer floods the log when a vendor tool is installed but has no card behind it (e.g. `nvidia-smi` left over after swapping the NVIDIA card for an AMD one). In auto mode each tool is probed once at startup and skipped, with a single warning, if it fails or reports no GPUs; a tool that fails later is logged once per outage and once on recovery, not on every poll. `nvidia-smi`'s actual error text (it prints it on stdout) is now included.
- The preset editor only offers the legacy "Separate CUDA build (build-cuda/bin)" option to presets already using it; new CUDA setups come from the Install page.
- The preset's Build choice was never sent with the launch request, so it had no effect; it is now.

### Added (continued)
- The preset's Backend field is now labelled **Build** and explained: it picks the configured binary or a separate `build-cuda` tree. The Devices picker lists the devices of whichever build is selected. A single llama.cpp build with both `-DGGML_CUDA=ON -DGGML_VULKAN=ON` needs no switching: pick `CUDA0` under Devices.
- Preset editor gains a **Devices** picker (GPU distribution section) listing what `llama-server --list-devices` reports; ticking one card passes `--device` so a model stays off the others. Shown as a `dev` chip on the Presets page.
- Models page pairs projector files with their models (same Hugging Face repo, or the model's name in the projector's filename): a vision model lists its `mmproj`, a projector lists the models it belongs to, and unmatched projectors say so. Picking such a model in the preset editor fills the projector field automatically.
- CPU card shows the package/die temperature from hwmon (`k10temp`/`zenpower` on AMD, `coretemp` on Intel, `cpu_thermal` on ARM boards; thermal zones as a fallback).
- Logs page (sidebar, under Monitor) showing the monitor's own event log: launches and why they failed, crashes it detected, update progress, GPU/telemetry problems and download results, with timestamps, a "Problems only" filter, Download, and an unread-problems count in the sidebar. llama-server's output stays in Process Output on Monitor. The crash notification's "View logs" button opens it.
- Update notifications: release builds check GitHub on load and every six hours, show an "Update" pill beside the version in the sidebar and a one-time toast per new release with an "Open updates" button that jumps to Settings > App Updates.

### Fixed
- When llama-server exits, the notification and the Logs entry now quote its last error line and add a hint for common causes (model + KV cache not fitting the selected device, unknown architecture, missing file, port in use, rejected flag) instead of only saying to check the logs. The notification's button opens Process Output.
- Generation and prompt speeds are now derived live from `/slots` token counters (smoothed), with the last finished task's average shown while idle. llama-server's `/metrics` gauges only carry a value on the scrape right after a task ends, so a long generation used to show "—".
- Starting on a port that is already in use reports a clear message (and how to free it) instead of a warp panic.
- The Settings dialog shows the llama-server path, working directory and models directory actually in effect, including values passed on the command line, instead of empty placeholders.
- A bare `llama-server` name that is not on PATH is reported as such at launch instead of "No such file or directory".
- Launch checks explain a llama-server path that is a directory, a file without the execute bit, or a missing working directory, and spawn errors name the path and directory tried instead of a bare "Permission denied".
- When the GitHub API is unavailable (its rate limit, most often), releases are read from the repository's Atom feed instead, so checking and installing updates keep working.
- Update checks are cached for 5 minutes server-side (page loads no longer each cost a GitHub API call), GitHub's rate limit is explained with a retry time, and `LLAMA_ADMIN_GITHUB_TOKEN` can raise the limit. "Check for updates" always fetches fresh.
- Rows hidden with the `hidden` attribute could still render when a class set `display`; `[hidden]` now always wins.
- Static assets are served with `Cache-Control: no-cache` and versioned URLs, so a browser can no longer pair a freshly updated binary with a cached script from the previous build (which made the Settings dialog misbehave after an update).

### Changed
- In-app updates no longer need a git checkout or shell tools. The binary knows its release tag and track (baked in by the release workflow), queries GitHub Releases directly, verifies the download's size and executable header, stops llama-server itself and swaps the binary atomically before re-executing. A failed download or swap leaves the running binary untouched.
- Tags containing `-beta` are published as GitHub pre-releases; the beta track installs those, so switching tracks now really changes the binary.
- The `/v1` proxy strips hop-by-hop headers in both directions and reuses one pooled HTTP client.
- Benchmark sweeps larger than six runs ask for confirmation with the run count; the default thread count now follows the machine's cores instead of a fixed 8.
- The dashboard HTML is rendered once at startup instead of shelling out to `git` on every page load.
- CI also runs on the `beta` branch.

## [2026.9.20]
### Fixed
- Fixed an issue where the `BETA` UI badge was hardcoded into stable release binaries.
- Overhauled the App Updates tab to separate track-switching from new updates.

## [2026.9.19]
### Added
- **OpenAI-Compatible API Proxy**: Exposes a transparent `/v1/*` proxy to route directly to the active `llama-server`. Compatible with SillyTavern and custom scripts.
- **In-App Updates**: A new System tab allowing direct OTA upgrades between `main` and `beta` branches natively from the web interface, utilizing zero-downtime Linux process replacement (`exec`).
- **Advanced Benchmarking Sweeps**: The benchmarking tool now supports full Cartesian sweeps of Batch Size, Ubatch Size, and Thread counts, processing combinations sequentially and identifying optimal performance configurations.
- **Dynamic Port Display**: The Chat UI now dynamically fetches and informs the user of the exact `http://ip:port/v1` address to use for third-party integrations.

### Fixed
- Stabilized memory display so RAM DIMM badges stack properly without overflowing layout borders.

## [2026.9.18]
### Added
- Process Output card now uses a native `<details>` toggle that persists its open/closed state to `localStorage` across reloads.
- CPU card now displays a per-core utilization grid dynamically styled with opacity based on CPU load.
- Memory card now displays physical DIMM module details (requires `sudo -n` access to `dmidecode`).

### Changed
- Converted versioning scheme to Calendar Versioning (CalVer).
- "GPU memory" and "VRAM usage" card headers renamed to "GPU Memory Pool" and "Total VRAM", respectively.
- DIMM module list now neatly stacks vertically instead of wrapping inline.
- Hidden the `sudo: a password is required` `dmidecode` fallback error behind a clean tooltip icon rather than disrupting the memory card layout.

### Removed
- Removed the redundant "Open Monitor" button from the sidebar since the main dashboard covers its functionality.
