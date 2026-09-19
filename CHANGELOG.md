# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to Calendar Versioning (CalVer) with the format `YYYY.MM.DD`. 
Backwards compatibility is preserved unless explicitly noted.

## [Unreleased]
### Added
- **Install page** (sidebar → Library → Install): installs prebuilt llama.cpp releases straight from ggml-org's GitHub releases, one directory per backend and version, no compiler needed. Offers Vulkan, CUDA 12.8, CUDA 13.3 (both with the CUDA runtime libraries bundled), ROCm 10.0 and CPU on Linux x64; Vulkan/CUDA/CPU on Linux arm64; Metal on Apple Silicon. Each install is smoke-tested with `--list-devices` and lists the devices it found. Builds can be removed, or set as the binary in Settings with one click.
- Presets choose a build under GPU distribution → Build (installed builds, the configured binary, or the legacy `build-cuda` tree); the Devices picker and the Benchmark page follow that choice.

### Fixed
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
