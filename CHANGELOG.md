# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to Calendar Versioning (CalVer) with the format `YYYY.MM.DD`. 
Backwards compatibility is preserved unless explicitly noted.

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
