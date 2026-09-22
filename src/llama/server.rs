use anyhow::{Context, Result};
use std::path::PathBuf;
use tokio::io::{AsyncBufReadExt, BufReader};
use tokio::process::Command as TokioCommand;

use crate::config::AppConfig;
use crate::gpu::env::{build_nvidia_env, build_rocm_env};
use crate::state::AppState;

#[derive(Debug, Clone, Default, serde::Serialize, serde::Deserialize)]
pub struct ServerConfig {
    pub model_path: String,
    #[serde(default)]
    pub mmproj: String,
    pub context_size: u64,
    pub ctk: String,
    pub ctv: String,
    pub tensor_split: String,
    pub batch_size: u32,
    pub ubatch_size: u32,
    pub no_mmap: bool,
    pub port: u16,
    pub ngram_spec: bool,
    pub parallel_slots: u32,
    // Model & memory
    #[serde(default)]
    pub gpu_layers: Option<i32>,
    #[serde(default)]
    pub mlock: bool,
    // Attention
    #[serde(default)]
    pub flash_attn: String,
    // GPU distribution
    #[serde(default)]
    pub backend: String,
    #[serde(default)]
    pub devices: String,
    #[serde(default)]
    pub split_mode: String,
    #[serde(default)]
    pub main_gpu: Option<u32>,
    // Threading
    #[serde(default)]
    pub threads: Option<u32>,
    #[serde(default)]
    pub threads_batch: Option<u32>,
    // Rope scaling
    #[serde(default)]
    pub rope_scaling: String,
    #[serde(default)]
    pub rope_freq_base: Option<f64>,
    #[serde(default)]
    pub rope_freq_scale: Option<f64>,
    // Speculative decoding
    #[serde(default)]
    pub draft_model: String,
    #[serde(default)]
    pub draft_min: Option<u32>,
    #[serde(default)]
    pub draft_max: Option<u32>,
    #[serde(default)]
    pub spec_ngram_size: Option<u32>,
    /// Speculative decoding type override (`--spec-type`); empty uses the
    /// ngram-spec checkbox (ngram-mod).
    #[serde(default)]
    pub spec_type: String,
    /// GPU layers for the draft model (`--spec-draft-ngl`).
    #[serde(default)]
    pub draft_ngl: Option<u32>,
    /// Device for the draft model (`--spec-draft-device`).
    #[serde(default)]
    pub draft_device: String,
    #[serde(default)]
    pub draft_threads: Option<u32>,
    #[serde(default)]
    pub draft_threads_batch: Option<u32>,
    /// Minimum probability for draft tokens (`--draft-p-min`).
    #[serde(default)]
    pub draft_p_min: Option<f64>,
    // Sampling (server-level defaults; requests may override per call)
    #[serde(default)]
    pub temperature: Option<f64>,
    #[serde(default)]
    pub top_p: Option<f64>,
    #[serde(default)]
    pub top_k: Option<u32>,
    #[serde(default)]
    pub min_p: Option<f64>,
    #[serde(default)]
    pub repeat_penalty: Option<f64>,
    #[serde(default)]
    pub presence_penalty: Option<f64>,
    #[serde(default)]
    pub frequency_penalty: Option<f64>,
    #[serde(default)]
    pub ignore_eos: bool,
    // Reasoning / thinking control
    /// Reasoning mode for hybrid thinking models (`--rea`): on/off/auto.
    #[serde(default)]
    pub reasoning: String,
    /// Thinking effort level (`--reasoning-effort`): xhigh/high/medium/low/none.
    #[serde(default)]
    pub reasoning_effort: String,
    /// Keep earlier thinking traces (`--reasoning-preserve`).
    #[serde(default)]
    pub reasoning_preserve: bool,
    /// Extra JSON args for the chat template (`--chat-template-kwargs`).
    #[serde(default)]
    pub chat_template_kwargs: String,
    // Memory & KV cache
    /// Auto-fit context to available memory (`--fit`): on/off.
    #[serde(default)]
    pub fit: String,
    /// Target memory per GPU in MiB when auto-fitting (`--fitt`), e.g. `8192,16384`.
    #[serde(default)]
    pub fit_target: String,
    /// Offload the KV cache to CPU (`--kv-offload`).
    #[serde(default)]
    pub kv_offload: bool,
    /// KV cache compression bits per element (`--cram`).
    #[serde(default)]
    pub cram: Option<u32>,
    /// Shift the context window as it fills (`--context-shift`).
    #[serde(default)]
    pub context_shift: bool,
    /// Skip the startup warmup pass (historical default: true).
    #[serde(default = "default_no_warmup")]
    pub no_warmup: bool,
    // Multimodal projector placement
    /// Device to run the projector on (`--mmdev`).
    #[serde(default)]
    pub mmproj_device: String,
    /// Offload the projector to CPU (`--mmproj-offload`).
    #[serde(default)]
    pub mmproj_offload: bool,
    #[serde(default)]
    pub image_min_tokens: Option<u32>,
    #[serde(default)]
    pub image_max_tokens: Option<u32>,
    // Advanced
    #[serde(default)]
    pub seed: Option<i64>,
    #[serde(default)]
    pub system_prompt_file: String,
    #[serde(default)]
    pub extra_args: String,
}

fn default_no_warmup() -> bool {
    true
}

/// Build the llama-server argument list for a config.
///
/// Pure and unit-testable: optional fields emit their flag only when set, so
/// older llama-server builds never receive flags they do not know. Sampling
/// and reasoning flags set here are server-level defaults; API clients can
/// still override them per request.
pub fn build_server_args(config: &ServerConfig, use_cuda: bool) -> Vec<String> {
    let mut a: Vec<String> = Vec::new();

    // Model & core
    a.push("-m".into());
    a.push(config.model_path.clone());
    if !config.mmproj.is_empty() {
        a.push("--mmproj".into());
        a.push(config.mmproj.clone());
    }
    // Multimodal projector placement
    if !config.mmproj_device.is_empty() {
        a.push("--mmdev".into());
        a.push(config.mmproj_device.clone());
    }
    if config.mmproj_offload {
        a.push("--mmproj-offload".into());
    }
    if let Some(v) = config.image_min_tokens {
        a.push("--image-min-tokens".into());
        a.push(v.to_string());
    }
    if let Some(v) = config.image_max_tokens {
        a.push("--image-max-tokens".into());
        a.push(v.to_string());
    }
    a.push("-ngl".into());
    a.push(config.gpu_layers.unwrap_or(99).to_string());
    a.push("-ctk".into());
    a.push(config.ctk.clone());
    a.push("-ctv".into());
    a.push(config.ctv.clone());
    a.push("--host".into());
    a.push("0.0.0.0".into());
    a.push("--port".into());
    a.push(config.port.to_string());
    a.push("-c".into());
    a.push(config.context_size.to_string());
    a.push("-b".into());
    a.push(config.batch_size.to_string());
    a.push("-ub".into());
    a.push(config.ubatch_size.to_string());
    if config.no_warmup {
        a.push("--no-warmup".into());
    } else {
        a.push("--warmup".into());
    }
    a.push("--jinja".into());
    a.push("--metrics".into());
    a.push("--webui-mcp-proxy".into());

    // Memory
    if config.no_mmap {
        a.push("--no-mmap".into());
    }
    if config.mlock {
        a.push("--mlock".into());
    }
    match config.fit.as_str() {
        "on" => {
            a.push("--fit".into());
            a.push("on".into());
        }
        "off" => {
            a.push("--fit".into());
            a.push("off".into());
        }
        _ => {}
    }
    if !config.fit_target.is_empty() {
        a.push("--fitt".into());
        a.push(config.fit_target.clone());
    }
    if config.kv_offload {
        a.push("--kv-offload".into());
    }
    if let Some(v) = config.cram {
        a.push("--cram".into());
        a.push(v.to_string());
    }
    if config.context_shift {
        a.push("--context-shift".into());
    }

    // Sampling (server-level defaults; requests may override per call)
    if let Some(v) = config.temperature {
        a.push("--temp".into());
        a.push(format!("{v}"));
    }
    if let Some(v) = config.top_p {
        a.push("--top-p".into());
        a.push(format!("{v}"));
    }
    if let Some(v) = config.top_k {
        a.push("--top-k".into());
        a.push(v.to_string());
    }
    if let Some(v) = config.min_p {
        a.push("--min-p".into());
        a.push(format!("{v}"));
    }
    if let Some(v) = config.repeat_penalty {
        a.push("--repeat-penalty".into());
        a.push(format!("{v}"));
    }
    if let Some(v) = config.presence_penalty {
        a.push("--presence-penalty".into());
        a.push(format!("{v}"));
    }
    if let Some(v) = config.frequency_penalty {
        a.push("--frequency-penalty".into());
        a.push(format!("{v}"));
    }
    if config.ignore_eos {
        a.push("--ignore-eos".into());
    }

    // Reasoning / thinking control
    if !config.reasoning.is_empty() {
        a.push("--rea".into());
        a.push(config.reasoning.clone());
    }
    if !config.reasoning_effort.is_empty() {
        a.push("--reasoning-effort".into());
        a.push(config.reasoning_effort.clone());
    }
    if config.reasoning_preserve {
        a.push("--reasoning-preserve".into());
    }
    if !config.chat_template_kwargs.is_empty() {
        a.push("--chat-template-kwargs".into());
        a.push(config.chat_template_kwargs.clone());
    }

    // Flash attention
    if !config.flash_attn.is_empty() {
        a.push("-fa".into());
        a.push(config.flash_attn.clone());
    }

    // Device selection: restricts offloading to the named ggml devices
    // (`llama-server --list-devices`), e.g. just the NVIDIA card.
    let devices = config.devices.trim();
    if !devices.is_empty() {
        a.push("--device".into());
        a.push(devices.to_string());
    }

    // GPU distribution -- a CUDA build only ever sees the NVIDIA card,
    // so a split would be meaningless and is deliberately not passed.
    if !use_cuda && !config.tensor_split.is_empty() {
        a.push("-ts".into());
        a.push(config.tensor_split.clone());
    }
    if !config.split_mode.is_empty() {
        a.push("--split-mode".into());
        a.push(config.split_mode.clone());
    }
    if let Some(mg) = config.main_gpu {
        a.push("-mg".into());
        a.push(mg.to_string());
    }

    // Threading
    if let Some(t) = config.threads {
        a.push("-t".into());
        a.push(t.to_string());
    }
    if let Some(tb) = config.threads_batch {
        a.push("-tb".into());
        a.push(tb.to_string());
    }

    // Rope scaling: explicit config takes priority, else auto-YaRN for large contexts
    if !config.rope_scaling.is_empty() {
        a.push("--rope-scaling".into());
        a.push(config.rope_scaling.clone());
    } else if config.context_size > 262144 {
        a.push("--rope-scaling".into());
        a.push("yarn".into());
    }
    if let Some(base) = config.rope_freq_base {
        a.push("--rope-freq-base".into());
        a.push(format!("{:.6}", base));
    }
    if let Some(scale) = config.rope_freq_scale {
        a.push("--rope-freq-scale".into());
        a.push(format!("{:.6}", scale));
    } else if config.rope_scaling.is_empty() && config.context_size > 262144 {
        // Auto-calculate YaRN scale when no explicit rope config
        let scale = 262144.0 / config.context_size as f64;
        a.push("--rope-freq-scale".into());
        a.push(format!("{:.6}", scale));
        a.push("--yarn-ext-factor".into());
        a.push("1.0".into());
        a.push("--yarn-attn-factor".into());
        a.push("1.0".into());
        a.push("--yarn-beta-fast".into());
        a.push("32".into());
        a.push("--yarn-beta-slow".into());
        a.push("1".into());
    }

    // Speculative decoding: an explicit spec type overrides the legacy
    // ngram-spec checkbox (which always means ngram-mod).
    let spec_type = if !config.spec_type.is_empty() {
        config.spec_type.as_str()
    } else if config.ngram_spec {
        "ngram-mod"
    } else {
        ""
    };
    if !spec_type.is_empty() && spec_type != "none" {
        a.push("--spec-type".into());
        a.push(spec_type.to_string());
        let ngram = matches!(
            spec_type,
            "ngram-mod" | "ngram-simple" | "ngram-map-k" | "ngram-map-k4v" | "ngram-cache"
        );
        if ngram {
            a.push("--spec-ngram-size-n".into());
            a.push(config.spec_ngram_size.unwrap_or(24).to_string());
        }
        a.push("--draft-min".into());
        a.push(config.draft_min.unwrap_or(8).to_string());
        a.push("--draft-max".into());
        a.push(config.draft_max.unwrap_or(24).to_string());
    }
    if !config.draft_model.is_empty() {
        a.push("-md".into());
        a.push(config.draft_model.clone());
    }
    if let Some(v) = config.draft_ngl {
        a.push("--spec-draft-ngl".into());
        a.push(v.to_string());
    }
    if !config.draft_device.is_empty() {
        a.push("--spec-draft-device".into());
        a.push(config.draft_device.clone());
    }
    if let Some(v) = config.draft_threads {
        a.push("--spec-draft-threads".into());
        a.push(v.to_string());
    }
    if let Some(v) = config.draft_threads_batch {
        a.push("--spec-draft-threads-batch".into());
        a.push(v.to_string());
    }
    if let Some(v) = config.draft_p_min {
        a.push("--draft-p-min".into());
        a.push(format!("{v}"));
    }

    // Slots
    if config.parallel_slots > 0 {
        a.push("--parallel".into());
        a.push(config.parallel_slots.to_string());
    }

    // Advanced
    if let Some(seed) = config.seed {
        a.push("--seed".into());
        a.push(seed.to_string());
    }
    if !config.system_prompt_file.is_empty() {
        a.push("--system-prompt-file".into());
        a.push(config.system_prompt_file.clone());
    }

    // Extra args (arbitrary flags)
    for arg in config.extra_args.split_whitespace() {
        a.push(arg.to_string());
    }

    a
}

pub async fn start_server(
    state: &AppState,
    mut config: ServerConfig,
    app_config: &AppConfig,
) -> Result<()> {
    // Validate model path before starting
    if config.model_path.is_empty() {
        anyhow::bail!("Model path is empty. Edit the preset and set a model path.");
    }
    if !std::path::Path::new(&config.model_path).exists() {
        anyhow::bail!("Model file not found: {}", config.model_path);
    }
    if !config.mmproj.is_empty() && !std::path::Path::new(&config.mmproj).exists() {
        let stale = config.mmproj.clone();
        match crate::models::find_mmproj_for(std::path::Path::new(&config.model_path), &stale) {
            Some(found) => {
                crate::applog::warn(format!(
                    "Preset projector {} not found; using {} instead",
                    stale,
                    found.display()
                ));
                // Point the preset(s) at the file that exists so the preset
                // editor and the next start don't repeat the mismatch.
                let mut presets = state.presets.lock().unwrap();
                let found_str = found.to_string_lossy().to_string();
                let mut changed = false;
                for p in presets.iter_mut() {
                    if p.model_path == config.model_path && p.mmproj == stale {
                        p.mmproj = found_str.clone();
                        changed = true;
                    }
                }
                if changed {
                    let _ = crate::presets::save_presets(&state.presets_path, &presets);
                }
                config.mmproj = found_str;
            }
            None => anyhow::bail!(
                "Multimodal projector not found: {}. No projector matching \
                 {} was found in the same directory; download a companion \
                 mmproj (HF downloads page) or set/clear the mmproj field \
                 in the preset editor.",
                stale,
                config.model_path
            ),
        }
    }

    // Validate the server binary: a path must exist, a bare name must be on PATH.
    let server_path = &app_config.llama_server_path;
    if server_path.components().count() > 1 {
        check_executable(server_path)?;
    } else if find_on_path(server_path).is_none() {
        anyhow::bail!(
            "`{}` is not on PATH. Set the full path to llama-server in Settings \
             (sidebar) or pass --llama-server-path.",
            server_path.display()
        );
    }

    // Clear old logs
    {
        let mut logs = state.server_logs.lock().unwrap();
        logs.clear();
    }

    // A CUDA preset runs a separate llama.cpp build. By convention the
    // CUDA binaries live in a sibling "build-cuda" directory next to the
    // default "build" one, since llama.cpp can only target one GPU
    // backend per build.
    let use_cuda = config.backend == "cuda";
    let target = launch_target(app_config, &config.backend)?;
    let binary_path = target.binary.clone();

    if !target.cwd.is_dir() {
        anyhow::bail!(
            "working directory {} does not exist or is not a directory. Fix it in Settings.",
            target.cwd.display()
        );
    }

    let mut cmd = TokioCommand::new(&binary_path);
    cmd.current_dir(&target.cwd);

    // Set GPU-specific environment variables
    let gpu_env = state.gpu_env.lock().unwrap().clone();
    let cwd = app_config.llama_server_cwd.display().to_string();
    let effective_backend = if use_cuda {
        "nvidia"
    } else {
        app_config.gpu_backend.as_str()
    };
    match effective_backend {
        "nvidia" => {
            for (key, val) in build_nvidia_env(&gpu_env) {
                cmd.env(key, val);
            }
        }
        "none" => {}
        _ => {
            for (key, val) in build_rocm_env(&gpu_env, &cwd) {
                cmd.env(key, val);
            }
        }
    }

    // Build args
    for arg in build_server_args(&config, use_cuda) {
        cmd.arg(arg);
    }

    cmd.stdout(std::process::Stdio::piped());
    cmd.stderr(std::process::Stdio::piped());

    let mut child = cmd.spawn().map_err(|e| {
        anyhow::anyhow!(
            "cannot launch {} (in {}): {e}",
            binary_path.display(),
            target.cwd.display()
        )
    })?;
    crate::applog::info(format!(
        "Launched {} on port {} with {}",
        binary_path.display(),
        config.port,
        config.model_path
    ));

    // Capture stdout
    if let Some(stdout) = child.stdout.take() {
        let state_clone = state.clone();
        tokio::spawn(async move {
            let reader = BufReader::new(stdout);
            let mut lines = reader.lines();
            while let Ok(Some(line)) = lines.next_line().await {
                state_clone.push_log(line);
            }
        });
    }

    // Capture stderr
    if let Some(stderr) = child.stderr.take() {
        let state_clone = state.clone();
        tokio::spawn(async move {
            let reader = BufReader::new(stderr);
            let mut lines = reader.lines();
            while let Ok(Some(line)) = lines.next_line().await {
                state_clone.push_log(line);
            }
        });
    }

    {
        let mut guard = state.server_child.lock().await;
        *guard = Some(child);
    }
    {
        let mut running = state.server_running.lock().unwrap();
        *running = true;
    }
    {
        let mut err = state.server_error.lock().unwrap();
        *err = None;
    }
    {
        let mut cfg = state.server_config.lock().unwrap();
        *cfg = Some(config);
    }

    // Watch for the process exiting on its own (e.g. a model that fails
    // to load). Without this, server_running stays true forever and the
    // UI keeps showing "Stop" for a server that is no longer there.
    {
        let watch_state = state.clone();
        let started = std::time::Instant::now();
        tokio::spawn(async move {
            loop {
                tokio::time::sleep(std::time::Duration::from_millis(500)).await;

                let mut guard = watch_state.server_child.lock().await;
                let exited = match guard.as_mut() {
                    // No child means stop_server() already cleaned up.
                    None => break,
                    Some(child) => match child.try_wait() {
                        Ok(Some(status)) => Some(status),
                        Ok(None) => None,
                        Err(_) => break,
                    },
                };

                if let Some(status) = exited {
                    *guard = None;
                    drop(guard);

                    *watch_state.server_running.lock().unwrap() = false;
                    *watch_state.server_config.lock().unwrap() = None;

                    let quick = started.elapsed() < std::time::Duration::from_secs(20);
                    let recent: Vec<String> = {
                        let logs = watch_state.server_logs.lock().unwrap();
                        logs.iter().rev().take(60).rev().cloned().collect()
                    };
                    let diagnosis = diagnose_exit(&recent);
                    let mut msg = if quick {
                        format!("llama-server exited during startup ({status})")
                    } else {
                        format!("llama-server exited unexpectedly ({status})")
                    };
                    match &diagnosis {
                        Some(d) => msg.push_str(&format!(": {d}")),
                        None => msg.push_str(". See Process Output on Monitor for its last lines."),
                    }
                    watch_state.push_log(format!("[monitor] {msg}"));
                    crate::applog::error(&msg);
                    *watch_state.server_error.lock().unwrap() = Some(msg);
                    break;
                }
            }
        });
    }

    // Notify the llama poller to start
    state.llama_poll_notify.notify_one();

    Ok(())
}

/// Turns llama-server's last error lines into one sentence for the crash
/// notification and the event log, with a hint for the common causes.
/// Returns None when nothing in `recent` looks like an error.
pub fn diagnose_exit(recent: &[String]) -> Option<String> {
    let is_error = |l: &str| {
        let lower = l.to_ascii_lowercase();
        l.contains(" E ")
            || lower.contains("error")
            || lower.contains("failed")
            || lower.contains("not found")
            || lower.contains("no such file")
    };
    let cleaned: Vec<String> = recent
        .iter()
        .filter(|l| !l.starts_with("[monitor]"))
        .filter(|l| is_error(l))
        .map(|l| strip_log_prefix(l))
        .collect();
    let joined = cleaned.join(" | ").to_ascii_lowercase();

    let hint = if joined.contains("outofdevicememory")
        || joined.contains("out of memory")
        || joined.contains("failed to allocate")
    {
        Some(
            "the model plus its KV cache does not fit in the selected device's memory. Lower the \
             context size, quantise the KV cache (q8_0), pick a smaller quant, or add a device",
        )
    } else if joined.contains("does not support split buffers") {
        Some(
            "split mode 'row' only works on CUDA; set GPU distribution -> Split mode to 'layer'              (or leave it blank) for Vulkan and ROCm builds",
        )
    } else if joined.contains("invalid device") || joined.contains("no device named") {
        Some(
            "the preset's Devices do not exist in this build (a CUDA build lists CUDA0, a Vulkan              build Vulkan0...); re-pick them under GPU distribution -> Devices",
        )
    } else if joined.contains("unknown model architecture")
        || joined.contains("unknown architecture")
    {
        Some("this llama.cpp build does not know the model's architecture; update llama.cpp")
    } else if joined.contains("no such file") || joined.contains("failed to open") {
        Some("a file the launch needs is missing; check the model, projector and draft paths")
    } else if joined.contains("address already in use") {
        Some("the port is taken by another process; change the port or stop that process")
    } else if joined.contains("invalid argument") || joined.contains("unknown argument") {
        Some("a flag was rejected; check Extra args and the other preset fields")
    } else {
        None
    };

    // The most specific line is usually the earliest error; ggml prints the
    // allocation failure before llama-server's generic "exiting" line.
    let first = cleaned
        .iter()
        .find(|l| !l.to_ascii_lowercase().contains("exiting due to"))
        .or_else(|| cleaned.first())?;
    let mut out = first.clone();
    if out.len() > 220 {
        out.truncate(217);
        out.push_str("...");
    }
    if let Some(h) = hint {
        out.push_str(" \u{2014} ");
        out.push_str(h);
    }
    Some(out)
}

/// Drops llama-server's `0.03.046.031 E srv ` timestamp/level/module prefix.
fn strip_log_prefix(line: &str) -> String {
    let mut parts = line.split_whitespace();
    let first = parts.next().unwrap_or("");
    let looks_timestamped =
        first.chars().all(|c| c.is_ascii_digit() || c == '.') && first.contains('.');
    if !looks_timestamped {
        return line.trim().to_string();
    }
    // timestamp, level letter, module tag, then the message
    let rest: Vec<&str> = parts.collect();
    let skip = rest.iter().take(2).take_while(|w| w.len() <= 3).count();
    rest[skip..].join(" ")
}

/// Where a launch runs: the binary and its working directory. A preset's
/// `backend` is one of: "" / "vulkan" (the configured binary and cwd),
/// "cuda" (legacy sibling build-cuda tree), or "build:<id>" for a build
/// installed from the Install page, which runs inside its own directory
/// so its bundled libraries resolve.
pub struct LaunchTarget {
    pub binary: PathBuf,
    pub cwd: PathBuf,
}

pub fn launch_target(app_config: &AppConfig, backend: &str) -> Result<LaunchTarget> {
    if let Some(id) = backend.strip_prefix("build:") {
        let build = crate::llama::builds::installed_build(id).with_context(|| {
            format!("the preset uses llama.cpp build {id}, which is not installed (see Install)")
        })?;
        return Ok(LaunchTarget {
            binary: build.server_path,
            cwd: build.dir,
        });
    }
    if backend != "cuda" {
        return Ok(LaunchTarget {
            binary: app_config.llama_server_path.clone(),
            cwd: app_config.llama_server_cwd.clone(),
        });
    }
    let s = app_config.llama_server_path.display().to_string();
    let swapped = s.replace("/build/bin/", "/build-cuda/bin/");
    let p = PathBuf::from(&swapped);
    if !p.exists() {
        anyhow::bail!(
            "CUDA build not found at {}. Either build it (cmake -B build-cuda -DGGML_CUDA=ON \
             && cmake --build build-cuda --config Release) or rebuild the main binary with \
             -DGGML_CUDA=ON -DGGML_VULKAN=ON and pick CUDA0 under Devices instead.",
            p.display()
        );
    }
    Ok(LaunchTarget {
        binary: p,
        cwd: app_config.llama_server_cwd.clone(),
    })
}

/// One offload device as printed by `llama-server --list-devices`.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct GgmlDevice {
    pub id: String,
    pub name: String,
    pub total_mib: Option<u64>,
    pub free_mib: Option<u64>,
}

/// Parses `--list-devices` output: lines like
/// `  Vulkan1: NVIDIA GeForce RTX 3080 (10240 MiB, 9800 MiB free)`.
pub fn parse_device_list(text: &str) -> Vec<GgmlDevice> {
    text.lines()
        .filter_map(|raw| {
            // Device lines are indented; backend chatter ("ggml_vulkan: ...")
            // and the header are not.
            if !raw.starts_with("  ") {
                return None;
            }
            let (id, rest) = raw.trim().split_once(": ")?;
            if id.contains(' ') || id.is_empty() {
                return None;
            }
            let (name, mem) = match rest.rsplit_once(" (") {
                Some((n, m)) if m.ends_with(')') => (n, Some(m.trim_end_matches(')'))),
                _ => (rest, None),
            };
            let mut nums = mem
                .unwrap_or("")
                .split(',')
                .filter_map(|part| part.trim().split(' ').next()?.parse::<u64>().ok());
            Some(GgmlDevice {
                id: id.to_string(),
                name: name.trim().to_string(),
                total_mib: nums.next(),
                free_mib: nums.next(),
            })
        })
        .collect()
}

/// Runs `llama-server --list-devices` with the same environment a launch
/// would get, so the names match what `--device` accepts.
pub async fn list_devices(
    state: &AppState,
    app_config: &AppConfig,
    backend: &str,
) -> Result<Vec<GgmlDevice>> {
    let target = launch_target(app_config, backend)?;
    let server_path = target.binary;
    if server_path.components().count() > 1 {
        check_executable(&server_path)?;
    }
    let mut cmd = TokioCommand::new(&server_path);
    cmd.arg("--list-devices");
    if target.cwd.is_dir() {
        cmd.current_dir(&target.cwd);
    }
    let gpu_env = state.gpu_env.lock().unwrap().clone();
    let cwd = app_config.llama_server_cwd.display().to_string();
    let effective_backend = if backend == "cuda" {
        "nvidia"
    } else {
        app_config.gpu_backend.as_str()
    };
    match effective_backend {
        "nvidia" => {
            for (key, val) in build_nvidia_env(&gpu_env) {
                cmd.env(key, val);
            }
        }
        "none" => {}
        _ => {
            for (key, val) in build_rocm_env(&gpu_env, &cwd) {
                cmd.env(key, val);
            }
        }
    }
    let output = tokio::time::timeout(std::time::Duration::from_secs(20), cmd.output())
        .await
        .map_err(|_| anyhow::anyhow!("llama-server --list-devices timed out"))??;
    let text = String::from_utf8_lossy(&output.stdout);
    let devices = parse_device_list(&text);
    if devices.is_empty() && !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        anyhow::bail!(
            "llama-server --list-devices failed: {}",
            stderr.lines().last().unwrap_or("unknown error")
        );
    }
    Ok(devices)
}

/// Explains the usual reasons a configured binary cannot run, before the
/// OS reduces them all to "Permission denied".
fn check_executable(path: &std::path::Path) -> Result<()> {
    if !path.exists() {
        anyhow::bail!(
            "llama-server binary not found: {}. Set it in Settings.",
            path.display()
        );
    }
    if path.is_dir() {
        anyhow::bail!(
            "{} is a directory. Point the llama-server setting at the executable \
             inside it (usually {}/llama-server).",
            path.display(),
            path.display()
        );
    }
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        let mode = std::fs::metadata(path)?.permissions().mode();
        if mode & 0o111 == 0 {
            anyhow::bail!(
                "{} is not executable. Run: chmod +x {}",
                path.display(),
                path.display()
            );
        }
    }
    Ok(())
}

/// Resolves a bare executable name against PATH, like a shell would.
pub fn find_on_path(name: &std::path::Path) -> Option<PathBuf> {
    let paths = std::env::var_os("PATH")?;
    std::env::split_paths(&paths)
        .map(|dir| dir.join(name))
        .find(|candidate| candidate.is_file())
}

pub async fn stop_server(state: &AppState) -> Result<()> {
    let mut guard = state.server_child.lock().await;
    if let Some(ref mut child) = *guard {
        child.kill().await.ok();
        child.wait().await.ok();
    }
    *guard = None;
    {
        let mut running = state.server_running.lock().unwrap();
        *running = false;
    }
    {
        let mut cfg = state.server_config.lock().unwrap();
        *cfg = None;
    }
    {
        let mut m = state.llama_metrics.lock().unwrap();
        *m = crate::llama::metrics::LlamaMetrics::default();
    }
    {
        let mut err = state.server_error.lock().unwrap();
        *err = None;
    }
    state.push_log("[monitor] Server stopped.".into());
    crate::applog::info("llama-server stopped");
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn list_devices_parses_names_and_memory() {
        let text = concat!(
            "ggml_vulkan: Found 2 Vulkan devices:\n",
            "Available devices:\n",
            "  Vulkan0: AMD Radeon RX 9070 XT (RADV GFX1201) (16368 MiB, 16000 MiB free)\n",
            "  Vulkan1: NVIDIA GeForce RTX 3080 (10240 MiB, 9800 MiB free)\n",
        );
        let devices = parse_device_list(text);
        assert_eq!(devices.len(), 2);
        assert_eq!(devices[0].id, "Vulkan0");
        assert_eq!(devices[0].name, "AMD Radeon RX 9070 XT (RADV GFX1201)");
        assert_eq!(devices[0].total_mib, Some(16368));
        assert_eq!(devices[1].id, "Vulkan1");
        assert_eq!(devices[1].free_mib, Some(9800));
        assert!(parse_device_list("Available devices:\n  (none)\n").is_empty());
    }

    #[test]
    fn diagnoses_out_of_memory_with_hint() {
        let recent = vec![
            "0.00.186.478 I srv    load_model: loading model 'x.gguf'".to_string(),
            "ggml_vulkan: Device memory allocation of size 811008000 failed.".to_string(),
            "ggml_vulkan: vk::Device::allocateMemory: ErrorOutOfDeviceMemory".to_string(),
            "0.03.010.681 E alloc_tensor_range: failed to allocate Vulkan1 buffer of size 811008000"
                .to_string(),
            "0.03.046.561 E srv  llama_server: exiting due to model loading error".to_string(),
        ];
        let d = diagnose_exit(&recent).unwrap();
        assert!(
            d.starts_with("ggml_vulkan: Device memory allocation of size 811008000 failed."),
            "{d}"
        );
        assert!(d.contains("does not fit"), "{d}");
    }

    #[test]
    fn diagnoses_row_split_on_vulkan() {
        let recent = vec![
            "0.00.483.274 E llama_model_load: error loading model: device Vulkan0 does not support split buffers".to_string(),
            "0.00.483.295 E llama_model_load_from_file_impl: failed to load model".to_string(),
            "0.00.759.180 E srv  llama_server: exiting due to model loading error".to_string(),
        ];
        let d = diagnose_exit(&recent).unwrap();
        assert!(
            d.starts_with("llama_model_load: error loading model"),
            "{d}"
        );
        assert!(d.contains("Split mode to 'layer'"), "{d}");
    }

    #[test]
    fn diagnose_returns_none_without_errors() {
        let recent = vec!["0.00.1 I srv  main: model loaded".to_string()];
        assert_eq!(diagnose_exit(&recent), None);
    }

    #[test]
    fn strips_timestamp_level_and_module() {
        assert_eq!(
            strip_log_prefix("0.03.046.031 E srv  llama_server: exiting"),
            "llama_server: exiting"
        );
        assert_eq!(
            strip_log_prefix("ggml_vulkan: allocateMemory failed"),
            "ggml_vulkan: allocateMemory failed"
        );
    }

    // ---- build_server_args ----

    fn cfg() -> ServerConfig {
        let mut c = ServerConfig::default();
        c.model_path = "/models/test.gguf".into();
        c.port = 8080;
        c.parallel_slots = 1;
        c.no_warmup = true; // historical default; serde default on deserialization
        c
    }

    fn has(a: &[String], flag: &str) -> bool {
        a.iter().any(|x| x == flag)
    }

    /// Value following `flag` in the arg list, if present.
    fn pair(a: &[String], flag: &str) -> Option<String> {
        let i = a.iter().position(|x| x == flag)?;
        a.get(i + 1).cloned()
    }

    #[test]
    fn build_args_core_always_present_and_optins_absent() {
        let a = build_server_args(&cfg(), false);
        for flag in [
            "-m",
            "--no-warmup",
            "--jinja",
            "--metrics",
            "--webui-mcp-proxy",
            "-ngl",
            "-ctk",
            "-ctv",
            "--host",
            "--port",
            "-c",
            "-b",
            "-ub",
            "--parallel",
        ] {
            assert!(has(&a, flag), "missing {flag} in {a:?}");
        }
        // Optional flags must not appear when unset
        for flag in [
            "--temp",
            "--top-p",
            "--top-k",
            "--min-p",
            "--repeat-penalty",
            "--presence-penalty",
            "--frequency-penalty",
            "--ignore-eos",
            "--rea",
            "--reasoning-effort",
            "--reasoning-preserve",
            "--chat-template-kwargs",
            "--spec-type",
            "--spec-draft-ngl",
            "--fit",
            "--fitt",
            "--kv-offload",
            "--cram",
            "--context-shift",
            "--mmdev",
            "--mmproj-offload",
            "--image-min-tokens",
            "--image-max-tokens",
        ] {
            assert!(!has(&a, flag), "unexpected {flag} in {a:?}");
        }
    }

    #[test]
    fn build_args_sampling_flags() {
        let mut c = cfg();
        c.temperature = Some(1.0);
        c.top_p = Some(0.95);
        c.top_k = Some(20);
        c.min_p = Some(0.0);
        c.repeat_penalty = Some(1.0);
        c.presence_penalty = Some(1.5);
        c.frequency_penalty = Some(0.0);
        c.ignore_eos = true;
        let a = build_server_args(&c, false);
        assert_eq!(pair(&a, "--temp"), Some("1".into()));
        assert_eq!(pair(&a, "--top-p"), Some("0.95".into()));
        assert_eq!(pair(&a, "--top-k"), Some("20".into()));
        assert_eq!(pair(&a, "--min-p"), Some("0".into()));
        assert_eq!(pair(&a, "--repeat-penalty"), Some("1".into()));
        assert_eq!(pair(&a, "--presence-penalty"), Some("1.5".into()));
        assert_eq!(pair(&a, "--frequency-penalty"), Some("0".into()));
        assert!(has(&a, "--ignore-eos"));
    }

    #[test]
    fn build_args_reasoning_flags() {
        let mut c = cfg();
        c.reasoning = "on".into();
        c.reasoning_effort = "medium".into();
        c.reasoning_preserve = true;
        c.chat_template_kwargs = r#"{"reasoning_effort":"medium"}"#.into();
        let a = build_server_args(&c, false);
        assert_eq!(pair(&a, "--rea"), Some("on".into()));
        assert_eq!(pair(&a, "--reasoning-effort"), Some("medium".into()));
        assert!(has(&a, "--reasoning-preserve"));
        assert_eq!(
            pair(&a, "--chat-template-kwargs"),
            Some(r#"{"reasoning_effort":"medium"}"#.into())
        );
    }

    #[test]
    fn build_args_spec_type_override_and_legacy() {
        // Legacy: ngram-spec checkbox alone means ngram-mod with default sizes
        let mut c = cfg();
        c.ngram_spec = true;
        let a = build_server_args(&c, false);
        assert_eq!(pair(&a, "--spec-type"), Some("ngram-mod".into()));
        assert_eq!(pair(&a, "--spec-ngram-size-n"), Some("24".into()));
        assert_eq!(pair(&a, "--draft-min"), Some("8".into()));
        assert_eq!(pair(&a, "--draft-max"), Some("24".into()));

        // Explicit override wins: draft-mtp gets no ngram-size flag
        c.spec_type = "draft-mtp".into();
        c.spec_ngram_size = Some(48);
        let a = build_server_args(&c, false);
        assert_eq!(pair(&a, "--spec-type"), Some("draft-mtp".into()));
        assert_eq!(pair(&a, "--spec-ngram-size-n"), None);

        // "none" disables speculation entirely
        c.spec_type = "none".into();
        let a = build_server_args(&c, false);
        assert_eq!(pair(&a, "--spec-type"), None);
        assert_eq!(pair(&a, "--draft-min"), None);

        // No spec at all, but a draft model still loads
        c.spec_type.clear();
        c.ngram_spec = false;
        c.draft_model = "/models/draft.gguf".into();
        let a = build_server_args(&c, false);
        assert_eq!(pair(&a, "--spec-type"), None);
        assert_eq!(pair(&a, "-md"), Some("/models/draft.gguf".into()));
    }

    #[test]
    fn build_args_draft_tuning() {
        let mut c = cfg();
        c.spec_type = "draft-mtp".into();
        c.draft_ngl = Some(40);
        c.draft_device = "CPU".into();
        c.draft_threads = Some(8);
        c.draft_threads_batch = Some(16);
        c.draft_p_min = Some(0.05);
        let a = build_server_args(&c, false);
        assert_eq!(pair(&a, "--spec-draft-ngl"), Some("40".into()));
        assert_eq!(pair(&a, "--spec-draft-device"), Some("CPU".into()));
        assert_eq!(pair(&a, "--spec-draft-threads"), Some("8".into()));
        assert_eq!(pair(&a, "--spec-draft-threads-batch"), Some("16".into()));
        assert_eq!(pair(&a, "--draft-p-min"), Some("0.05".into()));
    }

    #[test]
    fn build_args_memory_and_kv() {
        let mut c = cfg();
        c.fit = "on".into();
        c.fit_target = "8192,16384".into();
        c.kv_offload = true;
        c.cram = Some(2);
        c.context_shift = true;
        c.no_warmup = false;
        let a = build_server_args(&c, false);
        assert_eq!(pair(&a, "--fit"), Some("on".into()));
        assert_eq!(pair(&a, "--fitt"), Some("8192,16384".into()));
        assert!(has(&a, "--kv-offload"));
        assert_eq!(pair(&a, "--cram"), Some("2".into()));
        assert!(has(&a, "--context-shift"));
        assert!(has(&a, "--warmup") && !has(&a, "--no-warmup"));
    }

    #[test]
    fn build_args_projector_placement() {
        let mut c = cfg();
        c.mmproj = "/models/mmproj.gguf".into();
        c.mmproj_device = "Vulkan0".into();
        c.mmproj_offload = true;
        c.image_min_tokens = Some(256);
        c.image_max_tokens = Some(4096);
        let a = build_server_args(&c, false);
        assert_eq!(pair(&a, "--mmproj"), Some("/models/mmproj.gguf".into()));
        assert_eq!(pair(&a, "--mmdev"), Some("Vulkan0".into()));
        assert!(has(&a, "--mmproj-offload"));
        assert_eq!(pair(&a, "--image-min-tokens"), Some("256".into()));
        assert_eq!(pair(&a, "--image-max-tokens"), Some("4096".into()));
    }

    #[test]
    fn build_args_cuda_skips_tensor_split() {
        let mut c = cfg();
        c.tensor_split = "7,8".into();
        assert_eq!(pair(&build_server_args(&c, true), "-ts"), None);
        assert_eq!(
            pair(&build_server_args(&c, false), "-ts"),
            Some("7,8".into())
        );
    }
}
