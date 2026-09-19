use anyhow::{Context, Result};
use std::path::PathBuf;
use tokio::io::{AsyncBufReadExt, BufReader};
use tokio::process::Command as TokioCommand;

use crate::config::AppConfig;
use crate::gpu::env::{build_nvidia_env, build_rocm_env};
use crate::state::AppState;

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
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
    // Advanced
    #[serde(default)]
    pub seed: Option<i64>,
    #[serde(default)]
    pub system_prompt_file: String,
    #[serde(default)]
    pub extra_args: String,
}

pub async fn start_server(
    state: &AppState,
    config: ServerConfig,
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
        anyhow::bail!("Multimodal projector not found: {}", config.mmproj);
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

    // Build args — model & core
    cmd.arg("-m").arg(&config.model_path);
    if !config.mmproj.is_empty() {
        cmd.arg("--mmproj").arg(&config.mmproj);
    }
    cmd.arg("-ngl")
        .arg(config.gpu_layers.unwrap_or(99).to_string());
    cmd.arg("-ctk").arg(&config.ctk);
    cmd.arg("-ctv").arg(&config.ctv);
    cmd.arg("--host").arg("0.0.0.0");
    cmd.arg("--port").arg(config.port.to_string());
    cmd.arg("-c").arg(config.context_size.to_string());
    cmd.arg("-b").arg(config.batch_size.to_string());
    cmd.arg("-ub").arg(config.ubatch_size.to_string());
    cmd.arg("--no-warmup");
    cmd.arg("--jinja");
    cmd.arg("--metrics");
    cmd.arg("--webui-mcp-proxy");

    // Memory
    if config.no_mmap {
        cmd.arg("--no-mmap");
    }
    if config.mlock {
        cmd.arg("--mlock");
    }

    // Flash attention
    if !config.flash_attn.is_empty() {
        cmd.arg("-fa").arg(&config.flash_attn);
    }

    // Device selection: restricts offloading to the named ggml devices
    // (`llama-server --list-devices`), e.g. just the NVIDIA card.
    let devices = config.devices.trim();
    if !devices.is_empty() {
        cmd.arg("--device").arg(devices);
    }

    // GPU distribution -- a CUDA build only ever sees the NVIDIA card,
    // so a split would be meaningless and is deliberately not passed.
    if !use_cuda && !config.tensor_split.is_empty() {
        cmd.arg("-ts").arg(&config.tensor_split);
    }
    if !config.split_mode.is_empty() {
        cmd.arg("--split-mode").arg(&config.split_mode);
    }
    if let Some(mg) = config.main_gpu {
        cmd.arg("-mg").arg(mg.to_string());
    }

    // Threading
    if let Some(t) = config.threads {
        cmd.arg("-t").arg(t.to_string());
    }
    if let Some(tb) = config.threads_batch {
        cmd.arg("-tb").arg(tb.to_string());
    }

    // Rope scaling: explicit config takes priority, else auto-YaRN for large contexts
    if !config.rope_scaling.is_empty() {
        cmd.arg("--rope-scaling").arg(&config.rope_scaling);
    } else if config.context_size > 262144 {
        cmd.arg("--rope-scaling").arg("yarn");
    }
    if let Some(base) = config.rope_freq_base {
        cmd.arg("--rope-freq-base").arg(format!("{:.6}", base));
    }
    if let Some(scale) = config.rope_freq_scale {
        cmd.arg("--rope-freq-scale").arg(format!("{:.6}", scale));
    } else if config.rope_scaling.is_empty() && config.context_size > 262144 {
        // Auto-calculate YaRN scale when no explicit rope config
        let scale = 262144.0 / config.context_size as f64;
        cmd.arg("--rope-freq-scale").arg(format!("{:.6}", scale));
        cmd.arg("--yarn-ext-factor").arg("1.0");
        cmd.arg("--yarn-attn-factor").arg("1.0");
        cmd.arg("--yarn-beta-fast").arg("32");
        cmd.arg("--yarn-beta-slow").arg("1");
    }

    // Speculative decoding
    if config.ngram_spec {
        cmd.arg("--spec-type").arg("ngram-mod");
        cmd.arg("--spec-ngram-size-n")
            .arg(config.spec_ngram_size.unwrap_or(24).to_string());
        cmd.arg("--draft-min")
            .arg(config.draft_min.unwrap_or(8).to_string());
        cmd.arg("--draft-max")
            .arg(config.draft_max.unwrap_or(24).to_string());
    }
    if !config.draft_model.is_empty() {
        cmd.arg("-md").arg(&config.draft_model);
    }

    // Slots
    if config.parallel_slots > 0 {
        cmd.arg("--parallel").arg(config.parallel_slots.to_string());
    }

    // Advanced
    if let Some(seed) = config.seed {
        cmd.arg("--seed").arg(seed.to_string());
    }
    if !config.system_prompt_file.is_empty() {
        cmd.arg("--system-prompt-file")
            .arg(&config.system_prompt_file);
    }

    // Extra args (arbitrary flags)
    for arg in config.extra_args.split_whitespace() {
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
}
