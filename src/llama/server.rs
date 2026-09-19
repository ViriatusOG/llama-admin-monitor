use anyhow::Result;
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
    let binary_path = if use_cuda {
        let s = app_config.llama_server_path.display().to_string();
        let swapped = s.replace("/build/bin/", "/build-cuda/bin/");
        let p = PathBuf::from(&swapped);
        if !p.exists() {
            anyhow::bail!(
                "CUDA build not found at {}. Build it with: cmake -B build-cuda -DGGML_CUDA=ON && cmake --build build-cuda --config Release",
                p.display()
            );
        }
        p
    } else {
        app_config.llama_server_path.clone()
    };

    if !app_config.llama_server_cwd.is_dir() {
        anyhow::bail!(
            "working directory {} does not exist or is not a directory. Fix it in Settings.",
            app_config.llama_server_cwd.display()
        );
    }

    let mut cmd = TokioCommand::new(&binary_path);
    cmd.current_dir(&app_config.llama_server_cwd);

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
            app_config.llama_server_cwd.display()
        )
    })?;

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
                    let msg = if quick {
                        format!(
                            "llama-server exited during startup ({status}). Check the Logs tab."
                        )
                    } else {
                        format!("llama-server exited unexpectedly ({status}).")
                    };
                    watch_state.push_log(format!("[monitor] {msg}"));
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
    Ok(())
}
