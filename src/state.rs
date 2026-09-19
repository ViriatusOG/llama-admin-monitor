use std::collections::{BTreeMap, VecDeque};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use crate::gpu::GpuMetrics;
use crate::gpu::env::GpuEnv;
use crate::llama::bench::{BenchProgress, SharedBenchProgress};
use crate::llama::metrics::LlamaMetrics;
use crate::llama::server::ServerConfig;
use crate::models::DiscoveredModel;
use crate::models::hf::SharedDownloadProgress;
use crate::presets::ModelPreset;
use crate::system::SystemStats;

const MAX_LOG_LINES: usize = 500;

/// Persisted UI control-bar settings (survives page reload).
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct UiSettings {
    #[serde(default)]
    pub preset_id: String,
    #[serde(default = "default_port")]
    pub port: u16,
    #[serde(default)]
    pub llama_server_path: String,
    #[serde(default)]
    pub llama_server_cwd: String,
    #[serde(default)]
    pub models_dir: String,
}

fn default_port() -> u16 {
    8080
}

impl Default for UiSettings {
    fn default() -> Self {
        Self {
            preset_id: String::new(),
            port: 8080,
            llama_server_path: String::new(),
            llama_server_cwd: String::new(),
            models_dir: String::new(),
        }
    }
}

pub fn load_ui_settings(path: &Path) -> UiSettings {
    if path.exists()
        && let Ok(contents) = std::fs::read_to_string(path)
        && let Ok(s) = serde_json::from_str::<UiSettings>(&contents)
    {
        return s;
    }
    UiSettings::default()
}

pub fn save_ui_settings(path: &Path, settings: &UiSettings) -> anyhow::Result<()> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let tmp = path.with_extension("json.tmp");
    let json = serde_json::to_string_pretty(settings)?;
    std::fs::write(&tmp, json)?;
    std::fs::rename(&tmp, path)?;
    Ok(())
}

#[derive(Clone)]
pub struct AppState {
    pub gpu_metrics: Arc<Mutex<BTreeMap<String, GpuMetrics>>>,
    /// Processes using the GPUs (nvtop-style), refreshed every few seconds.
    pub gpu_processes: Arc<Mutex<Vec<crate::gpu::procs::GpuProcess>>>,
    pub llama_metrics: Arc<Mutex<LlamaMetrics>>,
    pub server_logs: Arc<Mutex<VecDeque<String>>>,
    pub server_child: Arc<tokio::sync::Mutex<Option<tokio::process::Child>>>,
    pub server_running: Arc<Mutex<bool>>,
    pub server_error: Arc<Mutex<Option<String>>>,
    pub server_config: Arc<Mutex<Option<ServerConfig>>>,
    pub llama_poll_notify: Arc<tokio::sync::Notify>,
    pub presets: Arc<Mutex<Vec<ModelPreset>>>,
    pub presets_path: PathBuf,
    pub discovered_models: Arc<Mutex<Vec<DiscoveredModel>>>,
    pub models_dir: Arc<Mutex<Option<PathBuf>>>,
    pub gpu_env: Arc<Mutex<GpuEnv>>,
    pub gpu_env_path: PathBuf,
    pub ui_settings: Arc<Mutex<UiSettings>>,
    pub ui_settings_path: PathBuf,
    pub hf_download_progress: SharedDownloadProgress,
    pub bench_progress: SharedBenchProgress,
    pub system_stats: Arc<Mutex<SystemStats>>,
    /// Progress of an in-app update in flight (see update.rs), or None.
    pub update_phase: Arc<Mutex<Option<String>>>,
    /// A llama.cpp build being installed from the Install page, if any.
    pub build_install: Arc<Mutex<Option<crate::llama::builds::InstallProgress>>>,
}

impl AppState {
    pub fn new(
        presets: Vec<ModelPreset>,
        presets_path: PathBuf,
        models_dir: Option<PathBuf>,
        gpu_env: GpuEnv,
        gpu_env_path: PathBuf,
        ui_settings: UiSettings,
        ui_settings_path: PathBuf,
    ) -> Self {
        let discovered = models_dir
            .as_ref()
            .and_then(|dir| crate::models::scan_models_dir(dir).ok())
            .unwrap_or_default();

        Self {
            gpu_metrics: Arc::new(Mutex::new(BTreeMap::new())),
            gpu_processes: Arc::new(Mutex::new(Vec::new())),
            llama_metrics: Arc::new(Mutex::new(LlamaMetrics::default())),
            server_logs: Arc::new(Mutex::new(VecDeque::new())),
            server_child: Arc::new(tokio::sync::Mutex::new(None)),
            server_running: Arc::new(Mutex::new(false)),
            server_error: Arc::new(Mutex::new(None)),
            server_config: Arc::new(Mutex::new(None)),
            llama_poll_notify: Arc::new(tokio::sync::Notify::new()),
            presets: Arc::new(Mutex::new(presets)),
            presets_path,
            discovered_models: Arc::new(Mutex::new(discovered)),
            models_dir: Arc::new(Mutex::new(models_dir)),
            gpu_env: Arc::new(Mutex::new(gpu_env)),
            gpu_env_path,
            ui_settings: Arc::new(Mutex::new(ui_settings)),
            ui_settings_path,
            hf_download_progress: Arc::new(Mutex::new(None)),
            bench_progress: Arc::new(Mutex::new(BenchProgress::default())),
            system_stats: Arc::new(Mutex::new(SystemStats::default())),
            update_phase: Arc::new(Mutex::new(None)),
            build_install: Arc::new(Mutex::new(None)),
        }
    }

    pub fn push_log(&self, line: String) {
        let mut logs = self.server_logs.lock().unwrap();
        if logs.len() >= MAX_LOG_LINES {
            logs.pop_front();
        }
        logs.push_back(line);
    }
}
