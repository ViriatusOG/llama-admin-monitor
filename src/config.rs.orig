use std::path::PathBuf;

use crate::cli::AppArgs;

#[derive(Debug, Clone)]
#[allow(dead_code)]
pub struct AppConfig {
    pub llama_server_path: PathBuf,
    pub llama_server_cwd: PathBuf,
    pub port: u16,
    pub gpu_backend: String,
    pub models_dir: Option<PathBuf>,
    pub presets_file: PathBuf,
    pub gpu_env_file: PathBuf,
    pub gpu_arch_override: Option<String>,
    pub gpu_devices_override: Option<String>,
    pub ui_settings_file: PathBuf,
}

impl AppConfig {
    pub fn from_args(args: AppArgs) -> Self {
        let default_server_path = PathBuf::from("llama-server");
        let default_server_cwd = PathBuf::from(".");

        let config_dir = resolve_config_dir(dirs::config_dir());

        let presets_file = args
            .presets_file
            .unwrap_or_else(|| config_dir.join("presets.json"));

        Self {
            llama_server_path: args.llama_server_path.unwrap_or(default_server_path),
            llama_server_cwd: args.llama_server_cwd.unwrap_or(default_server_cwd),
            port: args.port,
            gpu_backend: args.gpu_backend,
            models_dir: args.models_dir,
            presets_file,
            gpu_env_file: config_dir.join("gpu-env.json"),
            gpu_arch_override: args.gpu_arch,
            gpu_devices_override: args.gpu_devices,
            ui_settings_file: config_dir.join("ui-settings.json"),
        }
    }
}

/// Config lives under `llama-admin-monitor`. Installs upgraded from the
/// original `llama-monitor` fork keep their presets and settings: if the new
/// directory does not exist yet but the legacy one does, the legacy one is used.
fn resolve_config_dir(base: Option<PathBuf>) -> PathBuf {
    let base = base.unwrap_or_else(|| PathBuf::from("."));
    let current = base.join("llama-admin-monitor");
    let legacy = base.join("llama-monitor");
    if !current.exists() && legacy.exists() {
        legacy
    } else {
        current
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config() {
        let args = AppArgs {
            llama_server_path: None,
            llama_server_cwd: None,
            port: 7778,
            models_dir: None,
            presets_file: None,
            gpu_backend: "auto".into(),
            gpu_arch: None,
            gpu_devices: None,
        };
        let config = AppConfig::from_args(args);
        assert_eq!(config.port, 7778);
        assert_eq!(config.gpu_backend, "auto");
        // The config dir is either the new name or, on upgraded installs, the
        // legacy fork name -- both are acceptable here.
        let presets_dir = config
            .presets_file
            .parent()
            .and_then(|p| p.file_name())
            .and_then(|n| n.to_str())
            .unwrap();
        assert!(presets_dir == "llama-admin-monitor" || presets_dir == "llama-monitor");
        assert!(config.presets_file.ends_with("presets.json"));
        assert!(config.gpu_env_file.to_str().unwrap().contains("gpu-env"));
        assert!(
            config
                .ui_settings_file
                .to_str()
                .unwrap()
                .contains("ui-settings")
        );
    }

    #[test]
    fn test_config_with_overrides() {
        let args = AppArgs {
            llama_server_path: Some(PathBuf::from("/usr/bin/llama-server")),
            llama_server_cwd: Some(PathBuf::from("/tmp")),
            port: 9999,
            models_dir: Some(PathBuf::from("/models")),
            presets_file: Some(PathBuf::from("/custom/presets.json")),
            gpu_backend: "nvidia".into(),
            gpu_arch: Some("gfx1100".into()),
            gpu_devices: Some("0,1".into()),
        };
        let config = AppConfig::from_args(args);
        assert_eq!(
            config.llama_server_path,
            PathBuf::from("/usr/bin/llama-server")
        );
        assert_eq!(config.port, 9999);
        assert_eq!(config.gpu_arch_override, Some("gfx1100".into()));
        assert_eq!(config.gpu_devices_override, Some("0,1".into()));
    }

    #[test]
    fn test_config_dir_prefers_new_location() {
        let base = std::env::temp_dir().join("llama-admin-monitor-cfg-new");
        let _ = std::fs::remove_dir_all(&base);
        std::fs::create_dir_all(base.join("llama-admin-monitor")).unwrap();
        std::fs::create_dir_all(base.join("llama-monitor")).unwrap();
        assert_eq!(resolve_config_dir(Some(base.clone())), base.join("llama-admin-monitor"));
        let _ = std::fs::remove_dir_all(&base);
    }

    #[test]
    fn test_config_dir_falls_back_to_legacy() {
        let base = std::env::temp_dir().join("llama-admin-monitor-cfg-legacy");
        let _ = std::fs::remove_dir_all(&base);
        std::fs::create_dir_all(base.join("llama-monitor")).unwrap();
        assert_eq!(resolve_config_dir(Some(base.clone())), base.join("llama-monitor"));
        let _ = std::fs::remove_dir_all(&base);
    }

    #[test]
    fn test_config_dir_defaults_to_new_when_neither_exists() {
        let base = std::env::temp_dir().join("llama-admin-monitor-cfg-none");
        let _ = std::fs::remove_dir_all(&base);
        assert_eq!(resolve_config_dir(Some(base.clone())), base.join("llama-admin-monitor"));
    }
}
