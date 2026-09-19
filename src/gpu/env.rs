use anyhow::Result;
use std::path::Path;
use std::process::Command;

use super::amd_names::{parse_hex_id, resolve_amd_name};

// ── GPU Architecture Database ──────────────────────────────────────────────

#[derive(Debug, Clone, serde::Serialize)]
pub struct GpuArch {
    pub id: &'static str,
    pub name: &'static str,
    pub hsa_version: &'static str,
}

pub const GPU_ARCHITECTURES: &[GpuArch] = &[
    GpuArch {
        id: "auto",
        name: "Auto-detect",
        hsa_version: "",
    },
    GpuArch {
        id: "gfx900",
        name: "gfx900 (Vega 56/64, MI25)",
        hsa_version: "9.0.0",
    },
    GpuArch {
        id: "gfx906",
        name: "gfx906 (Radeon VII, MI50/60)",
        hsa_version: "9.0.6",
    },
    GpuArch {
        id: "gfx908",
        name: "gfx908 (MI100)",
        hsa_version: "9.0.8",
    },
    GpuArch {
        id: "gfx90a",
        name: "gfx90a (MI210, MI250, MI250X)",
        hsa_version: "9.0.10",
    },
    GpuArch {
        id: "gfx942",
        name: "gfx942 (MI300, MI300X, MI300A)",
        hsa_version: "9.4.2",
    },
    GpuArch {
        id: "gfx1010",
        name: "gfx1010 (RX 5700/5700 XT)",
        hsa_version: "10.1.0",
    },
    GpuArch {
        id: "gfx1030",
        name: "gfx1030 (RX 6800/6900 XT)",
        hsa_version: "10.3.0",
    },
    GpuArch {
        id: "gfx1100",
        name: "gfx1100 (RX 7900 XTX/XT)",
        hsa_version: "11.0.0",
    },
    GpuArch {
        id: "gfx1101",
        name: "gfx1101 (RX 7800/7700 XT)",
        hsa_version: "11.0.1",
    },
    GpuArch {
        id: "gfx1102",
        name: "gfx1102 (RX 7600)",
        hsa_version: "11.0.2",
    },
    GpuArch {
        id: "gfx1150",
        name: "gfx1150 (Ryzen AI APU, 890M)",
        hsa_version: "11.5.0",
    },
    GpuArch {
        id: "gfx1200",
        name: "gfx1200 (RX 9060 XT)",
        hsa_version: "12.0.0",
    },
    GpuArch {
        id: "gfx1201",
        name: "gfx1201 (RX 9070/9070 XT/9070 GRE)",
        hsa_version: "12.0.1",
    },
];

/// Lookup HSA_OVERRIDE_GFX_VERSION for a given architecture ID.
pub fn hsa_version_for_arch(arch: &str) -> Option<&'static str> {
    GPU_ARCHITECTURES
        .iter()
        .find(|a| a.id == arch)
        .map(|a| a.hsa_version)
        .filter(|v| !v.is_empty())
}

// ── GPU Environment Config ─────────────────────────────────────────────────

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct GpuEnv {
    #[serde(default = "default_arch")]
    pub arch: String,
    #[serde(default)]
    pub devices: String,
    #[serde(default = "default_rocm_path")]
    pub rocm_path: String,
    #[serde(default)]
    pub extra_env: Vec<(String, String)>,
}

fn default_arch() -> String {
    "auto".into()
}
fn default_rocm_path() -> String {
    "/opt/rocm".into()
}

impl Default for GpuEnv {
    fn default() -> Self {
        Self {
            arch: "auto".into(),
            devices: String::new(),
            rocm_path: "/opt/rocm".into(),
            extra_env: Vec::new(),
        }
    }
}

pub fn load_gpu_env(path: &Path) -> GpuEnv {
    if path.exists()
        && let Ok(contents) = std::fs::read_to_string(path)
        && let Ok(env) = serde_json::from_str::<GpuEnv>(&contents)
    {
        return env;
    }
    GpuEnv::default()
}

pub fn save_gpu_env(path: &Path, env: &GpuEnv) -> Result<()> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let tmp = path.with_extension("json.tmp");
    let json = serde_json::to_string_pretty(env)?;
    std::fs::write(&tmp, json)?;
    std::fs::rename(&tmp, path)?;
    Ok(())
}

// ── Auto-detection ─────────────────────────────────────────────────────────

#[derive(Debug, Clone, serde::Serialize)]
pub struct DetectedGpu {
    pub arch: String,
    pub count: usize,
    pub names: Vec<String>,
}

/// Detect AMD GPUs via rocminfo, returning the gfx architecture and count.
pub fn detect_rocm_gpus() -> Option<DetectedGpu> {
    let output = Command::new("rocminfo")
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::null())
        .output()
        .ok()?;

    if !output.status.success() {
        return None;
    }

    let stdout = String::from_utf8_lossy(&output.stdout);
    parse_rocminfo(&stdout)
}

pub fn parse_rocminfo(output: &str) -> Option<DetectedGpu> {
    // GPU agents have Name: gfxNNN, CPU agents have Name: AMD EPYC / Intel etc.
    // Each GPU agent is followed by a "Marketing Name:" line with the
    // human-readable product name and, further down, "Chip ID: 30032(0x7550)".
    // Prefer the marketing name for display; when the ROCm tools do not know
    // the card it is a generic "AMD Radeon Graphics", which the chip id can
    // usually resolve (see amd_names). Fall back to the raw gfx id when the
    // marketing line is absent (older rocminfo, test fixtures).
    struct Agent {
        gfx: String,
        marketing: String,
        chip: Option<u32>,
    }
    fn finish(agent: Agent) -> String {
        if agent.marketing.is_empty() && agent.chip.is_none() {
            return agent.gfx;
        }
        resolve_amd_name(&agent.marketing, &agent.gfx, agent.chip, None, None)
    }

    let mut arch = String::new();
    let mut names = Vec::new();
    let mut current: Option<Agent> = None;

    for line in output.lines() {
        let trimmed = line.trim();
        if let Some(name_val) = trimmed.strip_prefix("Name:") {
            if let Some(agent) = current.take() {
                names.push(finish(agent));
            }
            let name = name_val.trim().to_string();
            if name.starts_with("gfx") {
                if arch.is_empty() {
                    arch = name.clone();
                }
                current = Some(Agent {
                    gfx: name,
                    marketing: String::new(),
                    chip: None,
                });
            }
        } else if let Some(agent) = current.as_mut() {
            if let Some(marketing) = trimmed.strip_prefix("Marketing Name:") {
                agent.marketing = marketing.trim().to_string();
            } else if let Some(chip) = trimmed.strip_prefix("Chip ID:") {
                agent.chip = parse_hex_id(chip);
            }
        }
    }
    if let Some(agent) = current.take() {
        names.push(finish(agent));
    }

    if names.is_empty() {
        return None;
    }

    Some(DetectedGpu {
        arch,
        count: names.len(),
        names,
    })
}

/// Detect NVIDIA GPUs via nvidia-smi.
pub fn detect_nvidia_gpus() -> Option<DetectedGpu> {
    let output = Command::new("nvidia-smi")
        .args(["--query-gpu=name", "--format=csv,noheader"])
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::null())
        .output()
        .ok()?;

    if !output.status.success() {
        return None;
    }

    let stdout = String::from_utf8_lossy(&output.stdout);
    let names: Vec<String> = stdout
        .lines()
        .map(|l| l.trim().to_string())
        .filter(|l| !l.is_empty())
        .collect();

    if names.is_empty() {
        return None;
    }

    Some(DetectedGpu {
        arch: "nvidia".into(),
        count: names.len(),
        names,
    })
}

/// Detect GPUs (try ROCm first, then NVIDIA).
pub fn detect_gpus() -> Option<DetectedGpu> {
    let rocm = detect_rocm_gpus();
    let nvidia = detect_nvidia_gpus();
    match (rocm, nvidia) {
        (Some(mut r), Some(n)) => {
            r.count += n.count;
            r.names.extend(n.names);
            Some(r)
        }
        (Some(r), None) => Some(r),
        (None, Some(n)) => Some(n),
        (None, None) => None,
    }
}

/// Generate a device list string like "0,1,2,3" for N devices.
pub fn device_list_for_count(count: usize) -> String {
    (0..count)
        .map(|i| i.to_string())
        .collect::<Vec<_>>()
        .join(",")
}

// ── Build environment variables ────────────────────────────────────────────

pub fn build_rocm_env(gpu_env: &GpuEnv, llama_server_cwd: &str) -> Vec<(String, String)> {
    let rocm = &gpu_env.rocm_path;
    let mut env = vec![
        ("ROCM_PATH".into(), rocm.clone()),
        ("HIP_PATH".into(), rocm.clone()),
        ("HIP_PLATFORM".into(), "amd".into()),
        ("HIP_CLANG_PATH".into(), format!("{rocm}/llvm/bin")),
        (
            "PATH".into(),
            format!(
                "{rocm}/bin:{rocm}/llvm/bin:{}",
                std::env::var("PATH").unwrap_or_default()
            ),
        ),
        (
            "LD_LIBRARY_PATH".into(),
            format!(
                "{llama_server_cwd}/build/bin:{rocm}/lib:{rocm}/lib64:{rocm}/llvm/lib:{}",
                std::env::var("LD_LIBRARY_PATH").unwrap_or_default()
            ),
        ),
        ("GGML_BACKEND_HIP".into(), "1".into()),
    ];

    // Architecture-specific overrides (skip for "auto" — let ROCm detect natively)
    if gpu_env.arch != "auto" {
        if let Some(hsa_ver) = hsa_version_for_arch(&gpu_env.arch) {
            env.push(("HSA_OVERRIDE_GFX_VERSION".into(), hsa_ver.into()));
        }
        env.push(("HCC_AMDGPU_TARGET".into(), gpu_env.arch.clone()));
    }

    // Device visibility
    if !gpu_env.devices.is_empty() {
        env.push(("HIP_VISIBLE_DEVICES".into(), gpu_env.devices.clone()));
        env.push(("CUDA_VISIBLE_DEVICES".into(), gpu_env.devices.clone()));
        env.push(("ROCR_VISIBLE_DEVICES".into(), gpu_env.devices.clone()));
    }

    // Custom env vars
    for (k, v) in &gpu_env.extra_env {
        env.push((k.clone(), v.clone()));
    }

    env
}

pub fn build_nvidia_env(gpu_env: &GpuEnv) -> Vec<(String, String)> {
    let mut env = Vec::new();
    if !gpu_env.devices.is_empty() {
        env.push(("CUDA_VISIBLE_DEVICES".into(), gpu_env.devices.clone()));
    }
    for (k, v) in &gpu_env.extra_env {
        env.push((k.clone(), v.clone()));
    }
    env
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hsa_version_lookup() {
        assert_eq!(hsa_version_for_arch("gfx906"), Some("9.0.6"));
        assert_eq!(hsa_version_for_arch("gfx1100"), Some("11.0.0"));
        assert_eq!(hsa_version_for_arch("gfx942"), Some("9.4.2"));
        assert_eq!(hsa_version_for_arch("auto"), None);
        assert_eq!(hsa_version_for_arch("unknown"), None);
    }

    #[test]
    fn test_device_list_for_count() {
        assert_eq!(device_list_for_count(1), "0");
        assert_eq!(device_list_for_count(4), "0,1,2,3");
        assert_eq!(device_list_for_count(0), "");
    }

    #[test]
    fn test_build_rocm_env_auto() {
        let gpu_env = GpuEnv {
            arch: "auto".into(),
            devices: "".into(),
            rocm_path: "/opt/rocm".into(),
            extra_env: vec![],
        };
        let env = build_rocm_env(&gpu_env, "/tmp");
        // Should NOT contain HSA_OVERRIDE_GFX_VERSION or HCC_AMDGPU_TARGET
        assert!(!env.iter().any(|(k, _)| k == "HSA_OVERRIDE_GFX_VERSION"));
        assert!(!env.iter().any(|(k, _)| k == "HCC_AMDGPU_TARGET"));
        // Should NOT contain device visibility vars
        assert!(!env.iter().any(|(k, _)| k == "HIP_VISIBLE_DEVICES"));
    }

    #[test]
    fn test_build_rocm_env_specific_arch() {
        let gpu_env = GpuEnv {
            arch: "gfx1100".into(),
            devices: "0,1".into(),
            rocm_path: "/opt/rocm".into(),
            extra_env: vec![("MY_VAR".into(), "hello".into())],
        };
        let env = build_rocm_env(&gpu_env, "/tmp");
        assert!(
            env.iter()
                .any(|(k, v)| k == "HSA_OVERRIDE_GFX_VERSION" && v == "11.0.0")
        );
        assert!(
            env.iter()
                .any(|(k, v)| k == "HCC_AMDGPU_TARGET" && v == "gfx1100")
        );
        assert!(
            env.iter()
                .any(|(k, v)| k == "HIP_VISIBLE_DEVICES" && v == "0,1")
        );
        assert!(env.iter().any(|(k, v)| k == "MY_VAR" && v == "hello"));
    }

    #[test]
    fn test_build_nvidia_env() {
        let gpu_env = GpuEnv {
            arch: "auto".into(),
            devices: "0,2".into(),
            rocm_path: String::new(),
            extra_env: vec![],
        };
        let env = build_nvidia_env(&gpu_env);
        assert!(
            env.iter()
                .any(|(k, v)| k == "CUDA_VISIBLE_DEVICES" && v == "0,2")
        );
    }

    #[test]
    fn test_parse_rocminfo() {
        let output = r#"
ROCk module is loaded
=====================
HSA System Attributes
=====================
Agent 1
*******
  Name:                    gfx906
  Device Type:             GPU
  Feature:                 KERNEL_DISPATCH
Agent 2
*******
  Name:                    gfx906
  Device Type:             GPU
  Feature:                 KERNEL_DISPATCH
Agent 3
*******
  Name:                    gfx906
  Device Type:             GPU
  Feature:                 KERNEL_DISPATCH
Agent 4
*******
  Name:                    gfx906
  Device Type:             GPU
  Feature:                 KERNEL_DISPATCH
"#;
        let detected = parse_rocminfo(output).unwrap();
        assert_eq!(detected.arch, "gfx906");
        assert_eq!(detected.count, 4);
    }

    #[test]
    fn rocminfo_generic_marketing_name_resolves_from_chip_id() {
        let output = r#"
Agent 2
*******
  Name:                    gfx1201
  Uuid:                    GPU-1
  Marketing Name:          AMD Radeon Graphics
  Vendor Name:             AMD
  Chip ID:                 30033(0x7551)
Agent 3
*******
  Name:                    gfx1201
  Uuid:                    GPU-2
  Marketing Name:          AMD Radeon RX 9070 XT
  Vendor Name:             AMD
  Chip ID:                 30032(0x7550)
"#;
        let detected = parse_rocminfo(output).unwrap();
        assert_eq!(
            detected.names,
            vec!["AMD Radeon AI PRO R9700", "AMD Radeon RX 9070 XT"]
        );
    }

    #[test]
    fn test_parse_rocminfo_no_gpu() {
        let output = "ROCk module is loaded\n";
        assert!(parse_rocminfo(output).is_none());
    }

    #[test]
    fn test_gpu_env_serialization() {
        let env = GpuEnv::default();
        let json = serde_json::to_string(&env).unwrap();
        let parsed: GpuEnv = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed.arch, "auto");
        assert_eq!(parsed.rocm_path, "/opt/rocm");
    }

    #[test]
    fn test_load_save_roundtrip() {
        let path = std::env::temp_dir().join("llama-admin-monitor-gpu-env-test.json");
        let env = GpuEnv {
            arch: "gfx1100".into(),
            devices: "0,1".into(),
            rocm_path: "/opt/rocm".into(),
            extra_env: vec![("FOO".into(), "bar".into())],
        };
        save_gpu_env(&path, &env).unwrap();
        let loaded = load_gpu_env(&path);
        assert_eq!(loaded.arch, "gfx1100");
        assert_eq!(loaded.devices, "0,1");
        assert_eq!(loaded.extra_env.len(), 1);
        std::fs::remove_file(&path).ok();
    }
}
