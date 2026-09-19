//! Which processes are using the GPUs, nvtop-style. NVIDIA reports this
//! through `nvidia-smi --query-compute-apps`; amdgpu (and other DRM
//! drivers) expose per-client memory and engine time in each process's
//! `/proc/<pid>/fdinfo/<fd>`, which is what nvtop reads too. Only
//! processes whose fdinfo is readable show up: the monitor's own user's,
//! or everyone's when it runs as root.

use std::collections::HashMap;
use std::path::Path;
use std::time::Instant;

#[derive(Debug, Clone, Default, PartialEq, serde::Serialize)]
pub struct GpuProcess {
    pub pid: u32,
    pub name: String,
    /// PCI bus id (`0000:03:00.0`) of the GPU, to match a metrics card.
    pub bus: Option<String>,
    pub vendor: String,
    pub vram_bytes: u64,
    /// Share of the sample interval the process kept the 3D/compute engine
    /// busy; None on the first sample and where the driver does not say.
    pub busy_percent: Option<f32>,
}

/// One DRM client as parsed from an fdinfo file.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct DrmClient {
    pub driver: String,
    pub pdev: String,
    pub client_id: String,
    pub vram_bytes: u64,
    /// Busy nanoseconds on the gfx (or compute, when gfx is absent) engine.
    pub engine_ns: Option<u64>,
}

/// Parses one `/proc/<pid>/fdinfo/<fd>` file. Returns None for fds that
/// are not DRM clients.
pub fn parse_fdinfo(text: &str) -> Option<DrmClient> {
    let mut c = DrmClient::default();
    let mut gfx_ns = None;
    let mut compute_ns = None;
    for line in text.lines() {
        let Some((key, value)) = line.split_once(':') else {
            continue;
        };
        let value = value.trim();
        match key.trim() {
            "drm-driver" => c.driver = value.to_string(),
            "drm-pdev" => c.pdev = value.to_string(),
            "drm-client-id" => c.client_id = value.to_string(),
            "drm-memory-vram" | "drm-total-vram" => c.vram_bytes = parse_size(value).unwrap_or(0),
            "drm-engine-gfx" => gfx_ns = parse_ns(value),
            "drm-engine-compute" => compute_ns = parse_ns(value),
            _ => {}
        }
    }
    if c.driver.is_empty() {
        return None;
    }
    c.engine_ns = gfx_ns.or(compute_ns);
    Some(c)
}

/// "1234 KiB" / "12 MiB" / "5 GiB" -> bytes.
fn parse_size(v: &str) -> Option<u64> {
    let mut parts = v.split_whitespace();
    let n: u64 = parts.next()?.parse().ok()?;
    let mult = match parts.next().unwrap_or("B") {
        "KiB" => 1024,
        "MiB" => 1024 * 1024,
        "GiB" => 1024 * 1024 * 1024,
        _ => 1,
    };
    Some(n * mult)
}

/// "123456789 ns" -> 123456789.
fn parse_ns(v: &str) -> Option<u64> {
    v.split_whitespace().next()?.parse().ok()
}

/// Keeps the previous engine-time reading per (pid, client) so busy% can
/// be derived from the delta.
#[derive(Default)]
pub struct ProcessProbe {
    prev: HashMap<(u32, String), (u64, Instant)>,
}

impl ProcessProbe {
    pub fn sample(&mut self) -> Vec<GpuProcess> {
        let now = Instant::now();
        let mut out = self.sample_drm(now);
        out.extend(nvidia_processes());
        out.sort_by(|a, b| b.vram_bytes.cmp(&a.vram_bytes).then(a.pid.cmp(&b.pid)));
        out
    }

    fn sample_drm(&mut self, now: Instant) -> Vec<GpuProcess> {
        let mut seen: HashMap<(u32, String), ()> = HashMap::new();
        let mut per_proc: HashMap<(u32, String), GpuProcess> = HashMap::new();
        let mut next_prev = HashMap::new();
        let Ok(proc_dir) = std::fs::read_dir("/proc") else {
            return Vec::new();
        };
        for entry in proc_dir.flatten() {
            let Ok(pid) = entry.file_name().to_string_lossy().parse::<u32>() else {
                continue;
            };
            let fd_dir = entry.path().join("fd");
            let Ok(fds) = std::fs::read_dir(&fd_dir) else {
                continue;
            };
            for fd in fds.flatten() {
                // Only DRM device fds carry GPU accounting.
                let Ok(target) = std::fs::read_link(fd.path()) else {
                    continue;
                };
                if !target.starts_with("/dev/dri/") {
                    continue;
                }
                let info_path = entry.path().join("fdinfo").join(fd.file_name());
                let Ok(text) = std::fs::read_to_string(&info_path) else {
                    continue;
                };
                let Some(client) = parse_fdinfo(&text) else {
                    continue;
                };
                // Several fds share one client; count it once.
                let key = (pid, client.client_id.clone());
                if seen.insert(key.clone(), ()).is_some() {
                    continue;
                }
                let busy = client.engine_ns.and_then(|ns| {
                    next_prev.insert(key.clone(), (ns, now));
                    let (prev_ns, prev_at) = self.prev.get(&key)?;
                    let wall = now.duration_since(*prev_at).as_nanos() as f64;
                    if wall <= 0.0 {
                        return None;
                    }
                    Some(((ns.saturating_sub(*prev_ns) as f64 / wall) * 100.0).min(100.0) as f32)
                });
                let slot = per_proc
                    .entry((pid, client.pdev.clone()))
                    .or_insert_with(|| GpuProcess {
                        pid,
                        name: process_name(pid),
                        bus: Some(client.pdev.clone()).filter(|b| !b.is_empty()),
                        vendor: vendor_for_driver(&client.driver).to_string(),
                        ..Default::default()
                    });
                slot.vram_bytes += client.vram_bytes;
                if let Some(b) = busy {
                    slot.busy_percent = Some(slot.busy_percent.unwrap_or(0.0) + b);
                }
            }
        }
        self.prev = next_prev;
        per_proc
            .into_values()
            .filter(|p| p.vram_bytes > 0 || p.busy_percent.unwrap_or(0.0) > 0.5)
            .collect()
    }
}

fn vendor_for_driver(driver: &str) -> &'static str {
    match driver {
        "amdgpu" | "radeon" => "AMD",
        "i915" | "xe" => "Intel",
        "nvidia-drm" | "nouveau" => "NVIDIA",
        _ => "GPU",
    }
}

fn process_name(pid: u32) -> String {
    std::fs::read_to_string(Path::new("/proc").join(pid.to_string()).join("comm"))
        .map(|s| s.trim().to_string())
        .unwrap_or_else(|_| format!("pid {pid}"))
}

/// `nvidia-smi --query-compute-apps`, empty when the tool is absent or
/// reports nothing.
fn nvidia_processes() -> Vec<GpuProcess> {
    let Ok(out) = std::process::Command::new("nvidia-smi")
        .args([
            "--query-compute-apps=pid,process_name,used_memory,gpu_bus_id",
            "--format=csv,noheader,nounits",
        ])
        .output()
    else {
        return Vec::new();
    };
    if !out.status.success() {
        return Vec::new();
    }
    parse_nvidia_apps(&String::from_utf8_lossy(&out.stdout))
}

pub fn parse_nvidia_apps(csv: &str) -> Vec<GpuProcess> {
    csv.lines()
        .filter_map(|line| {
            let f: Vec<&str> = line.split(',').map(str::trim).collect();
            if f.len() < 4 {
                return None;
            }
            Some(GpuProcess {
                pid: f[0].parse().ok()?,
                name: f[1].rsplit('/').next().unwrap_or(f[1]).to_string(),
                vram_bytes: f[2].parse::<u64>().ok()? * 1024 * 1024,
                bus: Some(normalize_bus(f[3])),
                vendor: "NVIDIA".to_string(),
                busy_percent: None,
            })
        })
        .collect()
}

/// nvidia-smi prints `00000000:03:00.0`; the kernel uses `0000:03:00.0`.
pub fn normalize_bus(bus: &str) -> String {
    let b = bus.trim().to_ascii_lowercase();
    match b.split_once(':') {
        Some((domain, rest)) if domain.len() > 4 => {
            format!("{}:{rest}", &domain[domain.len() - 4..])
        }
        _ => b,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fdinfo_amdgpu_client() {
        let text = "pos:\t0\nflags:\t02100002\nmnt_id:\t28\nino:\t1234\n\
                    drm-driver:\tamdgpu\ndrm-client-id:\t42\ndrm-pdev:\t0000:03:00.0\n\
                    drm-memory-vram:\t1048576 KiB\ndrm-memory-gtt:\t2048 KiB\n\
                    drm-engine-gfx:\t123456789 ns\ndrm-engine-compute:\t0 ns\n";
        let c = parse_fdinfo(text).unwrap();
        assert_eq!(c.driver, "amdgpu");
        assert_eq!(c.client_id, "42");
        assert_eq!(c.pdev, "0000:03:00.0");
        assert_eq!(c.vram_bytes, 1024 * 1024 * 1024);
        assert_eq!(c.engine_ns, Some(123456789));
    }

    #[test]
    fn fdinfo_non_drm_fd_is_ignored() {
        assert_eq!(parse_fdinfo("pos:\t0\nflags:\t02\nmnt_id:\t28\n"), None);
    }

    #[test]
    fn nvidia_apps_csv() {
        let procs = parse_nvidia_apps("12345, /usr/bin/llama-server, 8192, 00000000:0B:00.0\n");
        assert_eq!(procs.len(), 1);
        assert_eq!(procs[0].pid, 12345);
        assert_eq!(procs[0].name, "llama-server");
        assert_eq!(procs[0].vram_bytes, 8192 * 1024 * 1024);
        assert_eq!(procs[0].bus.as_deref(), Some("0000:0b:00.0"));
    }

    #[test]
    fn bus_ids_normalise() {
        assert_eq!(normalize_bus("00000000:03:00.0"), "0000:03:00.0");
        assert_eq!(normalize_bus("0000:03:00.0"), "0000:03:00.0");
    }
}
