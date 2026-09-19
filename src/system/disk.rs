//! The physical disk behind the models directory: identity from sysfs,
//! temperature from the kernel's hwmon (NVMe and `drivetemp` expose it
//! without root), and SMART health/wear from `smartctl` when available.

use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

/// SMART is polled far less often than throughput: it costs a subprocess
/// and the values move slowly.
const SMART_INTERVAL: Duration = Duration::from_secs(60);

#[derive(Debug, Clone, Default, PartialEq, serde::Serialize)]
pub struct DiskInfo {
    /// Block device name, e.g. `nvme0n1` or `sda`.
    pub device: String,
    pub model: String,
    /// "NVMe", "SSD" or "HDD" (from the rotational flag and device name).
    pub kind: String,
    pub size_bytes: u64,
    pub temp_c: Option<f32>,
    /// SMART overall assessment: true = passed. None when unknown.
    pub health_passed: Option<bool>,
    /// Why SMART data is missing, when it is (usually smartctl/permissions).
    pub smart_error: Option<String>,
    pub power_on_hours: Option<u64>,
    /// NVMe "Percentage Used" (or an SSD wear attribute): 0 = new, 100 = rated life.
    pub wear_percent: Option<u32>,
    pub available_spare_percent: Option<u32>,
    pub media_errors: Option<u64>,
    pub reallocated_sectors: Option<u64>,
    pub data_written_bytes: Option<u64>,
    /// NVMe critical warning bitmask, when non-zero.
    pub critical_warning: Option<u32>,
}

/// Rolling state: the device is resolved once per models directory and
/// SMART is refreshed on its own timer.
#[derive(Default)]
pub struct DiskProbe {
    resolved_for: Option<PathBuf>,
    device: Option<String>,
    smart_at: Option<Instant>,
    smart: Option<SmartData>,
    smart_error: Option<String>,
}

impl DiskProbe {
    pub fn sample(&mut self, target: &Path) -> Option<DiskInfo> {
        if self.resolved_for.as_deref() != Some(target) {
            self.resolved_for = Some(target.to_path_buf());
            self.device = resolve_block_device(target);
            self.smart_at = None;
            self.smart = None;
            self.smart_error = None;
        }
        let device = self.device.clone()?;
        let sys = PathBuf::from("/sys/block").join(&device);
        let mut info = DiskInfo {
            device: device.clone(),
            model: read_trimmed(&sys.join("device/model")).unwrap_or_default(),
            size_bytes: read_trimmed(&sys.join("size"))
                .and_then(|s| s.parse::<u64>().ok())
                .map(|sectors| sectors * 512)
                .unwrap_or(0),
            ..Default::default()
        };
        let rotational = read_trimmed(&sys.join("queue/rotational")).as_deref() == Some("1");
        info.kind = disk_kind(&device, rotational).to_string();
        info.temp_c = hwmon_disk_temp(&sys);

        if self
            .smart_at
            .is_none_or(|at| at.elapsed() >= SMART_INTERVAL)
        {
            self.smart_at = Some(Instant::now());
            match probe_smart(&device) {
                Ok(data) => {
                    self.smart = Some(data);
                    self.smart_error = None;
                }
                Err(e) => {
                    if self.smart.is_none() {
                        self.smart_error = Some(e);
                    }
                }
            }
        }
        if let Some(s) = &self.smart {
            info.health_passed = s.passed;
            info.power_on_hours = s.power_on_hours;
            info.wear_percent = s.wear_percent;
            info.available_spare_percent = s.available_spare_percent;
            info.media_errors = s.media_errors;
            info.reallocated_sectors = s.reallocated_sectors;
            info.data_written_bytes = s.data_written_bytes;
            info.critical_warning = s.critical_warning;
            if info.temp_c.is_none() {
                info.temp_c = s.temp_c;
            }
        }
        info.smart_error = self.smart_error.clone();
        Some(info)
    }
}

fn read_trimmed(path: &Path) -> Option<String> {
    std::fs::read_to_string(path)
        .ok()
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
}

pub fn disk_kind(device: &str, rotational: bool) -> &'static str {
    if device.starts_with("nvme") {
        "NVMe"
    } else if rotational {
        "HDD"
    } else {
        "SSD"
    }
}

/// Walks from the filesystem holding `path` up to its whole disk:
/// `findmnt` gives the source (a partition, or an LVM/dm device), and
/// `lsblk -no PKNAME` climbs one level at a time.
fn resolve_block_device(path: &Path) -> Option<String> {
    let out = std::process::Command::new("findmnt")
        .args(["-n", "-o", "SOURCE", "--target"])
        .arg(path)
        .output()
        .ok()?;
    if !out.status.success() {
        return None;
    }
    let source = String::from_utf8_lossy(&out.stdout).trim().to_string();
    // "/dev/nvme0n1p2[/@home]" for btrfs subvolumes
    let source = source.split('[').next().unwrap_or("").trim().to_string();
    if !source.starts_with("/dev/") {
        return None;
    }
    let mut name = source.trim_start_matches("/dev/").to_string();
    // dm devices report as /dev/mapper/x; lsblk understands the path
    let mut current = source;
    for _ in 0..4 {
        let short = name.rsplit('/').next().unwrap_or(&name).to_string();
        if super::is_whole_disk(&short) && Path::new("/sys/block").join(&short).exists() {
            return Some(short);
        }
        let out = std::process::Command::new("lsblk")
            .args(["-no", "PKNAME"])
            .arg(&current)
            .output()
            .ok()?;
        let parent = String::from_utf8_lossy(&out.stdout)
            .lines()
            .map(str::trim)
            .find(|l| !l.is_empty())
            .map(str::to_string)?;
        name = parent.clone();
        current = format!("/dev/{parent}");
    }
    None
}

/// Temperature from the drive's own hwmon node. NVMe controllers put it
/// at `device/hwmonN/` (the block device's `device` link is the
/// controller); SATA drives with the `drivetemp` module at
/// `device/hwmon/hwmonN/`. Both are readable without root.
fn hwmon_disk_temp(sys: &Path) -> Option<f32> {
    let mut candidates = Vec::new();
    for base in [sys.join("device"), sys.join("device/hwmon")] {
        if let Ok(entries) = std::fs::read_dir(&base) {
            for e in entries.flatten() {
                let n = e.file_name().to_string_lossy().to_string();
                if n.starts_with("hwmon") {
                    candidates.push(e.path());
                }
            }
        }
    }
    for dir in candidates {
        // temp1 is the composite/drive temperature on both drivers
        if let Some(v) = read_trimmed(&dir.join("temp1_input")).and_then(|s| s.parse::<f32>().ok())
        {
            return Some(v / 1000.0);
        }
    }
    None
}

#[derive(Debug, Clone, Default, PartialEq)]
pub struct SmartData {
    pub passed: Option<bool>,
    pub temp_c: Option<f32>,
    pub power_on_hours: Option<u64>,
    pub wear_percent: Option<u32>,
    pub available_spare_percent: Option<u32>,
    pub media_errors: Option<u64>,
    pub reallocated_sectors: Option<u64>,
    pub data_written_bytes: Option<u64>,
    pub critical_warning: Option<u32>,
}

/// Runs `smartctl -j -H -A` on the device, plain and then via `sudo -n`
/// (a NOPASSWD sudoers rule for that command makes the second work; see
/// the README).
fn probe_smart(device: &str) -> Result<SmartData, String> {
    let dev = format!("/dev/{device}");
    let attempts: [&[&str]; 2] = [
        &["smartctl", "-j", "-H", "-A", dev.as_str()],
        &["sudo", "-n", "smartctl", "-j", "-H", "-A", dev.as_str()],
    ];
    let mut last_err = String::from("smartctl not found (install smartmontools)");
    for argv in attempts {
        let output = match std::process::Command::new(argv[0])
            .args(&argv[1..])
            .output()
        {
            Ok(o) => o,
            Err(e) => {
                last_err = format!("{}: {e}", argv.join(" "));
                continue;
            }
        };
        // smartctl's exit status is a bitmask: bit 0/1 = command line or
        // device open errors, bit 3 = SMART failing (still valid output).
        let code = output.status.code().unwrap_or(1);
        if code & 0b11 != 0 {
            let text = String::from_utf8_lossy(&output.stdout);
            let msg = serde_json::from_str::<serde_json::Value>(&text)
                .ok()
                .and_then(|v| {
                    v.get("smartctl")?
                        .get("messages")?
                        .as_array()?
                        .iter()
                        .filter_map(|m| m.get("string")?.as_str().map(str::to_string))
                        .next()
                })
                .or_else(|| {
                    String::from_utf8_lossy(&output.stderr)
                        .lines()
                        .next()
                        .map(|l| l.trim().to_string())
                })
                .unwrap_or_else(|| "permission denied".to_string());
            last_err = msg;
            continue;
        }
        let text = String::from_utf8_lossy(&output.stdout);
        return parse_smartctl_json(&text).ok_or_else(|| "unreadable smartctl output".to_string());
    }
    Err(last_err)
}

/// Parses `smartctl -j -H -A` for NVMe and ATA drives.
pub fn parse_smartctl_json(text: &str) -> Option<SmartData> {
    let v: serde_json::Value = serde_json::from_str(text).ok()?;
    let mut s = SmartData {
        passed: v.pointer("/smart_status/passed").and_then(|p| p.as_bool()),
        temp_c: v
            .pointer("/temperature/current")
            .and_then(|t| t.as_f64())
            .map(|t| t as f32),
        power_on_hours: v.pointer("/power_on_time/hours").and_then(|h| h.as_u64()),
        ..Default::default()
    };
    if let Some(log) = v.get("nvme_smart_health_information_log") {
        s.wear_percent = log
            .get("percentage_used")
            .and_then(|x| x.as_u64())
            .map(|x| x.min(u32::MAX as u64) as u32);
        s.available_spare_percent = log
            .get("available_spare")
            .and_then(|x| x.as_u64())
            .map(|x| x as u32);
        s.media_errors = log.get("media_errors").and_then(|x| x.as_u64());
        // Data units are 1000 x 512-byte blocks.
        s.data_written_bytes = log
            .get("data_units_written")
            .and_then(|x| x.as_u64())
            .map(|u| u.saturating_mul(512_000));
        s.critical_warning = log
            .get("critical_warning")
            .and_then(|x| x.as_u64())
            .filter(|&x| x != 0)
            .map(|x| x as u32);
    }
    if let Some(table) = v.pointer("/ata_smart_attributes/table").and_then(|t| t.as_array()) {
        for attr in table {
            let id = attr.get("id").and_then(|i| i.as_u64()).unwrap_or(0);
            let raw = attr.pointer("/raw/value").and_then(|r| r.as_u64());
            let normalized = attr.get("value").and_then(|n| n.as_u64());
            match id {
                5 => s.reallocated_sectors = raw,
                9 if s.power_on_hours.is_none() => s.power_on_hours = raw,
                // Wear levelling / SSD life left are reported as "life
                // remaining" normalised 100 -> 0.
                177 | 231 | 233 if s.wear_percent.is_none() => {
                    s.wear_percent = normalized.map(|n| 100u64.saturating_sub(n.min(100)) as u32)
                }
                _ => {}
            }
        }
    }
    Some(s)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn nvme_smart_json() {
        let text = r#"{
            "smart_status": {"passed": true},
            "temperature": {"current": 41},
            "power_on_time": {"hours": 8760},
            "nvme_smart_health_information_log": {
                "critical_warning": 0,
                "temperature": 41,
                "available_spare": 100,
                "percentage_used": 3,
                "data_units_written": 100000000,
                "media_errors": 0
            }
        }"#;
        let s = parse_smartctl_json(text).unwrap();
        assert_eq!(s.passed, Some(true));
        assert_eq!(s.temp_c, Some(41.0));
        assert_eq!(s.power_on_hours, Some(8760));
        assert_eq!(s.wear_percent, Some(3));
        assert_eq!(s.available_spare_percent, Some(100));
        assert_eq!(s.media_errors, Some(0));
        assert_eq!(s.data_written_bytes, Some(51_200_000_000_000));
        assert_eq!(s.critical_warning, None);
    }

    #[test]
    fn ata_smart_json() {
        let text = r#"{
            "smart_status": {"passed": false},
            "temperature": {"current": 35},
            "ata_smart_attributes": {"table": [
                {"id": 5, "name": "Reallocated_Sector_Ct", "value": 100, "raw": {"value": 12}},
                {"id": 9, "name": "Power_On_Hours", "value": 98, "raw": {"value": 20000}},
                {"id": 177, "name": "Wear_Leveling_Count", "value": 91, "raw": {"value": 300}}
            ]}
        }"#;
        let s = parse_smartctl_json(text).unwrap();
        assert_eq!(s.passed, Some(false));
        assert_eq!(s.reallocated_sectors, Some(12));
        assert_eq!(s.power_on_hours, Some(20000));
        assert_eq!(s.wear_percent, Some(9));
    }

    #[test]
    fn kinds() {
        assert_eq!(disk_kind("nvme0n1", false), "NVMe");
        assert_eq!(disk_kind("sda", true), "HDD");
        assert_eq!(disk_kind("sdb", false), "SSD");
    }
}
