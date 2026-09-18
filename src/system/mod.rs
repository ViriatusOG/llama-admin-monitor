//! Host telemetry for the Monitor page: CPU, memory and disk. Read from
//! `/proc` on Linux; on other platforms every reading stays "unavailable"
//! rather than zero, so the UI can say so instead of showing 0 %.

use std::path::Path;
use std::time::Instant;

#[derive(Debug, Clone, Default, serde::Serialize)]
pub struct SystemStats {
    /// False until the first successful sample (and always on non-Linux).
    pub available: bool,
    pub cpu_percent: Option<f32>,
    /// Per-core utilisation in `/proc/stat` order; None until a baseline exists.
    pub core_percent: Vec<Option<f32>>,
    pub cpu_cores: u32,
    pub cpu_model: String,
    pub load_avg_1m: Option<f32>,
    pub mem_total_bytes: u64,
    pub mem_used_bytes: u64,
    pub swap_total_bytes: u64,
    pub swap_used_bytes: u64,
    /// Sampled over the poll interval, summed across whole physical disks.
    pub disk_read_bytes_per_sec: Option<f64>,
    pub disk_write_bytes_per_sec: Option<f64>,
    /// Free space on the filesystem holding the models directory.
    pub disk_total_bytes: Option<u64>,
    pub disk_free_bytes: Option<u64>,
    pub disk_mount: String,
    pub sample_secs: f32,
    /// Physical memory slots, from `dmidecode -t 17`. Static, read once.
    pub dimm_slots_total: u32,
    pub dimms: Vec<DimmInfo>,
    /// Why DIMM details are missing, when they are (usually permissions).
    pub dimm_error: Option<String>,
}

/// One populated memory slot.
#[derive(Debug, Clone, Default, PartialEq, serde::Serialize)]
pub struct DimmInfo {
    pub locator: String,
    pub size_bytes: u64,
    pub mem_type: String,
    /// Rated speed in MT/s (dmidecode reports MHz on older firmware).
    pub speed_mts: Option<u32>,
    /// Speed the slot actually runs at, when the firmware reports it.
    pub configured_speed_mts: Option<u32>,
    pub manufacturer: String,
    pub part_number: String,
}

/// Rolling state between samples: the deltas need a previous reading.
#[derive(Default)]
pub struct SystemSampler {
    prev_cpu: Option<(u64, u64)>,
    prev_cores: Vec<(u64, u64)>,
    prev_disk: Option<(u64, u64)>,
    prev_at: Option<Instant>,
    /// (slots, populated, error) once probed; probing costs a subprocess.
    dimms: Option<(u32, Vec<DimmInfo>, Option<String>)>,
    cpu_model: Option<String>,
}

impl SystemSampler {
    pub fn new() -> Self {
        Self::default()
    }

    /// Takes one sample. `models_dir` picks the filesystem for the free-space
    /// reading; `/` is used when it is unset.
    pub fn sample(&mut self, models_dir: Option<&Path>) -> SystemStats {
        let now = Instant::now();
        let elapsed = self
            .prev_at
            .map(|t| now.duration_since(t).as_secs_f32())
            .unwrap_or(0.0);
        self.prev_at = Some(now);

        let mut stats = SystemStats {
            cpu_cores: std::thread::available_parallelism()
                .map(|n| n.get() as u32)
                .unwrap_or(0),
            sample_secs: elapsed,
            ..Default::default()
        };

        if let Ok(text) = std::fs::read_to_string("/proc/stat")
            && let Some((idle, total)) = parse_proc_stat(&text)
        {
            if let Some((prev_idle, prev_total)) = self.prev_cpu {
                stats.cpu_percent = busy_percent((prev_idle, prev_total), (idle, total));
            }
            self.prev_cpu = Some((idle, total));

            let cores = parse_proc_stat_cores(&text);
            stats.core_percent = cores
                .iter()
                .enumerate()
                .map(|(i, cur)| {
                    self.prev_cores
                        .get(i)
                        .and_then(|prev| busy_percent(*prev, *cur))
                })
                .collect();
            self.prev_cores = cores;
            stats.available = true;
        }

        if self.cpu_model.is_none() {
            self.cpu_model = Some(
                std::fs::read_to_string("/proc/cpuinfo")
                    .ok()
                    .and_then(|t| parse_cpu_model(&t))
                    .unwrap_or_default(),
            );
        }
        stats.cpu_model = self.cpu_model.clone().unwrap_or_default();

        if self.dimms.is_none() {
            self.dimms = Some(probe_dimms());
        }
        if let Some((slots, dimms, err)) = &self.dimms {
            stats.dimm_slots_total = *slots;
            stats.dimms = dimms.clone();
            stats.dimm_error = err.clone();
        }

        if let Ok(text) = std::fs::read_to_string("/proc/loadavg") {
            stats.load_avg_1m = parse_loadavg(&text);
        }

        if let Ok(text) = std::fs::read_to_string("/proc/meminfo") {
            let m = parse_meminfo(&text);
            stats.mem_total_bytes = m.total;
            stats.mem_used_bytes = m.total.saturating_sub(m.available);
            stats.swap_total_bytes = m.swap_total;
            stats.swap_used_bytes = m.swap_total.saturating_sub(m.swap_free);
            stats.available = true;
        }

        if let Ok(text) = std::fs::read_to_string("/proc/diskstats") {
            let (read_bytes, write_bytes) = parse_diskstats(&text);
            if let Some((prev_r, prev_w)) = self.prev_disk
                && elapsed > 0.0
            {
                stats.disk_read_bytes_per_sec =
                    Some(read_bytes.saturating_sub(prev_r) as f64 / elapsed as f64);
                stats.disk_write_bytes_per_sec =
                    Some(write_bytes.saturating_sub(prev_w) as f64 / elapsed as f64);
            }
            self.prev_disk = Some((read_bytes, write_bytes));
        }

        let target = models_dir
            .filter(|p| p.exists())
            .map(|p| p.to_path_buf())
            .unwrap_or_else(|| Path::new("/").to_path_buf());
        if let Some(space) = disk_space(&target) {
            stats.disk_total_bytes = Some(space.total);
            stats.disk_free_bytes = Some(space.free);
            stats.disk_mount = space.mount;
        }

        stats
    }
}

/// Returns (idle, total) jiffies from the aggregate `cpu` line.
pub fn parse_proc_stat(text: &str) -> Option<(u64, u64)> {
    let line = text.lines().find(|l| l.starts_with("cpu "))?;
    let nums: Vec<u64> = line
        .split_whitespace()
        .skip(1)
        .filter_map(|v| v.parse().ok())
        .collect();
    if nums.len() < 4 {
        return None;
    }
    // user nice system idle iowait irq softirq steal ...
    let idle = nums[3] + nums.get(4).copied().unwrap_or(0);
    let total: u64 = nums.iter().take(8).sum();
    Some((idle, total))
}

/// Utilisation between two (idle, total) readings, None when no time passed.
fn busy_percent(prev: (u64, u64), cur: (u64, u64)) -> Option<f32> {
    let d_total = cur.1.saturating_sub(prev.1);
    let d_idle = cur.0.saturating_sub(prev.0);
    if d_total == 0 {
        return None;
    }
    Some((1.0 - d_idle as f32 / d_total as f32) * 100.0)
}

/// (idle, total) per `cpuN` line, in file order.
pub fn parse_proc_stat_cores(text: &str) -> Vec<(u64, u64)> {
    text.lines()
        .filter(|l| l.starts_with("cpu") && !l.starts_with("cpu "))
        .filter_map(|line| {
            let nums: Vec<u64> = line
                .split_whitespace()
                .skip(1)
                .filter_map(|v| v.parse().ok())
                .collect();
            if nums.len() < 4 {
                return None;
            }
            let idle = nums[3] + nums.get(4).copied().unwrap_or(0);
            Some((idle, nums.iter().take(8).sum()))
        })
        .collect()
}

pub fn parse_cpu_model(text: &str) -> Option<String> {
    text.lines()
        .find(|l| l.starts_with("model name"))
        .and_then(|l| l.split_once(':'))
        .map(|(_, v)| v.split_whitespace().collect::<Vec<_>>().join(" "))
        .filter(|v| !v.is_empty())
}

/// Reads memory-slot details. `dmidecode` needs root to read the SMBIOS
/// tables, so a plain call is tried first and then `sudo -n` (which works
/// when a NOPASSWD sudoers rule exists for it, see the README).
fn probe_dimms() -> (u32, Vec<DimmInfo>, Option<String>) {
    let attempts: [&[&str]; 2] = [
        &["dmidecode", "-t", "17"],
        &["sudo", "-n", "dmidecode", "-t", "17"],
    ];
    let mut last_err = String::from("dmidecode not found");
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
        if output.status.success() {
            let text = String::from_utf8_lossy(&output.stdout);
            let (slots, dimms) = parse_dmidecode_memory(&text);
            if slots > 0 {
                return (slots, dimms, None);
            }
            last_err = "dmidecode returned no memory devices".to_string();
        } else {
            let stderr = String::from_utf8_lossy(&output.stderr);
            last_err = stderr
                .lines()
                .next()
                .unwrap_or("permission denied")
                .trim()
                .to_string();
        }
    }
    (0, Vec::new(), Some(last_err))
}

/// Parses `dmidecode -t 17` into (slot count, populated slots).
pub fn parse_dmidecode_memory(text: &str) -> (u32, Vec<DimmInfo>) {
    let mut slots = 0u32;
    let mut dimms = Vec::new();
    let mut current: Option<DimmInfo> = None;
    let mut populated = false;

    let flush = |cur: &mut Option<DimmInfo>, populated: &mut bool, dimms: &mut Vec<DimmInfo>| {
        if let Some(d) = cur.take()
            && *populated
        {
            dimms.push(d);
        }
        *populated = false;
    };

    for line in text.lines() {
        if line.starts_with("Memory Device") {
            flush(&mut current, &mut populated, &mut dimms);
            slots += 1;
            current = Some(DimmInfo::default());
            continue;
        }
        let Some(d) = current.as_mut() else { continue };
        let Some((key, value)) = line.trim().split_once(':') else { continue };
        let value = value.trim();
        match key.trim() {
            "Size" => {
                if let Some(bytes) = parse_dmi_size(value) {
                    d.size_bytes = bytes;
                    populated = bytes > 0;
                }
            }
            "Locator" => d.locator = value.to_string(),
            "Type" => d.mem_type = value.to_string(),
            "Speed" => d.speed_mts = parse_dmi_speed(value),
            "Configured Memory Speed" | "Configured Clock Speed" => {
                d.configured_speed_mts = parse_dmi_speed(value)
            }
            "Manufacturer" => d.manufacturer = value.to_string(),
            "Part Number" => d.part_number = value.to_string(),
            _ => {}
        }
    }
    flush(&mut current, &mut populated, &mut dimms);
    (slots, dimms)
}

/// "16 GB" / "16384 MB" -> bytes; "No Module Installed" -> Some(0).
fn parse_dmi_size(value: &str) -> Option<u64> {
    if value.starts_with("No Module") || value == "Unknown" {
        return Some(0);
    }
    let (num, unit) = value.split_once(' ')?;
    let n: u64 = num.parse().ok()?;
    let mult = match unit {
        "GB" => 1 << 30,
        "MB" => 1 << 20,
        "TB" => 1u64 << 40,
        _ => return None,
    };
    Some(n * mult)
}

/// "4800 MT/s" / "2133 MHz" -> 4800 / 2133; "Unknown" -> None.
fn parse_dmi_speed(value: &str) -> Option<u32> {
    value.split_whitespace().next()?.parse().ok()
}

pub fn parse_loadavg(text: &str) -> Option<f32> {
    text.split_whitespace().next()?.parse().ok()
}

#[derive(Debug, Default, PartialEq)]
pub struct MemInfo {
    pub total: u64,
    pub available: u64,
    pub swap_total: u64,
    pub swap_free: u64,
}

/// Values in /proc/meminfo are kB; returned in bytes.
pub fn parse_meminfo(text: &str) -> MemInfo {
    let mut m = MemInfo::default();
    for line in text.lines() {
        let mut parts = line.split_whitespace();
        let key = parts.next().unwrap_or("");
        let kb: u64 = parts.next().and_then(|v| v.parse().ok()).unwrap_or(0);
        let bytes = kb * 1024;
        match key {
            "MemTotal:" => m.total = bytes,
            "MemAvailable:" => m.available = bytes,
            "SwapTotal:" => m.swap_total = bytes,
            "SwapFree:" => m.swap_free = bytes,
            _ => {}
        }
    }
    m
}

/// Sums sectors read/written across whole disks (not partitions, not
/// loop/ram devices) and returns bytes, assuming 512-byte sectors as
/// /proc/diskstats does regardless of the device's real sector size.
pub fn parse_diskstats(text: &str) -> (u64, u64) {
    let mut read = 0u64;
    let mut write = 0u64;
    for line in text.lines() {
        let f: Vec<&str> = line.split_whitespace().collect();
        if f.len() < 10 || !is_whole_disk(f[2]) {
            continue;
        }
        read += f[5].parse::<u64>().unwrap_or(0) * 512;
        write += f[9].parse::<u64>().unwrap_or(0) * 512;
    }
    (read, write)
}

/// True for device names like `sda`, `nvme0n1`, `vda`, `mmcblk0`, `md0`;
/// false for partitions (`sda1`, `nvme0n1p2`) and virtual devices.
pub fn is_whole_disk(name: &str) -> bool {
    let ends_with_digit = name.ends_with(|c: char| c.is_ascii_digit());
    if let Some(rest) = name.strip_prefix("nvme") {
        // nvme0n1 is a disk, nvme0n1p1 a partition
        return !rest.contains('p') && rest.contains('n');
    }
    if let Some(rest) = name.strip_prefix("mmcblk") {
        return !rest.contains('p');
    }
    if name.starts_with("md") {
        return !name.contains('p');
    }
    for prefix in ["sd", "vd", "xvd", "hd"] {
        if let Some(rest) = name.strip_prefix(prefix) {
            return !rest.is_empty()
                && rest.chars().all(|c| c.is_ascii_lowercase())
                && !ends_with_digit;
        }
    }
    false
}

pub struct DiskSpace {
    pub total: u64,
    pub free: u64,
    pub mount: String,
}

/// Free space via `df`, which keeps this dependency-free. Any failure (no
/// `df`, a BSD `df` without `--output`) yields None and the card says so.
fn disk_space(path: &Path) -> Option<DiskSpace> {
    let output = std::process::Command::new("df")
        .args(["-B1", "--output=size,avail,target"])
        .arg(path)
        .output()
        .ok()?;
    if !output.status.success() {
        return None;
    }
    parse_df_output(&String::from_utf8_lossy(&output.stdout))
}

pub fn parse_df_output(text: &str) -> Option<DiskSpace> {
    let line = text.lines().nth(1)?;
    let mut parts = line.split_whitespace();
    let total = parts.next()?.parse().ok()?;
    let free = parts.next()?.parse().ok()?;
    let mount = parts.collect::<Vec<_>>().join(" ");
    Some(DiskSpace { total, free, mount })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn proc_stat_idle_and_total() {
        let text = "cpu  100 5 50 800 20 0 5 0 0 0\ncpu0 50 2 25 400 10 0 2 0 0 0\n";
        assert_eq!(parse_proc_stat(text), Some((820, 980)));
    }

    #[test]
    fn proc_stat_missing_line() {
        assert_eq!(parse_proc_stat("intr 1 2 3\n"), None);
    }

    #[test]
    fn meminfo_in_bytes() {
        let text = concat!(
            "MemTotal:       32768 kB\n",
            "MemFree:        1000 kB\n",
            "MemAvailable:   16384 kB\n",
            "SwapTotal:      2048 kB\n",
            "SwapFree:       1024 kB\n",
        );
        assert_eq!(
            parse_meminfo(text),
            MemInfo {
                total: 32768 * 1024,
                available: 16384 * 1024,
                swap_total: 2048 * 1024,
                swap_free: 1024 * 1024,
            }
        );
    }

    #[test]
    fn diskstats_sums_whole_disks_only() {
        let text = "\
   8       0 sda 100 0 2000 0 50 0 1000 0 0 0 0 0 0 0 0 0 0
   8       1 sda1 100 0 1500 0 50 0 800 0 0 0 0 0 0 0 0 0 0
 259       0 nvme0n1 10 0 4000 0 5 0 3000 0 0 0 0 0 0 0 0 0 0
 259       1 nvme0n1p1 10 0 3000 0 5 0 2000 0 0 0 0 0 0 0 0 0 0
   7       0 loop0 1 0 64 0 0 0 0 0 0 0 0 0 0 0 0 0 0
";
        assert_eq!(
            parse_diskstats(text),
            ((2000 + 4000) * 512, (1000 + 3000) * 512)
        );
    }

    #[test]
    fn whole_disk_names() {
        for d in [
            "sda", "sdb", "vda", "xvda", "nvme0n1", "nvme1n2", "mmcblk0", "md0",
        ] {
            assert!(is_whole_disk(d), "{d} should be a whole disk");
        }
        for p in [
            "sda1",
            "nvme0n1p1",
            "mmcblk0p2",
            "loop0",
            "ram0",
            "dm-0",
            "sr0",
            "zram0",
        ] {
            assert!(!is_whole_disk(p), "{p} should not be a whole disk");
        }
    }

    #[test]
    fn per_core_stats_skip_aggregate_line() {
        let text = concat!(
            "cpu  100 5 50 800 20 0 5 0 0 0\n",
            "cpu0 50 2 25 400 10 0 2 0 0 0\n",
            "cpu1 50 3 25 400 10 0 3 0 0 0\n",
            "intr 1 2\n",
        );
        assert_eq!(parse_proc_stat_cores(text), vec![(410, 489), (410, 491)]);
    }

    #[test]
    fn busy_percent_from_deltas() {
        assert_eq!(busy_percent((100, 200), (150, 300)), Some(50.0));
        assert_eq!(busy_percent((100, 200), (100, 200)), None);
    }

    #[test]
    fn cpu_model_name() {
        let text = "processor\t: 0\nmodel name\t: AMD Ryzen 9 7950X 16-Core Processor\n";
        assert_eq!(parse_cpu_model(text).as_deref(), Some("AMD Ryzen 9 7950X 16-Core Processor"));
        assert_eq!(parse_cpu_model("flags: fpu"), None);
    }

    #[test]
    fn dmidecode_memory_devices() {
        let text = concat!(
            "Handle 0x0040, DMI type 17, 92 bytes\n",
            "Memory Device\n",
            "\tSize: 16 GB\n",
            "\tLocator: DIMM_A1\n",
            "\tType: DDR5\n",
            "\tSpeed: 5600 MT/s\n",
            "\tManufacturer: Kingston\n",
            "\tPart Number: KF556C36-16\n",
            "\tConfigured Memory Speed: 4800 MT/s\n",
            "\n",
            "Handle 0x0041, DMI type 17, 92 bytes\n",
            "Memory Device\n",
            "\tSize: No Module Installed\n",
            "\tLocator: DIMM_A2\n",
            "\tType: Unknown\n",
            "\tSpeed: Unknown\n",
        );
        let (slots, dimms) = parse_dmidecode_memory(text);
        assert_eq!(slots, 2);
        assert_eq!(
            dimms,
            vec![DimmInfo {
                locator: "DIMM_A1".into(),
                size_bytes: 16 << 30,
                mem_type: "DDR5".into(),
                speed_mts: Some(5600),
                configured_speed_mts: Some(4800),
                manufacturer: "Kingston".into(),
                part_number: "KF556C36-16".into(),
            }]
        );
    }

    #[test]
    fn dmi_sizes_and_speeds() {
        assert_eq!(parse_dmi_size("16384 MB"), Some(16 << 30));
        assert_eq!(parse_dmi_size("No Module Installed"), Some(0));
        assert_eq!(parse_dmi_size("weird"), None);
        assert_eq!(parse_dmi_speed("2133 MHz"), Some(2133));
        assert_eq!(parse_dmi_speed("Unknown"), None);
    }

    #[test]
    fn loadavg_first_field() {
        assert_eq!(parse_loadavg("0.52 0.58 0.59 1/1234 5678\n"), Some(0.52));
    }

    #[test]
    fn df_output_parsed() {
        let text = "1B-blocks        Avail Mounted on\n1000000000000 250000000000 /home\n";
        let d = parse_df_output(text).unwrap();
        assert_eq!(d.total, 1_000_000_000_000);
        assert_eq!(d.free, 250_000_000_000);
        assert_eq!(d.mount, "/home");
    }

    #[test]
    fn sampler_reports_cpu_only_after_two_samples() {
        // The first sample has no baseline, so cpu_percent stays None on
        // Linux; on other platforms it is None throughout. Either way this
        // must not panic.
        let mut s = SystemSampler::new();
        let first = s.sample(None);
        assert!(first.cpu_percent.is_none());
    }
}
