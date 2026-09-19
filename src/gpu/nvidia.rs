use anyhow::Result;
use std::collections::BTreeMap;

use super::{GpuBackend, GpuMetrics, unique_card_key};

pub struct NvidiaBackend;

impl GpuBackend for NvidiaBackend {
    fn read_metrics(&self) -> Result<BTreeMap<String, GpuMetrics>> {
        let output = std::process::Command::new("nvidia-smi")
            .args([
                "--query-gpu=index,name,temperature.gpu,utilization.gpu,power.draw,power.limit,memory.used,memory.total,clocks.gr,clocks.mem",
                "--format=csv,noheader,nounits",
            ])
            .output()
            .map_err(|e| anyhow::anyhow!("failed to run nvidia-smi: {e}"))?;

        if !output.status.success() {
            // nvidia-smi prints "couldn't communicate with the NVIDIA driver"
            // and friends on stdout, so fall back to it when stderr is empty.
            let stderr = String::from_utf8_lossy(&output.stderr);
            let stdout = String::from_utf8_lossy(&output.stdout);
            let msg = if stderr.trim().is_empty() { stdout } else { stderr };
            anyhow::bail!("nvidia-smi failed: {}", msg.trim());
        }

        let stdout = String::from_utf8_lossy(&output.stdout);
        parse_nvidia_csv(&stdout)
    }

    fn name(&self) -> &str {
        "nvidia"
    }
}

pub fn parse_nvidia_csv(csv: &str) -> Result<BTreeMap<String, GpuMetrics>> {
    let mut metrics = BTreeMap::new();

    for line in csv.lines() {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }

        let fields: Vec<&str> = line.split(',').map(|s| s.trim()).collect();
        if fields.len() < 10 {
            continue;
        }

        let _index = fields[0];
        let name = fields[1];
        let temp = fields[2].parse::<f32>().unwrap_or(0.0);
        let load = fields[3].parse::<u32>().unwrap_or(0);
        let power_consumption = fields[4].parse::<f32>().unwrap_or(0.0);
        let power_limit = fields[5].parse::<f64>().unwrap_or(0.0) as u32;
        let vram_used = fields[6].parse::<u64>().unwrap_or(0); // MiB from nvidia-smi
        let vram_total = fields[7].parse::<u64>().unwrap_or(0); // MiB from nvidia-smi
        let sclk_mhz = fields[8].parse::<u32>().unwrap_or(0);
        let mclk_mhz = fields[9].parse::<u32>().unwrap_or(0);

        let card_name = unique_card_key(&metrics, name);
        metrics.insert(
            card_name,
            GpuMetrics {
                temp,
                load,
                power_consumption,
                power_limit,
                vram_used,
                vram_total,
                sclk_mhz,
                mclk_mhz,
            },
        );
    }

    Ok(metrics)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_nvidia_csv() {
        let csv = include_str!("../../tests/fixtures/nvidia_smi_csv.txt");
        let metrics = parse_nvidia_csv(csv).unwrap();
        assert_eq!(metrics.len(), 2);

        // Two identical cards must both survive: the second gets a suffix.
        let gpu1 = metrics.get("NVIDIA GeForce RTX 4090 #2").unwrap();
        assert_eq!(gpu1.load, 75);
        assert_eq!(gpu1.vram_used, 16384);

        let gpu0 = metrics.get("NVIDIA GeForce RTX 4090").unwrap();
        assert!((gpu0.temp - 45.0).abs() < 0.1);
        assert_eq!(gpu0.load, 87);
        assert!((gpu0.power_consumption - 320.5).abs() < 0.1);
        assert_eq!(gpu0.power_limit, 450);
        assert_eq!(gpu0.vram_used, 20480);
        assert_eq!(gpu0.vram_total, 24564);
        assert_eq!(gpu0.sclk_mhz, 2520);
        assert_eq!(gpu0.mclk_mhz, 10501);
    }

    #[test]
    fn test_parse_nvidia_csv_empty() {
        let metrics = parse_nvidia_csv("").unwrap();
        assert!(metrics.is_empty());
    }
}
