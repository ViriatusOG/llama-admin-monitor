pub mod amd_names;
pub mod dummy;
pub mod env;
pub mod nvidia;
pub mod rocm;

use anyhow::Result;
use std::collections::BTreeMap;
use std::sync::{Arc, Mutex};

#[derive(Debug, Clone, serde::Serialize)]
pub struct GpuMetrics {
    /// Headline temperature: junction (hotspot) when the card reports one,
    /// otherwise edge. The individual sensors follow when known.
    pub temp: f32,
    pub temp_edge: Option<f32>,
    pub temp_junction: Option<f32>,
    pub temp_memory: Option<f32>,
    pub load: u32,
    pub power_consumption: f32,
    pub power_limit: u32,
    pub vram_used: u64,
    pub vram_total: u64,
    pub sclk_mhz: u32,
    pub mclk_mhz: u32,
}

/// Returns a map key for `name` that is not yet present in `metrics`.
/// Two identical cards (a pair of RTX 4090s, say) share a marketing name, and
/// keying by name alone silently drops all but one of them. The first card
/// keeps the plain name; later duplicates get ` #2`, ` #3`, and so on.
pub fn unique_card_key(metrics: &BTreeMap<String, GpuMetrics>, name: &str) -> String {
    if !metrics.contains_key(name) {
        return name.to_string();
    }
    let mut n = 2;
    loop {
        let candidate = format!("{name} #{n}");
        if !metrics.contains_key(&candidate) {
            return candidate;
        }
        n += 1;
    }
}

pub trait GpuBackend: Send + Sync + 'static {
    fn read_metrics(&self) -> Result<BTreeMap<String, GpuMetrics>>;
    #[allow(dead_code)]
    fn name(&self) -> &str;
}

/// Polls several backends and merges their metrics into one map, so machines
/// with a mix of vendors (e.g. AMD + NVIDIA cards) report every GPU. A failure
/// in one backend is skipped without hiding the others, and logged only when
/// that backend changes state (working -> failing, failing -> working) so a
/// tool that is installed but has no card to talk to cannot flood the log.
pub struct MultiBackend {
    backends: Vec<Arc<dyn GpuBackend>>,
    failing: Mutex<Vec<bool>>,
}

impl MultiBackend {
    pub fn new(backends: Vec<Arc<dyn GpuBackend>>) -> Self {
        let failing = Mutex::new(vec![false; backends.len()]);
        Self { backends, failing }
    }
}

impl GpuBackend for MultiBackend {
    fn read_metrics(&self) -> Result<BTreeMap<String, GpuMetrics>> {
        let mut all = BTreeMap::new();
        let mut failing = self.failing.lock().unwrap();
        for (i, backend) in self.backends.iter().enumerate() {
            match backend.read_metrics() {
                Ok(metrics) => {
                    if failing[i] {
                        failing[i] = false;
                        crate::applog::info(format!("GPU metrics ({}) recovered", backend.name()));
                    }
                    all.extend(metrics);
                }
                Err(e) => {
                    if !failing[i] {
                        failing[i] = true;
                        crate::applog::error(format!(
                            "GPU metrics ({}): {e} (further failures are not logged until it recovers)",
                            backend.name()
                        ));
                    }
                }
            }
        }
        Ok(all)
    }

    fn name(&self) -> &str {
        "multi"
    }
}

pub fn detect_backend(force: &str) -> Arc<dyn GpuBackend> {
    match force {
        "rocm" => Arc::new(rocm::RocmBackend),
        "nvidia" => Arc::new(nvidia::NvidiaBackend),
        "none" => Arc::new(dummy::DummyBackend),
        // "auto" / "all" / anything else: monitor every vendor whose tool is
        // present AND actually reports a card. A leftover driver package
        // (nvidia-smi after the NVIDIA card was pulled, say) would otherwise
        // fail on every poll.
        _ => {
            let mut candidates: Vec<(&str, Arc<dyn GpuBackend>)> = Vec::new();
            if command_exists("rocm-smi") {
                candidates.push(("rocm-smi", Arc::new(rocm::RocmBackend)));
            }
            if command_exists("nvidia-smi") {
                candidates.push(("nvidia-smi", Arc::new(nvidia::NvidiaBackend)));
            }
            let mut backends: Vec<Arc<dyn GpuBackend>> = Vec::new();
            for (tool, backend) in candidates {
                match backend.read_metrics() {
                    Ok(m) if !m.is_empty() => {
                        crate::applog::info(format!(
                            "GPU monitoring via {tool}: {} card(s)",
                            m.len()
                        ));
                        backends.push(backend);
                    }
                    Ok(_) => crate::applog::warn(format!(
                        "{tool} is installed but reports no GPUs; not monitoring {}",
                        backend.name()
                    )),
                    Err(e) => crate::applog::warn(format!(
                        "{tool} is installed but not working ({e}); not monitoring {}",
                        backend.name()
                    )),
                }
            }
            match backends.len() {
                0 => {
                    crate::applog::warn(
                        "No working GPU monitoring tool found (rocm-smi / nvidia-smi)",
                    );
                    Arc::new(dummy::DummyBackend)
                }
                1 => backends.into_iter().next().unwrap(),
                _ => Arc::new(MultiBackend::new(backends)),
            }
        }
    }
}

fn command_exists(cmd: &str) -> bool {
    std::process::Command::new("which")
        .arg(cmd)
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .status()
        .is_ok_and(|s| s.success())
}

#[cfg(test)]
mod tests {
    use super::*;

    struct StubBackend {
        name: &'static str,
        cards: Vec<&'static str>,
        fail: bool,
    }

    impl GpuBackend for StubBackend {
        fn read_metrics(&self) -> Result<BTreeMap<String, GpuMetrics>> {
            if self.fail {
                anyhow::bail!("stub failure");
            }
            Ok(self
                .cards
                .iter()
                .map(|c| {
                    (
                        c.to_string(),
                        GpuMetrics {
                            temp: 0.0,
                            temp_edge: None,
                            temp_junction: None,
                            temp_memory: None,
                            load: 0,
                            power_consumption: 0.0,
                            power_limit: 0,
                            vram_used: 0,
                            vram_total: 0,
                            sclk_mhz: 0,
                            mclk_mhz: 0,
                        },
                    )
                })
                .collect())
        }

        fn name(&self) -> &str {
            self.name
        }
    }

    #[test]
    fn multi_backend_merges_all_vendors() {
        let multi = MultiBackend::new(vec![
            Arc::new(StubBackend {
                name: "rocm",
                cards: vec!["card0", "card1"],
                fail: false,
            }),
            Arc::new(StubBackend {
                name: "nvidia",
                cards: vec!["GPU0 NVIDIA"],
                fail: false,
            }),
        ]);
        let metrics = multi.read_metrics().unwrap();
        assert_eq!(metrics.len(), 3);
        assert!(metrics.contains_key("card0"));
        assert!(metrics.contains_key("card1"));
        assert!(metrics.contains_key("GPU0 NVIDIA"));
    }

    #[test]
    fn multi_backend_skips_failing_backend() {
        let multi = MultiBackend::new(vec![
            Arc::new(StubBackend {
                name: "rocm",
                cards: vec!["card0"],
                fail: true,
            }),
            Arc::new(StubBackend {
                name: "nvidia",
                cards: vec!["GPU0 NVIDIA"],
                fail: false,
            }),
        ]);
        let metrics = multi.read_metrics().unwrap();
        assert_eq!(metrics.len(), 1);
        assert!(metrics.contains_key("GPU0 NVIDIA"));
    }

    #[test]
    fn unique_card_key_suffixes_duplicates() {
        let mut metrics = BTreeMap::new();
        let zero = GpuMetrics {
            temp: 0.0,
            temp_edge: None,
            temp_junction: None,
            temp_memory: None,
            load: 0,
            power_consumption: 0.0,
            power_limit: 0,
            vram_used: 0,
            vram_total: 0,
            sclk_mhz: 0,
            mclk_mhz: 0,
        };
        assert_eq!(unique_card_key(&metrics, "RTX 4090"), "RTX 4090");
        metrics.insert("RTX 4090".to_string(), zero.clone());
        assert_eq!(unique_card_key(&metrics, "RTX 4090"), "RTX 4090 #2");
        metrics.insert("RTX 4090 #2".to_string(), zero);
        assert_eq!(unique_card_key(&metrics, "RTX 4090"), "RTX 4090 #3");
    }
}
