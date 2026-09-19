use anyhow::Result;
use std::collections::BTreeMap;
use std::process::Command;

use super::{GpuBackend, GpuMetrics, unique_card_key};

pub struct RocmBackend;

impl GpuBackend for RocmBackend {
    fn read_metrics(&self) -> Result<BTreeMap<String, GpuMetrics>> {
        let output = Command::new("rocm-smi")
            .args([
                "--json",
                "--showclocks",
                "--showtemp",
                "--showuse",
                "--showpower",
                "--showmaxpower",
                "--showproductname",
                "--showbus",
                "--showmeminfo",
                "vram",
            ])
            .output()
            .map_err(|e| anyhow::anyhow!("failed to run rocm-smi: {e}"))?;

        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            anyhow::bail!("rocm-smi failed: {stderr}");
        }

        let json: serde_json::Value = serde_json::from_slice(&output.stdout)
            .map_err(|e| anyhow::anyhow!("failed to parse rocm-smi JSON: {e}"))?;

        parse_rocm_json(&json)
    }

    fn name(&self) -> &str {
        "rocm"
    }
}

pub fn parse_rocm_json(json: &serde_json::Value) -> Result<BTreeMap<String, GpuMetrics>> {
    let mut metrics = BTreeMap::new();
    let object = json
        .as_object()
        .ok_or_else(|| anyhow::anyhow!("expected JSON object"))?;

    for (card_name, card) in object {
        let sensor = |key: &str| -> Option<f32> {
            card.get(key)
                .and_then(|v| v.as_str())
                .and_then(|t| t.parse::<f32>().ok())
        };
        let temp_edge = sensor("Temperature (Sensor edge) (C)");
        let temp_junction = sensor("Temperature (Sensor junction) (C)");
        let temp_memory = sensor("Temperature (Sensor memory) (C)");
        let temp = temp_junction.or(temp_edge).unwrap_or(0.0);

        let load = card
            .get("GPU use (%)")
            .and_then(|v| v.as_str())
            .and_then(|u| u.parse::<u32>().ok())
            .unwrap_or(0);

        let power_consumption = card
            .get("Current Socket Graphics Package Power (W)")
            .or_else(|| card.get("Average Graphics Package Power (W)"))
            .or_else(|| card.get("Average Package Power (W)"))
            .and_then(|v| v.as_str())
            .and_then(|p| p.parse::<f32>().ok())
            .unwrap_or(0.0);

        let power_limit = card
            .get("Max Graphics Package Power (W)")
            .or_else(|| card.get("Power Limit (W)"))
            .or_else(|| card.get("Power Limit (mW)"))
            .and_then(|v| v.as_str())
            .and_then(|p| p.parse::<f64>().ok())
            .map(|p| {
                if p > 1000.0 {
                    (p / 1000.0) as u32
                } else {
                    p as u32
                }
            })
            .unwrap_or(0);

        let vram_used = card
            .get("VRAM Total Used Memory (B)")
            .and_then(|v| v.as_str())
            .and_then(|v| v.parse::<u64>().ok())
            .map(|v| v / 1024 / 1024)
            .unwrap_or(0);

        let vram_total = card
            .get("VRAM Total Memory (B)")
            .and_then(|v| v.as_str())
            .and_then(|v| v.parse::<u64>().ok())
            .map(|v| v / 1024 / 1024)
            .unwrap_or(0);

        let parse_clock = |key: &str| -> u32 {
            card.get(key)
                .and_then(|v| v.as_str())
                .and_then(|s| {
                    let cleaned = s.replace("(", "").replace(")", "").replace("Mhz", "");
                    cleaned.trim().parse::<u32>().ok()
                })
                .unwrap_or(0)
        };

        let display_name = amd_display_name(card).unwrap_or_else(|| card_name.clone());
        let display_name = unique_card_key(&metrics, &display_name);
        let sclk_mhz = parse_clock("sclk clock speed:");
        let mclk_mhz = parse_clock("mclk clock speed:");

        metrics.insert(
            display_name,
            GpuMetrics {
                temp,
                temp_edge,
                temp_junction,
                temp_memory,
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

/// Marketing names for cards that `rocm-smi` (via libdrm's amdgpu.ids) does
/// not know yet and reports as a bare "AMD Radeon Graphics". Keyed by PCI
/// device id and revision; a `None` revision matches any.
const KNOWN_AMD_CARDS: &[(u32, Option<u32>, &str)] = &[
    (0x7550, Some(0xc0), "AMD Radeon RX 9070 XT"),
    (0x7550, Some(0xc3), "AMD Radeon RX 9070"),
    (0x7550, None, "AMD Radeon RX 9070 / 9070 XT"),
    (0x7551, None, "AMD Radeon AI PRO R9700"),
    (0x7590, Some(0xc0), "AMD Radeon RX 9060 XT"),
];

/// True for the placeholder names rocm-smi falls back to when the card is
/// missing from its id database.
fn is_generic_amd_name(name: &str) -> bool {
    let n = name.trim();
    n.is_empty()
        || n.eq_ignore_ascii_case("AMD Radeon Graphics")
        || n.eq_ignore_ascii_case("Radeon Graphics")
        || n.starts_with("Navi ")
        || n.starts_with("0x")
}

fn parse_hex(v: Option<&serde_json::Value>) -> Option<u32> {
    let s = v?.as_str()?.trim();
    let s = s
        .strip_prefix("0x")
        .or_else(|| s.strip_prefix("0X"))
        .unwrap_or(s);
    u32::from_str_radix(s, 16).ok()
}

fn read_sysfs_hex(bus: &str, file: &str) -> Option<u32> {
    let raw = std::fs::read_to_string(format!("/sys/bus/pci/devices/{bus}/{file}")).ok()?;
    parse_hex(Some(&serde_json::Value::String(raw.trim().to_string())))
}

fn known_amd_name(device: u32, rev: Option<u32>) -> Option<&'static str> {
    KNOWN_AMD_CARDS
        .iter()
        .find(|(d, r, _)| *d == device && r.is_some() && *r == rev)
        .or_else(|| {
            KNOWN_AMD_CARDS
                .iter()
                .find(|(d, r, _)| *d == device && r.is_none())
        })
        .map(|(_, _, name)| *name)
}

/// `lspci -s <bus>` prints e.g.
/// `03:00.0 VGA compatible controller: Advanced Micro Devices, Inc. [AMD/ATI] Navi 48 [Radeon RX 9070/9070 XT] (rev c0)`;
/// the last bracketed group is the marketing name, when the system's pci.ids
/// knows the card at all.
fn lspci_name(bus: &str) -> Option<String> {
    let out = Command::new("lspci").args(["-s", bus]).output().ok()?;
    if !out.status.success() {
        return None;
    }
    let line = String::from_utf8_lossy(&out.stdout);
    parse_lspci_name(&line)
}

fn parse_lspci_name(line: &str) -> Option<String> {
    let (_, rest) = line.split_once(": ")?;
    let rest = rest.split(" (rev ").next()?;
    let start = rest.rfind('[')?;
    let end = rest[start..].find(']')? + start;
    let name = rest[start + 1..end].trim();
    if name.is_empty() || name.eq_ignore_ascii_case("AMD/ATI") || name.starts_with("Device ") {
        return None;
    }
    if name.starts_with("AMD ") {
        Some(name.to_string())
    } else {
        Some(format!("AMD {name}"))
    }
}

/// The best name available for one rocm-smi card entry: its "Card Series"
/// when that is a real product name, otherwise the device id looked up in
/// the built-in table, then the system's pci.ids via lspci, and finally the
/// generic name tagged with the gfx target and device id so two unknown
/// cards are at least distinguishable. `None` when rocm-smi gave no name.
fn amd_display_name(card: &serde_json::Value) -> Option<String> {
    let series = card
        .get("Card Series")
        .and_then(|v| v.as_str())
        .map(str::trim)
        .unwrap_or("");
    if !is_generic_amd_name(series) {
        return Some(series.to_string());
    }
    let bus = card.get("PCI Bus").and_then(|v| v.as_str()).map(str::trim);
    let device =
        parse_hex(card.get("Card Model")).or_else(|| bus.and_then(|b| read_sysfs_hex(b, "device")));
    let rev = parse_hex(card.get("Device Rev"))
        .or_else(|| bus.and_then(|b| read_sysfs_hex(b, "revision")));
    if let Some(name) = device.and_then(|d| known_amd_name(d, rev)) {
        return Some(name.to_string());
    }
    if let Some(name) = bus.and_then(lspci_name) {
        return Some(name);
    }
    if series.is_empty() {
        // Nothing to go on: let the caller use rocm-smi's own key (card0...).
        return None;
    }
    let base = series;
    let gfx = card
        .get("GFX Version")
        .and_then(|v| v.as_str())
        .unwrap_or("");
    match (gfx.is_empty(), device) {
        (false, Some(d)) => Some(format!("{base} ({gfx}, 0x{d:04x})")),
        (false, None) => Some(format!("{base} ({gfx})")),
        (true, Some(d)) => Some(format!("{base} (0x{d:04x})")),
        (true, None) => Some(base.to_string()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_rocm_json() {
        let json: serde_json::Value =
            serde_json::from_str(include_str!("../../tests/fixtures/rocm_smi_output.json"))
                .unwrap();

        let metrics = parse_rocm_json(&json).unwrap();
        assert_eq!(metrics.len(), 1);

        let card = metrics.get("card0").unwrap();
        assert!((card.temp - 45.0).abs() < 0.1);
        assert_eq!(card.load, 87);
        assert!((card.power_consumption - 180.5).abs() < 0.1);
        assert_eq!(card.power_limit, 300);
        assert_eq!(card.vram_used, 15360); // 16106127360 / 1024 / 1024
        assert_eq!(card.vram_total, 16384); // 17179869184 / 1024 / 1024
        assert_eq!(card.sclk_mhz, 1725);
        assert_eq!(card.mclk_mhz, 1200);
    }

    #[test]
    fn test_parse_rocm_json_empty() {
        let json: serde_json::Value = serde_json::from_str("{}").unwrap();
        let metrics = parse_rocm_json(&json).unwrap();
        assert!(metrics.is_empty());
    }

    #[test]
    fn test_parse_rocm_json_missing_fields() {
        let json: serde_json::Value =
            serde_json::from_str(r#"{"card0": {"GPU use (%)": "50"}}"#).unwrap();
        let metrics = parse_rocm_json(&json).unwrap();
        let card = metrics.get("card0").unwrap();
        assert_eq!(card.load, 50);
        assert_eq!(card.temp, 0.0);
        assert_eq!(card.power_consumption, 0.0);
    }

    #[test]
    fn test_parse_rocm_json_all_sensors() {
        let json: serde_json::Value = serde_json::from_str(
            r#"{"card0": {
                "Temperature (Sensor edge) (C)": "48.0",
                "Temperature (Sensor junction) (C)": "61.0",
                "Temperature (Sensor memory) (C)": "70.0"
            }}"#,
        )
        .unwrap();
        let card = parse_rocm_json(&json).unwrap().remove("card0").unwrap();
        assert_eq!(card.temp, 61.0);
        assert_eq!(card.temp_edge, Some(48.0));
        assert_eq!(card.temp_junction, Some(61.0));
        assert_eq!(card.temp_memory, Some(70.0));
    }

    #[test]
    fn generic_card_series_is_resolved_from_device_id() {
        let card: serde_json::Value = serde_json::from_str(
            r#"{"Card Series": "AMD Radeon Graphics", "Card Model": "0x7551",
                "Device Rev": "0xc0", "GFX Version": "gfx1201"}"#,
        )
        .unwrap();
        assert_eq!(
            amd_display_name(&card).as_deref(),
            Some("AMD Radeon AI PRO R9700")
        );

        let card: serde_json::Value = serde_json::from_str(
            r#"{"Card Series": "AMD Radeon Graphics", "Card Model": "0x7550", "Device Rev": "0xc3"}"#,
        )
        .unwrap();
        assert_eq!(
            amd_display_name(&card).as_deref(),
            Some("AMD Radeon RX 9070")
        );

        let card: serde_json::Value = serde_json::from_str(
            r#"{"Card Series": "AMD Radeon Graphics", "Card Model": "0x7550"}"#,
        )
        .unwrap();
        assert_eq!(
            amd_display_name(&card).as_deref(),
            Some("AMD Radeon RX 9070 / 9070 XT")
        );
    }

    #[test]
    fn real_card_series_is_kept() {
        let card: serde_json::Value = serde_json::from_str(
            r#"{"Card Series": "AMD Radeon RX 9070 XT", "Card Model": "0x7551"}"#,
        )
        .unwrap();
        assert_eq!(
            amd_display_name(&card).as_deref(),
            Some("AMD Radeon RX 9070 XT")
        );
    }

    #[test]
    fn unknown_generic_card_gets_gfx_and_device_id() {
        let card: serde_json::Value = serde_json::from_str(
            r#"{"Card Series": "AMD Radeon Graphics", "Card Model": "0x7fff", "GFX Version": "gfx1250"}"#,
        )
        .unwrap();
        assert_eq!(
            amd_display_name(&card).as_deref(),
            Some("AMD Radeon Graphics (gfx1250, 0x7fff)")
        );
    }

    #[test]
    fn lspci_line_yields_bracketed_name() {
        let line = "03:00.0 VGA compatible controller: Advanced Micro Devices, Inc. [AMD/ATI] Navi 48 [Radeon RX 9070/9070 XT] (rev c0)";
        assert_eq!(
            parse_lspci_name(line).as_deref(),
            Some("AMD Radeon RX 9070/9070 XT")
        );
        let unknown = "03:00.0 VGA compatible controller: Advanced Micro Devices, Inc. [AMD/ATI] Device 7551 (rev c0)";
        assert_eq!(parse_lspci_name(unknown), None);
    }
}
