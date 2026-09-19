//! Product names for AMD cards that the ROCm tools only know generically.
//!
//! `rocm-smi` and `rocminfo` take their marketing names from libdrm's
//! `amdgpu.ids`, which lags new hardware: an unknown card shows up as a bare
//! "AMD Radeon Graphics". These helpers resolve such names from the PCI
//! device id (built-in table, then the system's `pci.ids` via `lspci`) or
//! at least tag the generic name so two unknown cards can be told apart.

use std::process::Command;

/// Keyed by PCI device id and revision; a `None` revision matches any.
const KNOWN_AMD_CARDS: &[(u32, Option<u32>, &str)] = &[
    (0x7550, Some(0xc0), "AMD Radeon RX 9070 XT"),
    (0x7550, Some(0xc3), "AMD Radeon RX 9070"),
    (0x7550, None, "AMD Radeon RX 9070 / 9070 XT"),
    (0x7551, None, "AMD Radeon AI PRO R9700"),
    (0x7590, Some(0xc0), "AMD Radeon RX 9060 XT"),
];

/// True for the placeholder names the ROCm tools fall back to when the
/// card is missing from their id database.
pub fn is_generic_amd_name(name: &str) -> bool {
    let n = name.trim();
    n.is_empty()
        || n.eq_ignore_ascii_case("AMD Radeon Graphics")
        || n.eq_ignore_ascii_case("Radeon Graphics")
        || n.starts_with("Navi ")
        || n.starts_with("0x")
}

/// Parses "0x7551", "7551" or rocminfo's "30033(0x7551)".
pub fn parse_hex_id(s: &str) -> Option<u32> {
    let s = s.trim();
    let s = match (s.find('('), s.rfind(')')) {
        (Some(a), Some(b)) if a < b => &s[a + 1..b],
        _ => s,
    };
    let s = s
        .strip_prefix("0x")
        .or_else(|| s.strip_prefix("0X"))
        .unwrap_or(s);
    u32::from_str_radix(s, 16).ok()
}

pub fn read_sysfs_hex(bus: &str, file: &str) -> Option<u32> {
    let raw = std::fs::read_to_string(format!("/sys/bus/pci/devices/{bus}/{file}")).ok()?;
    parse_hex_id(&raw)
}

pub fn known_amd_name(device: u32, rev: Option<u32>) -> Option<&'static str> {
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
pub fn lspci_name(bus: &str) -> Option<String> {
    let out = Command::new("lspci").args(["-s", bus]).output().ok()?;
    if !out.status.success() {
        return None;
    }
    let line = String::from_utf8_lossy(&out.stdout);
    parse_lspci_name(&line)
}

pub fn parse_lspci_name(line: &str) -> Option<String> {
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

/// The generic name tagged with whatever identifies the card, e.g.
/// "AMD Radeon Graphics (gfx1201, 0x7551)".
pub fn tagged_generic(base: &str, gfx: &str, device: Option<u32>) -> String {
    match (gfx.is_empty(), device) {
        (false, Some(d)) => format!("{base} ({gfx}, 0x{d:04x})"),
        (false, None) => format!("{base} ({gfx})"),
        (true, Some(d)) => format!("{base} (0x{d:04x})"),
        (true, None) => base.to_string(),
    }
}

/// Resolves a marketing name reported by a ROCm tool. Real names pass
/// through; generic ones are looked up by device id (and PCI bus, when
/// known), and failing that tagged with the gfx target and id.
pub fn resolve_amd_name(
    marketing: &str,
    gfx: &str,
    device: Option<u32>,
    rev: Option<u32>,
    bus: Option<&str>,
) -> String {
    let marketing = marketing.trim();
    if !is_generic_amd_name(marketing) {
        return marketing.to_string();
    }
    let device = device.or_else(|| bus.and_then(|b| read_sysfs_hex(b, "device")));
    let rev = rev.or_else(|| bus.and_then(|b| read_sysfs_hex(b, "revision")));
    if let Some(name) = device.and_then(|d| known_amd_name(d, rev)) {
        return name.to_string();
    }
    if let Some(name) = bus.and_then(lspci_name) {
        return name;
    }
    let base = if marketing.is_empty() {
        "AMD Radeon Graphics"
    } else {
        marketing
    };
    tagged_generic(base, gfx, device)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hex_ids_in_every_spelling() {
        assert_eq!(parse_hex_id("0x7551"), Some(0x7551));
        assert_eq!(parse_hex_id("7551"), Some(0x7551));
        assert_eq!(parse_hex_id("30033(0x7551)"), Some(0x7551));
        assert_eq!(parse_hex_id("0xc0\n"), Some(0xc0));
        assert_eq!(parse_hex_id("zz"), None);
    }

    #[test]
    fn generic_names_resolve_from_device_id() {
        assert_eq!(
            resolve_amd_name(
                "AMD Radeon Graphics",
                "gfx1201",
                Some(0x7551),
                Some(0xc0),
                None
            ),
            "AMD Radeon AI PRO R9700"
        );
        assert_eq!(
            resolve_amd_name("AMD Radeon Graphics", "", Some(0x7550), Some(0xc3), None),
            "AMD Radeon RX 9070"
        );
        assert_eq!(
            resolve_amd_name("AMD Radeon Graphics", "", Some(0x7550), None, None),
            "AMD Radeon RX 9070 / 9070 XT"
        );
    }

    #[test]
    fn real_names_are_kept() {
        assert_eq!(
            resolve_amd_name("AMD Radeon RX 9070 XT", "gfx1201", Some(0x7551), None, None),
            "AMD Radeon RX 9070 XT"
        );
    }

    #[test]
    fn unknown_generic_card_gets_gfx_and_device_id() {
        assert_eq!(
            resolve_amd_name("AMD Radeon Graphics", "gfx1250", Some(0x7fff), None, None),
            "AMD Radeon Graphics (gfx1250, 0x7fff)"
        );
        assert_eq!(
            resolve_amd_name("", "gfx1250", None, None, None),
            "AMD Radeon Graphics (gfx1250)"
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
