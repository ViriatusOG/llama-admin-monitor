pub mod hf;

use anyhow::Result;
use std::path::{Path, PathBuf};

#[derive(Debug, Clone, serde::Serialize)]
pub struct DiscoveredModel {
    pub path: PathBuf,
    pub filename: String,
    pub size_bytes: u64,
    pub size_display: String,
    pub quant_type: Option<String>,
    pub model_name: Option<String>,
    pub is_split: bool,
    pub hf_repo: Option<String>,
    pub downloaded_at: Option<u64>,
    pub hf_downloads: Option<u64>,
    pub hf_last_modified: Option<String>,
}

/// Scan a directory for .gguf model files.
/// For split models (e.g. -00001-of-00003.gguf), only the first shard is listed.
pub fn scan_models_dir(dir: &Path) -> Result<Vec<DiscoveredModel>> {
    let mut models = Vec::new();

    let entries = std::fs::read_dir(dir)
        .map_err(|e| anyhow::anyhow!("failed to read models directory '{}': {e}", dir.display()))?;

    for entry in entries.flatten() {
        let path = entry.path();
        if !path.is_file() {
            continue;
        }
        let filename = match path.file_name().and_then(|n| n.to_str()) {
            Some(n) => n.to_string(),
            None => continue,
        };
        if !filename.ends_with(".gguf") {
            continue;
        }

        // Skip split shards beyond the first
        let is_split = is_split_shard(&filename);
        if is_split && !is_first_shard(&filename) {
            continue;
        }

        let size_bytes = entry.metadata().map(|m| m.len()).unwrap_or(0);
        let (model_name, quant_type) = parse_gguf_filename(&filename);

        let meta = read_model_metadata(dir, &filename);

        models.push(DiscoveredModel {
            path: path.clone(),
            filename,
            size_bytes,
            size_display: format_size(size_bytes),
            quant_type,
            model_name,
            is_split,
            hf_repo: meta.as_ref().map(|m| m.repo.clone()),
            downloaded_at: meta.as_ref().map(|m| m.downloaded_at),
            hf_downloads: meta.as_ref().and_then(|m| m.hf_downloads),
            hf_last_modified: meta.as_ref().and_then(|m| m.hf_last_modified.clone()),
        });
    }

    models.sort_by(|a, b| a.filename.cmp(&b.filename));
    Ok(models)
}

/// Parse a GGUF filename to extract model name and quantization type.
/// Examples:
///   "Qwen3.5-27B-Q4_0.gguf" -> ("Qwen3.5-27B", "Q4_0")
///   "Qwen3-Coder-Next-Q4_1-00001-of-00003.gguf" -> ("Qwen3-Coder-Next", "Q4_1")
///   "model.gguf" -> ("model", None)
pub fn parse_gguf_filename(filename: &str) -> (Option<String>, Option<String>) {
    // Strip .gguf extension
    let stem = filename.strip_suffix(".gguf").unwrap_or(filename);

    // Strip split suffix like -00001-of-00003
    let stem = strip_split_suffix(stem);

    // Try to find a quant type pattern: Q followed by digits, underscores, and letters
    // Common patterns: Q4_0, Q4_1, Q8_0, Q4_K_M, Q4_K_XL, Q2_K_XL, UD-Q8_K_XL
    // Look for the last occurrence of a quant pattern
    let quant_patterns = [
        "-UD-Q", "-UD-IQ", "-Q", "-IQ", "_Q", "_IQ", // with separator
    ];

    for pattern in &quant_patterns {
        if let Some(pos) = stem.rfind(pattern) {
            let sep_len = pattern.len() - 1; // length of separator before Q/IQ
            let quant_start = pos + 1 + sep_len; // skip separator, include Q/IQ
            let model_name = &stem[..pos];
            let quant_str = if pattern.starts_with("-UD-") {
                // Include "UD-" prefix in quant type
                &stem[pos + 1..]
            } else {
                &stem[quant_start - pattern.trim_start_matches('-').trim_start_matches('_').len()..]
            };

            if !model_name.is_empty() && quant_str.len() >= 3 {
                return (Some(model_name.to_string()), Some(quant_str.to_string()));
            }
        }
    }

    // No quant type found
    if !stem.is_empty() {
        (Some(stem.to_string()), None)
    } else {
        (None, None)
    }
}

fn is_split_shard(filename: &str) -> bool {
    // Pattern: -NNNNN-of-NNNNN.gguf
    let stem = filename.strip_suffix(".gguf").unwrap_or(filename);
    if let Some(of_pos) = stem.rfind("-of-") {
        let after_of = &stem[of_pos + 4..];
        let before_of = &stem[..of_pos];
        if after_of.chars().all(|c| c.is_ascii_digit())
            && after_of.len() == 5
            && let Some(dash_pos) = before_of.rfind('-')
        {
            let shard_num = &before_of[dash_pos + 1..];
            return shard_num.chars().all(|c| c.is_ascii_digit()) && shard_num.len() == 5;
        }
    }
    false
}

fn is_first_shard(filename: &str) -> bool {
    filename.contains("-00001-of-")
}

fn strip_split_suffix(stem: &str) -> &str {
    // Remove -NNNNN-of-NNNNN from the end
    if let Some(of_pos) = stem.rfind("-of-") {
        let before_of = &stem[..of_pos];
        if let Some(dash_pos) = before_of.rfind('-') {
            let shard_num = &before_of[dash_pos + 1..];
            if shard_num.chars().all(|c| c.is_ascii_digit()) && shard_num.len() == 5 {
                return &stem[..dash_pos];
            }
        }
    }
    stem
}

fn read_model_metadata(dir: &Path, filename: &str) -> Option<crate::models::hf::ModelMetadata> {
    let meta_path = dir.join(format!("{filename}.meta.json"));
    let contents = std::fs::read_to_string(meta_path).ok()?;
    serde_json::from_str(&contents).ok()
}

fn format_size(bytes: u64) -> String {
    if bytes >= 1_073_741_824 {
        format!("{:.1} GB", bytes as f64 / 1_073_741_824.0)
    } else if bytes >= 1_048_576 {
        format!("{:.1} MB", bytes as f64 / 1_048_576.0)
    } else {
        format!("{} KB", bytes / 1024)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_simple_filename() {
        let (name, quant) = parse_gguf_filename("Qwen3.5-27B-Q4_0.gguf");
        assert_eq!(name.as_deref(), Some("Qwen3.5-27B"));
        assert_eq!(quant.as_deref(), Some("Q4_0"));
    }

    #[test]
    fn test_parse_split_filename() {
        let (name, quant) = parse_gguf_filename("Qwen3-Coder-Next-Q4_1-00001-of-00003.gguf");
        assert_eq!(name.as_deref(), Some("Qwen3-Coder-Next"));
        assert_eq!(quant.as_deref(), Some("Q4_1"));
    }

    #[test]
    fn test_parse_k_quant() {
        let (name, quant) = parse_gguf_filename("Devstral-Small-2-24B-Q4_K_M.gguf");
        assert_eq!(name.as_deref(), Some("Devstral-Small-2-24B"));
        assert_eq!(quant.as_deref(), Some("Q4_K_M"));
    }

    #[test]
    fn test_parse_ud_quant() {
        let (name, quant) = parse_gguf_filename("Qwen3.5-122B-A10B-UD-Q2_K_XL.gguf");
        assert_eq!(name.as_deref(), Some("Qwen3.5-122B-A10B"));
        assert_eq!(quant.as_deref(), Some("UD-Q2_K_XL"));
    }

    #[test]
    fn test_parse_iq_quant() {
        let (name, quant) = parse_gguf_filename("Gemma-4-E4B-Uncensored-HauhauCS-Aggressive-IQ3_M.gguf");
        assert_eq!(name.as_deref(), Some("Gemma-4-E4B-Uncensored-HauhauCS-Aggressive"));
        assert_eq!(quant.as_deref(), Some("IQ3_M"));
    }

    #[test]
    fn test_parse_no_quant() {
        let (name, quant) = parse_gguf_filename("model.gguf");
        assert_eq!(name.as_deref(), Some("model"));
        assert_eq!(quant, None);
    }

    #[test]
    fn test_is_split_shard() {
        assert!(is_split_shard("model-Q4_1-00001-of-00003.gguf"));
        assert!(is_split_shard("model-Q4_1-00002-of-00003.gguf"));
        assert!(!is_split_shard("model-Q4_1.gguf"));
    }

    #[test]
    fn test_is_first_shard() {
        assert!(is_first_shard("model-Q4_1-00001-of-00003.gguf"));
        assert!(!is_first_shard("model-Q4_1-00002-of-00003.gguf"));
    }

    #[test]
    fn test_scan_models_dir() {
        let dir = std::env::temp_dir().join("llama-admin-monitor-model-test");
        std::fs::create_dir_all(&dir).unwrap();

        // Create test files
        std::fs::write(dir.join("TestModel-Q4_0.gguf"), "fake model data").unwrap();
        std::fs::write(dir.join("Split-Q8_0-00001-of-00002.gguf"), "shard1").unwrap();
        std::fs::write(dir.join("Split-Q8_0-00002-of-00002.gguf"), "shard2").unwrap();
        std::fs::write(dir.join("readme.txt"), "not a model").unwrap();

        let models = scan_models_dir(&dir).unwrap();

        // Should find TestModel and first shard of Split, but not shard2 or readme
        assert_eq!(models.len(), 2);

        let names: Vec<&str> = models.iter().map(|m| m.filename.as_str()).collect();
        assert!(names.contains(&"TestModel-Q4_0.gguf"));
        assert!(names.contains(&"Split-Q8_0-00001-of-00002.gguf"));

        // Cleanup
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn test_format_size() {
        assert_eq!(format_size(1_500_000_000), "1.4 GB");
        assert_eq!(format_size(50_000_000), "47.7 MB");
        assert_eq!(format_size(500_000), "488 KB");
    }
}
