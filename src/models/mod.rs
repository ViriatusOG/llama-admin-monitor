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
    /// A multimodal projector (mmproj) rather than a language model.
    pub is_mmproj: bool,
    /// For a model: the projector file in the same directory that belongs to
    /// it, if one was found. For a projector: the models it belongs to.
    /// See `pair_projectors`.
    pub projector: Option<String>,
    pub pairs_with: Vec<String>,
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
        let is_mmproj = is_mmproj_filename(&filename);

        models.push(DiscoveredModel {
            path: path.clone(),
            filename,
            size_bytes,
            size_display: format_size(size_bytes),
            quant_type,
            model_name,
            is_split,
            is_mmproj,
            projector: None,
            pairs_with: Vec::new(),
            hf_repo: meta.as_ref().map(|m| m.repo.clone()),
            downloaded_at: meta.as_ref().map(|m| m.downloaded_at),
            hf_downloads: meta.as_ref().and_then(|m| m.hf_downloads),
            hf_last_modified: meta.as_ref().and_then(|m| m.hf_last_modified.clone()),
        });
    }

    models.sort_by(|a, b| a.filename.cmp(&b.filename));
    pair_projectors(&mut models);
    Ok(models)
}

/// Lower-cased model name with separators removed, so
/// "gemma-3-12b-it" and "Gemma_3_12B_IT" compare equal.
fn name_key(name: &str) -> String {
    name.to_ascii_lowercase()
        .chars()
        .filter(|c| c.is_ascii_alphanumeric())
        .collect()
}

/// Links projector files to the models they belong to. Two signals, in
/// order: both files were downloaded from the same Hugging Face repo (the
/// `.meta.json` sidecars record it), or the projector's filename contains
/// the model's name (`mmproj-gemma-3-12b-it-f16.gguf` for
/// `gemma-3-12b-it-Q6_K.gguf`). A model keeps the first projector that
/// matches by repo, else by name; a projector lists every model it fits.
pub fn pair_projectors(models: &mut [DiscoveredModel]) {
    let projectors: Vec<(usize, String, Option<String>)> = models
        .iter()
        .enumerate()
        .filter(|(_, m)| m.is_mmproj)
        .map(|(i, m)| (i, name_key(&m.filename), m.hf_repo.clone()))
        .collect();
    if projectors.is_empty() {
        return;
    }

    let mut links: Vec<(usize, usize)> = Vec::new(); // (model index, projector index)
    for (mi, model) in models.iter().enumerate() {
        if model.is_mmproj {
            continue;
        }
        let by_repo = model.hf_repo.as_ref().and_then(|repo| {
            projectors
                .iter()
                .find(|(_, _, prepo)| prepo.as_ref() == Some(repo))
                .map(|(pi, _, _)| *pi)
        });
        let by_name = model.model_name.as_ref().and_then(|name| {
            let key = name_key(name);
            if key.len() < 4 {
                return None;
            }
            projectors
                .iter()
                .find(|(_, pkey, _)| pkey.contains(&key))
                .map(|(pi, _, _)| *pi)
        });
        if let Some(pi) = by_repo.or(by_name) {
            links.push((mi, pi));
        }
    }

    for (mi, pi) in links {
        let projector_name = models[pi].filename.clone();
        let model_name = models[mi].filename.clone();
        models[mi].projector = Some(projector_name);
        models[pi].pairs_with.push(model_name);
    }
}

/// Vision projectors are conventionally named `mmproj-*.gguf` or
/// `*-mmproj-*.gguf`; they pair with a model via `--mmproj` and cannot be
/// launched on their own.
pub fn is_mmproj_filename(filename: &str) -> bool {
    filename.to_ascii_lowercase().contains("mmproj")
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
                &stem[quant_start
                    - pattern
                        .trim_start_matches('-')
                        .trim_start_matches('_')
                        .len()..]
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

    fn discovered(filename: &str, repo: Option<&str>) -> DiscoveredModel {
        let (model_name, quant_type) = parse_gguf_filename(filename);
        DiscoveredModel {
            path: PathBuf::from(filename),
            filename: filename.to_string(),
            size_bytes: 0,
            size_display: String::new(),
            quant_type,
            model_name,
            is_split: false,
            is_mmproj: is_mmproj_filename(filename),
            projector: None,
            pairs_with: Vec::new(),
            hf_repo: repo.map(str::to_string),
            downloaded_at: None,
            hf_downloads: None,
            hf_last_modified: None,
        }
    }

    #[test]
    fn projectors_pair_by_repo_then_by_name() {
        let mut models = vec![
            discovered(
                "gemma-3-12b-it-Q6_K.gguf",
                Some("unsloth/gemma-3-12b-it-GGUF"),
            ),
            discovered("mmproj-F16.gguf", Some("unsloth/gemma-3-12b-it-GGUF")),
            discovered("Qwen2.5-VL-7B-Instruct-Q4_K_M.gguf", None),
            discovered("mmproj-Qwen2.5-VL-7B-Instruct-f16.gguf", None),
            discovered("Llama-3.3-70B-Instruct-IQ3_M.gguf", None),
        ];
        pair_projectors(&mut models);
        assert_eq!(models[0].projector.as_deref(), Some("mmproj-F16.gguf"));
        assert_eq!(models[1].pairs_with, vec!["gemma-3-12b-it-Q6_K.gguf"]);
        assert_eq!(
            models[2].projector.as_deref(),
            Some("mmproj-Qwen2.5-VL-7B-Instruct-f16.gguf")
        );
        assert_eq!(
            models[3].pairs_with,
            vec!["Qwen2.5-VL-7B-Instruct-Q4_K_M.gguf"]
        );
        assert_eq!(models[4].projector, None);
    }

    #[test]
    fn test_is_mmproj_filename() {
        assert!(is_mmproj_filename("mmproj-model-f16.gguf"));
        assert!(is_mmproj_filename("gemma-3-12b-it-MMPROJ-BF16.gguf"));
        assert!(!is_mmproj_filename("gemma-3-12b-it-Q6_K.gguf"));
    }

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
        let (name, quant) =
            parse_gguf_filename("Gemma-4-E4B-Uncensored-HauhauCS-Aggressive-IQ3_M.gguf");
        assert_eq!(
            name.as_deref(),
            Some("Gemma-4-E4B-Uncensored-HauhauCS-Aggressive")
        );
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
