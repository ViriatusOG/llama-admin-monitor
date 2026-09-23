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

/// Fallback projector for when a preset's `--mmproj` file is no longer on
/// disk (renamed, re-downloaded, moved). Scans the directory the preset's
/// path points into, then the model's own directory, for a projector that
/// pairs with the model — the same signals as [`pair_projectors`] (both
/// files downloaded from the same HF repo, else the model's name in the
/// projector's filename), first match in filename order winning.
pub fn find_mmproj_for(model_path: &Path, stale_mmproj: &str) -> Option<PathBuf> {
    let model_filename = model_path
        .file_name()
        .and_then(|n| n.to_str())
        .unwrap_or("");

    let mut dirs: Vec<PathBuf> = Vec::new();
    for p in [stale_mmproj, model_path.to_str().unwrap_or("")] {
        if let Some(parent) = Path::new(p).parent().filter(|d| d.is_dir()) {
            let parent = parent.to_path_buf();
            if !dirs.contains(&parent) {
                dirs.push(parent);
            }
        }
    }

    for dir in dirs {
        let Ok(models) = scan_models_dir(&dir) else {
            continue;
        };
        let Some(model) = models.iter().find(|m| {
            !m.is_mmproj
                && (m.path == model_path
                    || (!model_filename.is_empty()
                        && m.path.file_name().and_then(|n| n.to_str()) == Some(model_filename)))
        }) else {
            continue;
        };

        // Same signal order as pair_projectors: HF repo, then name.
        if let Some(repo) = &model.hf_repo
            && let Some(c) = models
                .iter()
                .find(|c| c.is_mmproj && c.hf_repo.as_deref() == Some(repo))
        {
            return Some(c.path.clone());
        }
        if let Some(name) = &model.model_name {
            let key = name_key(name);
            if key.len() >= 4
                && let Some(c) = models
                    .iter()
                    .find(|c| c.is_mmproj && name_key(&c.filename).contains(&key))
            {
                return Some(c.path.clone());
            }
        }
    }
    None
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

/// True when the model's GGUF `tokenizer.chat_template` supports hybrid
/// thinking, i.e. it references the `enable_thinking` template variable (the
/// Qwen3-style switch). Reading only the file's metadata header, so a
/// multi-gigabyte model costs a couple of megabytes of reads. `None` means the
/// file is not a readable GGUF (missing, truncated, wrong magic) and the
/// caller should fall back to its default.
pub fn gguf_supports_thinking(path: &Path) -> Option<bool> {
    use std::io::Read;
    let file = std::fs::File::open(path).ok()?;
    let mut r = std::io::BufReader::with_capacity(1 << 20, file);
    // Magic "GGUF" + version.
    let mut head = [0u8; 8];
    r.read_exact(&mut head).ok()?;
    if &head[..4] != b"GGUF" {
        return None;
    }
    let version = u32::from_le_bytes([head[4], head[5], head[6], head[7]]);
    if version > 3 {
        return None;
    }
    // tensor count, kv count (u64).
    let mut buf8 = [0u8; 8];
    r.read_exact(&mut buf8).ok()?;
    let _tensors = u64::from_le_bytes(buf8);
    r.read_exact(&mut buf8).ok()?;
    let kv_count = u64::from_le_bytes(buf8);
    for _ in 0..kv_count {
        // Key: u64 length + bytes (+ NUL in v2).
        r.read_exact(&mut buf8).ok()?;
        let klen = u64::from_le_bytes(buf8);
        if klen > 1 << 20 {
            return None;
        }
        let mut key = vec![0u8; klen as usize];
        r.read_exact(&mut key).ok()?;
        if version == 2 {
            let mut nul = [0u8; 1];
            r.read_exact(&mut nul).ok()?;
        }
        let key = String::from_utf8_lossy(&key);
        // Value type (u32) then value. GGUF value types (ggml gguf.h):
        //   0 u8, 1 i8, 2 u16, 3 i16, 4 u32, 5 i32, 6 f32, 7 bool,
        //   8 string, 9 array, 10 u64, 11 i64, 12 f64.
        let mut buf4 = [0u8; 4];
        r.read_exact(&mut buf4).ok()?;
        let vtype = u32::from_le_bytes(buf4);
        let is_template = key == "tokenizer.chat_template";
        // Value sizes (ggml gguf.cpp GGUF_TYPE_SIZE): u8/i8/bool = 1 byte,
        // u16/i16 = 2, u32/i32/f32 = 4, u64/i64/f64 = 8.
        match vtype {
            0 | 1 | 7 => r.read_exact(&mut buf4[..1]).ok()?,
            2 | 3 => r.read_exact(&mut buf4[..2]).ok()?,
            4..=6 => r.read_exact(&mut buf4).ok()?,
            8 => {
                r.read_exact(&mut buf8).ok()?;
                let slen = u64::from_le_bytes(buf8);
                if slen > 1 << 22 {
                    return None;
                }
                if is_template {
                    let mut s = vec![0u8; slen as usize];
                    r.read_exact(&mut s).ok()?;
                    return Some(String::from_utf8_lossy(&s).contains("enable_thinking"));
                }
                // Not the key we want: skip the bytes without retaining them.
                let mut left = slen;
                while left > 0 {
                    let n = (left.min(1 << 20)) as usize;
                    let mut tmp = vec![0u8; n];
                    r.read_exact(&mut tmp).ok()?;
                    left -= n as u64;
                }
            }
            10..=12 => r.read_exact(&mut buf8).ok()?,
            // Array: u32 element type + u64 count, then that many elements.
            // Token tables can be hundreds of thousands of entries, so skip
            // element by element rather than allocating.
            9 => {
                r.read_exact(&mut buf4).ok()?;
                let etype = u32::from_le_bytes(buf4);
                r.read_exact(&mut buf8).ok()?;
                let count = u64::from_le_bytes(buf8);
                if count > 1 << 26 {
                    return None;
                }
                for _ in 0..count {
                    match skip_gguf_value(&mut r, etype, &mut buf8, &mut buf4) {
                        Ok(true) => {}
                        _ => return None,
                    }
                }
            }
            _ => return None, // unknown type: stop rather than desync
        }
    }
    // No chat template key at all: not a thinking model.
    Some(false)
}

/// Skip one GGUF value of the given type without retaining it.
/// Returns `Ok(false)` on an unknown element type.
fn skip_gguf_value(
    r: &mut std::io::BufReader<std::fs::File>,
    vtype: u32,
    buf8: &mut [u8; 8],
    buf4: &mut [u8; 4],
) -> std::io::Result<bool> {
    use std::io::Read;
    match vtype {
        0 | 1 | 7 => {
            r.read_exact(&mut buf4[..1])?;
        }
        2 | 3 => {
            r.read_exact(&mut buf4[..2])?;
        }
        4..=6 => {
            r.read_exact(buf4)?;
        }
        8 => {
            r.read_exact(buf8)?;
            let slen = u64::from_le_bytes(*buf8);
            if slen > 1 << 22 {
                return Ok(false);
            }
            let mut s = vec![0u8; slen as usize];
            r.read_exact(&mut s)?;
        }
        10..=12 => {
            r.read_exact(buf8)?;
        }
        _ => return Ok(false),
    }
    Ok(true)
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
    fn test_find_mmproj_by_name_when_preset_path_is_stale() {
        let dir = std::env::temp_dir().join("lam-mmproj-resolve-name");
        std::fs::create_dir_all(&dir).unwrap();
        let model = dir.join("gemma-3-12b-it-Q6_K.gguf");
        let actual = dir.join("mmproj-gemma-3-12b-it-F16.gguf");
        std::fs::write(&model, b"").unwrap();
        std::fs::write(&actual, b"").unwrap();

        let stale = dir.join("mmproj-F16.gguf");
        assert_eq!(
            find_mmproj_for(&model, &stale.to_string_lossy()),
            Some(actual.clone())
        );
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn test_find_mmproj_prefers_same_hf_repo_over_name() {
        let dir = std::env::temp_dir().join("lam-mmproj-resolve-repo");
        std::fs::create_dir_all(&dir).unwrap();
        let model = dir.join("gemma-3-12b-it-Q6_K.gguf");
        let repo_proj = dir.join("mmproj-F16.gguf");
        let name_proj = dir.join("mmproj-gemma-3-12b-it-BF16.gguf");
        for p in [&model, &repo_proj, &name_proj] {
            std::fs::write(p, b"").unwrap();
        }
        let meta = crate::models::hf::ModelMetadata {
            repo: "unsloth/gemma-3-12b-it-GGUF".into(),
            filename: "gemma-3-12b-it-Q6_K.gguf".into(),
            downloaded_at: 0,
            hf_downloads: None,
            hf_last_modified: None,
        };
        std::fs::write(
            dir.join("gemma-3-12b-it-Q6_K.gguf.meta.json"),
            serde_json::to_string(&meta).unwrap(),
        )
        .unwrap();
        std::fs::write(
            dir.join("mmproj-F16.gguf.meta.json"),
            serde_json::to_string(&meta).unwrap(),
        )
        .unwrap();

        // Only the BF16 file matches by name, but the repo sidecars say
        // mmproj-F16.gguf is the one that belongs to the model.
        let stale = dir.join("mmproj-renamed.gguf");
        assert_eq!(
            find_mmproj_for(&model, &stale.to_string_lossy()),
            Some(repo_proj.clone())
        );
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn test_find_mmproj_falls_back_to_model_directory() {
        let base = std::env::temp_dir().join("lam-mmproj-resolve-fallback");
        let stale_dir = base.join("old");
        let model_dir = base.join("models");
        std::fs::create_dir_all(&stale_dir).unwrap();
        std::fs::create_dir_all(&model_dir).unwrap();
        let model = model_dir.join("Qwen2.5-VL-7B-Instruct-Q4_K_M.gguf");
        let proj = model_dir.join("mmproj-Qwen2.5-VL-7B-Instruct-f16.gguf");
        std::fs::write(&model, b"").unwrap();
        std::fs::write(&proj, b"").unwrap();

        let stale = stale_dir.join("mmproj-F16.gguf");
        assert_eq!(
            find_mmproj_for(&model, &stale.to_string_lossy()),
            Some(proj.clone())
        );
        std::fs::remove_dir_all(&base).ok();
    }

    #[test]
    fn test_find_mmproj_no_match_yields_none() {
        let dir = std::env::temp_dir().join("lam-mmproj-resolve-none");
        std::fs::create_dir_all(&dir).unwrap();
        let model = dir.join("Llama-3.3-70B-Instruct-IQ3_M.gguf");
        let proj = dir.join("mmproj-gemma-3-12b-it-F16.gguf");
        std::fs::write(&model, b"").unwrap();
        std::fs::write(&proj, b"").unwrap();

        let stale = dir.join("mmproj-F16.gguf");
        assert_eq!(find_mmproj_for(&model, &stale.to_string_lossy()), None);
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn test_format_size() {
        assert_eq!(format_size(1_500_000_000), "1.4 GB");
        assert_eq!(format_size(50_000_000), "47.7 MB");
        assert_eq!(format_size(500_000), "488 KB");
    }

    /// A minimal but structurally valid GGUF v3 header: magic, version,
    /// zero tensors, and the given string key/value pairs.
    fn write_gguf_v3(path: &Path, kvs: &[(&str, &str)]) {
        let mut b = Vec::new();
        b.extend_from_slice(b"GGUF");
        b.extend_from_slice(&3u32.to_le_bytes());
        b.extend_from_slice(&0u64.to_le_bytes()); // tensor count
        b.extend_from_slice(&(kvs.len() as u64).to_le_bytes());
        for (k, v) in kvs {
            let kb = k.as_bytes();
            b.extend_from_slice(&(kb.len() as u64).to_le_bytes());
            b.extend_from_slice(kb);
            b.extend_from_slice(&8u32.to_le_bytes()); // string
            let vb = v.as_bytes();
            b.extend_from_slice(&(vb.len() as u64).to_le_bytes());
            b.extend_from_slice(vb);
        }
        std::fs::write(path, b).unwrap();
    }

    #[test]
    fn gguf_thinking_detected() {
        let dir = std::env::temp_dir().join(format!("lam-gguf-thinking-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("think.gguf");
        write_gguf_v3(
            &path,
            &[
                ("general.architecture", "qwen3"),
                (
                    "tokenizer.chat_template",
                    "{% set enable_thinking = true %}{{ '<|im_start|>user' }}",
                ),
            ],
        );
        assert_eq!(gguf_supports_thinking(&path), Some(true));
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn gguf_array_values_are_skipped() {
        // A token-table-like string array before the chat template: the
        // parser must walk past it and still find the template.
        let dir = std::env::temp_dir().join(format!("lam-gguf-arr-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("arr.gguf");
        let mut b = Vec::new();
        b.extend_from_slice(b"GGUF");
        b.extend_from_slice(&3u32.to_le_bytes());
        b.extend_from_slice(&0u64.to_le_bytes());
        b.extend_from_slice(&3u64.to_le_bytes()); // 3 kvs
        // kv 1: a u32 scalar
        let k = b"general.sampling.top_k";
        b.extend_from_slice(&(k.len() as u64).to_le_bytes());
        b.extend_from_slice(k);
        b.extend_from_slice(&4u32.to_le_bytes());
        b.extend_from_slice(&40u32.to_le_bytes());
        // kv 2: an array of two strings
        let k = b"tokenizer.ggml.tokens";
        b.extend_from_slice(&(k.len() as u64).to_le_bytes());
        b.extend_from_slice(k);
        b.extend_from_slice(&9u32.to_le_bytes()); // array
        b.extend_from_slice(&8u32.to_le_bytes()); // element type: string
        b.extend_from_slice(&2u64.to_le_bytes()); // count
        for tok in [&b"<s>"[..], &b"hello"[..]] {
            b.extend_from_slice(&(tok.len() as u64).to_le_bytes());
            b.extend_from_slice(tok);
        }
        // kv 3: the chat template, after the array
        let k = b"tokenizer.chat_template";
        b.extend_from_slice(&(k.len() as u64).to_le_bytes());
        b.extend_from_slice(k);
        b.extend_from_slice(&8u32.to_le_bytes());
        let tpl = b"{% if enable_thinking %}{% endif %}";
        b.extend_from_slice(&(tpl.len() as u64).to_le_bytes());
        b.extend_from_slice(tpl);
        std::fs::write(&path, b).unwrap();
        assert_eq!(gguf_supports_thinking(&path), Some(true));
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn gguf_non_thinking_detected() {
        let dir = std::env::temp_dir().join(format!("lam-gguf-notthink-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("plain.gguf");
        write_gguf_v3(
            &path,
            &[
                ("general.architecture", "llama"),
                (
                    "tokenizer.chat_template",
                    "{{ bos_token }}{% for m in messages %}{{ m.content }}",
                ),
            ],
        );
        assert_eq!(gguf_supports_thinking(&path), Some(false));
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn gguf_missing_template_is_not_thinking() {
        let dir = std::env::temp_dir().join(format!("lam-gguf-notpl-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("notpl.gguf");
        write_gguf_v3(&path, &[("general.architecture", "qwen3")]);
        assert_eq!(gguf_supports_thinking(&path), Some(false));
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn gguf_wrong_magic_is_none() {
        let dir = std::env::temp_dir().join(format!("lam-gguf-bad-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("bad.gguf");
        std::fs::write(&path, b"NOTGGUF\x00\x00\x00\x00\x00\x00\x00\x00").unwrap();
        assert_eq!(gguf_supports_thinking(&path), None);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn gguf_missing_file_is_none() {
        let path =
            std::env::temp_dir().join(format!("lam-gguf-none-{}-absent.gguf", std::process::id()));
        assert_eq!(gguf_supports_thinking(&path), None);
        let _ = std::fs::remove_file(&path);
    }
}
