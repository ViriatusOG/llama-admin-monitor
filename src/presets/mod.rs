use anyhow::Result;
use std::path::Path;

#[derive(Debug, Clone, Default, serde::Serialize, serde::Deserialize)]
pub struct ModelPreset {
    #[serde(default = "next_id")]
    pub id: String,
    pub name: String,
    pub model_path: String,
    /// Multimodal projector for vision models (`--mmproj`); empty when unused.
    #[serde(default)]
    pub mmproj: String,
    pub context_size: u64,
    pub ctk: String,
    pub ctv: String,
    pub tensor_split: String,
    pub batch_size: u32,
    pub ubatch_size: u32,
    pub no_mmap: bool,
    pub ngram_spec: bool,
    pub parallel_slots: u32,
    // Model & memory
    #[serde(default)]
    pub gpu_layers: Option<i32>,
    #[serde(default)]
    pub mlock: bool,
    // Attention
    #[serde(default)]
    pub flash_attn: String,
    // GPU distribution
    /// Which llama-server build to launch: "" / "vulkan" for the configured
    /// binary, "cuda" for the legacy build-cuda tree, or "build:<id>" for
    /// a build from the Install page.
    #[serde(default)]
    pub backend: String,
    /// Comma-separated ggml device names to offload to (`--device`), e.g.
    /// `Vulkan1`; empty means every device.
    #[serde(default)]
    pub devices: String,
    #[serde(default)]
    pub split_mode: String,
    #[serde(default)]
    pub main_gpu: Option<u32>,
    // Threading
    #[serde(default)]
    pub threads: Option<u32>,
    #[serde(default)]
    pub threads_batch: Option<u32>,
    // Rope scaling (override auto-YaRN)
    #[serde(default)]
    pub rope_scaling: String,
    #[serde(default)]
    pub rope_freq_base: Option<f64>,
    #[serde(default)]
    pub rope_freq_scale: Option<f64>,
    // Speculative decoding (granular)
    #[serde(default)]
    pub draft_model: String,
    #[serde(default)]
    pub draft_min: Option<u32>,
    #[serde(default)]
    pub draft_max: Option<u32>,
    #[serde(default)]
    pub spec_ngram_size: Option<u32>,
    /// Speculative decoding type override (`--spec-type`); empty uses the
    /// ngram-spec checkbox (ngram-mod).
    #[serde(default)]
    pub spec_type: String,
    #[serde(default)]
    pub draft_ngl: Option<u32>,
    #[serde(default)]
    pub draft_device: String,
    #[serde(default)]
    pub draft_threads: Option<u32>,
    #[serde(default)]
    pub draft_threads_batch: Option<u32>,
    #[serde(default)]
    pub draft_p_min: Option<f64>,
    // Sampling (server-level defaults; requests may override per call)
    #[serde(default)]
    pub temperature: Option<f64>,
    #[serde(default)]
    pub top_p: Option<f64>,
    #[serde(default)]
    pub top_k: Option<u32>,
    #[serde(default)]
    pub min_p: Option<f64>,
    #[serde(default)]
    pub repeat_penalty: Option<f64>,
    #[serde(default)]
    pub presence_penalty: Option<f64>,
    #[serde(default)]
    pub frequency_penalty: Option<f64>,
    #[serde(default)]
    pub ignore_eos: bool,
    // Reasoning / thinking control
    #[serde(default)]
    pub reasoning: String,
    #[serde(default)]
    pub reasoning_effort: String,
    #[serde(default)]
    pub reasoning_preserve: bool,
    #[serde(default)]
    pub chat_template_kwargs: String,
    // Memory & KV cache
    #[serde(default)]
    pub fit: String,
    #[serde(default)]
    pub fit_target: String,
    #[serde(default)]
    pub kv_offload: bool,
    #[serde(default)]
    pub cram: Option<i32>,
    #[serde(default)]
    pub context_shift: bool,
    /// Skip the startup warmup pass (historical default: true).
    #[serde(default = "default_no_warmup")]
    pub no_warmup: bool,
    // Multimodal projector placement
    #[serde(default)]
    pub mmproj_device: String,
    #[serde(default)]
    pub mmproj_offload: bool,
    #[serde(default)]
    pub image_min_tokens: Option<u32>,
    #[serde(default)]
    pub image_max_tokens: Option<u32>,
    // Advanced
    #[serde(default)]
    pub seed: Option<i64>,
    #[serde(default)]
    pub system_prompt_file: String,
    #[serde(default)]
    pub extra_args: String,
}

fn default_no_warmup() -> bool {
    true
}

fn next_id() -> String {
    use std::time::{SystemTime, UNIX_EPOCH};
    let ts = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis();
    format!("p{ts}")
}

/// Load presets from disk, falling back to defaults if file doesn't exist.
pub fn load_presets(path: &Path) -> Vec<ModelPreset> {
    if path.exists() {
        match std::fs::read_to_string(path) {
            Ok(contents) => match serde_json::from_str::<Vec<ModelPreset>>(&contents) {
                Ok(presets) if !presets.is_empty() => return presets,
                Ok(_) => crate::applog::warn("Presets file is empty, using defaults"),
                Err(e) => crate::applog::warn(format!(
                    "Failed to parse presets file: {e}, using defaults"
                )),
            },
            Err(e) => {
                crate::applog::warn(format!("Failed to read presets file: {e}, using defaults"))
            }
        }
    }
    let presets = default_presets();
    // Try to save defaults to disk for future editing
    let _ = save_presets(path, &presets);
    presets
}

/// Save presets to disk atomically (write tmp, rename).
pub fn save_presets(path: &Path, presets: &[ModelPreset]) -> Result<()> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let tmp = path.with_extension("json.tmp");
    let json = serde_json::to_string_pretty(presets)?;
    std::fs::write(&tmp, json)?;
    std::fs::rename(&tmp, path)?;
    Ok(())
}

pub fn default_presets() -> Vec<ModelPreset> {
    vec![
        ModelPreset {
            id: "default-1".into(),
            name: "Example: Small Model 128K context".into(),
            model_path: String::new(),
            mmproj: String::new(),
            backend: String::new(),
            devices: String::new(),
            context_size: 128000,
            ctk: "f16".into(),
            ctv: "f16".into(),
            tensor_split: String::new(),
            batch_size: 2048,
            ubatch_size: 2048,
            no_mmap: false,
            ngram_spec: false,
            parallel_slots: 1,
            gpu_layers: None,
            mlock: false,
            flash_attn: String::new(),
            split_mode: String::new(),
            main_gpu: None,
            threads: None,
            threads_batch: None,
            rope_scaling: String::new(),
            rope_freq_base: None,
            rope_freq_scale: None,
            draft_model: String::new(),
            draft_min: None,
            draft_max: None,
            spec_ngram_size: None,
            seed: None,
            system_prompt_file: String::new(),
            extra_args: String::new(),
            no_warmup: true, // historical default; serde default on deserialization
            ..Default::default()
        },
        ModelPreset {
            id: "default-2".into(),
            name: "Example: Medium Model 256K turbo3 + ngram".into(),
            model_path: String::new(),
            mmproj: String::new(),
            backend: String::new(),
            devices: String::new(),
            context_size: 256000,
            ctk: "turbo3".into(),
            ctv: "turbo3".into(),
            tensor_split: String::new(),
            batch_size: 2048,
            ubatch_size: 2048,
            no_mmap: true,
            ngram_spec: true,
            parallel_slots: 1,
            gpu_layers: None,
            mlock: false,
            flash_attn: String::new(),
            split_mode: String::new(),
            main_gpu: None,
            threads: None,
            threads_batch: None,
            rope_scaling: String::new(),
            rope_freq_base: None,
            rope_freq_scale: None,
            draft_model: String::new(),
            draft_min: None,
            draft_max: None,
            spec_ngram_size: None,
            seed: None,
            system_prompt_file: String::new(),
            extra_args: String::new(),
            no_warmup: true, // historical default; serde default on deserialization
            ..Default::default()
        },
        ModelPreset {
            id: "default-3".into(),
            name: "Example: Large Model 512K YaRN multi-GPU".into(),
            model_path: String::new(),
            mmproj: String::new(),
            backend: String::new(),
            devices: String::new(),
            context_size: 524288,
            ctk: "turbo3".into(),
            ctv: "turbo3".into(),
            tensor_split: "7,8,8,8".into(),
            batch_size: 2048,
            ubatch_size: 2048,
            no_mmap: true,
            ngram_spec: false,
            parallel_slots: 1,
            gpu_layers: None,
            mlock: false,
            flash_attn: String::new(),
            split_mode: String::new(),
            main_gpu: None,
            threads: None,
            threads_batch: None,
            rope_scaling: String::new(),
            rope_freq_base: None,
            rope_freq_scale: None,
            draft_model: String::new(),
            draft_min: None,
            draft_max: None,
            spec_ngram_size: None,
            seed: None,
            system_prompt_file: String::new(),
            extra_args: String::new(),
            no_warmup: true, // historical default; serde default on deserialization
            ..Default::default()
        },
        ModelPreset {
            id: "default-4".into(),
            name: "Example: Max Context 1M YaRN".into(),
            model_path: String::new(),
            mmproj: String::new(),
            backend: String::new(),
            devices: String::new(),
            context_size: 1048576,
            ctk: "turbo3".into(),
            ctv: "turbo3".into(),
            tensor_split: String::new(),
            batch_size: 2048,
            ubatch_size: 2048,
            no_mmap: true,
            ngram_spec: false,
            parallel_slots: 1,
            gpu_layers: None,
            mlock: false,
            flash_attn: String::new(),
            split_mode: String::new(),
            main_gpu: None,
            threads: None,
            threads_batch: None,
            rope_scaling: String::new(),
            rope_freq_base: None,
            rope_freq_scale: None,
            draft_model: String::new(),
            draft_min: None,
            draft_max: None,
            spec_ngram_size: None,
            seed: None,
            system_prompt_file: String::new(),
            extra_args: String::new(),
            no_warmup: true, // historical default; serde default on deserialization
            ..Default::default()
        },
    ]
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    #[test]
    fn test_default_presets_not_empty() {
        let presets = default_presets();
        assert!(!presets.is_empty());
        assert_eq!(presets.len(), 4);
    }

    #[test]
    fn test_preset_serialization_roundtrip() {
        let presets = default_presets();
        let json = serde_json::to_string(&presets).unwrap();
        let deserialized: Vec<ModelPreset> = serde_json::from_str(&json).unwrap();
        assert_eq!(deserialized.len(), presets.len());
        assert_eq!(deserialized[0].name, presets[0].name);
        assert_eq!(deserialized[0].id, "default-1");
    }

    #[test]
    fn test_all_default_presets_have_ids() {
        let presets = default_presets();
        for (i, p) in presets.iter().enumerate() {
            assert_eq!(p.id, format!("default-{}", i + 1));
        }
    }

    #[test]
    fn test_load_save_roundtrip() {
        let dir = std::env::temp_dir().join("llama-admin-monitor-test");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("test-presets.json");

        let presets = default_presets();
        save_presets(&path, &presets).unwrap();

        let loaded = load_presets(&path);
        assert_eq!(loaded.len(), presets.len());
        assert_eq!(loaded[0].name, presets[0].name);

        std::fs::remove_file(&path).ok();
        std::fs::remove_dir(&dir).ok();
    }

    #[test]
    fn test_load_missing_file_returns_defaults() {
        let path = std::path::PathBuf::from("/tmp/nonexistent-llama-presets-12345.json");
        let loaded = load_presets(&path);
        assert_eq!(loaded.len(), 4);
        // Clean up the file that load_presets creates
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn test_load_corrupt_file_returns_defaults() {
        let path = std::env::temp_dir().join("corrupt-presets.json");
        let mut f = std::fs::File::create(&path).unwrap();
        f.write_all(b"not json").unwrap();
        let loaded = load_presets(&path);
        assert_eq!(loaded.len(), 4);
        std::fs::remove_file(&path).ok();
    }
}
