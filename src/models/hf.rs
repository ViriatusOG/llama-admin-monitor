use anyhow::Result;
use futures_util::StreamExt;
use serde::{Deserialize, Serialize};
use std::path::PathBuf;
use std::sync::{Arc, Mutex};

const HF_API_BASE: &str = "https://huggingface.co";

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HfRepo {
    pub id: String,
    #[serde(default)]
    pub downloads: u64,
    #[serde(default)]
    pub likes: u64,
}

#[derive(Debug, Clone, Serialize)]
pub struct HfFile {
    pub filename: String,
    pub size_bytes: u64,
    pub size_display: String,
    pub is_mmproj: bool,
}

#[derive(Debug, Clone, Serialize, Default)]
pub struct DownloadProgress {
    pub repo: String,
    /// The file currently transferring.
    pub filename: String,
    pub downloaded_bytes: u64,
    pub total_bytes: u64,
    /// 1-based position of `filename` in the job, and the job's length,
    /// so the UI can show "2 / 2" when a companion projector follows.
    pub file_index: usize,
    pub file_count: usize,
    /// Final on-disk paths of the files completed so far.
    pub completed_paths: Vec<String>,
    pub done: bool,
    /// Unix seconds when `done` became true, so a finished job can be
    /// dropped after a short TTL and a page refresh cannot re-surface the
    /// completion toast.
    pub done_at: Option<u64>,
    pub error: Option<String>,
}

pub type SharedDownloadProgress = Arc<Mutex<Option<DownloadProgress>>>;

pub async fn search_hf_models(query: &str) -> Result<Vec<HfRepo>> {
    #[derive(Deserialize)]
    struct RawRepo {
        id: String,
        #[serde(default)]
        downloads: u64,
        #[serde(default)]
        likes: u64,
    }

    let url = format!(
        "{HF_API_BASE}/api/models?search={}&filter=gguf&limit=20&sort=downloads&direction=-1",
        percent_encode(query)
    );
    let client = reqwest::Client::new();
    let resp = client
        .get(&url)
        .header("User-Agent", "llama-admin-monitor")
        .send()
        .await?
        .error_for_status()?;
    let bytes = resp.bytes().await?;
    let raw: Vec<RawRepo> = serde_json::from_slice(&bytes)?;

    Ok(raw
        .into_iter()
        .map(|r| HfRepo {
            id: r.id,
            downloads: r.downloads,
            likes: r.likes,
        })
        .collect())
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct HfRepoInfo {
    pub downloads: u64,
    pub last_modified: Option<String>,
}

pub async fn get_hf_repo_info(repo_id: &str) -> Result<HfRepoInfo> {
    #[derive(Deserialize)]
    struct RawInfo {
        #[serde(default)]
        downloads: u64,
        #[serde(rename = "lastModified", default)]
        last_modified: Option<String>,
    }

    let url = format!("{HF_API_BASE}/api/models/{repo_id}");
    let client = reqwest::Client::new();
    let resp = client
        .get(&url)
        .header("User-Agent", "llama-admin-monitor")
        .send()
        .await?
        .error_for_status()?;
    let bytes = resp.bytes().await?;
    let raw: RawInfo = serde_json::from_slice(&bytes)?;

    Ok(HfRepoInfo {
        downloads: raw.downloads,
        last_modified: raw.last_modified,
    })
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelMetadata {
    pub repo: String,
    pub filename: String,
    pub downloaded_at: u64,
    pub hf_downloads: Option<u64>,
    pub hf_last_modified: Option<String>,
}

pub async fn list_hf_gguf_files(repo_id: &str) -> Result<Vec<HfFile>> {
    #[derive(Deserialize)]
    struct LfsInfo {
        #[serde(default)]
        size: u64,
    }

    #[derive(Deserialize)]
    struct TreeEntry {
        #[serde(rename = "type")]
        entry_type: String,
        path: String,
        #[serde(default)]
        size: u64,
        #[serde(default)]
        lfs: Option<LfsInfo>,
    }

    let url = format!("{HF_API_BASE}/api/models/{repo_id}/tree/main");
    let client = reqwest::Client::new();
    let resp = client
        .get(&url)
        .header("User-Agent", "llama-admin-monitor")
        .send()
        .await?
        .error_for_status()?;
    let bytes = resp.bytes().await?;
    let entries: Vec<TreeEntry> = serde_json::from_slice(&bytes)?;

    let mut files: Vec<HfFile> = entries
        .into_iter()
        .filter(|e| e.entry_type == "file" && e.path.ends_with(".gguf"))
        .map(|e| {
            let size = e
                .lfs
                .as_ref()
                .map(|l| l.size)
                .filter(|&s| s > 0)
                .unwrap_or(e.size);
            HfFile {
                is_mmproj: crate::models::is_mmproj_filename(&e.path),
                filename: e.path,
                size_bytes: size,
                size_display: format_size(size),
            }
        })
        .collect();

    files.sort_by(|a, b| a.filename.cmp(&b.filename));
    Ok(files)
}

/// Downloads `filenames` from `repo_id` one after another into `dest_dir`,
/// reporting through `progress`. A model and its companion mmproj travel as
/// one job so the UI shows a single download and the projector lands beside
/// its model. Stops at the first failure.
pub async fn download_hf_files(
    repo_id: String,
    filenames: Vec<String>,
    dest_dir: PathBuf,
    progress: SharedDownloadProgress,
) {
    let repo_info = get_hf_repo_info(&repo_id).await.unwrap_or_default();
    let file_count = filenames.len();

    {
        let mut p = progress.lock().unwrap();
        *p = Some(DownloadProgress {
            repo: repo_id.clone(),
            filename: filenames.first().cloned().unwrap_or_default(),
            file_index: 1,
            file_count,
            ..Default::default()
        });
    }

    let mut result: Result<()> = Ok(());
    for (i, filename) in filenames.iter().enumerate() {
        if let Some(p) = progress.lock().unwrap().as_mut() {
            p.filename = filename.clone();
            p.file_index = i + 1;
            p.downloaded_bytes = 0;
            p.total_bytes = 0;
        }
        match download_one(&repo_id, filename, &dest_dir, &progress, &repo_info).await {
            Ok(path) => {
                if let Some(p) = progress.lock().unwrap().as_mut() {
                    p.completed_paths.push(path.display().to_string());
                }
            }
            Err(e) => {
                result = Err(e);
                break;
            }
        }
    }

    if let Some(p) = progress.lock().unwrap().as_mut() {
        p.done = true;
        p.done_at = Some(now_unix_secs());
        if let Err(e) = result {
            p.error = Some(e.to_string());
        }
    }
}

/// Fetches one file to `dest_dir`, writing a `.meta.json` sidecar, and
/// returns the final path.
async fn download_one(
    repo_id: &str,
    filename: &str,
    dest_dir: &std::path::Path,
    progress: &SharedDownloadProgress,
    repo_info: &HfRepoInfo,
) -> Result<PathBuf> {
    use tokio::io::AsyncWriteExt;

    let url = format!("{HF_API_BASE}/{repo_id}/resolve/main/{filename}?download=true");
    let client = reqwest::Client::new();
    let resp = client
        .get(&url)
        .header("User-Agent", "llama-admin-monitor")
        .send()
        .await?
        .error_for_status()?;

    let total = resp.content_length().unwrap_or(0);
    if let Some(p) = progress.lock().unwrap().as_mut() {
        p.total_bytes = total;
    }

    // Use only the final path component on disk, in case the repo
    // nests gguf files under a subfolder.
    let out_name = std::path::Path::new(filename)
        .file_name()
        .map(|n| n.to_string_lossy().to_string())
        .unwrap_or_else(|| filename.to_string());
    let tmp_path = dest_dir.join(format!("{out_name}.part"));
    let final_path = dest_dir.join(&out_name);

    let mut file = tokio::fs::File::create(&tmp_path).await?;
    let mut stream = resp.bytes_stream();
    let mut downloaded: u64 = 0;

    while let Some(chunk) = stream.next().await {
        let chunk = chunk?;
        file.write_all(&chunk).await?;
        downloaded += chunk.len() as u64;
        if let Some(p) = progress.lock().unwrap().as_mut() {
            p.downloaded_bytes = downloaded;
        }
    }
    file.flush().await?;
    drop(file);

    tokio::fs::rename(&tmp_path, &final_path).await?;

    let meta = ModelMetadata {
        repo: repo_id.to_string(),
        filename: out_name.clone(),
        downloaded_at: now_unix_secs(),
        hf_downloads: Some(repo_info.downloads),
        hf_last_modified: repo_info.last_modified.clone(),
    };
    let meta_path = dest_dir.join(format!("{out_name}.meta.json"));
    if let Ok(json) = serde_json::to_string_pretty(&meta) {
        let _ = tokio::fs::write(&meta_path, json).await;
    }

    Ok(final_path)
}

pub(crate) fn now_unix_secs() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
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

fn percent_encode(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for b in s.bytes() {
        match b {
            b'A'..=b'Z' | b'a'..=b'z' | b'0'..=b'9' | b'-' | b'_' | b'.' | b'~' => {
                out.push(b as char)
            }
            b' ' => out.push_str("%20"),
            _ => out.push_str(&format!("%{:02X}", b)),
        }
    }
    out
}
