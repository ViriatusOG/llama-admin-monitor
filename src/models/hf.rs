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
}

#[derive(Debug, Clone, Serialize, Default)]
pub struct DownloadProgress {
    pub repo: String,
    pub filename: String,
    pub downloaded_bytes: u64,
    pub total_bytes: u64,
    pub done: bool,
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
                filename: e.path,
                size_bytes: size,
                size_display: format_size(size),
            }
        })
        .collect();

    files.sort_by(|a, b| a.filename.cmp(&b.filename));
    Ok(files)
}

pub async fn download_hf_file(
    repo_id: String,
    filename: String,
    dest_dir: PathBuf,
    progress: SharedDownloadProgress,
) {
    use tokio::io::AsyncWriteExt;

    let repo_info = get_hf_repo_info(&repo_id).await.unwrap_or_default();

    {
        let mut p = progress.lock().unwrap();
        *p = Some(DownloadProgress {
            repo: repo_id.clone(),
            filename: filename.clone(),
            downloaded_bytes: 0,
            total_bytes: 0,
            done: false,
            error: None,
        });
    }

    let result: Result<()> = async {
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
        let out_name = std::path::Path::new(&filename)
            .file_name()
            .map(|n| n.to_string_lossy().to_string())
            .unwrap_or_else(|| filename.clone());
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
            repo: repo_id.clone(),
            filename: out_name.clone(),
            downloaded_at: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_secs())
                .unwrap_or(0),
            hf_downloads: Some(repo_info.downloads),
            hf_last_modified: repo_info.last_modified.clone(),
        };
        let meta_path = dest_dir.join(format!("{out_name}.meta.json"));
        if let Ok(json) = serde_json::to_string_pretty(&meta) {
            let _ = tokio::fs::write(&meta_path, json).await;
        }

        Ok(())
    }
    .await;

    if let Some(p) = progress.lock().unwrap().as_mut() {
        p.done = true;
        if let Err(e) = result {
            p.error = Some(e.to_string());
        }
    }
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
