//! Prebuilt llama.cpp installs from ggml-org's GitHub releases, so nobody
//! has to compile. Every release ships one tarball per backend; each is
//! extracted into its own directory under the data dir and recorded with
//! a small metadata file. Presets pick a build with `backend = "build:<id>"`.

use anyhow::{Context, Result, bail};
use futures_util::StreamExt;
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};

use crate::state::AppState;

const REPO: &str = "ggml-org/llama.cpp";
const USER_AGENT: &str = "llama-admin-monitor";
const META_FILE: &str = ".llama-admin-monitor.json";

/// One installable backend for the running OS/CPU.
#[derive(Debug, Clone, Serialize, PartialEq)]
pub struct BackendOption {
    pub id: &'static str,
    pub label: &'static str,
    /// The part between `llama-<tag>-bin-` and `.tar.gz` in the asset name.
    pub suffix: &'static str,
    /// True when a `cudart-…` bundle with the CUDA runtime libraries exists
    /// for this backend; it is installed alongside so the build runs on a
    /// machine without the toolkit.
    pub has_cudart: bool,
    pub requires: &'static str,
}

/// Backends upstream publishes for this platform, most generally useful
/// first. ROCm is offered but its runtime must match the build exactly.
pub fn backend_options() -> Vec<BackendOption> {
    match (std::env::consts::OS, std::env::consts::ARCH) {
        ("linux", "x86_64") => vec![
            BackendOption {
                id: "vulkan",
                label: "Vulkan",
                suffix: "ubuntu-vulkan-x64",
                has_cudart: false,
                requires: "A Vulkan driver (Mesa radv/amdvlk for AMD, NVIDIA's driver). Works with AMD and NVIDIA cards, and across both at once.",
            },
            BackendOption {
                id: "cuda-12.8",
                label: "CUDA 12.8",
                suffix: "ubuntu-cuda-12.8-x64",
                has_cudart: true,
                requires: "NVIDIA driver 570 or newer. The CUDA runtime libraries are bundled.",
            },
            BackendOption {
                id: "cuda-13.3",
                label: "CUDA 13.3",
                suffix: "ubuntu-cuda-13.3-x64",
                has_cudart: true,
                requires: "NVIDIA driver 580 or newer. The CUDA runtime libraries are bundled.",
            },
            BackendOption {
                id: "rocm-10.0",
                label: "ROCm 10.0",
                suffix: "ubuntu-rocm-10.0-x64",
                has_cudart: false,
                requires: "ROCm 10.0 runtime installed system-wide; other ROCm versions will not load it.",
            },
            BackendOption {
                id: "cpu",
                label: "CPU only",
                suffix: "ubuntu-x64",
                has_cudart: false,
                requires: "Nothing. Slow for large models.",
            },
        ],
        ("linux", "aarch64") => vec![
            BackendOption {
                id: "vulkan",
                label: "Vulkan",
                suffix: "ubuntu-vulkan-arm64",
                has_cudart: false,
                requires: "A Vulkan driver.",
            },
            BackendOption {
                id: "cuda-13.3",
                label: "CUDA 13.3",
                suffix: "ubuntu-cuda-13.3-arm64",
                has_cudart: true,
                requires: "NVIDIA driver 580 or newer (Jetson/Grace). CUDA runtime bundled.",
            },
            BackendOption {
                id: "cpu",
                label: "CPU only",
                suffix: "ubuntu-arm64",
                has_cudart: false,
                requires: "Nothing.",
            },
        ],
        ("macos", "aarch64") => vec![BackendOption {
            id: "metal",
            label: "Metal (Apple Silicon)",
            suffix: "macos-arm64",
            has_cudart: false,
            requires: "Nothing; uses the built-in GPU.",
        }],
        ("macos", "x86_64") => vec![BackendOption {
            id: "cpu",
            label: "CPU (Intel Mac)",
            suffix: "macos-x64",
            has_cudart: false,
            requires: "Nothing.",
        }],
        _ => Vec::new(),
    }
}

fn option_by_id(id: &str) -> Option<BackendOption> {
    backend_options().into_iter().find(|b| b.id == id)
}

#[derive(Debug, Clone, Serialize, PartialEq)]
pub struct UpstreamRelease {
    pub tag: String,
    pub published_at: String,
}

/// Root for installed builds: `<data dir>/llama-admin-monitor/llama.cpp`.
pub fn builds_root() -> PathBuf {
    dirs::data_local_dir()
        .unwrap_or_else(|| PathBuf::from("."))
        .join("llama-admin-monitor")
        .join("llama.cpp")
}

/// Directory name for a backend + tag, also the build's id in presets.
pub fn build_id(backend: &str, tag: &str) -> String {
    format!("{backend}-{tag}")
}

/// Metadata written next to an installed build.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct InstalledBuild {
    pub id: String,
    pub backend: String,
    pub tag: String,
    pub installed_at: u64,
    pub server_path: PathBuf,
    pub dir: PathBuf,
    #[serde(default)]
    pub devices: Vec<crate::llama::server::GgmlDevice>,
    #[serde(default)]
    pub device_check_error: Option<String>,
}

/// Progress of an install in flight, mirrored over the WebSocket.
#[derive(Debug, Clone, Serialize, Default)]
pub struct InstallProgress {
    pub id: String,
    pub backend: String,
    pub tag: String,
    pub phase: String,
    pub downloaded_bytes: u64,
    pub total_bytes: u64,
    pub done: bool,
    pub error: Option<String>,
}

fn client() -> Result<reqwest::Client> {
    reqwest::Client::builder()
        .user_agent(USER_AGENT)
        .timeout(std::time::Duration::from_secs(1800))
        .build()
        .context("cannot build HTTP client")
}

/// Upstream tags look like `b11050`; anything else (the occasional
/// `v0.4.1` marker release) carries no binaries.
fn is_build_tag(tag: &str) -> bool {
    tag.len() > 1 && tag.starts_with('b') && tag[1..].chars().all(|c| c.is_ascii_digit())
}

/// The number in a build tag, for "is this newer" comparisons.
pub fn tag_number(tag: &str) -> Option<u64> {
    if !is_build_tag(tag) {
        return None;
    }
    tag[1..].parse().ok()
}

/// Newest upstream build tag.
pub async fn latest_tag() -> Result<String> {
    recent_releases(1)
        .await?
        .into_iter()
        .next()
        .map(|r| r.tag)
        .context("no llama.cpp releases found")
}

/// Moves everything that referenced build `old` to build `new`: presets on
/// `build:<old>` and, when Settings points at the old binary, Settings too.
/// Returns how many presets changed.
pub fn repoint(state: &AppState, old: &InstalledBuild, new: &InstalledBuild) -> usize {
    let old_ref = format!("build:{}", old.id);
    let new_ref = format!("build:{}", new.id);
    let mut changed = 0;
    {
        let mut presets = state.presets.lock().unwrap();
        for preset in presets.iter_mut() {
            if preset.backend == old_ref {
                preset.backend = new_ref.clone();
                changed += 1;
            }
        }
        if changed > 0 {
            let _ = crate::presets::save_presets(&state.presets_path, &presets);
        }
    }
    let mut settings = state.ui_settings.lock().unwrap();
    if settings.llama_server_path == old.server_path.display().to_string() {
        settings.llama_server_path = new.server_path.display().to_string();
        settings.llama_server_cwd = new.dir.display().to_string();
        let _ = crate::state::save_ui_settings(&state.ui_settings_path, &settings);
    }
    changed
}

/// Recent upstream releases, newest first. The API listing is tried
/// first; its rate limit falls back to the Atom feed.
pub async fn recent_releases(limit: usize) -> Result<Vec<UpstreamRelease>> {
    let client = client()?;
    let api = format!("https://api.github.com/repos/{REPO}/releases?per_page=20");
    let via_api: Result<Vec<UpstreamRelease>> = async {
        #[derive(Deserialize)]
        struct Gh {
            tag_name: String,
            #[serde(default)]
            draft: bool,
            #[serde(default)]
            published_at: Option<String>,
        }
        let resp = client.get(&api).send().await?.error_for_status()?;
        let bytes = resp.bytes().await?;
        let releases: Vec<Gh> = serde_json::from_slice(&bytes)?;
        Ok(releases
            .into_iter()
            .filter(|r| !r.draft && is_build_tag(&r.tag_name))
            .map(|r| UpstreamRelease {
                tag: r.tag_name,
                published_at: r.published_at.unwrap_or_default(),
            })
            .collect())
    }
    .await;
    let releases = match via_api {
        Ok(r) if !r.is_empty() => r,
        Ok(_) | Err(_) => {
            let feed = format!("https://github.com/{REPO}/releases.atom");
            let text = client
                .get(&feed)
                .send()
                .await
                .context("llama.cpp releases feed request failed")?
                .error_for_status()
                .context("llama.cpp releases feed returned an error")?
                .text()
                .await?;
            parse_releases_atom(&text)
        }
    };
    if releases.is_empty() {
        bail!("no llama.cpp releases found");
    }
    Ok(releases.into_iter().take(limit).collect())
}

pub fn parse_releases_atom(xml: &str) -> Vec<UpstreamRelease> {
    fn tag_text<'a>(block: &'a str, tag: &str) -> Option<&'a str> {
        let open = format!("<{tag}>");
        let close = format!("</{tag}>");
        let start = block.find(&open)? + open.len();
        let end = block[start..].find(&close)? + start;
        Some(block[start..end].trim())
    }
    xml.split("<entry>")
        .skip(1)
        .filter_map(|entry| {
            let tag = tag_text(entry, "title")?.to_string();
            if !is_build_tag(&tag) {
                return None;
            }
            Some(UpstreamRelease {
                published_at: tag_text(entry, "updated").unwrap_or("").to_string(),
                tag,
            })
        })
        .collect()
}

pub fn asset_url(tag: &str, suffix: &str, cudart: bool) -> String {
    let name = if cudart {
        format!("cudart-llama-{tag}-bin-{suffix}.tar.gz")
    } else {
        format!("llama-{tag}-bin-{suffix}.tar.gz")
    };
    format!("https://github.com/{REPO}/releases/download/{tag}/{name}")
}

fn set_progress(state: &AppState, f: impl FnOnce(&mut InstallProgress)) {
    if let Some(p) = state.build_install.lock().unwrap().as_mut() {
        f(p);
    }
}

async fn download(
    state: &AppState,
    client: &reqwest::Client,
    url: &str,
    dest: &Path,
) -> Result<()> {
    use tokio::io::AsyncWriteExt;
    let resp = client
        .get(url)
        .send()
        .await
        .with_context(|| format!("download request failed: {url}"))?;
    if resp.status().as_u16() == 404 {
        bail!("this release has no such asset ({url})");
    }
    let resp = resp.error_for_status().context("download failed")?;
    let total = resp.content_length().unwrap_or(0);
    set_progress(state, |p| {
        p.total_bytes = total;
        p.downloaded_bytes = 0;
    });
    let mut file = tokio::fs::File::create(dest).await?;
    let mut stream = resp.bytes_stream();
    let mut written = 0u64;
    while let Some(chunk) = stream.next().await {
        let chunk = chunk?;
        file.write_all(&chunk).await?;
        written += chunk.len() as u64;
        set_progress(state, |p| p.downloaded_bytes = written);
    }
    file.flush().await?;
    if total > 0 && written != total {
        bail!("download incomplete: {written} of {total} bytes");
    }
    Ok(())
}

/// `tar` is present on every platform this runs on (bsdtar on macOS and
/// Windows), which avoids linking a decompressor into the binary.
async fn extract(archive: &Path, dest: &Path) -> Result<()> {
    tokio::fs::create_dir_all(dest).await?;
    let output = tokio::process::Command::new("tar")
        .arg("-xzf")
        .arg(archive)
        .arg("--strip-components=1")
        .arg("-C")
        .arg(dest)
        .output()
        .await
        .context("cannot run tar")?;
    if !output.status.success() {
        bail!(
            "extracting failed: {}",
            String::from_utf8_lossy(&output.stderr).trim()
        );
    }
    Ok(())
}

/// Downloads, extracts and smoke-tests one build. Progress goes through
/// `state.build_install`; the result is recorded in the build's metadata.
pub async fn install(
    state: AppState,
    app_config: std::sync::Arc<crate::config::AppConfig>,
    backend: String,
    tag: String,
) -> Result<InstalledBuild> {
    let option = option_by_id(&backend)
        .with_context(|| format!("backend {backend} is not available on this platform"))?;
    if !is_build_tag(&tag) {
        bail!("{tag} is not a llama.cpp build tag");
    }
    let id = build_id(&backend, &tag);
    let dir = builds_root().join(&id);
    if dir.join(META_FILE).exists() {
        bail!("{id} is already installed");
    }
    let staging = builds_root().join(format!(".{id}.tmp"));
    let _ = tokio::fs::remove_dir_all(&staging).await;
    tokio::fs::create_dir_all(&staging).await?;
    let client = client()?;

    let result: Result<InstalledBuild> = async {
        set_progress(&state, |p| {
            p.phase = format!("Downloading llama.cpp {tag} ({})", option.label)
        });
        let archive = staging.join("build.tar.gz");
        download(
            &state,
            &client,
            &asset_url(&tag, option.suffix, false),
            &archive,
        )
        .await?;
        set_progress(&state, |p| p.phase = "Extracting".to_string());
        extract(&archive, &staging).await?;
        tokio::fs::remove_file(&archive).await.ok();

        if option.has_cudart {
            set_progress(&state, |p| {
                p.phase = "Downloading CUDA runtime libraries".to_string()
            });
            let cudart = staging.join("cudart.tar.gz");
            download(
                &state,
                &client,
                &asset_url(&tag, option.suffix, true),
                &cudart,
            )
            .await?;
            set_progress(&state, |p| p.phase = "Extracting CUDA runtime".to_string());
            extract(&cudart, &staging).await?;
            tokio::fs::remove_file(&cudart).await.ok();
        }

        let server_path = staging.join("llama-server");
        if !server_path.is_file() {
            bail!("the archive did not contain llama-server");
        }
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            let mut entries = tokio::fs::read_dir(&staging).await?;
            while let Some(e) = entries.next_entry().await? {
                let name = e.file_name().to_string_lossy().to_string();
                if name.starts_with("llama-") && !name.contains('.') {
                    tokio::fs::set_permissions(e.path(), std::fs::Permissions::from_mode(0o755))
                        .await?;
                }
            }
        }

        // Move into place before the smoke test so the id resolves. A
        // directory without metadata is a leftover from a failed attempt.
        if dir.exists() {
            tokio::fs::remove_dir_all(&dir).await?;
        }
        tokio::fs::rename(&staging, &dir).await?;
        let mut build = InstalledBuild {
            id: id.clone(),
            backend: backend.clone(),
            tag: tag.clone(),
            installed_at: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_secs())
                .unwrap_or(0),
            server_path: dir.join("llama-server"),
            dir: dir.clone(),
            devices: Vec::new(),
            device_check_error: None,
        };
        write_meta(&build)?;

        set_progress(&state, |p| {
            p.phase = "Checking which devices it can see".to_string()
        });
        let backend_ref = format!("build:{id}");
        let probe = crate::llama::server::list_devices(&state, &app_config, &backend_ref);
        match probe.await {
            Ok(devices) => build.devices = devices,
            Err(e) => build.device_check_error = Some(format!("{e:#}")),
        }
        write_meta(&build)?;
        Ok(build)
    }
    .await;

    if result.is_err() {
        let _ = tokio::fs::remove_dir_all(&staging).await;
    }
    result
}

fn write_meta(build: &InstalledBuild) -> Result<()> {
    let json = serde_json::to_string_pretty(build)?;
    std::fs::write(build.dir.join(META_FILE), json)?;
    Ok(())
}

/// Every build with metadata under the builds root, newest first.
pub fn installed_builds() -> Vec<InstalledBuild> {
    let mut out = Vec::new();
    let Ok(entries) = std::fs::read_dir(builds_root()) else {
        return out;
    };
    for e in entries.flatten() {
        let meta = e.path().join(META_FILE);
        if let Ok(text) = std::fs::read_to_string(&meta)
            && let Ok(mut b) = serde_json::from_str::<InstalledBuild>(&text)
        {
            // Paths follow the directory even if it was moved.
            b.dir = e.path();
            b.server_path = e.path().join("llama-server");
            out.push(b);
        }
    }
    out.sort_by_key(|b| std::cmp::Reverse(b.installed_at));
    out
}

pub fn installed_build(id: &str) -> Option<InstalledBuild> {
    if id.contains('/') || id.contains("..") {
        return None;
    }
    installed_builds().into_iter().find(|b| b.id == id)
}

pub fn remove(id: &str) -> Result<()> {
    let build = installed_build(id).with_context(|| format!("no installed build {id}"))?;
    std::fs::remove_dir_all(&build.dir)
        .with_context(|| format!("cannot remove {}", build.dir.display()))?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn build_tags_only() {
        assert!(is_build_tag("b11050"));
        assert_eq!(tag_number("b11050"), Some(11050));
        assert_eq!(tag_number("v0.4.1"), None);
        assert!(!is_build_tag("v0.4.1"));
        assert!(!is_build_tag("b"));
        assert!(!is_build_tag("main"));
    }

    #[test]
    fn asset_urls_follow_upstream_naming() {
        assert_eq!(
            asset_url("b11050", "ubuntu-vulkan-x64", false),
            "https://github.com/ggml-org/llama.cpp/releases/download/b11050/llama-b11050-bin-ubuntu-vulkan-x64.tar.gz"
        );
        assert_eq!(
            asset_url("b11050", "ubuntu-cuda-12.8-x64", true),
            "https://github.com/ggml-org/llama.cpp/releases/download/b11050/cudart-llama-b11050-bin-ubuntu-cuda-12.8-x64.tar.gz"
        );
    }

    #[test]
    fn atom_feed_skips_marker_releases() {
        let xml = concat!(
            "<feed><entry><title>v0.4.1</title><updated>2026-09-14T00:00:00Z</updated></entry>",
            "<entry><title>b11050</title><updated>2026-09-19T08:00:00Z</updated></entry>",
            "<entry><title>b11049</title><updated>2026-09-19T07:00:00Z</updated></entry></feed>"
        );
        let r = parse_releases_atom(xml);
        assert_eq!(r.len(), 2);
        assert_eq!(r[0].tag, "b11050");
        assert_eq!(r[0].published_at, "2026-09-19T08:00:00Z");
    }

    #[test]
    fn ids_and_paths() {
        assert_eq!(build_id("vulkan", "b11050"), "vulkan-b11050");
        assert!(installed_build("../etc").is_none());
    }
}
