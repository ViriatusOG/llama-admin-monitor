//! In-app updates from GitHub Releases.
//!
//! The running binary knows which release track it came from and which tag
//! built it (both baked in by the release workflow). Checking for updates is
//! one call to the GitHub API; applying one downloads this platform's asset
//! to a temp file next to the executable, validates it, swaps it into place
//! atomically and re-executes. No git checkout or shell is involved, so it
//! works for a binary copied anywhere, including under systemd.

use anyhow::{Context, Result, bail};
use futures_util::StreamExt;
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};

use crate::state::AppState;

const REPO: &str = "ViriatusOG/llama-admin-monitor";
const USER_AGENT: &str = "llama-admin-monitor";
/// Anything smaller than this is an error page, not a build of this app.
const MIN_BINARY_BYTES: u64 = 1_000_000;

/// Release track this binary was built for: "main", "beta", or "dev" for a
/// local `cargo build` (which never offers updates to itself).
pub fn current_track() -> &'static str {
    option_env!("LLAMA_ADMIN_TRACK").unwrap_or("dev")
}

/// The git tag this binary was built from, e.g. `v2026.9.20-beta.1`, or the
/// crate version for local builds.
pub fn current_version() -> String {
    option_env!("LLAMA_ADMIN_RELEASE_TAG")
        .map(|t| t.trim_start_matches('v').to_string())
        .unwrap_or_else(|| env!("CARGO_PKG_VERSION").to_string())
}

/// Name of the release asset for this OS and CPU, matching release.yml.
pub fn platform_asset() -> Option<String> {
    let os = match std::env::consts::OS {
        "linux" => "linux",
        "macos" => "macos",
        _ => return None,
    };
    let arch = match std::env::consts::ARCH {
        "x86_64" => "x86_64",
        "aarch64" => "aarch64",
        _ => return None,
    };
    Some(format!("llama-admin-monitor-{os}-{arch}"))
}

#[derive(Debug, Clone, Serialize, PartialEq)]
pub struct ReleaseInfo {
    pub tag: String,
    pub version: String,
    pub prerelease: bool,
    pub published_at: String,
    pub url: String,
    /// Download URL of this platform's asset, when the release ships one.
    pub asset_url: Option<String>,
}

#[derive(Debug, Clone, Serialize)]
pub struct UpdateStatus {
    pub current_track: &'static str,
    pub current_version: String,
    pub platform_asset: Option<String>,
    pub stable: Option<ReleaseInfo>,
    pub beta: Option<ReleaseInfo>,
    /// True when the current track has a release newer than this build.
    pub update_available: bool,
}

#[derive(Deserialize)]
struct GhAsset {
    name: String,
    browser_download_url: String,
}

#[derive(Deserialize)]
struct GhRelease {
    tag_name: String,
    #[serde(default)]
    prerelease: bool,
    #[serde(default)]
    draft: bool,
    #[serde(default)]
    published_at: Option<String>,
    #[serde(default)]
    html_url: String,
    #[serde(default)]
    assets: Vec<GhAsset>,
}

fn client() -> Result<reqwest::Client> {
    reqwest::Client::builder()
        .user_agent(USER_AGENT)
        .timeout(std::time::Duration::from_secs(600))
        .build()
        .context("cannot build HTTP client")
}

async fn fetch_releases(client: &reqwest::Client) -> Result<Vec<GhRelease>> {
    let url = format!("https://api.github.com/repos/{REPO}/releases?per_page=30");
    let resp = client
        .get(&url)
        .send()
        .await
        .context("GitHub API request failed")?
        .error_for_status()
        .context("GitHub API returned an error")?;
    resp.json().await.context("unexpected GitHub API response")
}

fn to_info(r: &GhRelease, asset: Option<&str>) -> ReleaseInfo {
    ReleaseInfo {
        tag: r.tag_name.clone(),
        version: r.tag_name.trim_start_matches('v').to_string(),
        prerelease: r.prerelease,
        published_at: r.published_at.clone().unwrap_or_default(),
        url: r.html_url.clone(),
        asset_url: asset.and_then(|name| {
            r.assets
                .iter()
                .find(|a| a.name == name)
                .map(|a| a.browser_download_url.clone())
        }),
    }
}

/// Latest stable (non-prerelease) and latest beta (prerelease) releases.
/// GitHub lists releases newest first, so the first match of each kind wins.
fn latest_per_track(
    releases: &[GhRelease],
    asset: Option<&str>,
) -> (Option<ReleaseInfo>, Option<ReleaseInfo>) {
    let published = releases.iter().filter(|r| !r.draft);
    let stable = published
        .clone()
        .find(|r| !r.prerelease)
        .map(|r| to_info(r, asset));
    let beta = published
        .clone()
        .find(|r| r.prerelease)
        .map(|r| to_info(r, asset));
    (stable, beta)
}

pub async fn check_updates() -> Result<UpdateStatus> {
    let client = client()?;
    let releases = fetch_releases(&client).await?;
    let asset = platform_asset();
    let (stable, beta) = latest_per_track(&releases, asset.as_deref());
    let current_version = current_version();
    let latest_on_track = match current_track() {
        "beta" => beta.as_ref(),
        "main" => stable.as_ref(),
        _ => None,
    };
    let update_available = latest_on_track
        .map(|r| r.version != current_version)
        .unwrap_or(false);
    Ok(UpdateStatus {
        current_track: current_track(),
        current_version,
        platform_asset: asset,
        stable,
        beta,
        update_available,
    })
}

fn set_phase(state: &AppState, phase: &str) {
    *state.update_phase.lock().unwrap() = Some(phase.to_string());
    println!("[update] {phase}");
}

/// True when `bytes` starts like a native executable for this platform, so
/// an HTML error page or a truncated download is never swapped in.
pub fn looks_like_executable(bytes: &[u8]) -> bool {
    if bytes.len() < 4 {
        return false;
    }
    let magic = &bytes[..4];
    let elf = magic == [0x7f, b'E', b'L', b'F'];
    let macho = matches!(
        magic,
        [0xfe, 0xed, 0xfa, 0xce]
            | [0xfe, 0xed, 0xfa, 0xcf]
            | [0xce, 0xfa, 0xed, 0xfe]
            | [0xcf, 0xfa, 0xed, 0xfe]
            | [0xca, 0xfe, 0xba, 0xbe]
    );
    elf || macho
}

/// Streams `url` to `tmp`, checking the declared length and the file magic.
async fn download_binary(client: &reqwest::Client, url: &str, tmp: &Path) -> Result<u64> {
    use tokio::io::AsyncWriteExt;

    let resp = client
        .get(url)
        .send()
        .await
        .context("download request failed")?
        .error_for_status()
        .context("download failed")?;
    let expected = resp.content_length();

    let mut file = tokio::fs::File::create(tmp).await?;
    let mut stream = resp.bytes_stream();
    let mut head: Vec<u8> = Vec::with_capacity(4);
    let mut written: u64 = 0;
    while let Some(chunk) = stream.next().await {
        let chunk = chunk?;
        if head.len() < 4 {
            head.extend_from_slice(&chunk[..chunk.len().min(4 - head.len())]);
        }
        file.write_all(&chunk).await?;
        written += chunk.len() as u64;
    }
    file.flush().await?;
    drop(file);

    if let Some(expected) = expected
        && expected != written
    {
        bail!("download incomplete: {written} of {expected} bytes");
    }
    if written < MIN_BINARY_BYTES {
        bail!("downloaded file is too small to be a build ({written} bytes)");
    }
    if !looks_like_executable(&head) {
        bail!("downloaded file is not an executable for this platform");
    }
    Ok(written)
}

/// Downloads the latest release of `track`, swaps it in for the running
/// binary and re-executes. Reports progress through `state.update_phase`;
/// on any failure the running binary is left untouched.
pub async fn apply_update(state: AppState, track: String) -> Result<()> {
    if !cfg!(unix) {
        bail!("in-app updates are supported on Linux and macOS only");
    }
    let asset = platform_asset().context("no release asset for this OS/CPU")?;

    set_phase(&state, "Checking releases");
    let client = client()?;
    let releases = fetch_releases(&client).await?;
    let (stable, beta) = latest_per_track(&releases, Some(&asset));
    let target = match track.as_str() {
        "main" => stable,
        "beta" => beta,
        other => bail!("unknown track {other}"),
    }
    .with_context(|| format!("no published release on the {track} track yet"))?;
    let url = target
        .asset_url
        .clone()
        .with_context(|| format!("release {} has no {asset} asset", target.tag))?;

    let exe = std::env::current_exe().context("cannot locate the running executable")?;
    let exe_dir = exe.parent().context("executable has no parent directory")?;
    let tmp: PathBuf = exe_dir.join(".llama-admin-monitor.update");

    set_phase(&state, &format!("Downloading {}", target.tag));
    let result = download_binary(&client, &url, &tmp).await;
    if let Err(e) = result {
        let _ = tokio::fs::remove_file(&tmp).await;
        return Err(e);
    }

    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        tokio::fs::set_permissions(&tmp, std::fs::Permissions::from_mode(0o755)).await?;
    }

    set_phase(&state, "Stopping llama-server");
    crate::llama::server::stop_server(&state).await.ok();

    set_phase(&state, "Installing");
    if let Err(e) = tokio::fs::rename(&tmp, &exe).await {
        let _ = tokio::fs::remove_file(&tmp).await;
        return Err(e).with_context(|| {
            format!(
                "cannot replace {} (is it writable by this user?)",
                exe.display()
            )
        });
    }

    set_phase(&state, &format!("Restarting into {}", target.tag));
    restart(exe);
    Ok(())
}

/// Re-executes `exe` with this process's arguments after a short delay so
/// the HTTP reply and the final WebSocket frame get out first.
fn restart(exe: PathBuf) {
    std::thread::spawn(move || {
        std::thread::sleep(std::time::Duration::from_millis(700));
        #[cfg(unix)]
        {
            use std::os::unix::process::CommandExt;
            let args: Vec<String> = std::env::args().skip(1).collect();
            let err = std::process::Command::new(&exe).args(&args).exec();
            eprintln!("[update] exec failed: {err}");
        }
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    fn release(tag: &str, prerelease: bool, draft: bool) -> GhRelease {
        GhRelease {
            tag_name: tag.into(),
            prerelease,
            draft,
            published_at: Some("2026-09-20T00:00:00Z".into()),
            html_url: format!("https://github.com/{REPO}/releases/tag/{tag}"),
            assets: vec![GhAsset {
                name: "llama-admin-monitor-linux-x86_64".into(),
                browser_download_url: format!("https://example/{tag}/linux"),
            }],
        }
    }

    #[test]
    fn picks_newest_stable_and_beta_skipping_drafts() {
        let releases = vec![
            release("v2026.9.21-beta.1", true, true),
            release("v2026.9.20-beta.2", true, false),
            release("v2026.9.20", false, false),
            release("v2026.9.19", false, false),
        ];
        let (stable, beta) = latest_per_track(&releases, Some("llama-admin-monitor-linux-x86_64"));
        assert_eq!(stable.as_ref().map(|r| r.tag.as_str()), Some("v2026.9.20"));
        assert_eq!(
            beta.as_ref().map(|r| r.tag.as_str()),
            Some("v2026.9.20-beta.2")
        );
        assert_eq!(beta.unwrap().version, "2026.9.20-beta.2");
        assert_eq!(
            stable.unwrap().asset_url.as_deref(),
            Some("https://example/v2026.9.20/linux")
        );
    }

    #[test]
    fn missing_asset_gives_no_url() {
        let releases = vec![release("v1", false, false)];
        let (stable, _) = latest_per_track(&releases, Some("llama-admin-monitor-macos-aarch64"));
        assert_eq!(stable.unwrap().asset_url, None);
    }

    #[test]
    fn executable_magic() {
        assert!(looks_like_executable(&[0x7f, b'E', b'L', b'F', 2, 1]));
        assert!(looks_like_executable(&[0xcf, 0xfa, 0xed, 0xfe]));
        assert!(!looks_like_executable(b"<!DOCTYPE html>"));
        assert!(!looks_like_executable(b"\x7fEL"));
    }

    #[test]
    fn version_strips_v_prefix() {
        assert!(!current_version().starts_with('v'));
        assert!(matches!(current_track(), "main" | "beta" | "dev"));
    }
}
