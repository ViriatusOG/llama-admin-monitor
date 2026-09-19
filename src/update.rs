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
use std::sync::Mutex;
use std::time::{Duration, Instant};

const REPO: &str = "ViriatusOG/llama-admin-monitor";
const USER_AGENT: &str = "llama-admin-monitor";
/// Anything smaller than this is an error page, not a build of this app.
const MIN_BINARY_BYTES: u64 = 1_000_000;
/// How long a successful release listing is reused. Unauthenticated GitHub
/// API calls are limited to 60 per hour per address, and every page load
/// asks; without this a busy afternoon of reloads hits the limit.
const CHECK_CACHE_TTL: Duration = Duration::from_secs(5 * 60);

static CHECK_CACHE: Mutex<Option<(Instant, UpdateStatus)>> = Mutex::new(None);

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
    let mut headers = reqwest::header::HeaderMap::new();
    // Optional: raises the API limit from 60 to 5000 requests per hour.
    if let Ok(token) = std::env::var("LLAMA_ADMIN_GITHUB_TOKEN")
        && !token.trim().is_empty()
        && let Ok(value) = format!("Bearer {}", token.trim()).parse()
    {
        headers.insert(reqwest::header::AUTHORIZATION, value);
    }
    reqwest::Client::builder()
        .user_agent(USER_AGENT)
        .default_headers(headers)
        .timeout(Duration::from_secs(600))
        .build()
        .context("cannot build HTTP client")
}

/// Turns GitHub's 403/429 replies into a message that says when to retry.
fn rate_limit_message(resp: &reqwest::Response) -> Option<String> {
    let status = resp.status().as_u16();
    if status != 403 && status != 429 {
        return None;
    }
    let remaining = resp
        .headers()
        .get("x-ratelimit-remaining")
        .and_then(|v| v.to_str().ok())
        .and_then(|v| v.parse::<u64>().ok());
    if remaining.is_some_and(|r| r > 0) && status == 403 {
        return None;
    }
    let reset = resp
        .headers()
        .get("x-ratelimit-reset")
        .and_then(|v| v.to_str().ok())
        .and_then(|v| v.parse::<u64>().ok())
        .map(|epoch| {
            let now = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_secs())
                .unwrap_or(0);
            epoch.saturating_sub(now) / 60 + 1
        });
    Some(match reset {
        Some(mins) => format!(
            "GitHub API rate limit reached (60 unauthenticated requests per hour); \
             try again in about {mins} min, or set LLAMA_ADMIN_GITHUB_TOKEN"
        ),
        None => "GitHub API rate limit reached; try again later".to_string(),
    })
}

async fn fetch_releases(client: &reqwest::Client) -> Result<Vec<GhRelease>> {
    let url = format!("https://api.github.com/repos/{REPO}/releases?per_page=30");
    let resp = client
        .get(&url)
        .send()
        .await
        .context("GitHub API request failed")?;
    if let Some(msg) = rate_limit_message(&resp) {
        bail!("{msg}");
    }
    let resp = resp
        .error_for_status()
        .context("GitHub API returned an error")?;
    let bytes = resp
        .bytes()
        .await
        .context("GitHub API response cut short")?;
    serde_json::from_slice(&bytes).context("unexpected GitHub API response")
}

/// The releases Atom feed is served by github.com, not the API, so it is
/// not counted against the 60-per-hour unauthenticated API quota. It lists
/// every published release (pre-releases included) newest first, but does
/// not carry the prerelease flag or the asset list, so those are derived:
/// a tag containing "-beta" is a pre-release, and the platform asset is
/// assumed to live at the conventional download URL (the download step
/// verifies it really is a binary).
async fn fetch_releases_atom(client: &reqwest::Client) -> Result<Vec<GhRelease>> {
    let url = format!("https://github.com/{REPO}/releases.atom");
    let resp = client
        .get(&url)
        .send()
        .await
        .context("releases feed request failed")?
        .error_for_status()
        .context("releases feed returned an error")?;
    let text = resp.text().await.context("releases feed cut short")?;
    let releases = parse_releases_atom(&text);
    if releases.is_empty() {
        bail!("releases feed had no entries");
    }
    Ok(releases)
}

fn parse_releases_atom(xml: &str) -> Vec<GhRelease> {
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
            if !tag.starts_with('v') {
                return None;
            }
            let assets = [
                "linux-x86_64",
                "linux-aarch64",
                "macos-x86_64",
                "macos-aarch64",
            ]
            .iter()
            .map(|p| {
                let name = format!("llama-admin-monitor-{p}");
                GhAsset {
                    browser_download_url: format!(
                        "https://github.com/{REPO}/releases/download/{tag}/{name}"
                    ),
                    name,
                }
            })
            .collect();
            Some(GhRelease {
                prerelease: is_beta_tag(&tag),
                draft: false,
                published_at: tag_text(entry, "updated").map(str::to_string),
                html_url: format!("https://github.com/{REPO}/releases/tag/{tag}"),
                assets,
                tag_name: tag,
            })
        })
        .collect()
}

/// Release tags containing "-beta" belong to the beta track.
pub fn is_beta_tag(tag: &str) -> bool {
    tag.contains("-beta")
}

/// The API listing, falling back to the Atom feed when the API is
/// unavailable (typically its rate limit). The API error is kept in the
/// message if both fail.
async fn list_releases(client: &reqwest::Client) -> Result<Vec<GhRelease>> {
    match fetch_releases(client).await {
        Ok(releases) => Ok(releases),
        Err(api_err) => {
            crate::applog::warn(format!(
                "GitHub API unavailable ({api_err:#}); using the releases feed"
            ));
            fetch_releases_atom(client)
                .await
                .with_context(|| format!("{api_err:#}"))
        }
    }
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

/// Sortable form of a release version: `YYYY.M.D` with an optional
/// `-beta.N`. A stable release outranks any beta of the same day, and
/// `beta.12` outranks `beta.9` (a plain string compare gets that wrong,
/// and so does GitHub's own listing order, which sorts by tag name).
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct VersionKey {
    date: (u32, u32, u32),
    /// 1 for a stable release, 0 for a beta, so stable sorts higher.
    stable: u8,
    beta: u32,
}

pub fn version_key(version: &str) -> Option<VersionKey> {
    let v = version.trim().trim_start_matches('v');
    let (date, rest) = match v.split_once('-') {
        Some((d, r)) => (d, Some(r)),
        None => (v, None),
    };
    let mut parts = date.split('.').map(|n| n.parse::<u32>().ok());
    let y = parts.next().flatten()?;
    let m = parts.next().flatten()?;
    let d = parts.next().flatten()?;
    if parts.next().is_some() {
        return None;
    }
    let (stable, beta) = match rest {
        None => (1, 0),
        Some(r) => (0, r.strip_prefix("beta.")?.parse::<u32>().ok()?),
    };
    Some(VersionKey {
        date: (y, m, d),
        stable,
        beta,
    })
}

/// Latest stable (non-prerelease) and latest beta (prerelease) releases,
/// by version number; releases whose tag does not parse fall back to their
/// publish date, so a hand-made tag still sorts somewhere sensible.
fn latest_per_track(
    releases: &[GhRelease],
    asset: Option<&str>,
) -> (Option<ReleaseInfo>, Option<ReleaseInfo>) {
    let rank = |r: &GhRelease| {
        (
            version_key(&r.tag_name),
            r.published_at.clone().unwrap_or_default(),
        )
    };
    let published = releases.iter().filter(|r| !r.draft);
    let stable = published
        .clone()
        .filter(|r| !r.prerelease)
        .max_by_key(|r| rank(r))
        .map(|r| to_info(r, asset));
    let beta = published
        .clone()
        .filter(|r| r.prerelease)
        .max_by_key(|r| rank(r))
        .map(|r| to_info(r, asset));
    (stable, beta)
}

/// True when `latest` is a newer version than `current`. Unparseable
/// versions compare by string inequality, as before.
fn is_newer(latest: &str, current: &str) -> bool {
    match (version_key(latest), version_key(current)) {
        (Some(l), Some(c)) => l > c,
        _ => latest != current,
    }
}

/// Lists releases, reusing a recent answer (see CHECK_CACHE_TTL). `force`
/// skips the cache for an explicit "Check for updates" click.
pub async fn check_updates(force: bool) -> Result<UpdateStatus> {
    if !force
        && let Some((at, status)) = CHECK_CACHE.lock().unwrap().as_ref()
        && at.elapsed() < CHECK_CACHE_TTL
    {
        return Ok(status.clone());
    }
    let status = check_updates_uncached().await?;
    *CHECK_CACHE.lock().unwrap() = Some((Instant::now(), status.clone()));
    Ok(status)
}

async fn check_updates_uncached() -> Result<UpdateStatus> {
    let client = client()?;
    let releases = list_releases(&client).await?;
    let asset = platform_asset();
    let (stable, beta) = latest_per_track(&releases, asset.as_deref());
    let current_version = current_version();
    let latest_on_track = match current_track() {
        "beta" => beta.as_ref(),
        "main" => stable.as_ref(),
        _ => None,
    };
    let update_available = latest_on_track
        .map(|r| is_newer(&r.version, &current_version))
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
    crate::applog::info(format!("Update: {phase}"));
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
    let releases = list_releases(&client).await?;
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
            crate::applog::error(format!("Update: exec failed: {err}"));
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
    fn latest_is_by_version_not_listing_order() {
        // GitHub's API lists by tag name, which puts beta.9 above beta.12.
        let releases = vec![
            release("v2026.9.21-beta.9", true, false),
            release("v2026.9.21-beta.12", true, false),
            release("v2026.9.21-beta.10", true, false),
            release("v2026.9.20", false, false),
            release("v2026.9.9", false, false),
        ];
        let (stable, beta) = latest_per_track(&releases, None);
        assert_eq!(beta.unwrap().tag, "v2026.9.21-beta.12");
        assert_eq!(stable.unwrap().tag, "v2026.9.20");
    }

    #[test]
    fn version_keys_order_sensibly() {
        let k = |v: &str| version_key(v).unwrap();
        assert!(k("2026.9.21-beta.12") > k("2026.9.21-beta.9"));
        assert!(k("2026.9.21") > k("2026.9.21-beta.12"));
        assert!(k("2026.10.1") > k("2026.9.30"));
        assert!(k("v2026.9.21") == k("2026.9.21"));
        // Zero-padded tags (the format from v2026.09.22 on) equal unpadded.
        assert!(k("2026.09.22-beta.01") == k("2026.9.22-beta.1"));
        assert!(k("2026.09.22-beta.01") > k("2026.9.21-beta.14"));
        assert!(k("2026.10.01") > k("2026.09.30"));
        assert_eq!(version_key("nightly"), None);
        assert_eq!(version_key("2026.9.21-rc.1"), None);
        assert!(is_newer("2026.9.21-beta.12", "2026.9.21-beta.9"));
        assert!(!is_newer("2026.9.21-beta.9", "2026.9.21-beta.12"));
        assert!(!is_newer("2026.9.21-beta.12", "2026.9.21-beta.12"));
    }

    #[test]
    fn missing_asset_gives_no_url() {
        let releases = vec![release("v1", false, false)];
        let (stable, _) = latest_per_track(&releases, Some("llama-admin-monitor-macos-aarch64"));
        assert_eq!(stable.unwrap().asset_url, None);
    }

    #[test]
    fn atom_feed_yields_releases_newest_first() {
        let xml = concat!(
            "<?xml version=\"1.0\"?><feed><title>Release notes</title>",
            "<entry><id>x/v2026.9.21-beta.2</id><updated>2026-09-19T09:32:44Z</updated>",
            "<title>v2026.9.21-beta.2</title></entry>",
            "<entry><id>x/v2026.9.20</id><updated>2026-09-18T23:41:07Z</updated>",
            "<title>v2026.9.20</title></entry></feed>"
        );
        let releases = parse_releases_atom(xml);
        assert_eq!(releases.len(), 2);
        assert_eq!(releases[0].tag_name, "v2026.9.21-beta.2");
        assert!(releases[0].prerelease);
        assert!(!releases[1].prerelease);
        assert_eq!(
            releases[0].published_at.as_deref(),
            Some("2026-09-19T09:32:44Z")
        );
        let (stable, beta) = latest_per_track(&releases, Some("llama-admin-monitor-linux-x86_64"));
        assert_eq!(
            beta.unwrap().asset_url.as_deref(),
            Some(concat!(
                "https://github.com/ViriatusOG/llama-admin-monitor/releases/download/",
                "v2026.9.21-beta.2/llama-admin-monitor-linux-x86_64"
            ))
        );
        assert_eq!(stable.unwrap().tag, "v2026.9.20");
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
