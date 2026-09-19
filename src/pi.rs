//! The pi coding agent (https://pi.dev) running in a pseudo-terminal, with
//! its output fanned out to browser terminals over a WebSocket. One
//! session at a time; it outlives page loads, and a reconnecting page gets
//! the recent scrollback replayed so the screen is not blank.
//!
//! pi talks to whichever model llama-server has loaded through the
//! monitor's own OpenAI-compatible `/v1` proxy: `models.json` gets a
//! `llama-admin-monitor` provider pointing at it, refreshed every start.

use anyhow::{Context, Result, bail};
use portable_pty::{CommandBuilder, MasterPty, PtySize, native_pty_system};
use std::collections::VecDeque;
use std::io::{Read, Write};
use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use tokio::sync::broadcast;

/// Provider name written to pi's models.json.
pub const PROVIDER: &str = "llama-admin-monitor";
/// How much output a late-joining page gets replayed.
const SCROLLBACK_BYTES: usize = 256 * 1024;

pub struct Session {
    /// Behind a mutex so the session is Sync and can be shared with
    /// WebSocket tasks; only resize touches it.
    master: Mutex<Box<dyn MasterPty + Send>>,
    writer: Mutex<Box<dyn Write + Send>>,
    child: Mutex<Box<dyn portable_pty::Child + Send + Sync>>,
    /// Live output; browser sessions subscribe here.
    tx: broadcast::Sender<Vec<u8>>,
    scrollback: Mutex<VecDeque<u8>>,
    pub label: String,
    pub cwd: PathBuf,
    pub exited: Mutex<Option<i32>>,
}

#[derive(Debug, Clone, Default, serde::Serialize)]
pub struct Status {
    pub installed: bool,
    pub path: Option<String>,
    pub version: Option<String>,
    pub running: bool,
    pub label: Option<String>,
    pub cwd: Option<String>,
    pub exit_code: Option<i32>,
    pub models_json: String,
    pub provider: &'static str,
}

pub type Shared = Arc<Mutex<Option<Arc<Session>>>>;

/// Where pi is, if anywhere on PATH (or in the usual per-user npm/pi
/// locations that a login shell would add but a service does not).
pub fn find_pi() -> Option<PathBuf> {
    if let Some(p) = crate::llama::server::find_on_path(std::path::Path::new("pi")) {
        return Some(p);
    }
    let home = dirs::home_dir()?;
    for rel in [
        ".pi/bin/pi",
        ".local/bin/pi",
        ".npm-global/bin/pi",
        ".bun/bin/pi",
    ] {
        let p = home.join(rel);
        if p.is_file() {
            return Some(p);
        }
    }
    None
}

fn pi_version(path: &PathBuf) -> Option<String> {
    let out = std::process::Command::new(path)
        .arg("--version")
        .output()
        .ok()?;
    let text = String::from_utf8_lossy(&out.stdout);
    text.lines()
        .find(|l| !l.trim().is_empty())
        .map(|l| l.trim().to_string())
}

pub fn models_json_path() -> PathBuf {
    dirs::home_dir()
        .unwrap_or_else(|| PathBuf::from("."))
        .join(".pi")
        .join("agent")
        .join("models.json")
}

pub fn status(shared: &Shared) -> Status {
    let path = find_pi();
    let session = shared.lock().unwrap().clone();
    let (running, label, cwd, exit_code) = match &session {
        Some(s) => {
            let exit = *s.exited.lock().unwrap();
            (
                exit.is_none(),
                Some(s.label.clone()),
                Some(s.cwd.display().to_string()),
                exit,
            )
        }
        None => (false, None, None, None),
    };
    Status {
        installed: path.is_some(),
        version: path.as_ref().and_then(pi_version),
        path: path.map(|p| p.display().to_string()),
        running,
        label,
        cwd,
        exit_code,
        models_json: models_json_path().display().to_string(),
        provider: PROVIDER,
    }
}

/// Adds (or refreshes) the monitor's provider in pi's models.json without
/// touching anything else in the file. The model id is whatever is
/// loaded now; llama-server serves one model and ignores the id, but pi
/// needs an entry to pick.
pub fn write_models_json(
    monitor_port: u16,
    model_id: &str,
    model_name: &str,
    context_window: u64,
) -> Result<PathBuf> {
    write_models_json_at(
        models_json_path(),
        monitor_port,
        model_id,
        model_name,
        context_window,
    )
}

fn write_models_json_at(
    path: PathBuf,
    monitor_port: u16,
    model_id: &str,
    model_name: &str,
    context_window: u64,
) -> Result<PathBuf> {
    let mut root: serde_json::Value = match std::fs::read_to_string(&path) {
        Ok(text) => serde_json::from_str(&text)
            .with_context(|| format!("{} is not valid JSON", path.display()))?,
        Err(_) => serde_json::json!({}),
    };
    if !root.is_object() {
        bail!("{} is not a JSON object", path.display());
    }
    let providers = root
        .as_object_mut()
        .unwrap()
        .entry("providers")
        .or_insert_with(|| serde_json::json!({}));
    if !providers.is_object() {
        bail!("\"providers\" in {} is not an object", path.display());
    }
    providers[PROVIDER] = serde_json::json!({
        "baseUrl": format!("http://127.0.0.1:{monitor_port}/v1"),
        "api": "openai-completions",
        // The proxy needs no key, but pi hides keyless models from /model.
        "apiKey": "llama-admin-monitor",
        "models": [{
            "id": model_id,
            "name": model_name,
            "contextWindow": context_window,
            "maxTokens": std::cmp::min(context_window / 4, 32000).max(4096),
            "reasoning": false,
            "input": ["text"]
        }]
    });
    if let Some(dir) = path.parent() {
        std::fs::create_dir_all(dir)?;
    }
    let tmp = path.with_extension("json.tmp");
    std::fs::write(&tmp, serde_json::to_string_pretty(&root)?)?;
    std::fs::rename(&tmp, &path)?;
    Ok(path)
}

/// Spawns `program args` in a fresh PTY and installs it as the session,
/// replacing (and killing) any previous one.
pub fn start(
    shared: &Shared,
    program: &str,
    args: &[String],
    cwd: PathBuf,
    label: String,
    cols: u16,
    rows: u16,
) -> Result<()> {
    stop(shared);
    let pty = native_pty_system();
    let pair = pty
        .openpty(PtySize {
            rows: rows.max(4),
            cols: cols.max(20),
            pixel_width: 0,
            pixel_height: 0,
        })
        .context("cannot open a pseudo-terminal")?;
    let mut cmd = CommandBuilder::new(program);
    cmd.args(args);
    cmd.cwd(&cwd);
    cmd.env("TERM", "xterm-256color");
    cmd.env("COLORTERM", "truecolor");
    let child = pair
        .slave
        .spawn_command(cmd)
        .with_context(|| format!("cannot start {program}"))?;
    // The slave must go so the reader sees EOF when the child exits.
    drop(pair.slave);
    let writer = pair.master.take_writer()?;
    let mut reader = pair.master.try_clone_reader()?;
    let (tx, _) = broadcast::channel::<Vec<u8>>(512);
    let session = Arc::new(Session {
        master: Mutex::new(pair.master),
        writer: Mutex::new(writer),
        child: Mutex::new(child),
        tx,
        scrollback: Mutex::new(VecDeque::with_capacity(SCROLLBACK_BYTES)),
        label,
        cwd,
        exited: Mutex::new(None),
    });
    *shared.lock().unwrap() = Some(session.clone());

    // Pump PTY output into the scrollback and to subscribers.
    let pump = session.clone();
    std::thread::spawn(move || {
        let mut buf = [0u8; 8192];
        loop {
            match reader.read(&mut buf) {
                Ok(0) | Err(_) => break,
                Ok(n) => {
                    let chunk = buf[..n].to_vec();
                    {
                        let mut sb = pump.scrollback.lock().unwrap();
                        sb.extend(chunk.iter());
                        while sb.len() > SCROLLBACK_BYTES {
                            sb.pop_front();
                        }
                    }
                    let _ = pump.tx.send(chunk);
                }
            }
        }
        let code = pump
            .child
            .lock()
            .unwrap()
            .wait()
            .ok()
            .map(|s| s.exit_code() as i32);
        *pump.exited.lock().unwrap() = Some(code.unwrap_or(-1));
        let note = format!(
            "\r\n\x1b[2m[{} exited with code {}]\x1b[0m\r\n",
            pump.label,
            code.unwrap_or(-1)
        );
        let _ = pump.tx.send(note.into_bytes());
    });
    Ok(())
}

pub fn stop(shared: &Shared) {
    let session = shared.lock().unwrap().take();
    if let Some(s) = session
        && s.exited.lock().unwrap().is_none()
    {
        let _ = s.child.lock().unwrap().kill();
    }
}

impl Session {
    pub fn subscribe(&self) -> (Vec<u8>, broadcast::Receiver<Vec<u8>>) {
        // Subscribe before copying the scrollback so nothing falls between.
        let rx = self.tx.subscribe();
        let sb = self.scrollback.lock().unwrap().iter().copied().collect();
        (sb, rx)
    }

    pub fn write_input(&self, data: &[u8]) -> Result<()> {
        let mut w = self.writer.lock().unwrap();
        w.write_all(data)?;
        w.flush()?;
        Ok(())
    }

    pub fn resize(&self, cols: u16, rows: u16) -> Result<()> {
        self.master.lock().unwrap().resize(PtySize {
            rows: rows.max(4),
            cols: cols.max(20),
            pixel_width: 0,
            pixel_height: 0,
        })?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn models_json_merges_into_existing_file() {
        let dir = std::env::temp_dir().join(format!("lam-pi-test-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("models.json");
        std::fs::write(
            &path,
            r#"{"providers": {"other": {"baseUrl": "https://x", "api": "openai-completions", "models": []}}}"#,
        )
        .unwrap();
        write_models_json_at(path.clone(), 7778, "qwen3-32b", "Qwen3 32B", 128000).unwrap();
        let v: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(&path).unwrap()).unwrap();
        assert!(v["providers"]["other"].is_object(), "other provider kept");
        assert_eq!(
            v["providers"][PROVIDER]["baseUrl"],
            "http://127.0.0.1:7778/v1"
        );
        assert_eq!(v["providers"][PROVIDER]["models"][0]["id"], "qwen3-32b");
        assert_eq!(v["providers"][PROVIDER]["models"][0]["maxTokens"], 32000);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn models_json_created_when_missing() {
        let dir = std::env::temp_dir().join(format!("lam-pi-test2-{}", std::process::id()));
        let path = dir.join("agent").join("models.json");
        write_models_json_at(path.clone(), 7778, "m", "M", 8192).unwrap();
        let v: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(&path).unwrap()).unwrap();
        assert_eq!(v["providers"][PROVIDER]["models"][0]["maxTokens"], 4096);
        let _ = std::fs::remove_dir_all(&dir);
    }
}
