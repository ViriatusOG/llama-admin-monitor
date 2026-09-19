//! DeepSeek Harness (`dsh`, https://github.com/deepseek-ai/deepseek-harness)
//! next to the monitor. Unlike pi it is a web app: `dsh web` serves its
//! own UI on 127.0.0.1 only and refuses `--host 0.0.0.0`, so the monitor
//! runs it in a PTY (for its logs), forwards a LAN-facing port to it, and
//! embeds that in the dashboard. Its model provider lives in
//! `$DSH_HOME/settings.yaml` (`llm-pi-ai.providers`), which the monitor
//! refreshes with one model per preset before every start.

use anyhow::{Context, Result, bail};
use std::path::PathBuf;
use std::sync::{Arc, Mutex};

use crate::pi::{self, ModelEntry, Shared};

/// Provider id written to settings.yaml (lowercase, as dsh requires).
pub const PROVIDER: &str = "llama-admin-monitor";
/// Environment variable dsh reads the (dummy) API key from.
pub const API_KEY_ENV: &str = "LLAMA_ADMIN_MONITOR_API_KEY";
/// dsh's own listening port (loopback only).
pub const DSH_PORT: u16 = 3080;

/// The LAN-facing forwarder: `0.0.0.0:<port>` -> `127.0.0.1:DSH_PORT`.
pub struct Proxy {
    pub port: u16,
    task: tokio::task::JoinHandle<()>,
}

impl Drop for Proxy {
    fn drop(&mut self) {
        self.task.abort();
    }
}

pub type SharedProxy = Arc<Mutex<Option<Proxy>>>;

#[derive(Debug, Clone, Default, serde::Serialize)]
pub struct Status {
    pub installed: bool,
    pub path: Option<String>,
    pub version: Option<String>,
    pub npm: Option<String>,
    pub running: bool,
    pub label: Option<String>,
    pub cwd: Option<String>,
    pub exit_code: Option<i32>,
    pub proxy_port: Option<u16>,
    pub settings_yaml: String,
    pub provider: &'static str,
}

pub fn dsh_home() -> PathBuf {
    if let Some(h) = std::env::var_os("DSH_HOME").filter(|h| !h.is_empty()) {
        return PathBuf::from(h);
    }
    dirs::home_dir()
        .unwrap_or_else(|| PathBuf::from("."))
        .join(".dsh")
}

pub fn settings_path() -> PathBuf {
    dsh_home().join("settings.yaml")
}

pub fn find_dsh() -> Option<PathBuf> {
    pi::find_program("dsh")
}

pub fn status(session: &Shared, proxy: &SharedProxy) -> Status {
    let path = find_dsh();
    let s = session.lock().unwrap().clone();
    let (running, label, cwd, exit_code) = match &s {
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
        version: path.as_ref().and_then(pi::program_version),
        path: path.map(|p| p.display().to_string()),
        npm: pi::find_program("npm").map(|p| p.display().to_string()),
        running,
        label,
        cwd,
        exit_code,
        proxy_port: proxy.lock().unwrap().as_ref().map(|p| p.port),
        settings_yaml: settings_path().display().to_string(),
        provider: PROVIDER,
    }
}

/// Merges the monitor's provider into `settings.yaml` under
/// `llm-pi-ai.providers`, keeping everything else. The compat switches
/// are what llama-server needs: no `developer` role and `max_tokens`.
pub fn write_settings(monitor_port: u16, models: &[ModelEntry]) -> Result<PathBuf> {
    write_settings_at(settings_path(), monitor_port, models)
}

fn write_settings_at(path: PathBuf, monitor_port: u16, models: &[ModelEntry]) -> Result<PathBuf> {
    use serde_yaml_ng::{Mapping, Value};
    let mut root: Value = match std::fs::read_to_string(&path) {
        Ok(text) if !text.trim().is_empty() => serde_yaml_ng::from_str(&text)
            .with_context(|| format!("{} is not valid YAML", path.display()))?,
        _ => Value::Mapping(Mapping::new()),
    };
    if root.is_null() {
        root = Value::Mapping(Mapping::new());
    }
    let Value::Mapping(top) = &mut root else {
        bail!("{} is not a YAML mapping", path.display());
    };
    let llm = top
        .entry(Value::String("llm-pi-ai".into()))
        .or_insert_with(|| Value::Mapping(Mapping::new()));
    if llm.is_null() {
        *llm = Value::Mapping(Mapping::new());
    }
    let Value::Mapping(llm) = llm else {
        bail!("\"llm-pi-ai\" in {} is not a mapping", path.display());
    };
    let providers = llm
        .entry(Value::String("providers".into()))
        .or_insert_with(|| Value::Mapping(Mapping::new()));
    if providers.is_null() {
        *providers = Value::Mapping(Mapping::new());
    }
    let Value::Mapping(providers) = providers else {
        bail!(
            "\"llm-pi-ai.providers\" in {} is not a mapping",
            path.display()
        );
    };
    let model_list: Vec<serde_json::Value> = models
        .iter()
        .map(|m| {
            serde_json::json!({
                "id": m.id,
                "name": m.name,
                "contextWindow": m.context_window,
                "maxTokens": std::cmp::min(m.context_window / 4, 32000).max(4096),
            })
        })
        .collect();
    let provider = serde_json::json!({
        "apiKeyEnv": API_KEY_ENV,
        "api": "openai-completions",
        "baseURL": format!("http://127.0.0.1:{monitor_port}/v1"),
        "compat": {
            "supportsDeveloperRole": false,
            "maxTokensField": "max_tokens"
        },
        "models": model_list
    });
    let provider: Value = serde_yaml_ng::to_value(provider)?;
    providers.insert(Value::String(PROVIDER.into()), provider);
    if let Some(dir) = path.parent() {
        std::fs::create_dir_all(dir)?;
    }
    let tmp = path.with_extension("yaml.tmp");
    std::fs::write(&tmp, serde_yaml_ng::to_string(&root)?)?;
    std::fs::rename(&tmp, &path)?;
    Ok(path)
}

/// Authorities the dsh browser-trust fence should accept: whatever host
/// the dashboard was reached on (with the proxy port), this machine's
/// addresses and names, and localhost.
pub fn trusted_hosts(request_host: Option<&str>, proxy_port: u16) -> Vec<String> {
    let mut out: Vec<String> = Vec::new();
    let mut push = |h: &str| {
        let h = h.trim();
        if h.is_empty() {
            return;
        }
        let v = format!("{h}:{proxy_port}");
        if !out.contains(&v) {
            out.push(v);
        }
    };
    if let Some(h) = request_host {
        push(host_without_port(h));
    }
    if let Ok(out_) = std::process::Command::new("hostname").arg("-I").output() {
        for ip in String::from_utf8_lossy(&out_.stdout).split_whitespace() {
            if !ip.contains(':') {
                push(ip);
            }
        }
    }
    if let Ok(out_) = std::process::Command::new("hostname").output() {
        push(String::from_utf8_lossy(&out_.stdout).trim());
    }
    push("localhost");
    push("127.0.0.1");
    out
}

pub fn host_without_port(host: &str) -> &str {
    let h = host.trim();
    if let Some(rest) = h.strip_prefix('[') {
        return rest.split(']').next().unwrap_or(h);
    }
    h.rsplit_once(':')
        .filter(|(_, port)| port.chars().all(|c| c.is_ascii_digit()))
        .map(|(host, _)| host)
        .unwrap_or(h)
}

/// Starts forwarding `0.0.0.0:port` to `127.0.0.1:DSH_PORT`. Plain TCP,
/// so HTTP and WebSocket traffic both pass through untouched.
pub async fn start_proxy(shared: &SharedProxy, port: u16) -> Result<()> {
    stop_proxy(shared);
    let listener = tokio::net::TcpListener::bind(("0.0.0.0", port))
        .await
        .with_context(|| format!("cannot listen on port {port} for the dsh proxy"))?;
    let task = tokio::spawn(async move {
        loop {
            let Ok((mut inbound, _)) = listener.accept().await else {
                continue;
            };
            tokio::spawn(async move {
                let Ok(mut outbound) =
                    tokio::net::TcpStream::connect(("127.0.0.1", DSH_PORT)).await
                else {
                    return;
                };
                let _ = tokio::io::copy_bidirectional(&mut inbound, &mut outbound).await;
            });
        }
    });
    *shared.lock().unwrap() = Some(Proxy { port, task });
    Ok(())
}

pub fn stop_proxy(shared: &SharedProxy) {
    // Dropping the Proxy aborts its task.
    shared.lock().unwrap().take();
}

#[cfg(test)]
mod tests {
    use super::*;

    fn entry(id: &str, ctx: u64) -> ModelEntry {
        ModelEntry {
            id: id.to_string(),
            name: id.to_string(),
            context_window: ctx,
            preset_id: String::new(),
        }
    }

    #[test]
    fn settings_yaml_merges() {
        let dir = std::env::temp_dir().join(format!("lam-dsh-test-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("settings.yaml");
        std::fs::write(
            &path,
            "llm-deepseek:\n  reasoningEffort: max\nllm-pi-ai:\n  providers:\n    other:\n      api: openai-completions\n      baseURL: https://x/v1\n      models:\n        - id: m\n",
        )
        .unwrap();
        write_settings_at(path.clone(), 7778, &[entry("Big 128k", 131072)]).unwrap();
        let v: serde_yaml_ng::Value =
            serde_yaml_ng::from_str(&std::fs::read_to_string(&path).unwrap()).unwrap();
        assert_eq!(v["llm-deepseek"]["reasoningEffort"], "max");
        assert!(v["llm-pi-ai"]["providers"]["other"].is_mapping());
        let mine = &v["llm-pi-ai"]["providers"][PROVIDER];
        assert_eq!(mine["baseURL"], "http://127.0.0.1:7778/v1");
        assert_eq!(mine["apiKeyEnv"], API_KEY_ENV);
        assert_eq!(mine["compat"]["maxTokensField"], "max_tokens");
        assert_eq!(mine["models"][0]["id"], "Big 128k");
        assert_eq!(mine["models"][0]["maxTokens"], 32000);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn settings_yaml_created_when_missing() {
        let dir = std::env::temp_dir().join(format!("lam-dsh-test2-{}", std::process::id()));
        let path = dir.join("settings.yaml");
        write_settings_at(path.clone(), 7778, &[entry("m", 8192)]).unwrap();
        let v: serde_yaml_ng::Value =
            serde_yaml_ng::from_str(&std::fs::read_to_string(&path).unwrap()).unwrap();
        assert_eq!(
            v["llm-pi-ai"]["providers"][PROVIDER]["models"][0]["maxTokens"],
            4096
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn host_port_split() {
        assert_eq!(host_without_port("192.168.0.199:7778"), "192.168.0.199");
        assert_eq!(host_without_port("al"), "al");
        assert_eq!(host_without_port("[::1]:7778"), "::1");
        let hosts = trusted_hosts(Some("192.168.0.199:7778"), 7779);
        assert_eq!(hosts[0], "192.168.0.199:7779");
        assert!(hosts.contains(&"localhost:7779".to_string()));
    }
}
