use std::path::PathBuf;
use std::sync::Arc;
use warp::Filter;

use crate::config::AppConfig;
use crate::gpu::env::{self as gpu_env, GPU_ARCHITECTURES, GpuEnv};
use crate::llama::bench;
use crate::llama::server::{self, ServerConfig};
use crate::models;
use crate::models::hf;
use crate::presets::{self, ModelPreset};
use crate::state::{self as app_state, AppState, UiSettings};
use crate::update;

pub fn api_routes(
    state: AppState,
    app_config: Arc<AppConfig>,
) -> impl Filter<Extract = (impl warp::Reply,), Error = warp::Rejection> + Clone {
    let app_config_bench = app_config.clone();
    let start = api_start(state.clone(), app_config.clone());
    let stop = api_stop(state.clone());
    let get_presets = api_get_presets(state.clone());
    let create_preset = api_create_preset(state.clone());
    let update_preset = api_update_preset(state.clone());
    let delete_preset = api_delete_preset(state.clone());
    let reset_presets = api_reset_presets(state.clone());
    let get_models = api_get_models(state.clone());
    let refresh_models = api_refresh_models(state.clone());
    let delete_model = api_delete_model(state.clone());
    let bench_run = api_bench_run(state.clone(), app_config_bench);
    let get_gpu_env = api_get_gpu_env(state.clone());
    let put_gpu_env = api_put_gpu_env(state.clone());
    let get_settings = api_get_settings(state.clone(), app_config.clone());
    let put_settings = api_put_settings(state.clone());
    let browse = api_browse();
    let hf_search = api_hf_search();
    let hf_files = api_hf_files();
    let hf_download = api_hf_download(state.clone());
    let v1_proxy = api_v1_proxy(state.clone());
    let app_update_check = api_app_update_check();
    let app_logs = api_app_logs();
    let devices = api_devices(state.clone(), app_config.clone());
    let builds = api_builds(state.clone(), app_config.clone());
    let pi = api_pi(state.clone(), app_config.clone());
    let dsh = api_dsh(state.clone(), app_config.clone());
    let app_update_apply = api_app_update_apply(state);
    // Boxed so the outer .or() chain stays shallow enough for the compiler.
    let app = app_update_check
        .or(app_update_apply)
        .or(app_logs)
        .or(devices)
        .or(builds)
        .or(pi)
        .or(dsh)
        .boxed();

    start
        .or(stop)
        .or(create_preset)
        .or(update_preset)
        .or(delete_preset)
        .or(reset_presets)
        .or(get_presets)
        .or(get_models)
        .or(refresh_models)
        .or(delete_model)
        .or(bench_run)
        .or(put_gpu_env)
        .or(get_gpu_env)
        .or(put_settings)
        .or(get_settings)
        .or(browse)
        .or(hf_search)
        .or(hf_files)
        .or(hf_download)
        .or(v1_proxy)
        .or(app)
}

fn api_start(
    state: AppState,
    app_config: Arc<AppConfig>,
) -> impl Filter<Extract = (impl warp::Reply,), Error = warp::Rejection> + Clone {
    warp::path!("api" / "start")
        .and(warp::post())
        .and(warp::body::json())
        .and_then(move |config: ServerConfig| {
            let state = state.clone();
            let app_config = app_config.clone();
            async move {
                // Build effective config: UI settings override CLI defaults
                let ui = state.ui_settings.lock().unwrap().clone();
                let mut eff_config = (*app_config).clone();
                if !ui.llama_server_path.is_empty() {
                    eff_config.llama_server_path = PathBuf::from(&ui.llama_server_path);
                }
                if !ui.llama_server_cwd.is_empty() {
                    eff_config.llama_server_cwd = PathBuf::from(&ui.llama_server_cwd);
                }
                match server::start_server(&state, config, &eff_config).await {
                    Ok(()) => Ok::<_, warp::Rejection>(warp::reply::json(
                        &serde_json::json!({"ok": true}),
                    )),
                    Err(e) => {
                        crate::applog::error(format!("Start failed: {e:#}"));
                        Ok(warp::reply::json(
                            &serde_json::json!({"ok": false, "error": e.to_string()}),
                        ))
                    }
                }
            }
        })
}

fn api_stop(
    state: AppState,
) -> impl Filter<Extract = (impl warp::Reply,), Error = warp::Rejection> + Clone {
    warp::path!("api" / "stop")
        .and(warp::post())
        .and_then(move || {
            let state = state.clone();
            async move {
                match server::stop_server(&state).await {
                    Ok(()) => Ok::<_, warp::Rejection>(warp::reply::json(
                        &serde_json::json!({"ok": true}),
                    )),
                    Err(e) => Ok(warp::reply::json(
                        &serde_json::json!({"ok": false, "error": e.to_string()}),
                    )),
                }
            }
        })
}

fn api_get_presets(
    state: AppState,
) -> impl Filter<Extract = (impl warp::Reply,), Error = warp::Rejection> + Clone {
    warp::path!("api" / "presets")
        .and(warp::get())
        .map(move || {
            let presets = state.presets.lock().unwrap().clone();
            warp::reply::json(&presets)
        })
}

fn api_create_preset(
    state: AppState,
) -> impl Filter<Extract = (impl warp::Reply,), Error = warp::Rejection> + Clone {
    warp::path!("api" / "presets")
        .and(warp::post())
        .and(warp::body::json())
        .map(move |preset: ModelPreset| {
            let mut presets = state.presets.lock().unwrap();
            presets.push(preset.clone());
            let _ = presets::save_presets(&state.presets_path, &presets);
            warp::reply::json(&serde_json::json!({"ok": true, "preset": preset}))
        })
}

fn api_update_preset(
    state: AppState,
) -> impl Filter<Extract = (impl warp::Reply,), Error = warp::Rejection> + Clone {
    warp::path!("api" / "presets" / String)
        .and(warp::put())
        .and(warp::body::json())
        .map(move |id: String, updated: ModelPreset| {
            let mut presets = state.presets.lock().unwrap();
            if let Some(existing) = presets.iter_mut().find(|p| p.id == id) {
                *existing = updated.clone();
                let _ = presets::save_presets(&state.presets_path, &presets);
                warp::reply::json(&serde_json::json!({"ok": true, "preset": updated}))
            } else {
                warp::reply::json(&serde_json::json!({"ok": false, "error": "preset not found"}))
            }
        })
}

fn api_delete_preset(
    state: AppState,
) -> impl Filter<Extract = (impl warp::Reply,), Error = warp::Rejection> + Clone {
    warp::path!("api" / "presets" / String)
        .and(warp::delete())
        .map(move |id: String| {
            let mut presets = state.presets.lock().unwrap();
            let before = presets.len();
            presets.retain(|p| p.id != id);
            if presets.len() < before {
                let _ = presets::save_presets(&state.presets_path, &presets);
                warp::reply::json(&serde_json::json!({"ok": true}))
            } else {
                warp::reply::json(&serde_json::json!({"ok": false, "error": "preset not found"}))
            }
        })
}

fn api_reset_presets(
    state: AppState,
) -> impl Filter<Extract = (impl warp::Reply,), Error = warp::Rejection> + Clone {
    warp::path!("api" / "presets" / "reset")
        .and(warp::post())
        .map(move || {
            let defaults = presets::default_presets();
            let mut presets = state.presets.lock().unwrap();
            *presets = defaults;
            let _ = presets::save_presets(&state.presets_path, &presets);
            warp::reply::json(&serde_json::json!({"ok": true}))
        })
}

fn api_get_models(
    state: AppState,
) -> impl Filter<Extract = (impl warp::Reply,), Error = warp::Rejection> + Clone {
    warp::path!("api" / "models").and(warp::get()).map(move || {
        let models = state.discovered_models.lock().unwrap().clone();
        warp::reply::json(&models)
    })
}

fn api_refresh_models(
    state: AppState,
) -> impl Filter<Extract = (impl warp::Reply,), Error = warp::Rejection> + Clone {
    warp::path!("api" / "models" / "refresh")
        .and(warp::post())
        .map(move || {
            let dir_opt = state.models_dir.lock().unwrap().clone();
            if let Some(dir) = dir_opt {
                match models::scan_models_dir(&dir) {
                    Ok(discovered) => {
                        let count = discovered.len();
                        *state.discovered_models.lock().unwrap() = discovered;
                        warp::reply::json(&serde_json::json!({"ok": true, "count": count}))
                    }
                    Err(e) => {
                        warp::reply::json(&serde_json::json!({"ok": false, "error": e.to_string()}))
                    }
                }
            } else {
                warp::reply::json(
                    &serde_json::json!({"ok": false, "error": "no models directory configured (use --models-dir)"}),
                )
            }
        })
}

fn api_get_gpu_env(
    state: AppState,
) -> impl Filter<Extract = (impl warp::Reply,), Error = warp::Rejection> + Clone {
    warp::path!("api" / "gpu-env")
        .and(warp::get())
        .map(move || {
            let env = state.gpu_env.lock().unwrap().clone();
            let detected = gpu_env::detect_gpus();
            warp::reply::json(&serde_json::json!({
                "env": env,
                "architectures": GPU_ARCHITECTURES,
                "detected": detected,
            }))
        })
}

fn api_put_gpu_env(
    state: AppState,
) -> impl Filter<Extract = (impl warp::Reply,), Error = warp::Rejection> + Clone {
    warp::path!("api" / "gpu-env")
        .and(warp::put())
        .and(warp::body::json())
        .map(move |updated: GpuEnv| {
            let mut env = state.gpu_env.lock().unwrap();
            *env = updated;
            let _ = gpu_env::save_gpu_env(&state.gpu_env_path, &env);
            warp::reply::json(&serde_json::json!({"ok": true}))
        })
}

fn api_get_settings(
    state: AppState,
    app_config: Arc<AppConfig>,
) -> impl Filter<Extract = (impl warp::Reply,), Error = warp::Rejection> + Clone {
    warp::path!("api" / "settings")
        .and(warp::get())
        .map(move || {
            // Empty UI fields mean "use the CLI value"; report that value so
            // the Settings dialog shows what is actually in effect.
            let mut settings = state.ui_settings.lock().unwrap().clone();
            if settings.llama_server_path.is_empty() {
                settings.llama_server_path = app_config.llama_server_path.display().to_string();
            }
            if settings.llama_server_cwd.is_empty() {
                settings.llama_server_cwd = app_config.llama_server_cwd.display().to_string();
            }
            if settings.models_dir.is_empty()
                && let Some(dir) = &app_config.models_dir
            {
                settings.models_dir = dir.display().to_string();
            }
            warp::reply::json(&settings)
        })
}

fn api_put_settings(
    state: AppState,
) -> impl Filter<Extract = (impl warp::Reply,), Error = warp::Rejection> + Clone {
    warp::path!("api" / "settings")
        .and(warp::put())
        .and(warp::body::json())
        .map(move |updated: UiSettings| {
            // Check if models_dir changed to rescan
            let old_dir = state.ui_settings.lock().unwrap().models_dir.clone();
            let new_dir = updated.models_dir.clone();

            let mut settings = state.ui_settings.lock().unwrap();
            *settings = updated;
            let _ = app_state::save_ui_settings(&state.ui_settings_path, &settings);
            drop(settings);

            // Rescan models and update the live models_dir if it changed
            if new_dir != old_dir && !new_dir.is_empty() {
                let new_path = PathBuf::from(&new_dir);
                if let Ok(discovered) = crate::models::scan_models_dir(&new_path) {
                    *state.discovered_models.lock().unwrap() = discovered;
                }
                *state.models_dir.lock().unwrap() = Some(new_path);
            }

            warp::reply::json(&serde_json::json!({"ok": true}))
        })
}

fn api_browse() -> impl Filter<Extract = (impl warp::Reply,), Error = warp::Rejection> + Clone {
    warp::path!("api" / "browse")
        .and(warp::get())
        .and(warp::query::<std::collections::HashMap<String, String>>())
        .map(|query: std::collections::HashMap<String, String>| {
            let requested = query.get("path").cloned().unwrap_or_default();
            let filter = query.get("filter").cloned().unwrap_or_default();

            let dir = if requested.is_empty() {
                dirs::home_dir().unwrap_or_else(|| PathBuf::from("/"))
            } else {
                PathBuf::from(&requested)
            };

            let dir = match dir.canonicalize() {
                Ok(p) => p,
                Err(_) => {
                    return warp::reply::json(&serde_json::json!({
                        "path": requested,
                        "error": "Path not found"
                    }));
                }
            };

            if !dir.is_dir() {
                return warp::reply::json(&serde_json::json!({
                    "path": dir.display().to_string(),
                    "error": "Not a directory"
                }));
            }

            let parent = dir
                .parent()
                .map(|p| p.display().to_string())
                .unwrap_or_default();

            let mut entries: Vec<serde_json::Value> = Vec::new();
            if let Ok(read_dir) = std::fs::read_dir(&dir) {
                for entry in read_dir.flatten() {
                    let name = entry.file_name().to_string_lossy().to_string();
                    // Skip hidden files
                    if name.starts_with('.') {
                        continue;
                    }
                    let meta = entry.metadata().ok();
                    let is_dir = meta.as_ref().is_some_and(|m| m.is_dir());

                    // Apply filter
                    if !is_dir && !filter.is_empty() {
                        let pass = match filter.as_str() {
                            "gguf" => name.ends_with(".gguf"),
                            "executable" => {
                                #[cfg(unix)]
                                {
                                    use std::os::unix::fs::PermissionsExt;
                                    meta.as_ref()
                                        .is_some_and(|m| m.permissions().mode() & 0o111 != 0)
                                }
                                #[cfg(not(unix))]
                                {
                                    true
                                }
                            }
                            _ => true,
                        };
                        if !pass {
                            continue;
                        }
                    }

                    let size = if is_dir {
                        0
                    } else {
                        meta.as_ref().map(|m| m.len()).unwrap_or(0)
                    };
                    let size_display = if is_dir {
                        String::new()
                    } else if size >= 1_000_000_000 {
                        format!("{:.1} GB", size as f64 / 1_000_000_000.0)
                    } else if size >= 1_000_000 {
                        format!("{:.0} MB", size as f64 / 1_000_000.0)
                    } else {
                        format!("{:.0} KB", size as f64 / 1_000.0)
                    };

                    entries.push(serde_json::json!({
                        "name": name,
                        "is_dir": is_dir,
                        "size": size,
                        "size_display": size_display,
                        "path": entry.path().display().to_string(),
                    }));
                }
            }

            // Sort: directories first, then alphabetical
            entries.sort_by(|a, b| {
                let a_dir = a["is_dir"].as_bool().unwrap_or(false);
                let b_dir = b["is_dir"].as_bool().unwrap_or(false);
                b_dir.cmp(&a_dir).then_with(|| {
                    a["name"]
                        .as_str()
                        .unwrap_or("")
                        .to_lowercase()
                        .cmp(&b["name"].as_str().unwrap_or("").to_lowercase())
                })
            });

            warp::reply::json(&serde_json::json!({
                "path": dir.display().to_string(),
                "parent": parent,
                "entries": entries,
            }))
        })
}

/// Headers that describe one hop of a connection rather than the message.
/// Forwarding them from the upstream reply would let hyper re-frame a body
/// that already carries a length or chunking declaration.
const HOP_BY_HOP: [&str; 9] = [
    "connection",
    "keep-alive",
    "proxy-authenticate",
    "proxy-authorization",
    "te",
    "trailer",
    "transfer-encoding",
    "upgrade",
    "content-length",
];

pub fn is_hop_by_hop(name: &str) -> bool {
    HOP_BY_HOP.iter().any(|h| h.eq_ignore_ascii_case(name)) || name.eq_ignore_ascii_case("host")
}

/// One pooled client for every proxied request; building a client per
/// request throws away its connection pool.
fn proxy_client() -> &'static reqwest::Client {
    static CLIENT: std::sync::OnceLock<reqwest::Client> = std::sync::OnceLock::new();
    CLIENT.get_or_init(|| {
        reqwest::Client::builder()
            .no_proxy()
            .build()
            .expect("reqwest client")
    })
}

pub fn api_v1_proxy(
    state: AppState,
) -> impl Filter<Extract = (impl warp::Reply,), Error = warp::Rejection> + Clone {
    warp::path("v1")
        .and(warp::path::tail())
        .and(warp::method())
        .and(
            warp::query::raw()
                .map(Some)
                .or_else(|_| async { Ok::<(Option<String>,), std::convert::Infallible>((None,)) }),
        )
        .and(warp::header::headers_cloned())
        .and(warp::body::bytes())
        .and_then(
            move |tail: warp::path::Tail,
                  method: warp::http::Method,
                  query: Option<String>,
                  headers: warp::http::HeaderMap,
                  body: bytes::Bytes| {
                let state = state.clone();
                async move {
                    let port = {
                        let cfg = state.server_config.lock().unwrap();
                        cfg.as_ref().map(|c| c.port).unwrap_or(8080)
                    };

                    let path = tail.as_str();
                    let url = if let Some(q) = query {
                        format!("http://127.0.0.1:{}/v1/{}?{}", port, path, q)
                    } else {
                        format!("http://127.0.0.1:{}/v1/{}", port, path)
                    };

                    let reqwest_method =
                        reqwest::Method::from_bytes(method.as_str().as_bytes()).unwrap();
                    let mut req = proxy_client()
                        .request(reqwest_method, &url)
                        .body(body.to_vec());

                    for (k, v) in headers.iter() {
                        if !is_hop_by_hop(k.as_str()) {
                            req = req.header(k.as_str(), v.as_bytes());
                        }
                    }

                    match req.send().await {
                        Ok(resp) => {
                            let status = resp.status().as_u16();
                            let mut builder = warp::http::Response::builder().status(status);

                            for (k, v) in resp.headers().iter() {
                                if !is_hop_by_hop(k.as_str()) {
                                    builder = builder.header(k.as_str(), v.as_bytes());
                                }
                            }

                            let stream = resp.bytes_stream();
                            let body_stream = warp::hyper::Body::wrap_stream(stream);
                            Ok::<_, warp::Rejection>(builder.body(body_stream).unwrap())
                        }
                        Err(e) => {
                            let err = format!(
                                r#"{{"error":{{"message":"{}","type":"proxy_error"}}}}"#,
                                e.to_string().replace('"', "'")
                            );
                            Ok(warp::http::Response::builder()
                                .status(502)
                                .header("content-type", "application/json")
                                .body(warp::hyper::Body::from(err))
                                .unwrap())
                        }
                    }
                }
            },
        )
}

fn api_hf_search() -> impl Filter<Extract = (impl warp::Reply,), Error = warp::Rejection> + Clone {
    warp::path!("api" / "hf" / "search")
        .and(warp::get())
        .and(warp::query::<std::collections::HashMap<String, String>>())
        .and_then(
            |query: std::collections::HashMap<String, String>| async move {
                let q = query.get("q").cloned().unwrap_or_default();
                if q.trim().is_empty() {
                    return Ok::<_, warp::Rejection>(warp::reply::json(
                        &serde_json::json!({"results": []}),
                    ));
                }
                match hf::search_hf_models(&q).await {
                    Ok(results) => Ok(warp::reply::json(&serde_json::json!({"results": results}))),
                    Err(e) => Ok(warp::reply::json(
                        &serde_json::json!({"error": e.to_string()}),
                    )),
                }
            },
        )
}

fn api_hf_files() -> impl Filter<Extract = (impl warp::Reply,), Error = warp::Rejection> + Clone {
    warp::path!("api" / "hf" / "files")
        .and(warp::get())
        .and(warp::query::<std::collections::HashMap<String, String>>())
        .and_then(
            |query: std::collections::HashMap<String, String>| async move {
                let repo = query.get("repo").cloned().unwrap_or_default();
                if repo.trim().is_empty() {
                    return Ok::<_, warp::Rejection>(warp::reply::json(
                        &serde_json::json!({"error": "missing repo"}),
                    ));
                }
                match hf::list_hf_gguf_files(&repo).await {
                    Ok(files) => Ok(warp::reply::json(&serde_json::json!({"files": files}))),
                    Err(e) => Ok(warp::reply::json(
                        &serde_json::json!({"error": e.to_string()}),
                    )),
                }
            },
        )
}

fn api_hf_download(
    state: AppState,
) -> impl Filter<Extract = (impl warp::Reply,), Error = warp::Rejection> + Clone {
    warp::path!("api" / "hf" / "download")
        .and(warp::post())
        .and(warp::body::json())
        .and(warp::any().map(move || state.clone()))
        .and_then(|body: serde_json::Value, state: AppState| async move {
            let repo = body
                .get("repo")
                .and_then(|v| v.as_str())
                .unwrap_or("")
                .to_string();
            let filename = body
                .get("filename")
                .and_then(|v| v.as_str())
                .unwrap_or("")
                .to_string();
            // Optional companion projector, downloaded after the model.
            let companion = body
                .get("companion")
                .and_then(|v| v.as_str())
                .unwrap_or("")
                .to_string();
            if repo.is_empty() || filename.is_empty() {
                return Ok::<_, warp::Rejection>(warp::reply::json(&serde_json::json!({
                    "ok": false,
                    "error": "repo and filename required"
                })));
            }
            let mut files = vec![filename];
            if !companion.is_empty() {
                files.push(companion);
            }
            let dest_dir = match state.models_dir.lock().unwrap().clone() {
                Some(d) => d,
                None => {
                    return Ok(warp::reply::json(&serde_json::json!({
                        "ok": false,
                        "error": "models directory not configured"
                    })));
                }
            };
            let progress = state.hf_download_progress.clone();
            let models_dir_state = state.models_dir.clone();
            let discovered_models = state.discovered_models.clone();
            tokio::spawn(async move {
                crate::applog::info(format!("Downloading {} from {repo}", files.join(", ")));
                hf::download_hf_files(repo, files, dest_dir, progress.clone()).await;
                match progress
                    .lock()
                    .unwrap()
                    .as_ref()
                    .and_then(|p| p.error.clone())
                {
                    Some(err) => crate::applog::error(format!("Download failed: {err}")),
                    None => crate::applog::info("Download finished"),
                }
                let dir_opt = models_dir_state.lock().unwrap().clone();
                if let Some(dir) = dir_opt
                    && let Ok(discovered) = crate::models::scan_models_dir(&dir)
                {
                    *discovered_models.lock().unwrap() = discovered;
                }
            });
            Ok(warp::reply::json(&serde_json::json!({"ok": true})))
        })
}

fn api_delete_model(
    state: AppState,
) -> impl Filter<Extract = (impl warp::Reply,), Error = warp::Rejection> + Clone {
    warp::path!("api" / "models" / "delete")
        .and(warp::post())
        .and(warp::body::json())
        .map(move |body: serde_json::Value| {
            let filename = body
                .get("filename")
                .and_then(|v| v.as_str())
                .unwrap_or("")
                .to_string();

            // Reject anything that isn't a bare filename -- no path traversal.
            if filename.is_empty()
                || filename.contains('/')
                || filename.contains("..")
                || !filename.ends_with(".gguf")
            {
                return warp::reply::json(&serde_json::json!({
                    "ok": false,
                    "error": "invalid filename"
                }));
            }

            let dir_opt = state.models_dir.lock().unwrap().clone();
            let dir = match dir_opt {
                Some(d) => d,
                None => {
                    return warp::reply::json(&serde_json::json!({
                        "ok": false,
                        "error": "no models directory configured"
                    }));
                }
            };

            let target = dir.join(&filename);
            // Re-verify the resolved path is actually inside the models dir.
            let inside = target
                .canonicalize()
                .ok()
                .and_then(|t| dir.canonicalize().ok().map(|d| t.starts_with(d)))
                .unwrap_or(false);
            if !inside {
                return warp::reply::json(&serde_json::json!({
                    "ok": false,
                    "error": "file not found in models directory"
                }));
            }

            match std::fs::remove_file(&target) {
                Ok(()) => {
                    if let Ok(discovered) = crate::models::scan_models_dir(&dir) {
                        *state.discovered_models.lock().unwrap() = discovered;
                    }
                    warp::reply::json(&serde_json::json!({"ok": true}))
                }
                Err(e) => warp::reply::json(&serde_json::json!({
                    "ok": false,
                    "error": e.to_string()
                })),
            }
        })
}

fn api_bench_run(
    state: AppState,
    app_config: Arc<AppConfig>,
) -> impl Filter<Extract = (impl warp::Reply,), Error = warp::Rejection> + Clone {
    warp::path!("api" / "bench" / "run")
        .and(warp::post())
        .and(warp::body::json())
        .and(warp::any().map(move || state.clone()))
        .and(warp::any().map(move || app_config.clone()))
        .map(
            |body: serde_json::Value, state: AppState, app_config: Arc<AppConfig>| {
                if state.bench_progress.lock().unwrap().running {
                    return warp::reply::json(&serde_json::json!({
                        "ok": false,
                        "error": "a benchmark is already running"
                    }));
                }
                if *state.server_running.lock().unwrap() {
                    return warp::reply::json(&serde_json::json!({
                        "ok": false,
                        "error": "stop the server before benchmarking -- it needs the GPUs"
                    }));
                }

                let model_path = body
                    .get("model_path")
                    .and_then(|v| v.as_str())
                    .unwrap_or("")
                    .to_string();
                if model_path.is_empty() {
                    return warp::reply::json(&serde_json::json!({
                        "ok": false,
                        "error": "model_path required"
                    }));
                }

                let splits: Vec<String> = body
                    .get("splits")
                    .and_then(|v| v.as_array())
                    .map(|arr| {
                        arr.iter()
                            .filter_map(|s| s.as_str().map(|s| s.to_string()))
                            .collect()
                    })
                    .unwrap_or_default();
                if splits.is_empty() {
                    return warp::reply::json(&serde_json::json!({
                        "ok": false,
                        "error": "at least one tensor split required"
                    }));
                }

                // Helper to parse comma-separated integers or use preset defaults
                let parse_ints = |key: &str, default_val: i32| -> Vec<i32> {
                    let vals: Vec<i32> = body
                        .get(key)
                        .and_then(|v| v.as_array())
                        .map(|arr| {
                            arr.iter()
                                .filter_map(|n| n.as_i64().map(|n| n as i32))
                                .collect()
                        })
                        .unwrap_or_default();
                    if vals.is_empty() {
                        vec![default_val]
                    } else {
                        vals
                    }
                };

                // Fall back to preset defaults if user didn't specify arrays for them
                // With no thread count configured, llama-bench's own
                // default is the physical core count; mirror that.
                let cpu_threads = std::thread::available_parallelism()
                    .map(|n| n.get() as u32)
                    .unwrap_or(8);
                let (default_batch, default_ubatch, default_threads) = {
                    let cfg = state.server_config.lock().unwrap();
                    if let Some(c) = cfg.as_ref() {
                        (
                            c.batch_size,
                            c.ubatch_size,
                            c.threads.unwrap_or(cpu_threads),
                        )
                    } else {
                        (2048, 512, cpu_threads)
                    }
                };

                let batch_sizes = parse_ints("batch_sizes", default_batch as i32);
                let ubatch_sizes = parse_ints("ubatch_sizes", default_ubatch as i32);
                let thread_counts = parse_ints("threads", default_threads as i32);

                let gpu_layers = body
                    .get("gpu_layers")
                    .and_then(|v| v.as_i64())
                    .unwrap_or(999) as i32;

                // llama-bench sits beside llama-server in whichever build the
                // active preset uses (an installed build, or the configured one).
                let backend = body
                    .get("backend")
                    .and_then(|v| v.as_str())
                    .unwrap_or("")
                    .to_string();
                let server_path = if backend.starts_with("build:") {
                    match server::launch_target(&app_config, &backend) {
                        Ok(target) => target.binary.display().to_string(),
                        Err(e) => {
                            return warp::reply::json(&serde_json::json!({
                                "ok": false,
                                "error": format!("{e:#}")
                            }));
                        }
                    }
                } else {
                    let ui = state.ui_settings.lock().unwrap();
                    if ui.llama_server_path.is_empty() {
                        app_config.llama_server_path.display().to_string()
                    } else {
                        ui.llama_server_path.clone()
                    }
                };
                let bench_bin = bench::bench_binary_path(&server_path);
                if !bench_bin.exists() {
                    return warp::reply::json(&serde_json::json!({
                        "ok": false,
                        "error": format!("llama-bench not found at {}", bench_bin.display())
                    }));
                }

                let progress = state.bench_progress.clone();
                tokio::spawn(async move {
                    let spec = bench::SweepSpec {
                        model_path,
                        splits,
                        batch_sizes,
                        ubatch_sizes,
                        thread_counts,
                        gpu_layers,
                    };
                    bench::run_benchmark_sweep(bench_bin, spec, progress).await;
                });

                warp::reply::json(&serde_json::json!({"ok": true}))
            },
        )
}

fn api_app_update_check()
-> impl Filter<Extract = (impl warp::Reply,), Error = warp::Rejection> + Clone {
    warp::path!("api" / "app" / "update" / "check")
        .and(warp::get())
        .and(warp::query::<std::collections::HashMap<String, String>>())
        .and_then(|q: std::collections::HashMap<String, String>| async move {
            let force = q.get("force").is_some_and(|v| v == "1" || v == "true");
            let reply = match update::check_updates(force).await {
                Ok(status) => warp::reply::json(&status),
                Err(e) => warp::reply::json(&serde_json::json!({"error": format!("{e:#}")})),
            };
            Ok::<_, warp::Rejection>(reply)
        })
}

fn api_app_update_apply(
    state: AppState,
) -> impl Filter<Extract = (impl warp::Reply,), Error = warp::Rejection> + Clone {
    warp::path!("api" / "app" / "update" / "apply")
        .and(warp::post())
        .and(warp::body::json())
        .and(warp::any().map(move || state.clone()))
        .and_then(|body: serde_json::Value, state: AppState| async move {
            let track = body
                .get("track")
                .and_then(|v| v.as_str())
                .unwrap_or("")
                .to_string();
            if track != "main" && track != "beta" {
                return Ok::<_, warp::Rejection>(warp::reply::json(
                    &serde_json::json!({"ok": false, "error": "track must be main or beta"}),
                ));
            }
            if state.update_phase.lock().unwrap().is_some() {
                return Ok(warp::reply::json(
                    &serde_json::json!({"ok": false, "error": "an update is already in progress"}),
                ));
            }
            *state.update_phase.lock().unwrap() = Some("Starting".to_string());
            // The work runs in the background; progress and errors reach the
            // UI through the `app_update` field of the WebSocket payload.
            tokio::spawn(async move {
                if let Err(e) = update::apply_update(state.clone(), track).await {
                    crate::applog::error(format!("Update failed: {e:#}"));
                    *state.update_phase.lock().unwrap() = Some(format!("Error: {e:#}"));
                    // Leave the error visible briefly, then clear so a retry
                    // is possible.
                    tokio::time::sleep(std::time::Duration::from_secs(8)).await;
                    *state.update_phase.lock().unwrap() = None;
                }
            });
            Ok(warp::reply::json(&serde_json::json!({"ok": true})))
        })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hop_by_hop_headers_are_stripped_case_insensitively() {
        for h in [
            "Transfer-Encoding",
            "content-length",
            "Connection",
            "Host",
            "keep-alive",
        ] {
            assert!(is_hop_by_hop(h), "{h}");
        }
        for h in [
            "Content-Type",
            "authorization",
            "x-request-id",
            "cache-control",
        ] {
            assert!(!is_hop_by_hop(h), "{h}");
        }
    }
}

/// The monitor's own event log, for the Logs page. `after` is the last
/// sequence number the client has; omit it for the most recent entries.
fn api_app_logs() -> impl Filter<Extract = (impl warp::Reply,), Error = warp::Rejection> + Clone {
    warp::path!("api" / "app" / "logs")
        .and(warp::get())
        .and(warp::query::<std::collections::HashMap<String, String>>())
        .map(|q: std::collections::HashMap<String, String>| {
            let after = q.get("after").and_then(|v| v.parse::<u64>().ok());
            let entries = match after {
                Some(after) => crate::applog::entries_after(after, 2000),
                None => {
                    let latest = crate::applog::latest_seq();
                    crate::applog::entries_after(latest.saturating_sub(500), 500)
                }
            };
            warp::reply::json(&serde_json::json!({
                "latest_seq": crate::applog::latest_seq(),
                "entries": entries,
            }))
        })
}

/// Offload devices as llama-server sees them, for the preset editor's
/// device picker. Honours the UI-configured binary path like a launch.
fn api_devices(
    state: AppState,
    app_config: Arc<AppConfig>,
) -> impl Filter<Extract = (impl warp::Reply,), Error = warp::Rejection> + Clone {
    warp::path!("api" / "devices")
        .and(warp::get())
        .and(warp::query::<std::collections::HashMap<String, String>>())
        .and(warp::any().map(move || (state.clone(), app_config.clone())))
        .and_then(
            |q: std::collections::HashMap<String, String>,
             (state, app_config): (AppState, Arc<AppConfig>)| async move {
                let backend = q.get("backend").cloned().unwrap_or_default();
                let ui = state.ui_settings.lock().unwrap().clone();
                let mut eff_config = (*app_config).clone();
                if !ui.llama_server_path.is_empty() {
                    eff_config.llama_server_path = PathBuf::from(&ui.llama_server_path);
                }
                if !ui.llama_server_cwd.is_empty() {
                    eff_config.llama_server_cwd = PathBuf::from(&ui.llama_server_cwd);
                }
                let reply = match server::list_devices(&state, &eff_config, &backend).await {
                    Ok(devices) => warp::reply::json(&serde_json::json!({"devices": devices})),
                    Err(e) => warp::reply::json(&serde_json::json!({"error": format!("{e:#}")})),
                };
                Ok::<_, warp::Rejection>(reply)
            },
        )
}

/// Install page: upstream releases and backends for this platform,
/// installed builds, install / remove, and "use this build in Settings".
fn api_builds(
    state: AppState,
    app_config: Arc<AppConfig>,
) -> impl Filter<Extract = (impl warp::Reply,), Error = warp::Rejection> + Clone {
    use crate::llama::builds;

    let catalog = warp::path!("api" / "builds" / "catalog")
        .and(warp::get())
        .and_then(|| async {
            let releases = builds::recent_releases(8).await;
            let reply = match releases {
                Ok(releases) => warp::reply::json(&serde_json::json!({
                    "backends": builds::backend_options(),
                    "releases": releases,
                    "root": builds::builds_root(),
                })),
                Err(e) => warp::reply::json(&serde_json::json!({
                    "backends": builds::backend_options(),
                    "releases": [],
                    "root": builds::builds_root(),
                    "error": format!("{e:#}"),
                })),
            };
            Ok::<_, warp::Rejection>(reply)
        });

    let installed = warp::path!("api" / "builds" / "installed")
        .and(warp::get())
        .map(|| warp::reply::json(&serde_json::json!({"builds": builds::installed_builds()})));

    // Starts an install in the background, reporting through
    // `state.build_install`. With `replace`, the named build is superseded
    // once the new one is in: presets and Settings move over and it is
    // removed.
    fn spawn_install(
        state: AppState,
        app_config: Arc<AppConfig>,
        backend: String,
        tag: String,
        replace: Option<builds::InstalledBuild>,
    ) -> Result<(), String> {
        let mut slot = state.build_install.lock().unwrap();
        if slot.as_ref().is_some_and(|p| !p.done) {
            return Err("an install is already running".to_string());
        }
        *slot = Some(builds::InstallProgress {
            id: builds::build_id(&backend, &tag),
            backend: backend.clone(),
            tag: tag.clone(),
            phase: "Starting".to_string(),
            ..Default::default()
        });
        drop(slot);
        crate::applog::info(format!("Installing llama.cpp {tag} ({backend})"));
        tokio::spawn(async move {
            let result = builds::install(state.clone(), app_config, backend, tag).await;
            let mut phase = None;
            if let (Ok(build), Some(old)) = (&result, replace.as_ref()) {
                let moved = builds::repoint(&state, old, build);
                match builds::remove(&old.id) {
                    Ok(()) => crate::applog::info(format!(
                        "Updated llama.cpp build {} -> {} ({moved} preset(s) moved)",
                        old.id, build.id
                    )),
                    Err(e) => crate::applog::warn(format!(
                        "Installed {} but could not remove {}: {e:#}",
                        build.id, old.id
                    )),
                }
                phase = Some(format!("Updated {} to {}", old.id, build.id));
            }
            let mut slot = state.build_install.lock().unwrap();
            if let Some(p) = slot.as_mut() {
                p.done = true;
                match &result {
                    Ok(build) => {
                        p.phase = phase.unwrap_or_else(|| format!("Installed {}", build.id));
                        crate::applog::info(format!(
                            "Installed llama.cpp build {} ({} device(s) found)",
                            build.id,
                            build.devices.len()
                        ));
                    }
                    Err(e) => {
                        p.error = Some(format!("{e:#}"));
                        crate::applog::error(format!("llama.cpp install failed: {e:#}"));
                    }
                }
            }
        });
        Ok(())
    }

    fn ok_or_error(result: Result<(), String>) -> warp::reply::Json {
        match result {
            Ok(()) => warp::reply::json(&serde_json::json!({"ok": true})),
            Err(e) => warp::reply::json(&serde_json::json!({"ok": false, "error": e})),
        }
    }

    let install_state = state.clone();
    let install_config = app_config.clone();
    let install = warp::path!("api" / "builds" / "install")
        .and(warp::post())
        .and(warp::body::json())
        .and(warp::any().map(move || (install_state.clone(), install_config.clone())))
        .map(
            |body: serde_json::Value, (state, app_config): (AppState, Arc<AppConfig>)| {
                let backend = body
                    .get("backend")
                    .and_then(|v| v.as_str())
                    .unwrap_or("")
                    .to_string();
                let tag = body
                    .get("tag")
                    .and_then(|v| v.as_str())
                    .unwrap_or("")
                    .to_string();
                if backend.is_empty() || tag.is_empty() {
                    return ok_or_error(Err("backend and tag required".to_string()));
                }
                ok_or_error(spawn_install(state, app_config, backend, tag, None))
            },
        );

    // Reinstall an installed build at the newest upstream tag, then move
    // presets/Settings over and drop the old one.
    let update_state = state.clone();
    let update_config = app_config.clone();
    let update = warp::path!("api" / "builds" / "update")
        .and(warp::post())
        .and(warp::body::json())
        .and(warp::any().map(move || (update_state.clone(), update_config.clone())))
        .and_then(
            |body: serde_json::Value, (state, app_config): (AppState, Arc<AppConfig>)| async move {
                let id = body.get("id").and_then(|v| v.as_str()).unwrap_or("");
                let Some(old) = builds::installed_build(id) else {
                    return Ok::<_, warp::Rejection>(ok_or_error(Err("no such build".to_string())));
                };
                let latest = match builds::latest_tag().await {
                    Ok(tag) => tag,
                    Err(e) => return Ok(ok_or_error(Err(format!("{e:#}")))),
                };
                if builds::tag_number(&latest) <= builds::tag_number(&old.tag) {
                    return Ok(ok_or_error(Err(format!(
                        "{} is already the newest release ({latest})",
                        old.tag
                    ))));
                }
                let backend = old.backend.clone();
                Ok(ok_or_error(spawn_install(
                    state,
                    app_config,
                    backend,
                    latest,
                    Some(old),
                )))
            },
        );

    let remove = warp::path!("api" / "builds" / "remove")
        .and(warp::post())
        .and(warp::body::json())
        .map(|body: serde_json::Value| {
            let id = body.get("id").and_then(|v| v.as_str()).unwrap_or("");
            match builds::remove(id) {
                Ok(()) => {
                    crate::applog::info(format!("Removed llama.cpp build {id}"));
                    warp::reply::json(&serde_json::json!({"ok": true}))
                }
                Err(e) => {
                    warp::reply::json(&serde_json::json!({"ok": false, "error": format!("{e:#}")}))
                }
            }
        });

    // Point Settings at an installed build so presets on "Configured
    // binary" use it.
    let use_state = state.clone();
    let use_build = warp::path!("api" / "builds" / "use")
        .and(warp::post())
        .and(warp::body::json())
        .and(warp::any().map(move || use_state.clone()))
        .map(|body: serde_json::Value, state: AppState| {
            let id = body.get("id").and_then(|v| v.as_str()).unwrap_or("");
            let Some(build) = builds::installed_build(id) else {
                return warp::reply::json(
                    &serde_json::json!({"ok": false, "error": "no such build"}),
                );
            };
            let mut settings = state.ui_settings.lock().unwrap();
            settings.llama_server_path = build.server_path.display().to_string();
            settings.llama_server_cwd = build.dir.display().to_string();
            let saved = app_state::save_ui_settings(&state.ui_settings_path, &settings);
            drop(settings);
            match saved {
                Ok(()) => {
                    crate::applog::info(format!("Settings now use llama.cpp build {id}"));
                    warp::reply::json(&serde_json::json!({"ok": true}))
                }
                Err(e) => {
                    warp::reply::json(&serde_json::json!({"ok": false, "error": format!("{e:#}")}))
                }
            }
        });

    catalog
        .or(installed)
        .or(install)
        .or(update)
        .or(remove)
        .or(use_build)
        .boxed()
}

/// The pi coding agent: status, start/stop in a PTY, and running pi's own
/// installer in that PTY when it is missing.
fn api_pi(
    state: AppState,
    app_config: Arc<AppConfig>,
) -> impl Filter<Extract = (impl warp::Reply,), Error = warp::Rejection> + Clone {
    use crate::pi;

    let status_state = state.clone();
    let status = warp::path!("api" / "pi" / "status")
        .and(warp::get())
        .and_then(move || {
            let state = status_state.clone();
            async move {
                let mut st = pi::status(&state.pi);
                // The default working directory is what Settings says, else home.
                if st.cwd.is_none() {
                    let wd = state.ui_settings.lock().unwrap().pi_workdir.clone();
                    st.cwd = Some(if wd.is_empty() {
                        dirs::home_dir()
                            .map(|h| h.display().to_string())
                            .unwrap_or_default()
                    } else {
                        wd
                    });
                }
                let st = pi::with_latest(st, pi::PI_NPM_PACKAGE).await;
                Ok::<_, warp::Rejection>(warp::reply::json(&st))
            }
        });

    // Update pi in place with npm (its installer uses npm underneath too).
    let update_state = state.clone();
    let update = warp::path!("api" / "pi" / "update")
        .and(warp::post())
        .and(warp::body::json())
        .map(move |body: serde_json::Value| {
            let cols = body.get("cols").and_then(|v| v.as_u64()).unwrap_or(120) as u16;
            let rows = body.get("rows").and_then(|v| v.as_u64()).unwrap_or(32) as u16;
            let home = dirs::home_dir().unwrap_or_else(|| std::path::PathBuf::from("."));
            let Some(npm) = pi::find_program("npm") else {
                return warp::reply::json(&serde_json::json!({
                    "ok": false,
                    "error": "npm was not found; re-run Install pi instead"
                }));
            };
            let args = vec![
                "install".to_string(),
                "-g".to_string(),
                "--ignore-scripts".to_string(),
                format!("{}@latest", pi::PI_NPM_PACKAGE),
            ];
            match pi::start(
                &update_state.pi,
                pi::Launch {
                    program: npm.display().to_string(),
                    args,
                    cwd: home,
                    label: "pi updater".to_string(),
                    cols,
                    rows,
                    env: Vec::new(),
                },
            ) {
                Ok(()) => {
                    crate::applog::info("Updating pi (npm install -g @earendil-works/pi-coding-agent@latest)");
                    warp::reply::json(&serde_json::json!({"ok": true}))
                }
                Err(e) => {
                    warp::reply::json(&serde_json::json!({"ok": false, "error": format!("{e:#}")}))
                }
            }
        });

    let start_state = state.clone();
    let start_config = app_config.clone();
    let start = warp::path!("api" / "pi" / "start")
        .and(warp::post())
        .and(warp::body::json())
        .map(move |body: serde_json::Value| {
            let state = start_state.clone();
            let cols = body.get("cols").and_then(|v| v.as_u64()).unwrap_or(120) as u16;
            let rows = body.get("rows").and_then(|v| v.as_u64()).unwrap_or(32) as u16;
            let cwd_str = body
                .get("cwd")
                .and_then(|v| v.as_str())
                .map(str::trim)
                .filter(|s| !s.is_empty())
                .map(str::to_string)
                .unwrap_or_else(|| state.ui_settings.lock().unwrap().pi_workdir.clone());
            let cwd = if cwd_str.is_empty() {
                dirs::home_dir().unwrap_or_else(|| std::path::PathBuf::from("."))
            } else {
                std::path::PathBuf::from(&cwd_str)
            };
            if !cwd.is_dir() {
                return warp::reply::json(&serde_json::json!({
                    "ok": false,
                    "error": format!("{} is not a directory", cwd.display())
                }));
            }
            // Remember the directory for next time.
            {
                let mut settings = state.ui_settings.lock().unwrap();
                if settings.pi_workdir != cwd_str {
                    settings.pi_workdir = cwd_str.clone();
                    let _ = app_state::save_ui_settings(&state.ui_settings_path, &settings);
                }
            }
            let Some(program) = pi::find_pi() else {
                return warp::reply::json(&serde_json::json!({
                    "ok": false,
                    "error": "pi is not installed; use Install pi first"
                }));
            };
            // Every preset is a model in pi's list; start on the active
            // preset (the one the sidebar launches), else the first.
            let presets = state.presets.lock().unwrap().clone();
            let entries = pi::entries_from_presets(&presets);
            let active_preset = state.ui_settings.lock().unwrap().preset_id.clone();
            let model_id = entries
                .iter()
                .find(|e| e.preset_id == active_preset)
                .map(|e| e.id.clone())
                .unwrap_or_else(|| entries[0].id.clone());
            if let Err(e) = pi::write_models_json(start_config.port, &entries) {
                return warp::reply::json(&serde_json::json!({
                    "ok": false,
                    "error": format!("cannot write pi models.json: {e:#}")
                }));
            }
            let args = vec![
                "--provider".to_string(),
                pi::PROVIDER.to_string(),
                "--model".to_string(),
                model_id,
            ];
            match pi::start(
                &state.pi,
                pi::Launch {
                    program: program.display().to_string(),
                    args,
                    cwd: cwd.clone(),
                    label: "pi".to_string(),
                    cols,
                    rows,
                    env: Vec::new(),
                },
            ) {
                Ok(()) => {
                    crate::applog::info(format!("Started pi in {}", cwd.display()));
                    warp::reply::json(&serde_json::json!({"ok": true}))
                }
                Err(e) => {
                    warp::reply::json(&serde_json::json!({"ok": false, "error": format!("{e:#}")}))
                }
            }
        });

    let install_state = state.clone();
    let install = warp::path!("api" / "pi" / "install")
        .and(warp::post())
        .and(warp::body::json())
        .map(move |body: serde_json::Value| {
            let cols = body.get("cols").and_then(|v| v.as_u64()).unwrap_or(120) as u16;
            let rows = body.get("rows").and_then(|v| v.as_u64()).unwrap_or(32) as u16;
            let home = dirs::home_dir().unwrap_or_else(|| std::path::PathBuf::from("."));
            // pi's own installer, exactly as its site documents it.
            let args = vec![
                "-c".to_string(),
                "curl -fsSL https://pi.dev/install.sh | sh".to_string(),
            ];
            match pi::start(
                &install_state.pi,
                pi::Launch {
                    program: "sh".to_string(),
                    args,
                    cwd: home,
                    label: "pi installer".to_string(),
                    cols,
                    rows,
                    env: Vec::new(),
                },
            ) {
                Ok(()) => {
                    crate::applog::info(
                        "Running the pi installer (curl -fsSL https://pi.dev/install.sh | sh)",
                    );
                    warp::reply::json(&serde_json::json!({"ok": true}))
                }
                Err(e) => {
                    warp::reply::json(&serde_json::json!({"ok": false, "error": format!("{e:#}")}))
                }
            }
        });

    let stop_state = state.clone();
    let stop = warp::path!("api" / "pi" / "stop")
        .and(warp::post())
        .map(move || {
            pi::stop(&stop_state.pi);
            warp::reply::json(&serde_json::json!({"ok": true}))
        });

    status.or(start).or(install).or(update).or(stop)
}

/// DeepSeek Harness: status, start/stop (`dsh web` in a PTY plus a port
/// forwarder so the LAN can reach its loopback-only UI), and install via
/// npm.
fn api_dsh(
    state: AppState,
    app_config: Arc<AppConfig>,
) -> impl Filter<Extract = (impl warp::Reply,), Error = warp::Rejection> + Clone {
    use crate::{dsh, pi};

    fn proxy_port(app_config: &AppConfig) -> u16 {
        app_config.port.wrapping_add(1).max(1024)
    }

    let status_state = state.clone();
    let status_config = app_config.clone();
    let status = warp::path!("api" / "dsh" / "status")
        .and(warp::get())
        .and_then(move || {
            let state = status_state.clone();
            let app_config = status_config.clone();
            async move {
                let mut st = dsh::status(&state.dsh, &state.dsh_proxy);
                if st.cwd.is_none() {
                    let wd = state.ui_settings.lock().unwrap().dsh_workdir.clone();
                    st.cwd = Some(if wd.is_empty() {
                        dirs::home_dir()
                            .map(|h| h.display().to_string())
                            .unwrap_or_default()
                    } else {
                        wd
                    });
                }
                let st = dsh::with_latest(st).await;
                let mut v = serde_json::to_value(&st).unwrap_or_default();
                v["planned_proxy_port"] = serde_json::json!(proxy_port(&app_config));
                Ok::<_, warp::Rejection>(warp::reply::json(&v))
            }
        });

    let start_state = state.clone();
    let start_config = app_config.clone();
    let start = warp::path!("api" / "dsh" / "start")
        .and(warp::post())
        .and(warp::header::optional::<String>("host"))
        .and(warp::body::json())
        .and_then(move |host: Option<String>, body: serde_json::Value| {
            let state = start_state.clone();
            let app_config = start_config.clone();
            async move {
                let cols = body.get("cols").and_then(|v| v.as_u64()).unwrap_or(120) as u16;
                let rows = body.get("rows").and_then(|v| v.as_u64()).unwrap_or(32) as u16;
                let cwd_str = body
                    .get("cwd")
                    .and_then(|v| v.as_str())
                    .map(str::trim)
                    .filter(|s| !s.is_empty())
                    .map(str::to_string)
                    .unwrap_or_else(|| state.ui_settings.lock().unwrap().dsh_workdir.clone());
                let cwd = if cwd_str.is_empty() {
                    dirs::home_dir().unwrap_or_else(|| std::path::PathBuf::from("."))
                } else {
                    std::path::PathBuf::from(&cwd_str)
                };
                let fail = |msg: String| {
                    Ok::<_, warp::Rejection>(warp::reply::json(
                        &serde_json::json!({"ok": false, "error": msg}),
                    ))
                };
                if !cwd.is_dir() {
                    return fail(format!("{} is not a directory", cwd.display()));
                }
                {
                    let mut settings = state.ui_settings.lock().unwrap();
                    if settings.dsh_workdir != cwd_str {
                        settings.dsh_workdir = cwd_str.clone();
                        let _ = app_state::save_ui_settings(&state.ui_settings_path, &settings);
                    }
                }
                let Some(program) = dsh::find_dsh() else {
                    return fail("dsh is not installed; use Install dsh first".to_string());
                };
                let presets = state.presets.lock().unwrap().clone();
                let entries = pi::entries_from_presets(&presets);
                if let Err(e) = dsh::write_settings(app_config.port, &entries) {
                    return fail(format!("cannot write dsh settings.yaml: {e:#}"));
                }
                let port = proxy_port(&app_config);
                if let Err(e) = dsh::start_proxy(&state.dsh_proxy, port).await {
                    return fail(format!("{e:#}"));
                }
                let mut args = vec![
                    "web".to_string(),
                    "--no-open".to_string(),
                    "--port".to_string(),
                    dsh::DSH_PORT.to_string(),
                ];
                for h in dsh::trusted_hosts(host.as_deref(), port) {
                    args.push("--trusted-host".to_string());
                    args.push(h);
                }
                match pi::start(
                    &state.dsh,
                    pi::Launch {
                        program: program.display().to_string(),
                        args,
                        cwd: cwd.clone(),
                        label: "dsh".to_string(),
                        cols,
                        rows,
                        env: vec![(
                            dsh::API_KEY_ENV.to_string(),
                            "llama-admin-monitor".to_string(),
                        )],
                    },
                ) {
                    Ok(()) => {
                        crate::applog::info(format!(
                            "Started dsh web in {} (forwarding port {port} to 127.0.0.1:{})",
                            cwd.display(),
                            dsh::DSH_PORT
                        ));
                        Ok(warp::reply::json(
                            &serde_json::json!({"ok": true, "proxy_port": port}),
                        ))
                    }
                    Err(e) => {
                        dsh::stop_proxy(&state.dsh_proxy);
                        fail(format!("{e:#}"))
                    }
                }
            }
        });

    // Install and update are the same npm command; update pins @latest and
    // stops a running dsh first so the new version is what starts next.
    fn npm_install_dsh(
        state: &AppState,
        body: &serde_json::Value,
        update: bool,
    ) -> warp::reply::Json {
        let cols = body.get("cols").and_then(|v| v.as_u64()).unwrap_or(120) as u16;
        let rows = body.get("rows").and_then(|v| v.as_u64()).unwrap_or(32) as u16;
        let home = dirs::home_dir().unwrap_or_else(|| std::path::PathBuf::from("."));
        let Some(npm) = pi::find_program("npm") else {
            return warp::reply::json(&serde_json::json!({
                "ok": false,
                "error": "npm was not found; install Node.js 22+ (installing pi also provides one) and try again"
            }));
        };
        if update {
            dsh::stop_proxy(&state.dsh_proxy);
        }
        let spec = if update {
            format!("{}@latest", dsh::NPM_PACKAGE)
        } else {
            dsh::NPM_PACKAGE.to_string()
        };
        let args = vec!["install".to_string(), "-g".to_string(), spec.clone()];
        match pi::start(
            &state.dsh,
            pi::Launch {
                program: npm.display().to_string(),
                args,
                cwd: home,
                label: if update { "dsh updater" } else { "dsh installer" }.to_string(),
                cols,
                rows,
                env: Vec::new(),
            },
        ) {
            Ok(()) => {
                crate::applog::info(format!("Running npm install -g {spec}"));
                warp::reply::json(&serde_json::json!({"ok": true}))
            }
            Err(e) => {
                warp::reply::json(&serde_json::json!({"ok": false, "error": format!("{e:#}")}))
            }
        }
    }

    let install_state = state.clone();
    let install = warp::path!("api" / "dsh" / "install")
        .and(warp::post())
        .and(warp::body::json())
        .map(move |body: serde_json::Value| npm_install_dsh(&install_state, &body, false));

    let update_state = state.clone();
    let update = warp::path!("api" / "dsh" / "update")
        .and(warp::post())
        .and(warp::body::json())
        .map(move |body: serde_json::Value| npm_install_dsh(&update_state, &body, true));

    let stop_state = state.clone();
    let stop = warp::path!("api" / "dsh" / "stop")
        .and(warp::post())
        .map(move || {
            pi::stop(&stop_state.dsh);
            dsh::stop_proxy(&stop_state.dsh_proxy);
            warp::reply::json(&serde_json::json!({"ok": true}))
        });

    status.or(start).or(install).or(update).or(stop)
}
