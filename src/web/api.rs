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

pub fn api_routes(
    state: AppState,
    app_config: Arc<AppConfig>,
) -> impl Filter<Extract = (impl warp::Reply,), Error = warp::Rejection> + Clone {
    let app_config_bench = app_config.clone();
    let start = api_start(state.clone(), app_config);
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
    let get_settings = api_get_settings(state.clone());
    let put_settings = api_put_settings(state.clone());
    let browse = api_browse();
    let hf_search = api_hf_search();
    let hf_files = api_hf_files();
    let hf_download = api_hf_download(state.clone());
    let v1_proxy = api_v1_proxy(state);

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
                    Err(e) => Ok(warp::reply::json(
                        &serde_json::json!({"ok": false, "error": e.to_string()}),
                    )),
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
) -> impl Filter<Extract = (impl warp::Reply,), Error = warp::Rejection> + Clone {
    warp::path!("api" / "settings")
        .and(warp::get())
        .map(move || {
            let settings = state.ui_settings.lock().unwrap().clone();
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

pub fn api_v1_proxy(
    state: AppState,
) -> impl Filter<Extract = (impl warp::Reply,), Error = warp::Rejection> + Clone {
    warp::path("v1")
        .and(warp::path::tail())
        .and(warp::method())
        .and(warp::query::raw().map(Some).or_else(|_| async { Ok::<(Option<String>,), std::convert::Infallible>((None,)) }))
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

                    let reqwest_method = reqwest::Method::from_bytes(method.as_str().as_bytes()).unwrap();
                    let client = reqwest::Client::new();
                    let mut req = client.request(reqwest_method, &url).body(body.to_vec());
                    
                    for (k, v) in headers.iter() {
                        if k.as_str().to_lowercase() != "host" {
                            req = req.header(k.as_str(), v.as_bytes());
                        }
                    }

                    match req.send().await {
                        Ok(resp) => {
                            let status = resp.status().as_u16();
                            let mut builder = warp::http::Response::builder().status(status);
                            
                            for (k, v) in resp.headers().iter() {
                                builder = builder.header(k.as_str(), v.as_bytes());
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
                hf::download_hf_files(repo, files, dest_dir, progress).await;
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
                let (default_batch, default_ubatch, default_threads) = {
                    let cfg = state.server_config.lock().unwrap();
                    if let Some(c) = cfg.as_ref() {
                        (c.batch_size, c.ubatch_size, c.threads.unwrap_or(8))
                    } else {
                        (512, 512, 8)
                    }
                };

                let batch_sizes = parse_ints("batch_sizes", default_batch as i32);
                let ubatch_sizes = parse_ints("ubatch_sizes", default_ubatch as i32);
                let thread_counts = parse_ints("threads", default_threads as i32);

                let gpu_layers = body
                    .get("gpu_layers")
                    .and_then(|v| v.as_i64())
                    .unwrap_or(999) as i32;

                let server_path = {
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
                    bench::run_benchmark_sweep(
                        bench_bin, 
                        model_path, 
                        splits, 
                        batch_sizes, 
                        ubatch_sizes, 
                        thread_counts, 
                        gpu_layers, 
                        progress
                    ).await;
                });

                warp::reply::json(&serde_json::json!({"ok": true}))
            },
        )
}
