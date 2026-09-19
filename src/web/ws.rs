use futures_util::{SinkExt, StreamExt};
use std::time::Duration;
use warp::Filter;
use warp::ws::{Message, Ws};

use crate::state::AppState;

const WS_PUSH_INTERVAL: Duration = Duration::from_millis(500);

pub fn ws_route(
    state: AppState,
) -> impl Filter<Extract = (impl warp::Reply,), Error = warp::Rejection> + Clone {
    let ws_state = state;
    warp::path("ws").and(warp::ws()).map(move |ws: Ws| {
        let state = ws_state.clone();
        ws.on_upgrade(move |socket| {
            let state = state.clone();
            async move {
                let (mut ws_tx, mut ws_rx) = socket.split();

                let update_task = tokio::spawn(async move {
                    let mut interval = tokio::time::interval(WS_PUSH_INTERVAL);
                    loop {
                        interval.tick().await;
                        let json = {
                            let gpu = state.gpu_metrics.lock().unwrap().clone();
                            let gpu_processes = state.gpu_processes.lock().unwrap().clone();
                            let llama = state.llama_metrics.lock().unwrap().clone();
                            let logs: Vec<String> =
                                state.server_logs.lock().unwrap().iter().cloned().collect();
                            let running = *state.server_running.lock().unwrap();
                            let hf_download = state.hf_download_progress.lock().unwrap().clone();
                            let model_path = state
                                .server_config
                                .lock()
                                .unwrap()
                                .as_ref()
                                .map(|c| c.model_path.clone());
                            let bench = state.bench_progress.lock().unwrap().clone();
                            let server_error = state.server_error.lock().unwrap().clone();
                            let system = state.system_stats.lock().unwrap().clone();
                            let app_update = state.update_phase.lock().unwrap().clone();
                            let app_log_seq = crate::applog::latest_seq();
                            let build_install = state.build_install.lock().unwrap().clone();
                            serde_json::json!({
                                "gpu": gpu,
                                "gpu_processes": gpu_processes,
                                "llama": llama,
                                "logs": logs,
                                "server_running": running,
                                "hf_download": hf_download,
                                "model_path": model_path,
                                "bench": bench,
                                "server_error": server_error,
                                "system": system,
                                "app_update": app_update,
                                "app_log_seq": app_log_seq,
                                "build_install": build_install,
                            })
                            .to_string()
                        };
                        if ws_tx.send(Message::text(&json)).await.is_err() {
                            break;
                        }
                    }
                });

                while let Some(_msg) = ws_rx.next().await {}
                update_task.abort();
            }
        })
    })
}

/// Terminal bridge for the pi page: PTY output goes out as binary frames
/// (scrollback first, then live), and the page sends either binary
/// keystrokes or a JSON text frame `{"resize": [cols, rows]}`.
pub fn pi_ws_route(
    state: AppState,
) -> impl Filter<Extract = (impl warp::Reply,), Error = warp::Rejection> + Clone {
    let pi = warp::path!("ws" / "pi").map(|| "pi");
    let dsh = warp::path!("ws" / "dsh").map(|| "dsh");
    pi.or(dsh).unify().and(warp::ws()).map(move |which: &'static str, ws: Ws| {
        let state = state.clone();
        ws.on_upgrade(move |socket| async move {
            let (mut ws_tx, mut ws_rx) = socket.split();
            let slot = if which == "dsh" { &state.dsh } else { &state.pi };
            let session = slot.lock().unwrap().clone();
            let Some(session) = session else {
                let _ = ws_tx
                    .send(Message::text(&format!(r#"{{"error":"no {which} session"}}"#)))
                    .await;
                return;
            };
            let (scrollback, mut rx) = session.subscribe();
            if !scrollback.is_empty() && ws_tx.send(Message::binary(scrollback)).await.is_err() {
                return;
            }
            let out = tokio::spawn(async move {
                loop {
                    match rx.recv().await {
                        Ok(chunk) => {
                            if ws_tx.send(Message::binary(chunk)).await.is_err() {
                                break;
                            }
                        }
                        Err(tokio::sync::broadcast::error::RecvError::Lagged(_)) => continue,
                        Err(_) => break,
                    }
                }
            });
            while let Some(Ok(msg)) = ws_rx.next().await {
                if msg.is_binary() {
                    let _ = session.write_input(msg.as_bytes());
                } else if msg.is_text() {
                    if let Ok(v) =
                        serde_json::from_str::<serde_json::Value>(msg.to_str().unwrap_or(""))
                        && let Some(size) = v.get("resize").and_then(|r| r.as_array())
                        && size.len() == 2
                    {
                        let cols = size[0].as_u64().unwrap_or(80) as u16;
                        let rows = size[1].as_u64().unwrap_or(24) as u16;
                        let _ = session.resize(cols, rows);
                    } else if let Ok(v) =
                        serde_json::from_str::<serde_json::Value>(msg.to_str().unwrap_or(""))
                        && let Some(text) = v.get("input").and_then(|t| t.as_str())
                    {
                        let _ = session.write_input(text.as_bytes());
                    }
                } else if msg.is_close() {
                    break;
                }
            }
            out.abort();
        })
    })
}
