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
                            serde_json::json!({
                                "gpu": gpu,
                                "llama": llama,
                                "logs": logs,
                                "server_running": running,
                                "hf_download": hf_download,
                                "model_path": model_path,
                                "bench": bench,
                                "server_error": server_error,
                                "system": system,
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
