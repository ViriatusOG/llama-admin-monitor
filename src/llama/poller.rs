use std::time::Duration;

use crate::state::AppState;

use super::metrics::parse_prometheus_metrics;

const LLAMA_POLL_INTERVAL: Duration = Duration::from_secs(1);

pub async fn llama_metrics_poller(state: AppState) {
    let client = reqwest::Client::builder()
        .timeout(Duration::from_secs(3))
        .build()
        .unwrap();

    loop {
        // Determine port: from config if started via UI, else default 8080
        let port = state
            .server_config
            .lock()
            .unwrap()
            .as_ref()
            .map(|c| c.port)
            .unwrap_or(8080);

        let base = format!("http://127.0.0.1:{port}");

        // Poll /health first to detect if any server is reachable
        let server_reachable = if let Ok(resp) = client.get(format!("{base}/health")).send().await {
            if let Ok(body) = resp.text().await {
                if let Ok(json) = serde_json::from_str::<serde_json::Value>(&body) {
                    let mut m = state.llama_metrics.lock().unwrap();
                    m.status = json
                        .get("status")
                        .and_then(|v| v.as_str())
                        .unwrap_or("unknown")
                        .to_string();
                    true
                } else {
                    false
                }
            } else {
                false
            }
        } else {
            false
        };

        if !server_reachable {
            {
                let mut m = state.llama_metrics.lock().unwrap();
                *m = super::metrics::LlamaMetrics::default();
            }
            tokio::time::sleep(LLAMA_POLL_INTERVAL).await;
            continue;
        }

        // Poll /metrics
        if let Ok(resp) = client.get(format!("{base}/metrics")).send().await
            && let Ok(body) = resp.text().await
        {
            let prom = parse_prometheus_metrics(&body);

            let prompt_tps = if prom.prompt_tokens_per_sec > 0.0 {
                prom.prompt_tokens_per_sec
            } else if prom.prompt_seconds_total > 0.0 {
                prom.prompt_tokens_total / prom.prompt_seconds_total
            } else {
                0.0
            };

            let gen_tps = if prom.predicted_tokens_per_sec > 0.0 {
                prom.predicted_tokens_per_sec
            } else if prom.predicted_seconds_total > 0.0 {
                prom.predicted_tokens_total / prom.predicted_seconds_total
            } else {
                0.0
            };

            let mut m = state.llama_metrics.lock().unwrap();
            m.prompt_tokens_per_sec = prompt_tps;
            m.generation_tokens_per_sec = gen_tps;
            m.prompt_tokens_total = prom.prompt_tokens_total as u64;
            m.predicted_tokens_total = prom.predicted_tokens_total as u64;
            m.kv_cache_tokens = prom.n_tokens_max;
            m.requests_processing = prom.requests_processing;
        }

        // Poll /slots — get per-slot processing state + total context
        if let Ok(resp) = client.get(format!("{base}/slots")).send().await
            && let Ok(body) = resp.text().await
            && let Ok(slots) = serde_json::from_str::<Vec<serde_json::Value>>(&body)
        {
            let mut idle = 0u32;
            let mut processing = 0u32;
            let num_slots = slots.len() as u64;
            let mut per_slot_ctx = 0u64;
            for slot in &slots {
                if slot
                    .get("is_processing")
                    .and_then(|v| v.as_bool())
                    .unwrap_or(false)
                {
                    processing += 1;
                } else {
                    idle += 1;
                }
                if let Some(n) = slot.get("n_ctx").and_then(|v| v.as_u64()) {
                    per_slot_ctx = n;
                }
            }
            let mut m = state.llama_metrics.lock().unwrap();
            m.slots_idle = idle;
            m.slots_processing = processing;
            if per_slot_ctx > 0 {
                m.kv_cache_max = per_slot_ctx * num_slots;
            }
        }

        tokio::time::sleep(LLAMA_POLL_INTERVAL).await;
    }
}
