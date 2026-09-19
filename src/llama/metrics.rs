#[derive(Debug, Clone, Default, serde::Serialize)]
pub struct LlamaMetrics {
    /// Live throughput while a slot is working, else the last finished
    /// task's average. See `SlotRates`.
    pub prompt_tokens_per_sec: f64,
    pub generation_tokens_per_sec: f64,
    pub prompt_tokens_total: u64,
    pub predicted_tokens_total: u64,
    pub kv_cache_tokens: u64,
    pub kv_cache_max: u64,
    pub slots_idle: u32,
    pub slots_processing: u32,
    pub requests_processing: u32,
    pub status: String,
}

#[derive(Debug, Clone, Default)]
pub struct PrometheusValues {
    pub prompt_tokens_per_sec: f64,
    pub predicted_tokens_per_sec: f64,
    pub prompt_tokens_total: f64,
    pub prompt_seconds_total: f64,
    pub predicted_tokens_total: f64,
    pub predicted_seconds_total: f64,
    pub n_tokens_max: u64,
    pub requests_processing: u32,
}

/// Parse Prometheus text format and extract the metrics we care about.
/// llama.cpp uses colon-separated names like `llamacpp:prompt_tokens_total`.
pub fn parse_prometheus_metrics(body: &str) -> PrometheusValues {
    let mut vals = PrometheusValues::default();
    for line in body.lines() {
        if line.starts_with('#') || line.is_empty() {
            continue;
        }
        let mut parts = line.split_whitespace();
        let name = match parts.next() {
            Some(n) => n,
            None => continue,
        };
        let value = match parts.next().and_then(|v| v.parse::<f64>().ok()) {
            Some(v) => v,
            None => continue,
        };
        match name {
            "llamacpp:prompt_tokens_seconds" => vals.prompt_tokens_per_sec = value,
            "llamacpp:predicted_tokens_seconds" => vals.predicted_tokens_per_sec = value,
            "llamacpp:prompt_tokens_total" => vals.prompt_tokens_total = value,
            "llamacpp:prompt_seconds_total" => vals.prompt_seconds_total = value,
            "llamacpp:tokens_predicted_total" => vals.predicted_tokens_total = value,
            "llamacpp:tokens_predicted_seconds_total" => vals.predicted_seconds_total = value,
            "llamacpp:n_tokens_max" => vals.n_tokens_max = value as u64,
            "llamacpp:requests_processing" => vals.requests_processing = value as u32,
            _ => {}
        }
    }
    vals
}

/// One slot's counters as reported by `/slots`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SlotSample {
    pub id: u64,
    pub id_task: Option<u64>,
    pub is_processing: bool,
    pub n_prompt_processed: u64,
    pub n_decoded: u64,
}

impl SlotSample {
    pub fn from_json(v: &serde_json::Value) -> Option<Self> {
        let next = v
            .get("next_token")
            .and_then(|n| n.as_array())
            .and_then(|a| a.first());
        Some(Self {
            id: v.get("id")?.as_u64()?,
            id_task: v.get("id_task").and_then(|t| t.as_u64()),
            is_processing: v
                .get("is_processing")
                .and_then(|b| b.as_bool())
                .unwrap_or(false),
            n_prompt_processed: v
                .get("n_prompt_tokens_processed")
                .and_then(|n| n.as_u64())
                .unwrap_or(0),
            n_decoded: next
                .and_then(|n| n.get("n_decoded"))
                .and_then(|n| n.as_u64())
                .unwrap_or(0),
        })
    }
}

/// Throughput from `/slots` counter deltas between polls. The `/metrics`
/// gauges only carry a value on the scrape right after a task finishes
/// (and then reset), so during a long generation they read zero; the slot
/// counters advance token by token instead.
///
/// While a slot works the live rate is reported (smoothed); when every
/// slot is idle the last finished task's average is kept so the card does
/// not blank between requests.
#[derive(Debug, Default)]
pub struct SlotRates {
    prev: Vec<SlotSample>,
    live_prompt: f64,
    live_gen: f64,
    pub last_prompt: f64,
    pub last_gen: f64,
}

/// Half-weight exponential smoothing; the first reading is taken as is.
fn smooth(old: f64, new: f64) -> f64 {
    if old > 0.0 {
        old * 0.5 + new * 0.5
    } else {
        new
    }
}

impl SlotRates {
    /// Feeds one poll. `elapsed_s` is the time since the previous poll.
    pub fn update(&mut self, slots: &[SlotSample], elapsed_s: f64) {
        let mut prompt_tokens = 0u64;
        let mut gen_tokens = 0u64;
        let mut any_processing = false;
        for cur in slots {
            if !cur.is_processing {
                continue;
            }
            any_processing = true;
            // Only compare against the same task; a new task restarts the
            // counters and a naive delta would go negative or spike.
            let Some(prev) = self
                .prev
                .iter()
                .find(|p| p.id == cur.id && p.id_task == cur.id_task && p.is_processing)
            else {
                continue;
            };
            prompt_tokens += cur
                .n_prompt_processed
                .saturating_sub(prev.n_prompt_processed);
            gen_tokens += cur.n_decoded.saturating_sub(prev.n_decoded);
        }

        if any_processing && elapsed_s > 0.0 {
            let prompt_rate = prompt_tokens as f64 / elapsed_s;
            let gen_rate = gen_tokens as f64 / elapsed_s;
            // A phase reports zero while the other one runs; keep each
            // figure until its phase moves again rather than flashing 0.
            if prompt_rate > 0.0 {
                self.live_prompt = smooth(self.live_prompt, prompt_rate);
                self.last_prompt = self.live_prompt;
            }
            if gen_rate > 0.0 {
                self.live_gen = smooth(self.live_gen, gen_rate);
                self.last_gen = self.live_gen;
            }
        } else if !any_processing {
            self.live_prompt = 0.0;
            self.live_gen = 0.0;
        }
        self.prev = slots.to_vec();
    }

    /// Records a finished task's averages from the `/metrics` gauges, which
    /// are exact for that task; they replace the smoothed live estimate.
    pub fn record_task_average(&mut self, prompt_tps: f64, gen_tps: f64) {
        if prompt_tps > 0.0 {
            self.last_prompt = prompt_tps;
        }
        if gen_tps > 0.0 {
            self.last_gen = gen_tps;
        }
    }

    pub fn prompt_tps(&self) -> f64 {
        if self.live_prompt > 0.0 {
            self.live_prompt
        } else {
            self.last_prompt
        }
    }

    pub fn gen_tps(&self) -> f64 {
        if self.live_gen > 0.0 {
            self.live_gen
        } else {
            self.last_gen
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_prometheus_metrics() {
        let body = include_str!("../../tests/fixtures/prometheus_metrics.txt");
        let vals = parse_prometheus_metrics(body);

        assert!((vals.prompt_tokens_per_sec - 1234.5).abs() < 0.1);
        assert!((vals.predicted_tokens_per_sec - 56.7).abs() < 0.1);
        assert!((vals.prompt_tokens_total - 10000.0).abs() < 0.1);
        assert!((vals.prompt_seconds_total - 8.1).abs() < 0.1);
        assert!((vals.predicted_tokens_total - 5000.0).abs() < 0.1);
        assert!((vals.predicted_seconds_total - 88.2).abs() < 0.1);
        assert_eq!(vals.n_tokens_max, 131072);
        assert_eq!(vals.requests_processing, 1);
    }

    #[test]
    fn test_parse_prometheus_metrics_empty() {
        let vals = parse_prometheus_metrics("");
        assert_eq!(vals.prompt_tokens_per_sec, 0.0);
        assert_eq!(vals.n_tokens_max, 0);
    }

    #[test]
    fn test_parse_prometheus_metrics_comments_only() {
        let body = "# HELP llamacpp:prompt_tokens_total Total prompt tokens\n# TYPE llamacpp:prompt_tokens_total counter\n";
        let vals = parse_prometheus_metrics(body);
        assert_eq!(vals.prompt_tokens_total, 0.0);
    }

    fn slot(id: u64, task: u64, processing: bool, prompt: u64, decoded: u64) -> SlotSample {
        SlotSample {
            id,
            id_task: Some(task),
            is_processing: processing,
            n_prompt_processed: prompt,
            n_decoded: decoded,
        }
    }

    #[test]
    fn live_rates_from_slot_deltas() {
        let mut r = SlotRates::default();
        r.update(&[slot(0, 7, true, 0, 0)], 1.0);
        assert_eq!(r.gen_tps(), 0.0);
        r.update(&[slot(0, 7, true, 0, 24)], 1.0);
        assert!((r.gen_tps() - 24.0).abs() < 1e-9);
        // Smoothed towards the new reading, prompt phase untouched.
        r.update(&[slot(0, 7, true, 0, 44)], 1.0);
        assert!((r.gen_tps() - 22.0).abs() < 1e-9);
        assert_eq!(r.prompt_tps(), 0.0);
    }

    #[test]
    fn new_task_does_not_produce_a_spike() {
        let mut r = SlotRates::default();
        r.update(&[slot(0, 1, true, 0, 500)], 1.0);
        r.update(&[slot(0, 2, true, 0, 10)], 1.0);
        assert_eq!(r.gen_tps(), 0.0);
        r.update(&[slot(0, 2, true, 0, 30)], 1.0);
        assert!((r.gen_tps() - 20.0).abs() < 1e-9);
    }

    #[test]
    fn idle_keeps_last_task_average() {
        let mut r = SlotRates::default();
        r.update(&[slot(0, 1, true, 0, 0)], 1.0);
        r.update(&[slot(0, 1, true, 0, 30)], 1.0);
        r.update(&[slot(0, 1, false, 0, 30)], 1.0);
        assert!((r.gen_tps() - 30.0).abs() < 1e-9);
        r.record_task_average(400.0, 27.5);
        assert!((r.gen_tps() - 27.5).abs() < 1e-9);
        assert!((r.prompt_tps() - 400.0).abs() < 1e-9);
    }

    #[test]
    fn slot_sample_from_json() {
        let v: serde_json::Value = serde_json::from_str(
            r#"{"id":1,"id_task":9,"is_processing":true,"n_ctx":4096,
                "n_prompt_tokens_processed":120,"next_token":[{"n_decoded":42}]}"#,
        )
        .unwrap();
        assert_eq!(SlotSample::from_json(&v), Some(slot(1, 9, true, 120, 42)));
    }
}
