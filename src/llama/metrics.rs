#[derive(Debug, Clone, Default, serde::Serialize)]
pub struct LlamaMetrics {
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
}
