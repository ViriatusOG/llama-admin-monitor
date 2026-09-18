use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::path::PathBuf;
use std::sync::{Arc, Mutex};

#[derive(Debug, Clone, Serialize)]
pub struct BenchResult {
    pub tensor_split: String,
    pub prompt_tps: f64,
    pub gen_tps: f64,
}

#[derive(Debug, Clone, Serialize, Default)]
pub struct BenchProgress {
    pub running: bool,
    pub current_split: String,
    pub completed: usize,
    pub total: usize,
    pub results: Vec<BenchResult>,
    pub best_split: Option<String>,
    pub error: Option<String>,
    pub done: bool,
}

pub type SharedBenchProgress = Arc<Mutex<BenchProgress>>;

/// One row of llama-bench's JSON output. It emits many fields; we only
/// need the test type and throughput.
#[derive(Deserialize)]
struct BenchRow {
    #[serde(default)]
    n_prompt: u32,
    #[serde(default)]
    n_gen: u32,
    #[serde(default)]
    avg_ts: f64,
}

/// Derive the llama-bench path from the configured llama-server path,
/// since they're built into the same directory.
pub fn bench_binary_path(server_path: &str) -> PathBuf {
    let p = PathBuf::from(server_path);
    match p.parent() {
        Some(dir) => dir.join("llama-bench"),
        None => PathBuf::from("llama-bench"),
    }
}

async fn run_one(
    bench_bin: &PathBuf,
    model_path: &str,
    tensor_split: &str,
    gpu_layers: i32,
) -> Result<(f64, f64)> {
    let mut cmd = tokio::process::Command::new(bench_bin);
    cmd.arg("-m")
        .arg(model_path)
        .arg("-ngl")
        .arg(gpu_layers.to_string())
        .arg("-o")
        .arg("json");
    if !tensor_split.is_empty() {
        cmd.arg("-ts").arg(tensor_split);
    }

    let output = cmd.output().await?;
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        anyhow::bail!("llama-bench failed: {}", stderr.lines().last().unwrap_or(""));
    }

    let rows: Vec<BenchRow> = serde_json::from_slice(&output.stdout)
        .map_err(|e| anyhow::anyhow!("failed to parse llama-bench JSON: {e}"))?;

    let mut prompt_tps = 0.0;
    let mut gen_tps = 0.0;
    for row in rows {
        if row.n_prompt > 0 && row.n_gen == 0 {
            prompt_tps = row.avg_ts;
        } else if row.n_gen > 0 {
            gen_tps = row.avg_ts;
        }
    }
    Ok((prompt_tps, gen_tps))
}

pub async fn run_benchmark_sweep(
    bench_bin: PathBuf,
    model_path: String,
    splits: Vec<String>,
    gpu_layers: i32,
    progress: SharedBenchProgress,
) {
    {
        let mut p = progress.lock().unwrap();
        *p = BenchProgress {
            running: true,
            current_split: String::new(),
            completed: 0,
            total: splits.len(),
            results: Vec::new(),
            best_split: None,
            error: None,
            done: false,
        };
    }

    for split in &splits {
        {
            let mut p = progress.lock().unwrap();
            p.current_split = split.clone();
        }

        match run_one(&bench_bin, &model_path, split, gpu_layers).await {
            Ok((prompt_tps, gen_tps)) => {
                let mut p = progress.lock().unwrap();
                p.results.push(BenchResult {
                    tensor_split: split.clone(),
                    prompt_tps,
                    gen_tps,
                });
                p.completed += 1;
            }
            Err(e) => {
                let mut p = progress.lock().unwrap();
                p.error = Some(e.to_string());
                p.completed += 1;
            }
        }
    }

    let mut p = progress.lock().unwrap();
    // Rank by generation throughput -- the metric that dominates
    // interactive use. Prompt speed is reported but not used to rank.
    p.best_split = p
        .results
        .iter()
        .max_by(|a, b| {
            a.gen_tps
                .partial_cmp(&b.gen_tps)
                .unwrap_or(std::cmp::Ordering::Equal)
        })
        .map(|r| r.tensor_split.clone());
    p.running = false;
    p.done = true;
    p.current_split = String::new();
}
