use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::path::PathBuf;
use std::sync::{Arc, Mutex};

#[derive(Debug, Clone, Serialize)]
pub struct BenchResult {
    pub tensor_split: String,
    pub batch_size: i32,
    pub ubatch_size: i32,
    pub threads: i32,
    pub prompt_tps: f64,
    pub gen_tps: f64,
}

#[derive(Debug, Clone, Serialize, Default)]
pub struct BenchProgress {
    pub running: bool,
    pub current_split: String,
    pub current_batch: i32,
    pub current_ubatch: i32,
    pub current_threads: i32,
    pub completed: usize,
    pub total: usize,
    pub results: Vec<BenchResult>,
    pub best_result: Option<BenchResult>,
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
    batch_size: i32,
    ubatch_size: i32,
    threads: i32,
) -> Result<(f64, f64)> {
    let mut cmd = tokio::process::Command::new(bench_bin);
    cmd.arg("-m")
        .arg(model_path)
        .arg("-ngl")
        .arg(gpu_layers.to_string())
        .arg("-b")
        .arg(batch_size.to_string())
        .arg("-ub")
        .arg(ubatch_size.to_string())
        .arg("-t")
        .arg(threads.to_string())
        .arg("-o")
        .arg("json");
    if !tensor_split.is_empty() {
        cmd.arg("-ts").arg(tensor_split);
    }

    let output = cmd.output().await?;
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        anyhow::bail!(
            "llama-bench failed: {}",
            stderr.lines().last().unwrap_or("")
        );
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

/// Everything a sweep varies: every combination of the four lists is run.
#[derive(Debug, Clone)]
pub struct SweepSpec {
    pub model_path: String,
    pub splits: Vec<String>,
    pub batch_sizes: Vec<i32>,
    pub ubatch_sizes: Vec<i32>,
    pub thread_counts: Vec<i32>,
    pub gpu_layers: i32,
}

impl SweepSpec {
    pub fn total_runs(&self) -> usize {
        self.splits.len()
            * self.batch_sizes.len()
            * self.ubatch_sizes.len()
            * self.thread_counts.len()
    }
}

pub async fn run_benchmark_sweep(
    bench_bin: PathBuf,
    spec: SweepSpec,
    progress: SharedBenchProgress,
) {
    let SweepSpec {
        model_path,
        splits,
        batch_sizes,
        ubatch_sizes,
        thread_counts,
        gpu_layers,
    } = spec.clone();
    let total_runs = spec.total_runs();
    {
        let mut p = progress.lock().unwrap();
        *p = BenchProgress {
            running: true,
            current_split: String::new(),
            current_batch: 0,
            current_ubatch: 0,
            current_threads: 0,
            completed: 0,
            total: total_runs,
            results: Vec::new(),
            best_result: None,
            error: None,
            done: false,
        };
    }

    for split in &splits {
        for &batch_size in &batch_sizes {
            for &ubatch_size in &ubatch_sizes {
                for &threads in &thread_counts {
                    {
                        let mut p = progress.lock().unwrap();
                        p.current_split = split.clone();
                        p.current_batch = batch_size;
                        p.current_ubatch = ubatch_size;
                        p.current_threads = threads;
                    }

                    match run_one(
                        &bench_bin,
                        &model_path,
                        split,
                        gpu_layers,
                        batch_size,
                        ubatch_size,
                        threads,
                    )
                    .await
                    {
                        Ok((prompt_tps, gen_tps)) => {
                            let mut p = progress.lock().unwrap();
                            p.results.push(BenchResult {
                                tensor_split: split.clone(),
                                batch_size,
                                ubatch_size,
                                threads,
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
            }
        }
    }

    let mut p = progress.lock().unwrap();
    // Rank by generation throughput -- the metric that dominates
    // interactive use. Prompt speed is reported but not used to rank.
    p.best_result = p
        .results
        .iter()
        .max_by(|a, b| {
            a.gen_tps
                .partial_cmp(&b.gen_tps)
                .unwrap_or(std::cmp::Ordering::Equal)
        })
        .cloned();
    p.running = false;
    p.done = true;
    p.current_split = String::new();
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bench_binary_path() {
        // When given a full path
        let p = bench_binary_path("/opt/llama.cpp/llama-server");
        assert_eq!(p.to_string_lossy(), "/opt/llama.cpp/llama-bench");

        // When given a relative path
        let p = bench_binary_path("./llama-server");
        assert_eq!(p.to_string_lossy(), "./llama-bench");

        // A bare name resolves next to it: parent() is Some(""), not None.
        let p = bench_binary_path("llama-server");
        assert_eq!(p.to_string_lossy(), "llama-bench");
    }
}
