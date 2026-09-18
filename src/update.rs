use anyhow::Result;
use serde::Serialize;
use std::process::Command;
use std::thread;
use std::time::Duration;

#[derive(Serialize, Debug)]
pub struct UpdateStatus {
    pub current_branch: String,
    pub current_commit: String,
    pub main_latest_commit: String,
    pub beta_latest_commit: String,
}

pub fn check_updates() -> Result<UpdateStatus> {
    // 1. git fetch origin
    Command::new("git").args(["fetch", "origin"]).output()?;

    // 2. get current branch
    let branch_out = Command::new("git")
        .args(["rev-parse", "--abbrev-ref", "HEAD"])
        .output()?;
    let current_branch = String::from_utf8_lossy(&branch_out.stdout).trim().to_string();

    // 3. get current commit
    let commit_out = Command::new("git")
        .args(["rev-parse", "--short", "HEAD"])
        .output()?;
    let current_commit = String::from_utf8_lossy(&commit_out.stdout).trim().to_string();

    // 4. get main commit
    let main_out = Command::new("git")
        .args(["rev-parse", "--short", "origin/main"])
        .output()?;
    let main_latest_commit = String::from_utf8_lossy(&main_out.stdout).trim().to_string();

    // 5. get beta commit
    let beta_out = Command::new("git")
        .args(["rev-parse", "--short", "origin/beta"])
        .output()?;
    let beta_latest_commit = String::from_utf8_lossy(&beta_out.stdout).trim().to_string();

    Ok(UpdateStatus {
        current_branch,
        current_commit,
        main_latest_commit,
        beta_latest_commit,
    })
}

pub fn apply_update(branch: String) {
    // We spawn a thread to give the HTTP response time to send (200 OK)
    // Then we replace the current process via `exec`.
    thread::spawn(move || {
        thread::sleep(Duration::from_millis(500));
        
        #[cfg(unix)]
        {
            use std::os::unix::process::CommandExt;
            let script = format!(
                "git fetch origin && git checkout {branch} && git reset --hard origin/{branch} && cargo build --release && exec ./target/release/llama-admin-monitor"
            );
            println!("[info] Executing update: {}", script);
            let err = Command::new("bash").arg("-c").arg(&script).exec();
            // If exec returns, it failed!
            eprintln!("[error] Failed to exec update script: {}", err);
        }
        
        #[cfg(not(unix))]
        {
            println!("[error] In-app updates are only supported on Unix systems.");
        }
    });
}
