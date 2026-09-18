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
            
            let exe_path = std::env::current_exe()
                .unwrap_or_else(|_| std::path::PathBuf::from("./target/release/llama-admin-monitor"))
                .to_string_lossy()
                .to_string();

            let script = if branch == "main" {
                format!(
                    r#"
                    echo "[info] Fetching latest release info..."
                    LATEST_TAG=$(curl -s https://api.github.com/repos/ViriatusOG/llama-admin-monitor/releases/latest | grep '"tag_name":' | sed -E 's/.*"([^"]+)".*/\1/')
                    if [ -z "$LATEST_TAG" ]; then
                        echo "[error] Could not determine latest release tag."
                        exit 1
                    fi
                    echo "[info] Latest release is $LATEST_TAG. Downloading..."
                    
                    ARCH=$(uname -m)
                    if [ "$ARCH" = "x86_64" ]; then
                        ASSET="llama-admin-monitor-linux-x86_64"
                    elif [ "$ARCH" = "aarch64" ] || [ "$ARCH" = "arm64" ]; then
                        ASSET="llama-admin-monitor-linux-aarch64"
                    else
                        echo "[error] Unsupported architecture $ARCH"
                        exit 1
                    fi
                    
                    DOWNLOAD_URL="https://github.com/ViriatusOG/llama-admin-monitor/releases/download/$LATEST_TAG/$ASSET"
                    
                    # Ensure git tree is synced
                    git fetch origin
                    git checkout main
                    git reset --hard origin/main
                    
                    echo "[info] Downloading $DOWNLOAD_URL to {exe_path}"
                    curl -L -o "{exe_path}" "$DOWNLOAD_URL"
                    chmod +x "{exe_path}"
                    
                    echo "[info] Restarting..."
                    exec "{exe_path}"
                    "#
                )
            } else {
                format!(
                    "git fetch origin && git checkout {branch} && git reset --hard origin/{branch} && cargo build --release && exec {exe_path}"
                )
            };

            println!("[info] Executing update: {}", script);
            let err = Command::new("bash").arg("-c").arg(&script).exec();
            eprintln!("[error] Failed to exec update script: {}", err);
        }
        
        #[cfg(not(unix))]
        {
            println!("[error] In-app updates are only supported on Unix systems.");
        }
    });
}
