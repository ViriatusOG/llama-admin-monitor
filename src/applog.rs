//! The monitor's own event log: launches and failures it detected, update
//! progress, telemetry problems, downloads. Distinct from llama-server's
//! process output, which lives in `AppState::server_logs`.
//!
//! Entries go to the terminal as before and into a bounded in-memory ring
//! that the Logs page reads through `/api/app/logs`. Global rather than on
//! `AppState` so startup code can log before the state exists.

use serde::Serialize;
use std::collections::VecDeque;
use std::sync::Mutex;
use std::sync::atomic::{AtomicU64, Ordering};

const CAPACITY: usize = 2000;

#[derive(Debug, Clone, Serialize)]
pub struct Entry {
    /// Monotonic, starts at 1; lets clients ask for "everything after N".
    pub seq: u64,
    pub ts_ms: u64,
    pub level: &'static str,
    pub message: String,
}

static ENTRIES: Mutex<VecDeque<Entry>> = Mutex::new(VecDeque::new());
static NEXT_SEQ: AtomicU64 = AtomicU64::new(1);

fn now_ms() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_millis() as u64)
        .unwrap_or(0)
}

fn push(level: &'static str, message: String) {
    match level {
        "error" | "warn" => eprintln!("[{level}] {message}"),
        _ => println!("[{level}] {message}"),
    }
    let entry = Entry {
        seq: NEXT_SEQ.fetch_add(1, Ordering::Relaxed),
        ts_ms: now_ms(),
        level,
        message,
    };
    let mut entries = ENTRIES.lock().unwrap();
    if entries.len() >= CAPACITY {
        entries.pop_front();
    }
    entries.push_back(entry);
}

pub fn info(message: impl Into<String>) {
    push("info", message.into());
}

pub fn warn(message: impl Into<String>) {
    push("warn", message.into());
}

pub fn error(message: impl Into<String>) {
    push("error", message.into());
}

/// Sequence number of the newest entry (0 when empty). Cheap to poll.
pub fn latest_seq() -> u64 {
    NEXT_SEQ.load(Ordering::Relaxed) - 1
}

/// Entries with `seq > after`, oldest first, at most `limit`. Entries that
/// have already rotated out of the ring are simply absent.
pub fn entries_after(after: u64, limit: usize) -> Vec<Entry> {
    let entries = ENTRIES.lock().unwrap();
    entries
        .iter()
        .filter(|e| e.seq > after)
        .take(limit)
        .cloned()
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn entries_are_sequential_and_filterable() {
        let before = latest_seq();
        info("first");
        error("second");
        let all = entries_after(before, 10);
        assert_eq!(all.len(), 2);
        assert_eq!(all[0].message, "first");
        assert_eq!(all[0].level, "info");
        assert_eq!(all[1].level, "error");
        assert_eq!(all[1].seq, all[0].seq + 1);
        assert_eq!(latest_seq(), all[1].seq);
        assert!(entries_after(all[1].seq, 10).is_empty());
    }
}
