// src/history.rs — persistence + helpers for the "Arrays" tab.
//
// Stores every shuffle result with >= MIN_SAVED_CORRECT correct positions
// to disk (as JSON, alongside config.toml) so the Array History tab can
// list, sort, and inspect them across runs.

use serde::{Deserialize, Serialize};
use std::path::PathBuf;
use std::time::{SystemTime, UNIX_EPOCH};

use crate::config::Config;

/// Minimum `correct` count for an array to be saved to history.
pub const MIN_SAVED_CORRECT: u32 = 16;

/// Maximum number of entries kept in history (oldest are dropped first).
pub const MAX_SAVED: usize = 1000;

/// A single "great" array, persisted to disk.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SavedArray {
    /// Unique id (nanosecond timestamp at time of insertion).
    pub id: u64,
    /// Number of correct positions (>= MIN_SAVED_CORRECT).
    pub correct: u32,
    /// The shuffled array itself.
    pub arr: [u8; 25],
    /// Seed string for the lease this array was found in.
    pub seed: String,
    /// Index within the seed's shuffle space.
    pub index: u64,
    /// Shuffles/sec at the time this tick was reported.
    pub rate: u64,
    /// Running total shuffles (lifetime) at the time this was found.
    pub total_shuffles: u64,
    /// Unix timestamp (seconds, UTC) when this array was found.
    pub timestamp: i64,
}

impl SavedArray {
    /// XP earned for this entry: total lifetime shuffles / 10,000.
    pub fn xp(&self) -> f64 {
        self.total_shuffles as f64 / 10_000.0
    }
}

// ── Time helpers ────────────────────────────────────────────────────────────

/// Current time as nanoseconds since the Unix epoch — used as a unique id.
pub fn now_nanos() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_nanos() as u64)
        .unwrap_or(0)
}

/// Current time as seconds since the Unix epoch (UTC).
pub fn now_unix() -> i64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs() as i64)
        .unwrap_or(0)
}

/// Render a relative time string ("just now", "5m ago", "3h ago", "2d ago").
pub fn format_relative(timestamp: i64) -> String {
    let now = now_unix();
    let diff = (now - timestamp).max(0);

    if diff < 5 {
        "just now".to_string()
    } else if diff < 60 {
        format!("{diff}s ago")
    } else if diff < 3600 {
        format!("{}m ago", diff / 60)
    } else if diff < 86_400 {
        format!("{}h ago", diff / 3600)
    } else if diff < 86_400 * 30 {
        format!("{}d ago", diff / 86_400)
    } else {
        format_timestamp(timestamp)
    }
}

/// Render an absolute UTC timestamp as "YYYY-MM-DD HH:MM:SS UTC".
///
/// Implemented without a chrono dependency using Howard Hinnant's
/// civil-from-days algorithm.
pub fn format_timestamp(timestamp: i64) -> String {
    let days = timestamp.div_euclid(86_400);
    let secs_of_day = timestamp.rem_euclid(86_400);

    let (y, m, d) = civil_from_days(days);
    let h = secs_of_day / 3600;
    let mi = (secs_of_day % 3600) / 60;
    let s = secs_of_day % 60;

    format!("{y:04}-{m:02}-{d:02} {h:02}:{mi:02}:{s:02} UTC")
}

/// Convert a day count since the Unix epoch (1970-01-01) into (year, month, day).
fn civil_from_days(z: i64) -> (i64, u32, u32) {
    let z = z + 719_468;
    let era = if z >= 0 { z } else { z - 146_096 } / 146_097;
    let doe = (z - era * 146_097) as u64; // [0, 146096]
    let yoe = (doe - doe / 1460 + doe / 36524 - doe / 146_096) / 365; // [0, 399]
    let y = yoe as i64 + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100); // [0, 365]
    let mp = (5 * doy + 2) / 153; // [0, 11]
    let d = (doy - (153 * mp + 2) / 5 + 1) as u32; // [1, 31]
    let m = if mp < 10 { mp + 3 } else { mp - 9 } as u32; // [1, 12]
    let y = if m <= 2 { y + 1 } else { y };
    (y, m, d)
}

// ── Persistence ──────────────────────────────────────────────────────────────

/// Path of the on-disk array history file (next to config.toml).
fn history_path() -> PathBuf {
    Config::config_dir().join("arrays.json")
}

/// Load saved arrays from disk. Returns an empty list if the file doesn't
/// exist or can't be parsed.
pub fn load() -> Vec<SavedArray> {
    let path = history_path();
    match std::fs::read_to_string(&path) {
        Ok(contents) => match serde_json::from_str::<Vec<SavedArray>>(&contents) {
            Ok(entries) => entries,
            Err(e) => {
                tracing::warn!("[history] failed to parse {path:?}: {e}");
                Vec::new()
            }
        },
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => Vec::new(),
        Err(e) => {
            tracing::warn!("[history] failed to read {path:?}: {e}");
            Vec::new()
        }
    }
}

/// Persist the given list of saved arrays to disk, overwriting any
/// previous contents.
pub fn save(entries: &[SavedArray]) {
    let dir = Config::config_dir();
    if let Err(e) = std::fs::create_dir_all(&dir) {
        tracing::warn!("[history] failed to create config dir {dir:?}: {e}");
        return;
    }

    let contents = match serde_json::to_string_pretty(entries) {
        Ok(s) => s,
        Err(e) => {
            tracing::warn!("[history] failed to serialize array history: {e}");
            return;
        }
    };

    if let Err(e) = std::fs::write(history_path(), contents) {
        tracing::warn!("[history] failed to write {:?}: {e}", history_path());
    }
}

// ── All-time best record ────────────────────────────────────────────────────

/// The single best array ever found, persisted across sessions independent
/// of the MIN_SAVED_CORRECT cutoff used for the Array History tab.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BestRecord {
    pub correct: u32,
    pub arr: [u8; 25],
    pub seed: String,
    pub index: u64,
    pub timestamp: i64,
}

/// Path of the on-disk all-time-best record file (next to config.toml).
fn best_path() -> PathBuf {
    Config::config_dir().join("best.json")
}

/// Load the persisted all-time-best record, if any.
pub fn load_best() -> Option<BestRecord> {
    let path = best_path();
    match std::fs::read_to_string(&path) {
        Ok(contents) => match serde_json::from_str::<BestRecord>(&contents) {
            Ok(record) => Some(record),
            Err(e) => {
                tracing::warn!("[history] failed to parse {path:?}: {e}");
                None
            }
        },
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => None,
        Err(e) => {
            tracing::warn!("[history] failed to read {path:?}: {e}");
            None
        }
    }
}

/// Persist a new all-time-best record to disk, overwriting any previous one.
pub fn save_best(record: &BestRecord) {
    let dir = Config::config_dir();
    if let Err(e) = std::fs::create_dir_all(&dir) {
        tracing::warn!("[history] failed to create config dir {dir:?}: {e}");
        return;
    }

    let contents = match serde_json::to_string_pretty(record) {
        Ok(s) => s,
        Err(e) => {
            tracing::warn!("[history] failed to serialize best record: {e}");
            return;
        }
    };

    if let Err(e) = std::fs::write(best_path(), contents) {
        tracing::warn!("[history] failed to write {:?}: {e}", best_path());
    }
}
