use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use anyhow::Result;
use futures_util::{SinkExt, StreamExt};
use http::header::HeaderValue;
use tokio::sync::mpsc;
use tokio_tungstenite::{
    connect_async,
    tungstenite::{client::IntoClientRequest, Message},
};
use tokio_util::sync::CancellationToken;

use crate::compute::Backend;
use crate::config::Config;

// ── Shared GUI state (lock-free reads from egui, locked writes from worker) ──

/// A single completed tick (one second of work).
#[derive(Clone, Debug)]
pub struct TickRecord {
    pub best_correct: u32,
    pub best_arr:     [u8; 25],
    pub shuffles:     u64,
}

/// Stats object shared between the tokio worker and the egui thread.
pub struct GuiStats {
    pub inner: Mutex<GuiStatsInner>,
}

pub struct GuiStatsInner {
    /// Rolling history — one entry per tick (second), newest last.
    pub ticks:           Vec<TickRecord>,
    /// Running total shuffles ever submitted.
    pub total_shuffles:  u64,
    /// Shuffles in the last second (for display).
    pub shuffles_per_sec: u64,
    /// All-time best score.
    pub all_time_best:   u32,
    /// All-time best array.
    pub all_time_arr:    [u8; 25],
    /// Connection status string.
    pub status:          String,
    /// Nickname (filled in on welcome).
    pub nickname:        String,
    /// Current seed string.
    pub seed_str:        String,
    /// Uptime start.
    pub start:           Instant,
}

impl Default for GuiStats {
    fn default() -> Self {
        GuiStats {
            inner: Mutex::new(GuiStatsInner {
                ticks:            Vec::new(),
                total_shuffles:   0,
                shuffles_per_sec: 0,
                all_time_best:    0,
                all_time_arr:     core::array::from_fn(|i| (i + 1) as u8),
                status:           "connecting…".into(),
                nickname:         String::new(),
                seed_str:         String::new(),
                start:            Instant::now(),
            }),
        }
    }
}

// ── Internal message types ────────────────────────────────────────────────────

#[derive(Debug, Clone)]
struct Lease {
    seed_str: String,
    seed:     u64,
    count:    u64,
}

#[derive(Debug, Clone)]
struct Chunk {
    seed: u64,
    lo:   u64,
    hi:   u64,
}

#[derive(Debug, Clone)]
struct RangeResult {
    lo:           u64,
    hi:           u64,
    best_correct: u32,
    best_arr:     [u8; 25],
    best_index:   u64,
}

#[derive(Debug, Clone)]
struct Report {
    seed_str:     String,
    total_done:   u64,
    best_correct: u32,
    best_arr:     [u8; 25],
    best_index:   u64,
}

// ── Worker ────────────────────────────────────────────────────────────────────

pub struct Worker {
    config: Config,
    stats:  Arc<GuiStats>,
}

impl Worker {
    pub fn new(config: Config, stats: Arc<GuiStats>) -> Self {
        Worker { config, stats }
    }

    pub async fn run(&self) -> Result<()> {
        eprintln!("[worker] init backend");
        let backend = tokio::task::spawn_blocking(|| Backend::new_default()).await??;
        eprintln!("[worker] backend ready");

        {
            let mut g = self.stats.inner.lock().unwrap();
            g.status = "backend ready, connecting…".into();
        }

        let cancel = CancellationToken::new();

        let (lease_tx, lease_rx)   = mpsc::channel::<Lease>(4);
        let (report_tx, report_rx) = mpsc::channel::<Report>(16);
        let (chunk_tx, chunk_rx)   = mpsc::channel::<Chunk>(2);
        let (done_tx, done_rx)     = mpsc::channel::<RangeResult>(16);

        {
            let done_tx = done_tx.clone();
            tokio::task::spawn_blocking(move || {
                run_compute_worker(backend, chunk_rx, done_tx);
            });
        }
        drop(done_tx);

        let sched = Scheduler {
            config:    std::sync::Arc::new(self.config.clone()),
            lease_rx,
            report_tx,
            chunk_tx,
            done_rx,
            stats:     Arc::clone(&self.stats),
        };
        let sched_handle = tokio::spawn(sched.run(cancel.clone()));

        let net = NetClient {
            config:    std::sync::Arc::new(self.config.clone()),
            lease_tx,
            report_rx,
            stats:     Arc::clone(&self.stats),
        };
        let net_handle = tokio::spawn(net.run(cancel.clone()));

        tokio::select! {
            res = net_handle   => { if let Ok(Err(e)) = res { tracing::error!("[net] {e}"); } }
            res = sched_handle => { if let Ok(Err(e)) = res { tracing::error!("[sched] {e}"); } }
            _ = cancel.cancelled() => {}
        }

        cancel.cancel();
        Ok(())
    }
}

// ── Compute worker ────────────────────────────────────────────────────────────

fn run_compute_worker(
    mut backend: Backend,
    mut chunk_rx: mpsc::Receiver<Chunk>,
    done_tx: mpsc::Sender<RangeResult>,
) {
    while let Some(chunk) = chunk_rx.blocking_recv() {
        let count = (chunk.hi - chunk.lo) as u32;
        let result = match backend.run_batch(chunk.seed, chunk.lo, count) {
            Ok(r) => r,
            Err(e) => {
                tracing::error!("[compute] run_batch error: {e}");
                continue;
            }
        };
        let range_result = RangeResult {
            lo:           chunk.lo,
            hi:           chunk.hi,
            best_correct: result.best_correct,
            best_arr:     result.best_arr,
            best_index:   result.best_index,
        };
        if done_tx.blocking_send(range_result).is_err() {
            break;
        }
    }
}

// ── Scheduler ─────────────────────────────────────────────────────────────────

const CHUNK_SIZE: u64 = 2_147_483_648;
const REPORT_INTERVAL: Duration = Duration::from_secs(1);

struct Scheduler {
    config:    std::sync::Arc<Config>,
    lease_rx:  mpsc::Receiver<Lease>,
    report_tx: mpsc::Sender<Report>,
    chunk_tx:  mpsc::Sender<Chunk>,
    done_rx:   mpsc::Receiver<RangeResult>,
    stats:     Arc<GuiStats>,
}

impl Scheduler {
    async fn run(mut self, cancel: CancellationToken) -> Result<()> {
        loop {
            tokio::select! {
                lease = self.lease_rx.recv() => {
                    match lease {
                        Some(l) => { let _ = self.process_lease(l, &cancel).await; }
                        None    => break,
                    }
                }
                _ = cancel.cancelled() => break,
            }
        }
        Ok(())
    }

    async fn process_lease(&mut self, lease: Lease, cancel: &CancellationToken) -> Result<()> {
        let mut next_dispatch: u64 = 0;
        let mut in_flight:     usize = 0;
        let mut total_done:    u64 = 0;  // cumulative for the whole lease — never reset
        let mut tick_done:     u64 = 0;  // shuffles since last report — for GUI display
        let mut win_best:      i32 = -1;
        let mut win_arr              = [0u8; 25];
        let mut win_index:     u64 = 0;
        let mut last_report         = tokio::time::Instant::now();

        // Update current seed in GUI stats.
        {
            let mut g = self.stats.inner.lock().unwrap();
            g.seed_str = lease.seed_str.clone();
        }

        if lease.count > 0 {
            let hi = CHUNK_SIZE.min(lease.count);
            self.chunk_tx.send(Chunk { seed: lease.seed, lo: 0, hi }).await?;
            next_dispatch = hi;
            in_flight += 1;
        }

        loop {
            if next_dispatch < lease.count && in_flight == 0 {
                let lo = next_dispatch;
                let hi = (lo + CHUNK_SIZE).min(lease.count);
                self.chunk_tx.send(Chunk { seed: lease.seed, lo, hi }).await?;
                next_dispatch = hi;
                in_flight += 1;
            }

            if in_flight == 0 { break; }

            let result = tokio::select! {
                r = self.done_rx.recv() => match r { Some(r) => r, None => break },
                _ = cancel.cancelled() => break,
            };

            in_flight  -= 1;
            let count   = result.hi - result.lo;
            total_done += count;
            tick_done  += count;

            if result.best_correct as i32 > win_best {
                win_best  = result.best_correct as i32;
                win_arr   = result.best_arr;
                win_index = result.best_index;
            }

            if next_dispatch < lease.count {
                let lo = next_dispatch;
                let hi = (lo + CHUNK_SIZE).min(lease.count);
                self.chunk_tx.send(Chunk { seed: lease.seed, lo, hi }).await?;
                next_dispatch = hi;
                in_flight += 1;
            }

            let lease_done = total_done >= lease.count && in_flight == 0;
            let report_due = last_report.elapsed() >= REPORT_INTERVAL;

            if (report_due || lease_done) && win_best >= 0 {
                // Push tick to GUI stats.
                {
                    let tick = TickRecord {
                        best_correct: win_best as u32,
                        best_arr:     win_arr,
                        shuffles:     tick_done,
                    };
                    let mut g = self.stats.inner.lock().unwrap();
                    if win_best as u32 > g.all_time_best {
                        g.all_time_best = win_best as u32;
                        g.all_time_arr  = win_arr;
                    }
                    g.ticks.push(tick);
                    if g.ticks.len() > 120 {
                        g.ticks.remove(0);
                    }
                    g.shuffles_per_sec = tick_done;
                    g.total_shuffles  += tick_done;
                }

                let _ = self.report_tx.send(Report {
                    seed_str:     lease.seed_str.clone(),
                    total_done,           // cumulative — server needs this to be monotonically increasing
                    best_correct: win_best as u32,
                    best_arr:     win_arr,
                    best_index:   win_index,
                }).await;

                // Reset only the per-tick win accumulators, NOT total_done.
                win_best  = -1;
                win_arr   = [0u8; 25];
                win_index = 0;
                tick_done = 0;
                last_report = tokio::time::Instant::now();
            }

            if lease_done { break; }
        }

        Ok(())
    }
}

// ── Net client ────────────────────────────────────────────────────────────────

struct NetClient {
    config:    std::sync::Arc<Config>,
    lease_tx:  mpsc::Sender<Lease>,
    report_rx: mpsc::Receiver<Report>,
    stats:     Arc<GuiStats>,
}

impl NetClient {
    async fn run(mut self, cancel: CancellationToken) -> Result<()> {
        let mut backoff = Duration::from_secs(2);
        loop {
            {
                let mut g = self.stats.inner.lock().unwrap();
                g.status = format!("connecting… (backoff {backoff:?})");
            }
            match self.connect_and_run(&cancel).await {
                Ok(())                          => break,
                Err(_) if cancel.is_cancelled() => break,
                Err(e) => {
                    tracing::warn!("[net] error: {e}, reconnecting in {backoff:?}");
                    {
                        let mut g = self.stats.inner.lock().unwrap();
                        g.status = format!("disconnected — retry in {backoff:?}");
                    }
                    tokio::select! {
                        _ = tokio::time::sleep(backoff) => {}
                        _ = cancel.cancelled()          => break,
                    }
                    backoff = (backoff * 2).min(Duration::from_secs(30));
                }
            }
        }
        Ok(())
    }

    async fn connect_and_run(&mut self, cancel: &CancellationToken) -> Result<()> {
        let socket_url = "wss://bogo.swapjs.dev/ws";

        let mut request = socket_url.into_client_request()?;
        {
            let h = request.headers_mut();
            h.insert("origin",     HeaderValue::from_static("https://bogo.swapjs.dev"));
            h.insert("referer",    HeaderValue::from_static("https://bogo.swapjs.dev/contribute"));
            h.insert("user-agent", HeaderValue::from_static("Mozilla/5.0 (X11; Linux x86_64) BogoForge/1.0"));
        }

        let (ws_stream, _) = connect_async(request).await?;
        tracing::info!("✅ Connected");
        {
            let mut g = self.stats.inner.lock().unwrap();
            g.status = "connected ✅".into();
        }

        let (mut write, mut read) = ws_stream.split();

        let hello = serde_json::json!({
            "type":     "hello",
            "v":        5,
            "uuid":     self.config.user.uuid,
            "nickname": self.config.user.nickname,
            "code":     self.config.user.code,
        }).to_string();
        write.send(Message::Text(hello)).await?;

        loop {
            tokio::select! {
                msg = read.next() => {
                    match msg {
                        Some(Ok(Message::Text(text))) => {
                            self.handle_message(&text, cancel).await?;
                        }
                        Some(Ok(Message::Ping(data))) => {
                            write.send(Message::Pong(data)).await?;
                        }
                        Some(Ok(Message::Close(_))) | None => {
                            return Err(anyhow::anyhow!("connection closed"));
                        }
                        Some(Err(e)) => return Err(anyhow::Error::from(e)),
                        _ => {}
                    }
                }

                report = self.report_rx.recv() => {
                    match report {
                        Some(r) => {
                            let arr: Vec<u32> = r.best_arr.iter().map(|&v| v as u32).collect();
                            let msg = serde_json::json!({
                                "type":         "result",
                                "seed":         r.seed_str,
                                "total_done":   r.total_done,
                                "best_correct": r.best_correct,
                                "best_arr":     arr,
                                "best_index":   r.best_index,
                            }).to_string();
                            write.send(Message::Text(msg)).await?;
                            tracing::info!(
                                "📊 sent report | done={} best={} idx={}",
                                r.total_done, r.best_correct, r.best_index
                            );
                        }
                        None => break,
                    }
                }

                _ = cancel.cancelled() => {
                    let _ = write.send(Message::Text(
                        serde_json::json!({"type":"stop"}).to_string()
                    )).await;
                    break;
                }
            }
        }
        Ok(())
    }

    async fn handle_message(&self, text: &str, cancel: &CancellationToken) -> Result<()> {
        let Ok(msg) = serde_json::from_str::<serde_json::Value>(text) else {
            return Ok(());
        };
        let msg_type = msg.get("type").and_then(|t| t.as_str()).unwrap_or("");

        match msg_type {
            "job" => {
                let seed_str = msg.get("seed").and_then(|s| s.as_str()).unwrap_or("").to_string();
                let count    = msg.get("count").and_then(|c| c.as_u64()).unwrap_or(0);
                if let Ok(seed) = seed_str.parse::<u64>() {
                    tracing::info!("🎯 Job: seed={seed_str} count={count}");
                    let _ = self.lease_tx.try_send(Lease { seed_str, seed, count });
                }
            }
            "welcome" => {
                let nick     = msg.get("nickname").and_then(|v| v.as_str()).unwrap_or("?");
                let lifetime = msg.get("lifetime_shuffles").and_then(|v| v.as_u64()).unwrap_or(0);
                tracing::info!("👋 Welcome {nick}! lifetime={lifetime}");
                {
                    let mut g = self.stats.inner.lock().unwrap();
                    g.nickname       = nick.to_string();
                    g.total_shuffles = lifetime;
                    g.status         = format!("connected ✅ — welcome, {nick}!");
                }
            }
            "credited" => {
                let credit = msg.get("credit").and_then(|v| v.as_u64()).unwrap_or(0);
                let rate   = msg.get("rate").and_then(|v| v.as_u64()).unwrap_or(0);
                let best   = msg.get("batch_best").and_then(|v| v.as_u64()).unwrap_or(0);
                tracing::info!("📊 {} shuffles | Best: {} | Rate: {}/s", credit, best, rate);
                {
                    let mut g = self.stats.inner.lock().unwrap();
                    g.shuffles_per_sec = rate;
                }
            }
            "rejected" => {
                let reason = msg.get("reason").and_then(|r| r.as_str()).unwrap_or(text);
                tracing::error!("❌ Rejected: {reason}");
                {
                    let mut g = self.stats.inner.lock().unwrap();
                    g.status = format!("❌ rejected: {reason}");
                }
            }
            "banned" => {
                let reason = msg.get("reason").and_then(|r| r.as_str()).unwrap_or(text);
                tracing::error!("🔨 Banned: {reason}");
                cancel.cancel();
            }
            other => {
                tracing::debug!("📨 Server ({other}): {text}");
            }
        }
        Ok(())
    }
}
