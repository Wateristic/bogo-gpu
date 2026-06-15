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
use crate::history::{self, SavedArray};
 
// Set this to 'true' for higher debugging and atleast itll FUCKING WORK NOW 
const VERBOSE: bool = false;
 
// ── Shared GUI state (lock-free reads from egui, locked writes from worker) ──
 
/// A single completed tick (one second of work).
#[derive(Clone, Debug)]
pub struct TickRecord {
    pub best_correct: u32,
    pub best_arr:     [u8; 25],
    pub shuffles:     u64,
}
 
/// Commands the GUI can send to the worker.
#[derive(Debug, Clone)]
pub enum WorkerCmd {
    /// Stop processing and disconnect.
    Stop,
    /// Start (or restart) with a (possibly new) config.
    Start(Config),
}
 
/// Stats object shared between the tokio worker and the egui thread.
pub struct GuiStats {
    pub inner: Mutex<GuiStatsInner>,
    /// GUI → worker command channel (sender side stored here so GUI can post).
    pub cmd_tx: Mutex<Option<mpsc::UnboundedSender<WorkerCmd>>>,
}
 
pub struct GuiStatsInner {
    /// Rolling history — one entry per tick (second), newest last.
    pub ticks:            Vec<TickRecord>,
    /// Running total shuffles ever submitted.
    pub total_shuffles:   u64,
    /// Shuffles in the last second (for display).
    pub shuffles_per_sec: u64,
    /// Best score so far *this session* (resets on app restart).
    pub session_best:     u32,
    /// Best array so far this session.
    pub session_best_arr: [u8; 25],
    /// All-time best score ever recorded, persisted to disk across sessions.
    pub all_time_best:     u32,
    /// All-time best array ever recorded.
    pub all_time_best_arr: [u8; 25],
    /// Connection status string.
    pub status:           String,
    /// Nickname (filled in on welcome).
    pub nickname:         String,
    /// Current seed string.
    pub seed_str:         String,
    /// History of "great" arrays (>= history::MIN_SAVED_CORRECT), persisted to disk.
    pub saved_arrays:     Vec<SavedArray>,
    /// Uptime start.
    pub start:            Instant,
    /// Whether the worker is currently running.
    pub running:          bool,
}
 
impl Default for GuiStats {
    fn default() -> Self {
        let best_record = history::load_best();
        let (all_time_best, all_time_best_arr) = match &best_record {
            Some(r) => (r.correct, r.arr),
            None    => (0, core::array::from_fn(|i| (i + 1) as u8)),
        };
        GuiStats {
            inner: Mutex::new(GuiStatsInner {
                ticks:            Vec::new(),
                total_shuffles:   0,
                shuffles_per_sec: 0,
                session_best:     0,
                session_best_arr: core::array::from_fn(|i| (i + 1) as u8),
                all_time_best,
                all_time_best_arr,
                status:           "stopped".into(),
                nickname:         String::new(),
                seed_str:         String::new(),
                saved_arrays:     history::load(),
                start:            Instant::now(),
                running:          false,
            }),
            cmd_tx: Mutex::new(None),
        }
    }
}
 
impl GuiStats {
    /// Send a command to the running worker (no-op if not connected).
    pub fn send_cmd(&self, cmd: WorkerCmd) {
        if let Ok(lock) = self.cmd_tx.lock() {
            if let Some(tx) = lock.as_ref() {
                let _ = tx.send(cmd);
            }
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
    stats: Arc<GuiStats>,
    /// If set, a `Start(cfg)` command is enqueued immediately when `run()`
    /// begins -- used by headless mode, where there's no GUI to press Start.
    autostart: Option<Config>,
}
 
impl Worker {
    pub fn new(stats: Arc<GuiStats>) -> Self {
        Worker { stats, autostart: None }
    }
 
    /// Like `Worker::new`, but immediately starts a session with `config`
    /// once `run()` begins -- for headless mode (no GUI to click Start).
    pub fn new_with_autostart(stats: Arc<GuiStats>, config: Config) -> Self {
        Worker { stats, autostart: Some(config) }
    }
 
    /// Main loop — waits for Start commands and runs sessions until Stop.
    pub async fn run(&self) -> Result<()> {
        let (cmd_tx, mut cmd_rx) = mpsc::unbounded_channel::<WorkerCmd>();
        {
            let mut lock = self.stats.cmd_tx.lock().unwrap();
            *lock = Some(cmd_tx.clone());
        }
 
        if let Some(cfg) = self.autostart.clone() {
            let _ = cmd_tx.send(WorkerCmd::Start(cfg));
        }
 
        loop {
            // Wait for a Start command.
            let config = loop {
                match cmd_rx.recv().await {
                    Some(WorkerCmd::Start(cfg)) => break cfg,
                    Some(WorkerCmd::Stop) | None => continue,
                }
            };
 
            {
                let mut g = self.stats.inner.lock().unwrap();
                g.running = true;
                g.status  = "starting…".into();
            }
 
            // Run one session; returns when stopped or fatally errored.
            self.run_session(config, &mut cmd_rx).await;
 
            {
                let mut g = self.stats.inner.lock().unwrap();
                g.running          = false;
                g.status           = "stopped".into();
                g.shuffles_per_sec = 0;
                g.seed_str         = String::new();
            }
        }
    }
 
    async fn run_session(
        &self,
        config: Config,
        cmd_rx: &mut mpsc::UnboundedReceiver<WorkerCmd>,
    ) {
        eprintln!("[worker] init backend");
        let backend = match tokio::task::spawn_blocking(|| Backend::new_default()).await {
            Ok(Ok(b)) => b,
            Ok(Err(e)) => {
                let mut g = self.stats.inner.lock().unwrap();
                g.status = format!("backend error: {e}");
                return;
            }
            Err(e) => {
                let mut g = self.stats.inner.lock().unwrap();
                g.status = format!("backend panic: {e}");
                return;
            }
        };
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
            config:    std::sync::Arc::new(config.clone()),
            lease_rx,
            report_tx,
            chunk_tx,
            done_rx,
            stats:     Arc::clone(&self.stats),
        };
        let sched_handle = tokio::spawn(sched.run(cancel.clone()));
 
        let net = NetClient {
            config:    std::sync::Arc::new(config.clone()),
            lease_tx,
            report_rx,
            stats:     Arc::clone(&self.stats),
            last_seed_str: None,
            last_report:   None,
        };
        let net_handle = tokio::spawn(net.run(cancel.clone()));
 
        // Watch for Stop / new Start commands from the GUI.
        loop {
            tokio::select! {
                res = net_handle => {
                    if let Ok(Err(e)) = res { tracing::error!("[net] {e}"); }
                    cancel.cancel();
                    break;
                }
                res = sched_handle => {
                    if let Ok(Err(e)) = res { tracing::error!("[sched] {e}"); }
                    cancel.cancel();
                    break;
                }
                cmd = cmd_rx.recv() => {
                    match cmd {
                        Some(WorkerCmd::Stop) | None => {
                            cancel.cancel();
                            break;
                        }
                        Some(WorkerCmd::Start(_)) => {
                            // Restart requested — stop this session first.
                            cancel.cancel();
                            break;
                        }
                    }
                }
                _ = cancel.cancelled() => { break; }
            }
        }
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
 
        if VERBOSE {
            eprintln!(
                "[compute] dispatch chunk lo={} hi={} count={}",
                chunk.lo, chunk.hi, count
            );
        }
 
        let started = Instant::now();
        let result = match backend.run_batch(chunk.seed, chunk.lo, count) {
            Ok(r) => r,
            Err(e) => {
                tracing::error!("[compute] run_batch error: {e}");
                continue;
            }
        };
 
        if VERBOSE {
            eprintln!(
                "[compute] chunk done lo={} hi={} best_correct={} best_index={} took={:?}",
                chunk.lo, chunk.hi, result.best_correct, result.best_index, started.elapsed()
            );
        }
 
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
        // `next_lease` carries a job that arrived *while* we were still
        // crunching the previous one, so we can hand off to it immediately
        // instead of waiting on `lease_rx` again.
        let mut next_lease: Option<Lease> = None;
 
        loop {
            let lease = match next_lease.take() {
                Some(l) => l,
                None => {
                    tokio::select! {
                        lease = self.lease_rx.recv() => {
                            match lease {
                                Some(l) => l,
                                None    => break,
                            }
                        }
                        _ = cancel.cancelled() => break,
                    }
                }
            };
 
            next_lease = self.process_lease(lease, &cancel).await?;
        }
        Ok(())
    }
 
    /// Run one lease to completion, or until a new job is announced — in
    /// which case the in-flight chunk is finished, a final report for this
    /// lease is flushed, and the new lease is returned so `run()` can switch
    /// to it right away (rather than continuing to compute/report against a
    /// seed the server has already moved on from).
    async fn process_lease(&mut self, lease: Lease, cancel: &CancellationToken) -> Result<Option<Lease>> {
        let mut next_dispatch: u64 = 0;
        let mut in_flight:     usize = 0;
        let mut total_done:    u64 = 0;
        let mut tick_done:     u64 = 0;
        let mut win_best:      i32 = -1;
        let mut win_arr              = [0u8; 25];
        let mut win_index:     u64 = 0;
        let mut last_report         = tokio::time::Instant::now();
        // Set once a fresher job arrives mid-lease; once set we stop
        // dispatching new chunks for `lease` and wrap up as soon as the
        // in-flight chunk completes.
        let mut next_lease:    Option<Lease> = None;
 
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
            if next_lease.is_none() && next_dispatch < lease.count && in_flight == 0 {
                let lo = next_dispatch;
                let hi = (lo + CHUNK_SIZE).min(lease.count);
                self.chunk_tx.send(Chunk { seed: lease.seed, lo, hi }).await?;
                next_dispatch = hi;
                in_flight += 1;
            }
 
            if in_flight == 0 { break; }
 
            let result = tokio::select! {
                r = self.done_rx.recv() => match r { Some(r) => r, None => break },
                job = self.lease_rx.recv(), if next_lease.is_none() => {
                    match job {
                        Some(nl) => {
                            tracing::info!(
                                "🎯 new job (seed={}) received — finishing current chunk for seed={} before switching",
                                nl.seed_str, lease.seed_str
                            );
                            next_lease = Some(nl);
                        }
                        None => break,
                    }
                    continue;
                }
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
 
            if next_lease.is_none() && next_dispatch < lease.count {
                let lo = next_dispatch;
                let hi = (lo + CHUNK_SIZE).min(lease.count);
                self.chunk_tx.send(Chunk { seed: lease.seed, lo, hi }).await?;
                next_dispatch = hi;
                in_flight += 1;
            }
 
            let lease_done = next_lease.is_some() || (total_done >= lease.count && in_flight == 0);
            let report_due = last_report.elapsed() >= REPORT_INTERVAL;
 
            if (report_due || lease_done) && win_best >= 0 {
                if VERBOSE {
                    eprintln!(
                        "[sched] tick: best={} shuffles={} total_done={}/{} since_last_report={:?}",
                        win_best, tick_done, total_done, lease.count, last_report.elapsed()
                    );
                }
 
                {
                    let tick = TickRecord {
                        best_correct: win_best as u32,
                        best_arr:     win_arr,
                        shuffles:     tick_done,
                    };
                    let mut g = self.stats.inner.lock().unwrap();
                    if win_best as u32 > g.session_best {
                        g.session_best     = win_best as u32;
                        g.session_best_arr = win_arr;
                    }
 
                    let mut best_record: Option<history::BestRecord> = None;
                    if win_best as u32 > g.all_time_best {
                        g.all_time_best     = win_best as u32;
                        g.all_time_best_arr = win_arr;
                        best_record = Some(history::BestRecord {
                            correct:   win_best as u32,
                            arr:       win_arr,
                            seed:      lease.seed_str.clone(),
                            index:     win_index,
                            timestamp: history::now_unix(),
                        });
                    }
 
                    g.ticks.push(tick);
                    if g.ticks.len() > 120 {
                        g.ticks.remove(0);
                    }
                    g.shuffles_per_sec = tick_done;
                    g.total_shuffles  += tick_done;
 
                    // Stash "great" arrays (>= MIN_SAVED_CORRECT) for the Array History tab.
                    let mut persist: Option<Vec<SavedArray>> = None;
                    if win_best as u32 >= history::MIN_SAVED_CORRECT {
                        let total_shuffles = g.total_shuffles;
                        g.saved_arrays.push(SavedArray {
                            id:             history::now_nanos(),
                            correct:        win_best as u32,
                            arr:            win_arr,
                            seed:           lease.seed_str.clone(),
                            index:          win_index,
                            rate:           tick_done,
                            total_shuffles,
                            timestamp:      history::now_unix(),
                        });
                        if g.saved_arrays.len() > history::MAX_SAVED {
                            let excess = g.saved_arrays.len() - history::MAX_SAVED;
                            g.saved_arrays.drain(0..excess);
                        }
                        persist = Some(g.saved_arrays.clone());
                    }
                    drop(g);
 
                    if let Some(entries) = persist {
                        history::save(&entries);
                    }
                    if let Some(record) = best_record {
                        history::save_best(&record);
                    }
                }
 
                let _ = self.report_tx.send(Report {
                    seed_str:     lease.seed_str.clone(),
                    total_done,
                    best_correct: win_best as u32,
                    best_arr:     win_arr,
                    best_index:   win_index,
                }).await;
 
                win_best  = -1;
                win_arr   = [0u8; 25];
                win_index = 0;
                tick_done = 0;
                last_report = tokio::time::Instant::now();
            }
 
            if lease_done { break; }
        }
 
        Ok(next_lease)
    }
}
 
// ── Net client ────────────────────────────────────────────────────────────────
 
struct NetClient {
    config:    std::sync::Arc<Config>,
    lease_tx:  mpsc::Sender<Lease>,
    report_rx: mpsc::Receiver<Report>,
    stats:     Arc<GuiStats>,
    /// Most recent seed string the server has sent us via a "job" message.
    /// Used to recover if the server rejects a "result" because the seed
    /// we reported against is no longer (or not yet) what it considers current.
    last_seed_str: Option<String>,
    /// The most recently sent "result" report, kept around so it can be
    /// resent under a different seed if the server rejects it for an
    /// unknown/stale seed.
    last_report:   Option<Report>,
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
        if VERBOSE {
            eprintln!("[net] -> {hello}");
        }
        write.send(Message::Text(hello)).await?;
 
        loop {
            tokio::select! {
                msg = read.next() => {
                    match msg {
                        Some(Ok(Message::Text(text))) => {
                            if let Some(retry) = self.handle_message(&text, cancel).await? {
                                write.send(retry).await?;
                            }
                        }
                        Some(Ok(Message::Ping(data))) => {
                            if VERBOSE {
                                eprintln!("[net] <- ping ({} bytes)", data.len());
                                eprintln!("[net] -> pong ({} bytes)", data.len());
                            }
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
                            write.send(Message::Text(msg.clone())).await?;
                            tracing::info!(
                                "📊 sent report | done={} best={} idx={}",
                                r.total_done, r.best_correct, r.best_index
                            );
                            if VERBOSE {
                                eprintln!("[net] -> {msg}");
                            }
                            self.last_report = Some(r);
                        }
                        None => break,
                    }
                }
 
                _ = cancel.cancelled() => {
                    let stop_msg = serde_json::json!({"type":"stop"}).to_string();
                    if VERBOSE {
                        eprintln!("[net] -> {stop_msg}");
                    }
                    let _ = write.send(Message::Text(stop_msg)).await;
                    break;
                }
            }
        }
        Ok(())
    }
 
    async fn handle_message(&mut self, text: &str, cancel: &CancellationToken) -> Result<Option<Message>> {
        if VERBOSE {
            eprintln!("[net] <- {text}");
        }
 
        let Ok(msg) = serde_json::from_str::<serde_json::Value>(text) else {
            return Ok(None);
        };
        let msg_type = msg.get("type").and_then(|t| t.as_str()).unwrap_or("");
 
        match msg_type {
            "job" => {
                let seed_str = msg.get("seed").and_then(|s| s.as_str()).unwrap_or("").to_string();
                let count    = msg.get("count").and_then(|c| c.as_u64()).unwrap_or(0);
                if let Ok(seed) = seed_str.parse::<u64>() {
                    tracing::info!("🎯 Job: seed={seed_str} count={count}");
                    self.last_seed_str = Some(seed_str.clone());
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
 
                // If the server rejected our result because it didn't
                // recognize the seed we reported against, retry once using
                // the most recent seed it has actually sent us via a "job"
                // message (it may have moved on to a new seed between the
                // time we received our lease and the time we reported).
                if reason.to_lowercase().contains("seed") {
                    if let (Some(retry_seed), Some(r)) =
                        (self.last_seed_str.clone(), self.last_report.clone())
                    {
                        if retry_seed != r.seed_str {
                            tracing::warn!(
                                "🔁 retrying report with most recently seen seed={} (was {})",
                                retry_seed, r.seed_str
                            );
                            let arr: Vec<u32> = r.best_arr.iter().map(|&v| v as u32).collect();
                            let retry_msg = serde_json::json!({
                                "type":         "result",
                                "seed":         retry_seed,
                                "total_done":   r.total_done,
                                "best_correct": r.best_correct,
                                "best_arr":     arr,
                                "best_index":   r.best_index,
                            }).to_string();
                            if VERBOSE {
                                eprintln!("[net] -> {retry_msg}");
                            }
                            return Ok(Some(Message::Text(retry_msg)));
                        }
                    }
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
        Ok(None)
    }
}
