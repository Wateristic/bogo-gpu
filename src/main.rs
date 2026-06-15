use std::io::{self, Write};
use std::sync::Arc;
use std::time::Duration;

mod config;
mod compute;
mod history;
mod messages;
mod rng;
mod worker;
mod gui;

use config::Config;
use worker::{Worker, GuiStats, WorkerCmd};

fn main() -> anyhow::Result<()> {
    // eframe must own the main thread (required on macOS + most platforms).
    // We spin up a dedicated OS thread for the tokio runtime instead.

    tracing_subscriber::fmt()
        .with_max_level(tracing::Level::INFO)
        .init();

    let args: Vec<String> = std::env::args().collect();
    if args.iter().any(|a| a == "-h" || a == "--help") {
        print_usage();
        return Ok(());
    }
    let headless = args.iter().any(|a| a == "--headless" || a == "-H");

    let config = match Config::load()? {
        Some(cfg) => {
            tracing::info!("Loaded config from {:?}", Config::config_path());
            cfg
        }
        None => {
            tracing::info!("No config found, prompting for user input");
            let uuid     = prompt("UUID: ")?;
            let nickname = prompt("Nickname: ")?;
            let code     = prompt("Code: ")?;

            let cfg = Config {
                user: config::UserConfig { uuid, nickname, code },
                compute: Default::default(),
            };
            cfg.save()?;
            tracing::info!("Config saved to {:?}", Config::config_path());
            cfg
        }
    };

    // Shared stats updated by the worker, read by egui (or printed in headless mode).
    let stats = Arc::new(GuiStats::default());

    if headless {
        return run_headless(stats, config);
    }

    // Spawn tokio + worker on a background OS thread.
    {
        let stats  = Arc::clone(&stats);
        let config = config.clone();
        std::thread::spawn(move || {
            let rt = tokio::runtime::Runtime::new().expect("tokio runtime");
            rt.block_on(async move {
                let worker = Worker::new(Arc::clone(&stats));
                if let Err(e) = worker.run().await {
                    tracing::error!("[worker] fatal: {e}");
                }
            });
        });
    }

    // Run egui on the main thread.
    let native_options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_title("bogo-gpu")
            .with_inner_size([1100.0, 680.0])
            .with_min_inner_size([800.0, 500.0]),
        ..Default::default()
    };

    eframe::run_native(
        "bogo-gpu",
        native_options,
        Box::new(move |cc| {
            Ok(Box::new(gui::BogoApp::new(cc, Arc::clone(&stats), config.clone())))
        }),
    ).map_err(|e| anyhow::anyhow!("eframe: {e}"))?;

    Ok(())
}

/// Run the worker without a GUI: starts immediately with the loaded config,
/// logs periodic status updates, and shuts down cleanly on Ctrl+C.
fn run_headless(stats: Arc<GuiStats>, config: Config) -> anyhow::Result<()> {
    tracing::info!("Starting in headless mode (no GUI) — worker will start automatically");

    let rt = tokio::runtime::Runtime::new().expect("tokio runtime");
    rt.block_on(async move {
        let worker = Worker::new_with_autostart(Arc::clone(&stats), config);

        // Periodically log a status line so headless runs have some visible feedback.
        let status_stats = Arc::clone(&stats);
        let status_task = tokio::spawn(async move {
            loop {
                tokio::time::sleep(Duration::from_secs(5)).await;
                let g = status_stats.inner.lock().unwrap();
                tracing::info!(
                    "status: {} | {}/s | session best {}/25 | all-time best {}/25 | total {}",
                    g.status, g.shuffles_per_sec, g.session_best, g.all_time_best, g.total_shuffles
                );
            }
        });

        tokio::select! {
            res = worker.run() => {
                if let Err(e) = res {
                    tracing::error!("[worker] fatal: {e}");
                }
            }
            _ = tokio::signal::ctrl_c() => {
                tracing::info!("Ctrl+C received, shutting down…");
                stats.send_cmd(WorkerCmd::Stop);
                // Give the net client a moment to send its "stop" message
                // and close the socket cleanly.
                tokio::time::sleep(Duration::from_millis(500)).await;
            }
        }

        status_task.abort();
    });

    Ok(())
}

fn print_usage() {
    println!("bogo-gpu");
    println!();
    println!("USAGE:");
    println!("    bogo-gpu [OPTIONS]");
    println!();
    println!("OPTIONS:");
    println!("    -H, --headless   Run without the GUI. The worker starts");
    println!("                     immediately using the saved config and");
    println!("                     logs periodic status updates. Stop with Ctrl+C.");
    println!("    -h, --help       Print this help message.");
}

fn prompt(msg: &str) -> io::Result<String> {
    print!("{}", msg);
    io::stdout().flush()?;
    let mut input = String::new();
    io::stdin().read_line(&mut input)?;
    Ok(input.trim().to_string())
}