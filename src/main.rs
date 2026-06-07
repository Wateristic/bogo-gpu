use std::io::{self, Write};
use std::sync::Arc;

mod config;
mod compute;
mod messages;
mod rng;
mod worker;
mod gui;

use config::Config;
use worker::{Worker, GuiStats};

fn main() -> anyhow::Result<()> {
    // eframe must own the main thread (required on macOS + most platforms).
    // We spin up a dedicated OS thread for the tokio runtime instead.

    tracing_subscriber::fmt()
        .with_max_level(tracing::Level::INFO)
        .init();

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
            };
            cfg.save()?;
            tracing::info!("Config saved to {:?}", Config::config_path());
            cfg
        }
    };

    // Shared stats updated by the worker, read by egui.
    let stats = Arc::new(GuiStats::default());

    // Spawn tokio + worker on a background OS thread.
    {
        let stats  = Arc::clone(&stats);
        let config = config.clone();
        std::thread::spawn(move || {
            let rt = tokio::runtime::Runtime::new().expect("tokio runtime");
            rt.block_on(async move {
                let worker = Worker::new(config, Arc::clone(&stats));
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
            Ok(Box::new(gui::BogoApp::new(cc, Arc::clone(&stats))))
        }),
    ).map_err(|e| anyhow::anyhow!("eframe: {e}"))?;

    Ok(())
}

fn prompt(msg: &str) -> io::Result<String> {
    print!("{}", msg);
    io::stdout().flush()?;
    let mut input = String::new();
    io::stdin().read_line(&mut input)?;
    Ok(input.trim().to_string())
}
