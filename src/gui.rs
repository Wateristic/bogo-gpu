// src/gui.rs — Bogo-GPU dashboard (egui 0.29 / eframe 0.29)
//
// Two top-level tabs:
//   Dashboard — live bar chart, history strip, stats panel, stop/start buttons.
//   Settings  — edit identity + compute config; Apply saves & hot-reloads worker.

use std::sync::Arc;
use std::time::Duration;

use egui::{
    Align, Color32, FontId, Frame, Layout, Pos2, Rect,
    RichText, Rounding, Sense, Stroke, Vec2, Visuals,
};

use crate::config::{ComputeBackend, ComputeConfig, Config, UserConfig};
use crate::history::{self, SavedArray};
use crate::worker::{GuiStats, WorkerCmd};

// ─── Palette ──────────────────────────────────────────────────────────────────

const BG:        Color32 = Color32::from_rgb(18, 18, 18);
const PANEL_BG:  Color32 = Color32::from_rgb(26, 26, 26);
const ORANGE:    Color32 = Color32::from_rgb(232, 107, 66);
const GREEN:     Color32 = Color32::from_rgb(100, 195, 140);
const DIM:       Color32 = Color32::from_rgb(100, 100, 100);
const TEXT:      Color32 = Color32::from_rgb(220, 220, 220);
const TEXT_DIM:  Color32 = Color32::from_rgb(130, 130, 130);
const STRIP_BG:  Color32 = Color32::from_rgb(36, 36, 36);
const STRIP_HL:  Color32 = Color32::from_rgb(70, 70, 70);
const BTN_RED:   Color32 = Color32::from_rgb(180, 60, 60);
const BTN_GREEN: Color32 = Color32::from_rgb(60, 150, 90);
const INPUT_BG:  Color32 = Color32::from_rgb(32, 32, 32);

// ─── Tabs ─────────────────────────────────────────────────────────────────────

#[derive(PartialEq)]
enum Tab { Dashboard, Arrays, Settings }

/// Sort order for the Array History tab.
#[derive(PartialEq, Clone, Copy)]
enum ArraySort { Recent, Highest }

// ─── Settings edit state (mirrored from Config so we can cancel) ──────────────

struct SettingsEdit {
    uuid:        String,
    nickname:    String,
    code:        String,
    backend:     ComputeBackend,
    gpu_arch:    String,
    blocks:      String,
    threads:     String,
    cpu_threads: String,
    /// Feedback message after Apply.
    feedback:    Option<(bool, String)>,  // (is_error, msg)
}

impl SettingsEdit {
    fn from_config(cfg: &Config) -> Self {
        SettingsEdit {
            uuid:        cfg.user.uuid.clone(),
            nickname:    cfg.user.nickname.clone(),
            code:        cfg.user.code.clone(),
            backend:     cfg.compute.backend.clone(),
            gpu_arch:    cfg.compute.gpu_arch.clone(),
            blocks:      cfg.compute.blocks.to_string(),
            threads:     cfg.compute.threads.to_string(),
            cpu_threads: cfg.compute.cpu_threads.to_string(),
            feedback:    None,
        }
    }

    fn to_config(&self) -> Result<Config, String> {
        let blocks = self.blocks.trim().parse::<u32>()
            .map_err(|_| "Blocks must be a positive integer".to_string())?;
        let threads = self.threads.trim().parse::<u32>()
            .map_err(|_| "Threads must be a positive integer".to_string())?;
        let cpu_threads = self.cpu_threads.trim().parse::<u32>()
            .map_err(|_| "CPU threads must be 0 or a positive integer".to_string())?;

        if self.uuid.trim().is_empty()     { return Err("UUID cannot be empty".into()); }
        if self.nickname.trim().is_empty() { return Err("Nickname cannot be empty".into()); }
        if self.code.trim().is_empty()     { return Err("Code cannot be empty".into()); }
        if blocks == 0   { return Err("Blocks must be > 0".into()); }
        if threads == 0  { return Err("Threads must be > 0".into()); }

        Ok(Config {
            user: UserConfig {
                uuid:     self.uuid.trim().to_string(),
                nickname: self.nickname.trim().to_string(),
                code:     self.code.trim().to_string(),
            },
            compute: ComputeConfig {
                backend:     self.backend.clone(),
                gpu_arch:    self.gpu_arch.trim().to_string(),
                blocks,
                threads,
                cpu_threads,
            },
        })
    }
}

// ─── Snapshot ─────────────────────────────────────────────────────────────────

#[derive(Default)]
struct Snapshot {
    ticks:             Vec<(u32, [u8; 25])>,
    total_shuffles:    u64,
    shuffles_per_sec:  u64,
    /// Best score this session (resets on app restart).
    session_best:      u32,
    session_best_arr:  [u8; 25],
    /// All-time best ever recorded, persisted across sessions.
    all_time_best:     u32,
    all_time_best_arr: [u8; 25],
    status:            String,
    nickname:          String,
    seed_str:          String,
    uptime_secs:       u64,
    running:           bool,
    saved_arrays:      Vec<SavedArray>,
}

// ─── App ──────────────────────────────────────────────────────────────────────

pub struct BogoApp {
    stats:         Arc<GuiStats>,
    snap:          Snapshot,
    strip_offset:  f32,
    last_tick:     std::time::Instant,
    tick_count:    usize,
    history:       Vec<u32>,
    active_tab:    Tab,
    settings:      SettingsEdit,
    /// The last known good config (used to seed settings on first open).
    current_cfg:   Config,
    /// Accumulated uptime in seconds (pauses when worker is stopped).
    uptime_secs:   u64,
    /// Whether we were running last frame (to detect start/stop transitions).
    was_running:   bool,
    /// Wall-clock instant when the current run started.
    run_started:   Option<std::time::Instant>,
    /// Sort order for the Array History tab.
    array_sort:    ArraySort,
    /// Id of the entry currently shown in the full-screen detail view, if any.
    selected_array: Option<u64>,
}

impl BogoApp {
    pub fn new(_cc: &eframe::CreationContext<'_>, stats: Arc<GuiStats>, initial_cfg: Config) -> Self {
        let settings = SettingsEdit::from_config(&initial_cfg);
        BogoApp {
            stats,
            snap: Snapshot::default(),
            strip_offset: 0.0,
            last_tick: std::time::Instant::now(),
            tick_count: 0,
            history: Vec::new(),
            active_tab: Tab::Dashboard,
            settings,
            current_cfg: initial_cfg,
            uptime_secs: 0,
            was_running: false,
            run_started: None,
            array_sort: ArraySort::Recent,
            selected_array: None,
        }
    }

    fn update_uptime(&mut self) {
        let running = self.snap.running;
        if running && !self.was_running {
            // Just started
            self.run_started = Some(std::time::Instant::now());
        } else if !running && self.was_running {
            // Just stopped — accumulate
            if let Some(t) = self.run_started.take() {
                self.uptime_secs += t.elapsed().as_secs();
            }
        }
        self.was_running = running;
    }

    fn refresh_snapshot(&mut self) {
        let g = self.stats.inner.lock().unwrap();
        let ticks: Vec<_> = g.ticks.iter().map(|t| (t.best_correct, t.best_arr)).collect();
        self.history = ticks.iter().rev().map(|(c, _)| *c).collect();
        self.snap = Snapshot {
            ticks,
            total_shuffles:    g.total_shuffles,
            shuffles_per_sec:  g.shuffles_per_sec,
            session_best:      g.session_best,
            session_best_arr:  g.session_best_arr,
            all_time_best:     g.all_time_best,
            all_time_best_arr: g.all_time_best_arr,
            status:            g.status.clone(),
            nickname:          g.nickname.clone(),
            seed_str:          g.seed_str.clone(),
            uptime_secs:       0,
            running:           g.running,
            saved_arrays:      g.saved_arrays.clone(),
        };
        drop(g);
        self.update_uptime();
    }

    fn tick_strip(&mut self) {
        if self.last_tick.elapsed() >= Duration::from_secs(1) {
            self.tick_count += 1;
            self.last_tick = std::time::Instant::now();
        }
    }

    // ── Dashboard sub-panels ──────────────────────────────────────────────────

    fn draw_bar_chart(&self, ui: &mut egui::Ui) {
        let (score, arr) = match self.snap.ticks.last() {
            Some((s, a)) => (*s, *a),
            None => {
                ui.centered_and_justified(|ui| {
                    ui.label(RichText::new("waiting for data…").color(DIM));
                });
                return;
            }
        };

        let rect    = ui.available_rect_before_wrap();
        let painter = ui.painter_at(rect);
        draw_score_bar_chart(&painter, rect, &arr, score);
        ui.allocate_rect(rect, Sense::hover());
    }

    fn draw_history_strip(&self, ui: &mut egui::Ui) {
        const CELL_W: f32 = 48.0;
        const CELL_H: f32 = 44.0;

        let rect    = ui.available_rect_before_wrap();
        let painter = ui.painter_at(rect);
        painter.rect_filled(rect, Rounding::ZERO, STRIP_BG);

        let history = &self.history;
        let n_cells = ((rect.width() / CELL_W) as usize + 1).min(history.len());

        let highlight_threshold = if !history.is_empty() {
            let avg = history.iter().map(|&s| s as f64).sum::<f64>() / history.len() as f64;
            (avg + 2.0).floor() as u32
        } else {
            u32::MAX
        };

        for (i, &score) in history.iter().take(n_cells + 4).enumerate() {
            let x  = rect.left() + i as f32 * CELL_W;
            let cr = Rect::from_min_size(
                Pos2::new(x + 2.0, rect.top() + 2.0),
                Vec2::new(CELL_W - 4.0, CELL_H - 4.0),
            );
            let highlighted = score >= highlight_threshold && score > 0;
            let bg = if highlighted { STRIP_HL } else { STRIP_BG };
            painter.rect_filled(cr, Rounding::same(4.0), bg);
            painter.text(
                cr.center(),
                egui::Align2::CENTER_CENTER,
                score.to_string(),
                FontId::monospace(14.0),
                if highlighted { GREEN } else { TEXT },
            );
        }

        if !history.is_empty() {
            let avg = history.iter().map(|&s| s as f64).sum::<f64>() / history.len() as f64;
            let avg_rect = Rect::from_min_size(
                Pos2::new(rect.right() - 60.0, rect.top()),
                Vec2::new(60.0, CELL_H),
            );
            painter.rect_filled(avg_rect, Rounding::ZERO, PANEL_BG);
            painter.text(
                Pos2::new(rect.right() - 30.0, rect.top() + 12.0),
                egui::Align2::CENTER_TOP,
                "avg",
                FontId::monospace(10.0),
                TEXT_DIM,
            );
            painter.text(
                Pos2::new(rect.right() - 30.0, rect.top() + 24.0),
                egui::Align2::CENTER_TOP,
                format!("{avg:.1}"),
                FontId::monospace(14.0),
                TEXT,
            );
        }

        ui.allocate_rect(rect, Sense::hover());
    }

    fn draw_right_panel(&self, ui: &mut egui::Ui) {
        let uptime = self.uptime_secs
            + self.run_started.map(|t| t.elapsed().as_secs()).unwrap_or(0);
        let h = uptime / 3600;
        let m = (uptime % 3600) / 60;
        let s = uptime % 60;

        ui.add_space(8.0);
        ui.label(RichText::new("UPTIME").small().color(TEXT_DIM));
        ui.label(RichText::new(format!("{h}h {m}m {s}s"))
            .font(FontId::monospace(22.0)).color(TEXT));

        ui.add_space(12.0);
        ui.separator();
        ui.add_space(8.0);

        ui.label(RichText::new("shuffles / sec").small().color(TEXT_DIM));
        ui.label(RichText::new(fmt_large(self.snap.shuffles_per_sec))
            .font(FontId::monospace(18.0)).color(GREEN));

        ui.add_space(10.0);
        ui.label(RichText::new("total shuffles").small().color(TEXT_DIM));
        ui.label(RichText::new(fmt_large(self.snap.total_shuffles))
            .font(FontId::monospace(16.0)).color(TEXT));

        ui.add_space(10.0);
        ui.label(RichText::new("session best").small().color(TEXT_DIM));
        ui.label(RichText::new(format!("{}/25", self.snap.session_best))
            .font(FontId::monospace(22.0)).color(ORANGE));

        ui.add_space(10.0);
        ui.label(RichText::new("all-time best").small().color(TEXT_DIM));
        ui.label(RichText::new(format!("{}/25", self.snap.all_time_best))
            .font(FontId::monospace(22.0)).color(GREEN));

        ui.add_space(10.0);
        if let Some((score, arr)) = self.snap.ticks.last() {
            ui.label(RichText::new(format!("this tick  {}/25", score)).small().color(TEXT_DIM));
            draw_array_grid(ui, arr);
        } else if self.snap.session_best > 0 {
            ui.label(RichText::new("best array (session)").small().color(TEXT_DIM));
            draw_array_grid(ui, &self.snap.session_best_arr);
        }

        ui.add_space(10.0);
        ui.separator();
        ui.add_space(6.0);

        if !self.snap.seed_str.is_empty() {
            ui.label(RichText::new("seed").small().color(TEXT_DIM));
            ui.label(RichText::new(&self.snap.seed_str)
                .font(FontId::monospace(11.0)).color(TEXT_DIM));
            ui.add_space(6.0);
        }

        ui.label(RichText::new(&self.snap.status).small().color(TEXT_DIM));
    }

    // ── Array History tab ─────────────────────────────────────────────────────

    fn draw_arrays_tab(&mut self, ui: &mut egui::Ui) {
        // Full-screen detail view for a single selected entry.
        if let Some(id) = self.selected_array {
            if let Some(entry) = self.snap.saved_arrays.iter().find(|e| e.id == id).cloned() {
                self.draw_array_detail(ui, &entry);
                return;
            } else {
                // Entry no longer exists (e.g. trimmed from history) — go back.
                self.selected_array = None;
            }
        }

        ui.add_space(12.0);

        ui.horizontal(|ui| {
            ui.label(RichText::new("ARRAY HISTORY").color(ORANGE).strong().font(FontId::proportional(15.0)));
            ui.add_space(8.0);
            ui.label(
                RichText::new(format!("{} saved ({}+/25)", self.snap.saved_arrays.len(), history::MIN_SAVED_CORRECT))
                    .small().color(TEXT_DIM),
            );

            ui.with_layout(Layout::right_to_left(Align::Center), |ui| {
                let recent_sel  = self.array_sort == ArraySort::Recent;
                let highest_sel = self.array_sort == ArraySort::Highest;

                let recent_btn = egui::Button::new(
                    RichText::new("Recent").color(if recent_sel { Color32::WHITE } else { TEXT_DIM })
                ).fill(if recent_sel { Color32::from_rgb(50, 50, 55) } else { Color32::TRANSPARENT })
                 .rounding(Rounding::same(5.0));

                let highest_btn = egui::Button::new(
                    RichText::new("Highest").color(if highest_sel { Color32::WHITE } else { TEXT_DIM })
                ).fill(if highest_sel { Color32::from_rgb(50, 50, 55) } else { Color32::TRANSPARENT })
                 .rounding(Rounding::same(5.0));

                if ui.add(recent_btn).clicked()  { self.array_sort = ArraySort::Recent; }
                if ui.add(highest_btn).clicked() { self.array_sort = ArraySort::Highest; }
                ui.label(RichText::new("Sort:").small().color(TEXT_DIM));
            });
        });

        ui.add_space(8.0);
        ui.separator();
        ui.add_space(8.0);

        if self.snap.saved_arrays.is_empty() {
            ui.add_space(40.0);
            ui.vertical_centered(|ui| {
                ui.label(RichText::new(format!(
                    "No arrays with {}+ correct yet — keep grinding!",
                    history::MIN_SAVED_CORRECT
                )).color(DIM));
            });
            return;
        }

        let mut entries: Vec<SavedArray> = self.snap.saved_arrays.clone();
        match self.array_sort {
            ArraySort::Recent  => entries.sort_by(|a, b| b.timestamp.cmp(&a.timestamp).then(b.id.cmp(&a.id))),
            ArraySort::Highest => entries.sort_by(|a, b| b.correct.cmp(&a.correct).then(b.timestamp.cmp(&a.timestamp))),
        }

        egui::ScrollArea::vertical()
            .auto_shrink([false, false])
            .show(ui, |ui| {
                for entry in &entries {
                    self.draw_array_card(ui, entry);
                    ui.add_space(6.0);
                }
                ui.add_space(8.0);
            });
    }

    fn draw_array_card(&mut self, ui: &mut egui::Ui, entry: &SavedArray) {
        let frame = Frame::none()
            .fill(STRIP_BG)
            .rounding(Rounding::same(6.0))
            .inner_margin(egui::Margin::symmetric(10.0, 8.0));

        let inner = frame.show(ui, |ui| {
            ui.set_width(ui.available_width());

            ui.horizontal(|ui| {
                draw_array_bars_sized(ui, &entry.arr, Vec2::new(150.0, 64.0));

                ui.add_space(10.0);

                ui.vertical(|ui| {
                    ui.add_space(2.0);
                    let score_color = if entry.correct >= 20 { GREEN } else { ORANGE };
                    ui.label(
                        RichText::new(format!("{}/25 correct", entry.correct))
                            .font(FontId::monospace(16.0))
                            .color(score_color)
                            .strong(),
                    );
                    ui.label(
                        RichText::new(history::format_relative(entry.timestamp))
                            .font(FontId::monospace(11.0))
                            .color(TEXT_DIM),
                    );
                });

                ui.with_layout(Layout::right_to_left(Align::Center), |ui| {
                    ui.label(RichText::new("›").font(FontId::proportional(18.0)).color(TEXT_DIM));
                });
            });
        });

        let response = ui.interact(inner.response.rect, ui.id().with(("array_card", entry.id)), Sense::click());
        if response.clicked() {
            self.selected_array = Some(entry.id);
        }
        if response.hovered() {
            ui.painter().rect_stroke(inner.response.rect, Rounding::same(6.0), Stroke::new(1.0, STRIP_HL));
        }
    }

    /// Full-screen detail view for a single saved array, with a big bar
    /// chart and a stats panel. Reached by clicking a card in the list.
    fn draw_array_detail(&mut self, ui: &mut egui::Ui, entry: &SavedArray) {
        ui.add_space(12.0);

        ui.horizontal(|ui| {
            let back_btn = egui::Button::new(
                RichText::new("←  Back").color(TEXT).strong()
            ).fill(Color32::from_rgb(40, 40, 40)).rounding(Rounding::same(6.0));

            if ui.add(back_btn).clicked() {
                self.selected_array = None;
            }

            ui.add_space(16.0);

            let score_color = if entry.correct >= 20 { GREEN } else { ORANGE };
            ui.label(
                RichText::new(format!("{}/25 correct", entry.correct))
                    .font(FontId::proportional(20.0))
                    .color(score_color)
                    .strong(),
            );

            ui.with_layout(Layout::right_to_left(Align::Center), |ui| {
                ui.label(RichText::new(history::format_relative(entry.timestamp)).small().color(TEXT_DIM));
            });
        });

        ui.add_space(12.0);
        ui.separator();
        ui.add_space(12.0);

        egui::ScrollArea::vertical()
            .auto_shrink([false, false])
            .show(ui, |ui| {
                // ── Big bar chart ─────────────────────────────────────────
                let avail_w = ui.available_width();
                let chart_h = (ui.available_height() * 0.5).clamp(180.0, 420.0);
                let (_, rect) = ui.allocate_space(Vec2::new(avail_w, chart_h));
                let painter = ui.painter_at(rect);
                painter.rect_filled(rect, Rounding::same(6.0), Color32::from_rgb(20, 20, 20));
                draw_score_bar_chart(&painter, rect.shrink(10.0), &entry.arr, entry.correct);

                ui.add_space(20.0);
                ui.separator();
                ui.add_space(16.0);

                // ── Stats panel ───────────────────────────────────────────
                ui.label(RichText::new("STATS").color(ORANGE).strong().font(FontId::proportional(14.0)));
                ui.add_space(10.0);

                egui::Grid::new(("array_detail_full", entry.id))
                    .num_columns(2)
                    .spacing([32.0, 12.0])
                    .show(ui, |ui| {
                        ui.label(RichText::new("Found").color(TEXT_DIM).monospace());
                        ui.label(RichText::new(history::format_timestamp(entry.timestamp))
                            .font(FontId::monospace(15.0)).color(TEXT));
                        ui.end_row();

                        ui.label(RichText::new("Seed").color(TEXT_DIM).monospace());
                        ui.label(RichText::new(&entry.seed)
                            .font(FontId::monospace(15.0)).color(TEXT));
                        ui.end_row();

                        ui.label(RichText::new("Index").color(TEXT_DIM).monospace());
                        ui.label(RichText::new(entry.index.to_string())
                            .font(FontId::monospace(15.0)).color(TEXT));
                        ui.end_row();

                        ui.label(RichText::new("Shuffles / sec").color(TEXT_DIM).monospace());
                        ui.label(RichText::new(fmt_large(entry.rate))
                            .font(FontId::monospace(15.0)).color(GREEN));
                        ui.end_row();

                        ui.label(RichText::new("Total shuffles").color(TEXT_DIM).monospace());
                        ui.label(RichText::new(fmt_large(entry.total_shuffles))
                            .font(FontId::monospace(15.0)).color(TEXT));
                        ui.end_row();

                        ui.label(RichText::new("XP").color(TEXT_DIM).monospace());
                        ui.label(RichText::new(format!("{:.2}", entry.xp()))
                            .font(FontId::monospace(15.0)).color(ORANGE).strong());
                        ui.end_row();
                    });

                ui.add_space(16.0);
            });
    }

    // ── Settings tab ──────────────────────────────────────────────────────────

    fn draw_settings(&mut self, ui: &mut egui::Ui) {
        egui::ScrollArea::vertical().show(ui, |ui| {
            ui.add_space(12.0);

            // ── Identity ────────────────────────────────────────────────────
            section_header(ui, "Identity");

            ui.add_space(6.0);
            field_row(ui, "UUID",     &mut self.settings.uuid);
            field_row(ui, "Nickname", &mut self.settings.nickname);
            field_row(ui, "Code",     &mut self.settings.code);

            ui.add_space(16.0);

            // ── Compute backend ─────────────────────────────────────────────
            section_header(ui, "Compute");

            ui.add_space(6.0);
            ui.horizontal(|ui| {
                ui.label(RichText::new("Backend").color(TEXT_DIM).monospace());
                ui.add_space(8.0);
                egui::ComboBox::from_id_salt("backend_combo")
                    .selected_text(self.settings.backend.to_string())
                    .show_ui(ui, |ui| {
                        ui.selectable_value(
                            &mut self.settings.backend,
                            ComputeBackend::Gpu,
                            "GPU",
                        );
                        ui.selectable_value(
                            &mut self.settings.backend,
                            ComputeBackend::Cpu,
                            "CPU",
                        );
                    });
            });

            ui.add_space(8.0);

            match self.settings.backend {
                ComputeBackend::Gpu => {
                    field_row(ui, "GPU Arch",   &mut self.settings.gpu_arch);
                    field_row(ui, "Blocks",     &mut self.settings.blocks);
                    field_row(ui, "Threads",    &mut self.settings.threads);
                    ui.add_space(4.0);
                    ui.label(RichText::new(
                        "  gpu_arch: e.g. gfx1201 (RDNA4), gfx1100 (RDNA3), sm_86 (CUDA Ampere)"
                    ).small().color(TEXT_DIM));
                    ui.label(RichText::new(
                        "  blocks × threads = total GPU threads launched per kernel"
                    ).small().color(TEXT_DIM));
                }
                ComputeBackend::Cpu => {
                    field_row(ui, "CPU Threads", &mut self.settings.cpu_threads);
                    ui.add_space(4.0);
                    ui.label(RichText::new(
                        "  0 = use all available CPU cores (via rayon)"
                    ).small().color(TEXT_DIM));
                }
            }

            ui.add_space(20.0);
            ui.separator();
            ui.add_space(12.0);

            // ── Apply / Reset ───────────────────────────────────────────────
            ui.horizontal(|ui| {
                let apply_btn = egui::Button::new(
                    RichText::new("  Save & Restart  ").color(Color32::WHITE).strong()
                ).fill(BTN_GREEN).rounding(Rounding::same(6.0));

                if ui.add(apply_btn).clicked() {
                    match self.settings.to_config() {
                        Ok(cfg) => {
                            match cfg.save() {
                                Ok(()) => {
                                    self.current_cfg = cfg.clone();
                                    // If worker is running, restart it with new config.
                                    // If stopped, just save — user presses Start themselves.
                                    if self.snap.running {
                                        self.stats.send_cmd(WorkerCmd::Stop);
                                        self.stats.send_cmd(WorkerCmd::Start(cfg));
                                    }
                                    self.settings.feedback = Some((false, "Saved ✓".into()));
                                }
                                Err(e) => {
                                    self.settings.feedback = Some((true, format!("Save failed: {e}")));
                                }
                            }
                        }
                        Err(e) => {
                            self.settings.feedback = Some((true, e));
                        }
                    }
                }

                ui.add_space(12.0);

                let reset_btn = egui::Button::new(
                    RichText::new("  Reset  ").color(TEXT_DIM)
                ).fill(Color32::from_rgb(40, 40, 40)).rounding(Rounding::same(6.0));

                if ui.add(reset_btn).clicked() {
                    self.settings = SettingsEdit::from_config(&self.current_cfg);
                }
            });

            // Feedback message
            if let Some((is_err, msg)) = &self.settings.feedback {
                ui.add_space(8.0);
                let color = if *is_err { Color32::from_rgb(220, 80, 80) } else { GREEN };
                ui.label(RichText::new(msg).color(color).monospace());
            }

            ui.add_space(20.0);
        });
    }
}

// ── Helpers ───────────────────────────────────────────────────────────────────

fn section_header(ui: &mut egui::Ui, title: &str) {
    ui.label(RichText::new(title).color(ORANGE).strong().font(FontId::proportional(15.0)));
    ui.add(egui::Separator::default().spacing(4.0));
}

fn field_row(ui: &mut egui::Ui, label: &str, value: &mut String) {
    ui.horizontal(|ui| {
        ui.label(RichText::new(format!("{:<12}", label)).color(TEXT_DIM).monospace());
        ui.add_space(4.0);
        let te = egui::TextEdit::singleline(value)
            .desired_width(320.0)
            .font(FontId::monospace(13.0))
            .text_color(TEXT);
        ui.add(te);
    });
    ui.add_space(4.0);
}

fn draw_array_grid(ui: &mut egui::Ui, arr: &[u8; 25]) {
    draw_array_grid_sized(ui, arr, 26.0);
}

fn draw_array_grid_sized(ui: &mut egui::Ui, arr: &[u8; 25], cell: f32) {
    const GAP: f32 = 3.0;
    let grid_w = 5.0 * cell + 4.0 * GAP;
    let grid_h = 5.0 * cell + 4.0 * GAP;
    let (_, rect) = ui.allocate_space(Vec2::new(grid_w, grid_h));
    let painter   = ui.painter_at(rect);

    for (i, &v) in arr.iter().enumerate() {
        let row = i / 5;
        let col = i % 5;
        let x   = rect.left() + col as f32 * (cell + GAP);
        let y   = rect.top()  + row as f32 * (cell + GAP);
        let cr  = Rect::from_min_size(Pos2::new(x, y), Vec2::splat(cell));
        let correct = v == (i + 1) as u8;
        let bg = if correct { Color32::from_rgb(30, 70, 45) } else { Color32::from_rgb(80, 38, 20) };
        painter.rect_filled(cr, Rounding::same(3.0), bg);
        let border = if correct { GREEN } else { ORANGE };
        painter.rect_stroke(cr, Rounding::same(3.0), Stroke::new(1.0, border));
        let font_size = (cell * 0.42).max(7.0);
        painter.text(
            cr.center(), egui::Align2::CENTER_CENTER,
            v.to_string(), FontId::monospace(font_size),
            if correct { GREEN } else { ORANGE },
        );
    }
}

/// Compact bar-chart rendering of a 25-element array, used in the Array
/// History tab. Bar height = value/25; green if the position is correct
/// (value == position), orange otherwise.
fn draw_array_bars_sized(ui: &mut egui::Ui, arr: &[u8; 25], size: Vec2) {
    let (_, rect) = ui.allocate_space(size);
    let painter = ui.painter_at(rect);

    painter.rect_filled(rect, Rounding::same(4.0), Color32::from_rgb(20, 20, 20));

    let n = arr.len();
    let bar_w = rect.width() / n as f32;
    let gap = (bar_w * 0.18).max(1.0);
    let bar_inner = (bar_w - gap).max(1.0);
    let baseline = rect.bottom() - 1.0;
    let chart_h = rect.height() - 2.0;

    for (i, &v) in arr.iter().enumerate() {
        let correct = v == (i + 1) as u8;
        let color = if correct { GREEN } else { ORANGE };
        let frac = v as f32 / 25.0;
        let h = (frac * chart_h).max(2.0);
        let x_left = rect.left() + i as f32 * bar_w + gap * 0.5;

        let bar_rect = Rect::from_min_max(
            Pos2::new(x_left, baseline - h),
            Pos2::new(x_left + bar_inner, baseline),
        );
        painter.rect_filled(bar_rect, Rounding::same(1.0), color);
    }
}

/// Full labeled 25-bar chart (value labels on top, position labels on the
/// bottom axis, "X/25 correct" caption in the top-left). Used for both the
/// live dashboard chart and the Array History detail view.
fn draw_score_bar_chart(painter: &egui::Painter, rect: Rect, arr: &[u8; 25], score: u32) {
    let baseline  = rect.bottom() - 24.0;
    let chart_h   = rect.height() - 24.0 - 28.0;
    let n         = 25usize;
    let bar_w     = rect.width() / n as f32;
    let gap       = (bar_w * 0.12).max(2.0);
    let bar_inner = bar_w - gap;

    for i in 0..n {
        let v       = arr[i];
        let correct = v == (i + 1) as u8;
        let color   = if correct { GREEN } else { ORANGE };
        let frac    = v as f32 / 25.0;
        let h       = (frac * chart_h).max(4.0);
        let x_left  = rect.left() + i as f32 * bar_w + gap * 0.5;

        let bar_rect = Rect::from_min_max(
            Pos2::new(x_left,             baseline - h),
            Pos2::new(x_left + bar_inner, baseline),
        );
        painter.rect_filled(bar_rect, Rounding::same(3.0), color);

        painter.text(
            Pos2::new(x_left + bar_inner * 0.5, baseline - h - 4.0),
            egui::Align2::CENTER_BOTTOM,
            v.to_string(),
            FontId::monospace(10.0),
            color,
        );
        painter.text(
            Pos2::new(x_left + bar_inner * 0.5, rect.bottom() - 12.0),
            egui::Align2::CENTER_CENTER,
            (i + 1).to_string(),
            FontId::monospace(9.0),
            DIM,
        );
    }

    painter.text(
        Pos2::new(rect.left() + 8.0, rect.top() + 4.0),
        egui::Align2::LEFT_TOP,
        format!("{}/25 correct", score),
        FontId::monospace(11.0),
        TEXT_DIM,
    );
}

fn fmt_large(n: u64) -> String {
    if n >= 1_000_000_000_000 { format!("{:.2}T", n as f64 / 1e12) }
    else if n >= 1_000_000_000 { format!("{:.2}B", n as f64 / 1e9) }
    else if n >= 1_000_000     { format!("{:.2}M", n as f64 / 1e6) }
    else if n >= 1_000         { format!("{:.1}K", n as f64 / 1e3) }
    else                       { n.to_string() }
}

// ── eframe App ────────────────────────────────────────────────────────────────

impl eframe::App for BogoApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        ctx.set_visuals(Visuals::dark());
        self.refresh_snapshot();
        self.tick_strip();
        ctx.request_repaint_after(Duration::from_millis(200));

        let mut style = (*ctx.style()).clone();
        style.visuals.panel_fill        = BG;
        style.visuals.window_fill       = PANEL_BG;
        style.visuals.widgets.inactive.bg_fill   = Color32::from_rgb(38, 38, 38);
        style.visuals.widgets.hovered.bg_fill    = Color32::from_rgb(50, 50, 50);
        style.visuals.widgets.active.bg_fill     = Color32::from_rgb(60, 60, 60);
        ctx.set_style(style);

        // ── Top bar ─────────────────────────────────────────────────────────
        egui::TopBottomPanel::top("topbar")
            .frame(Frame::none().fill(BG).inner_margin(egui::Margin::symmetric(12.0, 8.0)))
            .show(ctx, |ui| {
                ui.horizontal(|ui| {
                    // Logo
                    ui.label(RichText::new("bogo").font(FontId::proportional(22.0)).color(TEXT).strong());
                    ui.label(RichText::new("·").font(FontId::proportional(22.0)).color(ORANGE).strong());
                    ui.add_space(8.0);

                    // Tab buttons
                    let dash_sel = self.active_tab == Tab::Dashboard;
                    let arrays_sel = self.active_tab == Tab::Arrays;
                    let settings_sel = self.active_tab == Tab::Settings;

                    let dash_btn = egui::Button::new(
                        RichText::new("Dashboard").color(if dash_sel { Color32::WHITE } else { TEXT_DIM })
                    ).fill(if dash_sel { Color32::from_rgb(50, 50, 55) } else { Color32::TRANSPARENT })
                     .rounding(Rounding::same(5.0));

                    let arrays_label = if self.snap.saved_arrays.is_empty() {
                        "Arrays".to_string()
                    } else {
                        format!("Arrays ({})", self.snap.saved_arrays.len())
                    };
                    let arrays_btn = egui::Button::new(
                        RichText::new(arrays_label).color(if arrays_sel { Color32::WHITE } else { TEXT_DIM })
                    ).fill(if arrays_sel { Color32::from_rgb(50, 50, 55) } else { Color32::TRANSPARENT })
                     .rounding(Rounding::same(5.0));

                    let sett_btn = egui::Button::new(
                        RichText::new("Settings").color(if settings_sel { Color32::WHITE } else { TEXT_DIM })
                    ).fill(if settings_sel { Color32::from_rgb(50, 50, 55) } else { Color32::TRANSPARENT })
                     .rounding(Rounding::same(5.0));

                    if ui.add(dash_btn).clicked() { self.active_tab = Tab::Dashboard; }
                    if ui.add(arrays_btn).clicked() { self.active_tab = Tab::Arrays; }
                    if ui.add(sett_btn).clicked() {
                        // Re-sync edit state from current config when opening settings.
                        if self.active_tab != Tab::Settings {
                            self.settings = SettingsEdit::from_config(&self.current_cfg);
                        }
                        self.active_tab = Tab::Settings;
                    }

                    ui.add_space(16.0);

                    // Nickname
                    if !self.snap.nickname.is_empty() {
                        ui.label(RichText::new(format!("@{}", self.snap.nickname))
                            .font(FontId::monospace(13.0)).color(GREEN));
                    }

                    // Centre: current tick score (dashboard only)
                    if self.active_tab == Tab::Dashboard {
                        if let Some((latest_score, _)) = self.snap.ticks.last() {
                            ui.with_layout(Layout::right_to_left(Align::Center), |ui| {
                                // Stop / Start buttons on the right
                                let running = self.snap.running;
                                if running {
                                    let stop_btn = egui::Button::new(
                                        RichText::new("  ■ Stop  ").color(Color32::WHITE).strong()
                                    ).fill(BTN_RED).rounding(Rounding::same(6.0));
                                    if ui.add(stop_btn).clicked() {
                                        self.stats.send_cmd(WorkerCmd::Stop);
                                    }
                                } else {
                                    let start_btn = egui::Button::new(
                                        RichText::new("  ▶ Start  ").color(Color32::WHITE).strong()
                                    ).fill(BTN_GREEN).rounding(Rounding::same(6.0));
                                    if ui.add(start_btn).clicked() {
                                        self.stats.send_cmd(WorkerCmd::Start(self.current_cfg.clone()));
                                    }
                                }

                                ui.add_space(16.0);
                                ui.label(
                                    RichText::new(format!("● {} this tick", latest_score))
                                        .font(FontId::monospace(13.0)).color(GREEN),
                                );
                            });
                        } else {
                            // No ticks yet — still show stop/start on right
                            ui.with_layout(Layout::right_to_left(Align::Center), |ui| {
                                let running = self.snap.running;
                                if running {
                                    let stop_btn = egui::Button::new(
                                        RichText::new("  ■ Stop  ").color(Color32::WHITE).strong()
                                    ).fill(BTN_RED).rounding(Rounding::same(6.0));
                                    if ui.add(stop_btn).clicked() {
                                        self.stats.send_cmd(WorkerCmd::Stop);
                                    }
                                } else {
                                    let start_btn = egui::Button::new(
                                        RichText::new("  ▶ Start  ").color(Color32::WHITE).strong()
                                    ).fill(BTN_GREEN).rounding(Rounding::same(6.0));
                                    if ui.add(start_btn).clicked() {
                                        self.stats.send_cmd(WorkerCmd::Start(self.current_cfg.clone()));
                                    }
                                }
                            });
                        }
                    }
                });
            });

        match self.active_tab {
            Tab::Dashboard => {
                // ── Bottom history strip ─────────────────────────────────────
                egui::TopBottomPanel::bottom("strip")
                    .exact_height(48.0)
                    .frame(Frame::none().fill(STRIP_BG))
                    .show(ctx, |ui| {
                        self.draw_history_strip(ui);
                    });

                // ── Right panel ──────────────────────────────────────────────
                egui::SidePanel::right("right_panel")
                    .min_width(200.0)
                    .max_width(260.0)
                    .frame(Frame::none().fill(PANEL_BG).inner_margin(egui::Margin::symmetric(12.0, 8.0)))
                    .show(ctx, |ui| {
                        egui::ScrollArea::vertical().show(ui, |ui| {
                            self.draw_right_panel(ui);
                        });
                    });

                // ── Central chart ────────────────────────────────────────────
                egui::CentralPanel::default()
                    .frame(Frame::none().fill(BG).inner_margin(egui::Margin::symmetric(8.0, 8.0)))
                    .show(ctx, |ui| {
                        self.draw_bar_chart(ui);
                    });
            }

            Tab::Arrays => {
                egui::CentralPanel::default()
                    .frame(Frame::none().fill(BG).inner_margin(egui::Margin::symmetric(20.0, 8.0)))
                    .show(ctx, |ui| {
                        self.draw_arrays_tab(ui);
                    });
            }

            Tab::Settings => {
                egui::CentralPanel::default()
                    .frame(Frame::none().fill(BG).inner_margin(egui::Margin::symmetric(32.0, 16.0)))
                    .show(ctx, |ui| {
                        self.draw_settings(ui);
                    });
            }
        }
    }
}
