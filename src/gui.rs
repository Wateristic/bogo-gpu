// src/gui.rs — Bogo-GPU dashboard (egui 0.29 / eframe 0.29)
//
// Layout mirrors the reference screenshot:
//   ┌──────────────────────────────────────────────────────┬──────────────┐
//   │  bar chart (best_correct per tick, 30-bar window)   │  right panel │
//   ├──────────────────────────────────────────────────────┤   (stats)    │
//   │  scrolling history strip (all-time best scores)     │              │
//   └──────────────────────────────────────────────────────┴──────────────┘

use std::sync::Arc;
use std::time::Duration;

use egui::{
    Align, Color32, FontId, Frame, Layout, Pos2, Rect,
    RichText, Rounding, Sense, Stroke, Vec2, Visuals,
};

use crate::worker::GuiStats;

// ─── Palette ──────────────────────────────────────────────────────────────────

const BG:         Color32 = Color32::from_rgb(18, 18, 18);
const PANEL_BG:   Color32 = Color32::from_rgb(26, 26, 26);
const ORANGE:     Color32 = Color32::from_rgb(232, 107, 66);
const GREEN:      Color32 = Color32::from_rgb(100, 195, 140);
const DIM:        Color32 = Color32::from_rgb(100, 100, 100);
const TEXT:       Color32 = Color32::from_rgb(220, 220, 220);
const TEXT_DIM:   Color32 = Color32::from_rgb(130, 130, 130);
const STRIP_BG:   Color32 = Color32::from_rgb(36, 36, 36);
const STRIP_HL:   Color32 = Color32::from_rgb(70, 70, 70);

// ─── State ────────────────────────────────────────────────────────────────────

pub struct BogoApp {
    stats:        Arc<GuiStats>,
    /// Snapshot taken each frame to avoid holding the lock while painting.
    snap:         Snapshot,
    /// History strip scroll offset (auto-advances each second).
    strip_offset: f32,
    last_tick:    std::time::Instant,
    tick_count:   usize,
    /// Cached history scores for the strip (newest first).
    history:      Vec<u32>,
}

#[derive(Default)]
struct Snapshot {
    ticks:            Vec<(u32, [u8; 25])>,  // (best_correct, arr)
    total_shuffles:   u64,
    shuffles_per_sec: u64,
    all_time_best:    u32,
    all_time_arr:     [u8; 25],
    status:           String,
    nickname:         String,
    seed_str:         String,
    uptime_secs:      u64,
}

impl BogoApp {
    pub fn new(_cc: &eframe::CreationContext<'_>, stats: Arc<GuiStats>) -> Self {
        BogoApp {
            stats,
            snap: Snapshot::default(),
            strip_offset: 0.0,
            last_tick: std::time::Instant::now(),
            tick_count: 0,
            history: Vec::new(),
        }
    }

    fn refresh_snapshot(&mut self) {
        let g = self.stats.inner.lock().unwrap();
        let ticks: Vec<_> = g.ticks.iter().map(|t| (t.best_correct, t.best_arr)).collect();

        // Rebuild history from ticks (newest first for the strip).
        self.history = ticks.iter().rev().map(|(c, _)| *c).collect();

        self.snap = Snapshot {
            ticks,
            total_shuffles:   g.total_shuffles,
            shuffles_per_sec: g.shuffles_per_sec,
            all_time_best:    g.all_time_best,
            all_time_arr:     g.all_time_arr,
            status:           g.status.clone(),
            nickname:         g.nickname.clone(),
            seed_str:         g.seed_str.clone(),
            uptime_secs:      g.start.elapsed().as_secs(),
        };
    }

    /// Advance the strip scroll every second.
    fn tick_strip(&mut self) {
        if self.last_tick.elapsed() >= Duration::from_secs(1) {
            self.tick_count += 1;
            self.last_tick = std::time::Instant::now();
            // Each cell is CELL_W wide; advance one cell per second.
            // We just reset; the strip always renders newest-first from offset.
        }
    }

    // ── Sub-panels ─────────────────────────────────────────────────────────────

    fn draw_bar_chart(&self, ui: &mut egui::Ui) {
        // 25 bars — one per array position.
        // Bar height = the value stored at that position (1..=25).
        // Green = value matches position (correct). Orange = wrong.
        // Refreshes every second with the latest tick's best array.
        let (score, arr) = match self.snap.ticks.last() {
            Some((s, a)) => (*s, *a),
            None => {
                ui.centered_and_justified(|ui| {
                    ui.label(RichText::new("waiting for data...").color(DIM));
                });
                return;
            }
        };

        let rect    = ui.available_rect_before_wrap();
        let painter = ui.painter_at(rect);

        let baseline  = rect.bottom() - 24.0; // bottom of bars
        let chart_h   = rect.height() - 24.0 - 28.0; // usable bar height
        let n         = 25usize;
        let bar_w     = rect.width() / n as f32;
        let gap       = (bar_w * 0.12).max(2.0);
        let bar_inner = bar_w - gap;

        for i in 0..n {
            let v       = arr[i];
            let correct = v == (i + 1) as u8;
            let color   = if correct { GREEN } else { ORANGE };

            // Height proportional to value (1=shortest, 25=tallest).
            let frac   = v as f32 / 25.0;
            let h      = (frac * chart_h).max(4.0);
            let x_left = rect.left() + i as f32 * bar_w + gap * 0.5;

            let bar_rect = Rect::from_min_max(
                Pos2::new(x_left,             baseline - h),
                Pos2::new(x_left + bar_inner, baseline),
            );
            painter.rect_filled(bar_rect, Rounding::same(3.0), color);

            // Value label above bar.
            painter.text(
                Pos2::new(x_left + bar_inner * 0.5, baseline - h - 4.0),
                egui::Align2::CENTER_BOTTOM,
                v.to_string(),
                FontId::monospace(10.0),
                color,
            );

            // Position index label below bar (1-based).
            painter.text(
                Pos2::new(x_left + bar_inner * 0.5, rect.bottom() - 12.0),
                egui::Align2::CENTER_CENTER,
                (i + 1).to_string(),
                FontId::monospace(9.0),
                DIM,
            );
        }

        // Score overlay top-left.
        painter.text(
            Pos2::new(rect.left() + 8.0, rect.top() + 4.0),
            egui::Align2::LEFT_TOP,
            format!("{}/25 correct", score),
            FontId::monospace(11.0),
            TEXT_DIM,
        );

        ui.allocate_rect(rect, Sense::hover());
    }

    fn draw_history_strip(&self, ui: &mut egui::Ui) {
        const CELL_W: f32 = 48.0;
        const CELL_H: f32 = 44.0;

        let rect  = ui.available_rect_before_wrap();
        let painter = ui.painter_at(rect);
        painter.rect_filled(rect, Rounding::ZERO, STRIP_BG);

        let history = &self.history;
        let n_cells = ((rect.width() / CELL_W) as usize + 1).min(history.len());

        for (i, &score) in history.iter().take(n_cells + 4).enumerate() {
            let x  = rect.left() + i as f32 * CELL_W;
            let cr = Rect::from_min_size(Pos2::new(x + 2.0, rect.top() + 2.0),
                                          Vec2::new(CELL_W - 4.0, CELL_H - 4.0));

            let bg = if score == self.snap.all_time_best && score > 0 { STRIP_HL } else { STRIP_BG };
            painter.rect_filled(cr, Rounding::same(4.0), bg);

            // Score number
            painter.text(
                cr.center(),
                egui::Align2::CENTER_CENTER,
                score.to_string(),
                FontId::monospace(14.0),
                if score == self.snap.all_time_best && score > 0 { GREEN } else { TEXT },
            );
        }

        // Average label on the right.
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
        let uptime = self.snap.uptime_secs;
        let h = uptime / 3600;
        let m = (uptime % 3600) / 60;
        let s = uptime % 60;

        ui.add_space(8.0);

        // Uptime
        ui.label(RichText::new("UPTIME").small().color(TEXT_DIM));
        ui.label(RichText::new(format!("{h}h {m}m {s}s"))
            .font(FontId::monospace(22.0))
            .color(TEXT));

        ui.add_space(12.0);
        ui.separator();
        ui.add_space(8.0);

        // Shuffles/sec
        ui.label(RichText::new("shuffles / sec").small().color(TEXT_DIM));
        let sps = self.snap.shuffles_per_sec;
        let sps_str = fmt_large(sps);
        ui.label(RichText::new(&sps_str).font(FontId::monospace(18.0)).color(GREEN));

        ui.add_space(10.0);

        // Total shuffles
        ui.label(RichText::new("total shuffles").small().color(TEXT_DIM));
        ui.label(RichText::new(fmt_large(self.snap.total_shuffles))
            .font(FontId::monospace(16.0)).color(TEXT));

        ui.add_space(10.0);

        // All-time best score
        ui.label(RichText::new("all-time best").small().color(TEXT_DIM));
        ui.label(RichText::new(format!("{}/25", self.snap.all_time_best))
            .font(FontId::monospace(22.0)).color(ORANGE));

        ui.add_space(10.0);

        // This tick's best array — refreshes every second
        // orange cell = value is in the wrong position
        // green  cell = value is in the correct position
        if let Some((score, arr)) = self.snap.ticks.last() {
            ui.label(RichText::new(format!("this tick  {}/25", score)).small().color(TEXT_DIM));
            draw_array_grid(ui, arr);
        } else if self.snap.all_time_best > 0 {
            ui.label(RichText::new("best array").small().color(TEXT_DIM));
            draw_array_grid(ui, &self.snap.all_time_arr);
        }

        ui.add_space(10.0);
        ui.separator();
        ui.add_space(6.0);

        // Seed
        if !self.snap.seed_str.is_empty() {
            ui.label(RichText::new("seed").small().color(TEXT_DIM));
            ui.label(RichText::new(&self.snap.seed_str)
                .font(FontId::monospace(11.0)).color(TEXT_DIM));
            ui.add_space(6.0);
        }

        // Status
        ui.label(RichText::new(&self.snap.status).small().color(TEXT_DIM));
    }
}

// ── Draw a 5×5 array grid — green = correct position, orange = wrong position ──

fn draw_array_grid(ui: &mut egui::Ui, arr: &[u8; 25]) {
    const CELL: f32 = 26.0;
    const GAP:  f32 = 3.0;
    let grid_w = 5.0 * CELL + 4.0 * GAP;
    let grid_h = 5.0 * CELL + 4.0 * GAP;
    let (_, rect) = ui.allocate_space(Vec2::new(grid_w, grid_h));
    let painter   = ui.painter_at(rect);

    for (i, &v) in arr.iter().enumerate() {
        let row = i / 5;
        let col = i % 5;
        let x   = rect.left() + col as f32 * (CELL + GAP);
        let y   = rect.top()  + row as f32 * (CELL + GAP);
        let cr  = Rect::from_min_size(Pos2::new(x, y), Vec2::splat(CELL));

        let correct = v == (i + 1) as u8;

        // Background: dark green tint if correct, dark orange tint if not
        let bg = if correct {
            Color32::from_rgb(30, 70, 45)
        } else {
            Color32::from_rgb(80, 38, 20)
        };
        painter.rect_filled(cr, Rounding::same(3.0), bg);

        // Border in the accent colour
        let border = if correct { GREEN } else { ORANGE };
        painter.rect_stroke(cr, Rounding::same(3.0), Stroke::new(1.0, border));

        // Value text
        let fg = if correct { GREEN } else { ORANGE };
        painter.text(
            cr.center(),
            egui::Align2::CENTER_CENTER,
            v.to_string(),
            FontId::monospace(11.0),
            fg,
        );
    }
}

// ── Number formatting ─────────────────────────────────────────────────────────

fn fmt_large(n: u64) -> String {
    if n >= 1_000_000_000_000 {
        format!("{:.2}T", n as f64 / 1e12)
    } else if n >= 1_000_000_000 {
        format!("{:.2}B", n as f64 / 1e9)
    } else if n >= 1_000_000 {
        format!("{:.2}M", n as f64 / 1e6)
    } else if n >= 1_000 {
        format!("{:.1}K", n as f64 / 1e3)
    } else {
        n.to_string()
    }
}

// ── eframe App impl ───────────────────────────────────────────────────────────

impl eframe::App for BogoApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        // Dark theme.
        ctx.set_visuals(Visuals::dark());

        // Refresh data from the worker every frame (lock is brief).
        self.refresh_snapshot();
        self.tick_strip();

        // Request continuous repaints so we update as data arrives.
        ctx.request_repaint_after(Duration::from_millis(200));

        // Custom style tweaks.
        let mut style = (*ctx.style()).clone();
        style.visuals.panel_fill = BG;
        style.visuals.window_fill = PANEL_BG;
        ctx.set_style(style);

        // ── Top bar ─────────────────────────────────────────────────────────────
        egui::TopBottomPanel::top("topbar")
            .frame(Frame::none().fill(BG).inner_margin(egui::Margin::symmetric(12.0, 8.0)))
            .show(ctx, |ui| {
                ui.horizontal(|ui| {
                    ui.label(RichText::new("bogo").font(FontId::proportional(22.0)).color(TEXT).strong());
                    ui.label(RichText::new("·").font(FontId::proportional(22.0)).color(ORANGE).strong());

                    ui.add_space(16.0);

                    if !self.snap.nickname.is_empty() {
                        let nick = format!("@{}", self.snap.nickname);
                        ui.label(RichText::new(nick).font(FontId::monospace(13.0)).color(GREEN));
                    }

                    // Current tick best in the centre
                    if let Some((latest_score, _)) = self.snap.ticks.last() {
                        ui.with_layout(Layout::centered_and_justified(egui::Direction::LeftToRight), |ui| {
                            ui.label(
                                RichText::new(format!("● {} this tick", latest_score))
                                    .font(FontId::monospace(13.0))
                                    .color(GREEN),
                            );
                        });
                    }
                });
            });

        // ── Bottom history strip ─────────────────────────────────────────────────
        egui::TopBottomPanel::bottom("strip")
            .exact_height(48.0)
            .frame(Frame::none().fill(STRIP_BG))
            .show(ctx, |ui| {
                self.draw_history_strip(ui);
            });

        // ── Right panel ──────────────────────────────────────────────────────────
        egui::SidePanel::right("right_panel")
            .min_width(200.0)
            .max_width(260.0)
            .frame(Frame::none().fill(PANEL_BG).inner_margin(egui::Margin::symmetric(12.0, 8.0)))
            .show(ctx, |ui| {
                egui::ScrollArea::vertical().show(ui, |ui| {
                    self.draw_right_panel(ui);
                });
            });

        // ── Central chart ────────────────────────────────────────────────────────
        egui::CentralPanel::default()
            .frame(Frame::none().fill(BG).inner_margin(egui::Margin::symmetric(8.0, 8.0)))
            .show(ctx, |ui| {
                self.draw_bar_chart(ui);
            });
    }
}
