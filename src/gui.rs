#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

use eframe::egui;
use haruka_w4a8::cpu_backend::infer_cpu;
use haruka_w4a8::io::{
    filter_predictions, for_each_image, write_outputs, write_txt, ImagePrediction, InputFormat,
    OutputFormat, TagFilterOpts,
};
use haruka_w4a8::model::HarukaModel;
use haruka_w4a8::wgpu_backend::WgpuRunner;
use std::path::{Path, PathBuf};
use std::sync::mpsc::{self, Receiver};
use std::thread;
use std::time::Instant;

enum WorkerMsg {
    Status(String),
    Result(ImagePrediction),
    Done(String),
    Error(String),
}

struct HarukaGuiApp {
    weights: String,
    metadata: String,
    input: String,
    input_format: InputFormat,
    output_format: OutputFormat,
    output_path: String,
    force_cpu: bool,
    use_custom_threshold: bool,
    threshold: f32,
    top_k: usize,
    save_probs: bool,

    running: bool,
    status: String,
    results: Vec<ImagePrediction>,
    selected_idx: usize,
    rx: Option<Receiver<WorkerMsg>>,
    preview_tex: Option<(String, egui::TextureHandle)>,
}

impl Default for HarukaGuiApp {
    fn default() -> Self {
        Self {
            weights: "haruka-tagger-12L-w4a8-awq-qat-packed.safetensors".to_string(),
            metadata: "haruka-tagger-12L-w4a8-awq-qat_metadata.json".to_string(),
            input: String::new(),
            input_format: InputFormat::Auto,
            output_format: OutputFormat::Stdout,
            output_path: String::new(),
            force_cpu: false,
            use_custom_threshold: false,
            threshold: 0.35,
            top_k: 0,
            save_probs: true,
            running: false,
            status: "Ready.".to_string(),
            results: Vec::new(),
            selected_idx: 0,
            rx: None,
            preview_tex: None,
        }
    }
}

impl HarukaGuiApp {
    fn start_run(&mut self, ctx: egui::Context) {
        if self.running {
            return;
        }
        if self.input.trim().is_empty() {
            self.status = "Select an input image, folder, or WebDataset .tar shard first.".to_string();
            return;
        }

        let weights = PathBuf::from(self.weights.trim());
        let metadata = PathBuf::from(self.metadata.trim());
        let input = PathBuf::from(self.input.trim());
        let input_format = self.input_format;
        let output_format = self.output_format;
        let output_path = if self.output_path.trim().is_empty() {
            None
        } else {
            Some(PathBuf::from(self.output_path.trim()))
        };
        let force_cpu = self.force_cpu;
        let filter_opts = TagFilterOpts {
            threshold: self.use_custom_threshold.then_some(self.threshold),
            top_k: (self.top_k > 0).then_some(self.top_k),
            save_probs: self.save_probs,
        };

        let (tx, rx) = mpsc::channel();
        self.rx = Some(rx);
        self.running = true;
        self.results.clear();
        self.selected_idx = 0;
        self.preview_tex = None;
        self.status = "Loading model...".to_string();

        thread::spawn(move || {
            let run = || -> anyhow::Result<()> {
                let t_load = Instant::now();
                let model = HarukaModel::load(&weights, &metadata)?;
                let backend_name;
                let runner = if force_cpu {
                    backend_name = format!("CPU ({} threads)", rayon::current_num_threads());
                    None
                } else {
                    match WgpuRunner::try_new(&model) {
                        Ok(r) => {
                            backend_name = format!("wgpu ({})", r.adapter_name);
                            Some(r)
                        }
                        Err(err) => {
                            backend_name = format!("CPU fallback ({err})");
                            None
                        }
                    }
                };
                let _ = tx.send(WorkerMsg::Status(format!(
                    "Loaded {}L in {:.0} ms | {}",
                    model.num_layers,
                    t_load.elapsed().as_secs_f64() * 1000.0,
                    backend_name
                )));
                ctx.request_repaint();

                let mut collected = Vec::new();
                let collect = matches!(
                    output_format,
                    OutputFormat::Json | OutputFormat::Parquet | OutputFormat::Sqlite
                );

                let total_t0 = Instant::now();
                let count = for_each_image(&input, input_format, model.meta.img_size, |name, nchw| {
                    let t0 = Instant::now();
                    let probs = match &runner {
                        Some(r) => r.infer(&model, &nchw),
                        None => infer_cpu(&model, &nchw),
                    };
                    let dt = t0.elapsed().as_secs_f64() * 1000.0;
                    let pred = filter_predictions(name, &probs, &model, &filter_opts);
                    if output_format == OutputFormat::Txt {
                        write_txt(&pred, output_path.as_deref())?;
                    }
                    if collect {
                        collected.push(pred.clone());
                    }
                    let _ = tx.send(WorkerMsg::Status(format!("Tagged {name} ({dt:.0} ms)")));
                    let _ = tx.send(WorkerMsg::Result(pred));
                    ctx.request_repaint();
                    Ok(())
                })?;

                write_outputs(
                    &collected,
                    output_format,
                    output_path.as_deref(),
                    filter_opts.save_probs,
                )?;
                let _ = tx.send(WorkerMsg::Done(format!(
                    "Done: {count} image(s) in {:.2} s | {}",
                    total_t0.elapsed().as_secs_f64(),
                    backend_name
                )));
                Ok(())
            };

            if let Err(e) = run() {
                let _ = tx.send(WorkerMsg::Error(format!("{e:#}")));
            }
            ctx.request_repaint();
        });
    }

    fn poll_worker(&mut self) {
        let mut finished = false;
        if let Some(rx) = &self.rx {
            while let Ok(msg) = rx.try_recv() {
                match msg {
                    WorkerMsg::Status(s) => self.status = s,
                    WorkerMsg::Result(pred) => self.results.push(pred),
                    WorkerMsg::Done(s) => {
                        self.status = s;
                        finished = true;
                    }
                    WorkerMsg::Error(e) => {
                        self.status = format!("Error: {e}");
                        finished = true;
                    }
                }
            }
        }
        if finished {
            self.running = false;
            self.rx = None;
        }
    }

    fn ensure_preview(&mut self, ctx: &egui::Context, image_path: &str) {
        if self
            .preview_tex
            .as_ref()
            .is_some_and(|(p, _)| p == image_path)
        {
            return;
        }
        let p = Path::new(image_path);
        if !p.is_file() {
            return;
        }
        if let Ok(img) = image::open(p) {
            let thumb = img.thumbnail(420, 420).to_rgba8();
            let size = [thumb.width() as usize, thumb.height() as usize];
            let color = egui::ColorImage::from_rgba_unmultiplied(size, thumb.as_raw());
            let tex = ctx.load_texture("preview", color, egui::TextureOptions::LINEAR);
            self.preview_tex = Some((image_path.to_string(), tex));
        }
    }
}

impl eframe::App for HarukaGuiApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        self.poll_worker();

        egui::SidePanel::left("controls")
            .resizable(true)
            .default_width(340.0)
            .show(ctx, |ui| {
                ui.heading("Haruka-Tagger W4A8");
                ui.add_space(6.0);

                ui.label("Weights (.safetensors)");
                ui.horizontal(|ui| {
                    ui.text_edit_singleline(&mut self.weights);
                    if ui.button("...").clicked() {
                        if let Some(p) = rfd::FileDialog::new()
                            .add_filter("safetensors", &["safetensors"])
                            .pick_file()
                        {
                            self.weights = p.display().to_string();
                        }
                    }
                });

                ui.label("Metadata (.json)");
                ui.horizontal(|ui| {
                    ui.text_edit_singleline(&mut self.metadata);
                    if ui.button("...").clicked() {
                        if let Some(p) = rfd::FileDialog::new()
                            .add_filter("json", &["json"])
                            .pick_file()
                        {
                            self.metadata = p.display().to_string();
                        }
                    }
                });

                ui.checkbox(&mut self.force_cpu, "Force CPU fallback");
                ui.separator();

                ui.label("Input (Image / Folder / .tar shard)");
                ui.text_edit_singleline(&mut self.input);
                ui.horizontal(|ui| {
                    if ui.button("Image").clicked() {
                        if let Some(p) = rfd::FileDialog::new()
                            .add_filter("Images", &["jpg", "jpeg", "png", "webp", "bmp"])
                            .pick_file()
                        {
                            self.input = p.display().to_string();
                        }
                    }
                    if ui.button("Folder").clicked() {
                        if let Some(p) = rfd::FileDialog::new().pick_folder() {
                            self.input = p.display().to_string();
                        }
                    }
                    if ui.button("WebDataset").clicked() {
                        if let Some(p) = rfd::FileDialog::new()
                            .add_filter("WebDataset", &["tar", "gz", "tgz"])
                            .pick_file()
                        {
                            self.input = p.display().to_string();
                            self.input_format = InputFormat::Webdataset;
                        }
                    }
                });

                ui.horizontal(|ui| {
                    ui.label("Input format:");
                    egui::ComboBox::from_id_salt("in_fmt")
                        .selected_text(format!("{:?}", self.input_format))
                        .show_ui(ui, |ui| {
                            ui.selectable_value(&mut self.input_format, InputFormat::Auto, "Auto");
                            ui.selectable_value(&mut self.input_format, InputFormat::Folder, "Folder");
                            ui.selectable_value(&mut self.input_format, InputFormat::Webdataset, "Webdataset");
                        });
                });

                ui.separator();

                ui.horizontal(|ui| {
                    ui.label("Output format:");
                    egui::ComboBox::from_id_salt("out_fmt")
                        .selected_text(match self.output_format {
                            OutputFormat::Stdout => "Preview only",
                            OutputFormat::Txt => "Txt (.txt per image)",
                            OutputFormat::Json => "JSON",
                            OutputFormat::Parquet => "Parquet",
                            OutputFormat::Sqlite => "SQLite",
                        })
                        .show_ui(ui, |ui| {
                            ui.selectable_value(&mut self.output_format, OutputFormat::Stdout, "Preview only");
                            ui.selectable_value(&mut self.output_format, OutputFormat::Txt, "Txt (.txt per image)");
                            ui.selectable_value(&mut self.output_format, OutputFormat::Json, "JSON");
                            ui.selectable_value(&mut self.output_format, OutputFormat::Parquet, "Parquet");
                            ui.selectable_value(&mut self.output_format, OutputFormat::Sqlite, "SQLite");
                        });
                });

                if self.output_format != OutputFormat::Stdout {
                    ui.label("Output file / folder (optional)");
                    ui.horizontal(|ui| {
                        ui.text_edit_singleline(&mut self.output_path);
                        if ui.button("...").clicked() {
                            let picked = if self.output_format == OutputFormat::Txt {
                                rfd::FileDialog::new().pick_folder()
                            } else {
                                rfd::FileDialog::new().save_file()
                            };
                            if let Some(p) = picked {
                                self.output_path = p.display().to_string();
                            }
                        }
                    });
                }

                ui.separator();
                ui.checkbox(&mut self.use_custom_threshold, "Custom threshold");
                if self.use_custom_threshold {
                    ui.add(egui::Slider::new(&mut self.threshold, 0.05..=0.95).text("Threshold"));
                }
                ui.horizontal(|ui| {
                    ui.label("Top-K (0 = all):");
                    ui.add(egui::DragValue::new(&mut self.top_k).range(0..=500));
                });
                ui.checkbox(&mut self.save_probs, "Save probabilities");

                ui.add_space(8.0);
                if ui
                    .add_enabled(!self.running, egui::Button::new("Run Tagging"))
                    .clicked()
                {
                    self.start_run(ctx.clone());
                }
                ui.add_space(6.0);
                ui.label(&self.status);
            });

        egui::CentralPanel::default().show(ctx, |ui| {
            if self.results.is_empty() {
                ui.centered_and_justified(|ui| {
                    ui.label("Select an input image, folder, or WebDataset shard and click Run Tagging.");
                });
                return;
            }

            if self.results.len() > 1 {
                ui.horizontal(|ui| {
                    if ui.button("< Prev").clicked() && self.selected_idx > 0 {
                        self.selected_idx -= 1;
                    }
                    ui.label(format!(
                        "{} / {}",
                        self.selected_idx + 1,
                        self.results.len()
                    ));
                    if ui.button("Next >").clicked() && self.selected_idx + 1 < self.results.len() {
                        self.selected_idx += 1;
                    }
                });
                ui.separator();
            }

            let pred = self.results[self.selected_idx].clone();
            self.ensure_preview(ctx, &pred.image);

            egui::ScrollArea::vertical().show(ui, |ui| {
                ui.horizontal_top(|ui| {
                    if let Some((path, tex)) = &self.preview_tex {
                        if path == &pred.image {
                            ui.image((tex.id(), tex.size_vec2()));
                        }
                    }
                    ui.vertical(|ui| {
                        ui.heading(&pred.image);
                        let rating_txt = match pred.rating_prob {
                            Some(p) => format!("Rating: {} ({p:.3})", pred.rating),
                            None => format!("Rating: {}", pred.rating),
                        };
                        ui.strong(rating_txt);
                        ui.add_space(6.0);

                        let prompt: String = pred
                            .tags
                            .iter()
                            .map(|t| t.tag.as_str())
                            .collect::<Vec<_>>()
                            .join(", ");
                        if ui.button("Copy Prompt").clicked() {
                            ctx.copy_text(prompt.clone());
                        }
                        ui.add_space(4.0);
                        ui.label(prompt);
                    });
                });

                ui.separator();
                for cat in ["character", "copyright", "artist", "general"] {
                    let items: Vec<String> = pred
                        .tags
                        .iter()
                        .filter(|t| t.category == cat)
                        .map(|t| match t.prob {
                            Some(p) => format!("{} ({p:.2})", t.tag),
                            None => t.tag.clone(),
                        })
                        .collect();
                    if !items.is_empty() {
                        ui.strong(format!("[{cat}]"));
                        ui.label(items.join(", "));
                        ui.add_space(4.0);
                    }
                }
            });
        });
    }
}

fn main() -> eframe::Result<()> {
    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default().with_inner_size([1020.0, 680.0]),
        ..Default::default()
    };
    eframe::run_native(
        "Haruka-Tagger W4A8",
        options,
        Box::new(|_cc| Ok(Box::new(HarukaGuiApp::default()))),
    )
}
