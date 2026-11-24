use anyhow::Result;
use clap::Parser;
use haruka_w4a8::io::{
    filter_predictions, for_each_image, print_stdout, write_outputs, write_txt, ImagePrediction,
    InputFormat, OutputFormat, TagFilterOpts,
};
use haruka_w4a8::model::{preprocess_image, HarukaModel};
use haruka_w4a8::{cpu_backend, wgpu_backend};
use std::path::PathBuf;
use std::time::Instant;

#[derive(Parser, Debug)]
#[command(
    name = "haruka-w4a8",
    version,
    about = "Fast W4A8 AWQ-QAT Haruka-Tagger inference via wgpu (with automatic CPU fallback)"
)]
struct Args {
    /// Image file, directory of images, or WebDataset .tar/.tar.gz shard(s)
    input: PathBuf,

    /// Path to packed W4A8 .safetensors checkpoint
    #[arg(short, long, default_value = "haruka-tagger-12L-w4a8-awq-qat-packed.safetensors")]
    weights: PathBuf,

    /// Path to metadata JSON file
    #[arg(short, long, default_value = "haruka-tagger-12L-w4a8-awq-qat_metadata.json")]
    metadata: PathBuf,

    /// Input source format
    #[arg(long, value_enum, default_value_t = InputFormat::Auto)]
    input_format: InputFormat,

    /// Output format
    #[arg(short, long, value_enum, default_value_t = OutputFormat::Stdout)]
    format: OutputFormat,

    /// Output file (for json/parquet/sqlite) or directory (for txt)
    #[arg(short, long)]
    output: Option<PathBuf>,

    /// Shorthand for --format json
    #[arg(long)]
    json: bool,

    /// Global probability threshold override (defaults to per-category thresholds)
    #[arg(short, long)]
    threshold: Option<f32>,

    /// Keep only top-K non-rating tags per image
    #[arg(short = 'k', long)]
    top_k: Option<usize>,

    /// Omit tag probabilities from output
    #[arg(long)]
    no_probs: bool,

    /// Force CPU fallback backend instead of wgpu GPU
    #[arg(long)]
    cpu: bool,

    /// Run N benchmark iterations on the first image and print timing
    #[arg(long, default_value_t = 0)]
    bench: usize,
}

enum Backend {
    Wgpu(wgpu_backend::WgpuRunner),
    Cpu,
}

impl Backend {
    fn infer(&self, model: &HarukaModel, nchw: &[f32]) -> Vec<f32> {
        match self {
            Backend::Wgpu(runner) => runner.infer(model, nchw),
            Backend::Cpu => cpu_backend::infer_cpu(model, nchw),
        }
    }

    fn name(&self) -> String {
        match self {
            Backend::Wgpu(r) => format!("wgpu ({})", r.adapter_name),
            Backend::Cpu => format!("CPU (rayon {} threads)", rayon::current_num_threads()),
        }
    }
}

fn main() -> Result<()> {
    let args = Args::parse();
    let format = if args.json {
        OutputFormat::Json
    } else {
        args.format
    };
    let filter_opts = TagFilterOpts {
        threshold: args.threshold,
        top_k: args.top_k,
        save_probs: !args.no_probs,
    };

    let t_load = Instant::now();
    let model = HarukaModel::load(&args.weights, &args.metadata)?;
    eprintln!(
        "Loaded {}L Haruka-W4A8 ({} tags) in {:.1} ms",
        model.num_layers,
        model.meta.total_tags,
        t_load.elapsed().as_secs_f64() * 1000.0
    );

    let backend = if args.cpu {
        Backend::Cpu
    } else {
        match wgpu_backend::WgpuRunner::try_new(&model) {
            Ok(runner) => Backend::Wgpu(runner),
            Err(err) => {
                eprintln!("wgpu unavailable ({err:#}); falling back to CPU.");
                Backend::Cpu
            }
        }
    };
    eprintln!("Backend: {}", backend.name());

    if args.bench > 0 && args.input.is_file() && args.input_format != InputFormat::Webdataset {
        if let Ok(nchw) = preprocess_image(&args.input, model.meta.img_size) {
            let _ = backend.infer(&model, &nchw);
            let t0 = Instant::now();
            for _ in 0..args.bench {
                let _ = backend.infer(&model, &nchw);
            }
            let avg_ms = t0.elapsed().as_secs_f64() * 1000.0 / (args.bench as f64);
            eprintln!(
                "Benchmark ({} iters on {}): {:.2} ms/img ({:.2} img/s)",
                args.bench,
                backend.name(),
                avg_ms,
                1000.0 / avg_ms
            );
        }
    }

    let mut preds: Vec<ImagePrediction> = Vec::new();
    let collect_preds = matches!(
        format,
        OutputFormat::Json | OutputFormat::Parquet | OutputFormat::Sqlite
    );

    let count = for_each_image(&args.input, args.input_format, model.meta.img_size, |name, nchw| {
        let t0 = Instant::now();
        let probs = backend.infer(&model, &nchw);
        let dt_ms = t0.elapsed().as_secs_f64() * 1000.0;
        eprintln!("Inferred {name} in {dt_ms:.1} ms");

        let pred = filter_predictions(name, &probs, &model, &filter_opts);
        match format {
            OutputFormat::Stdout => print_stdout(&pred),
            OutputFormat::Txt => write_txt(&pred, args.output.as_deref())?,
            OutputFormat::Json | OutputFormat::Parquet | OutputFormat::Sqlite => {
                if collect_preds {
                    preds.push(pred);
                }
            }
        }
        Ok(true)
    })?;

    write_outputs(&preds, format, args.output.as_deref(), filter_opts.save_probs)?;
    eprintln!("Processed {count} image(s).");
    Ok(())
}
