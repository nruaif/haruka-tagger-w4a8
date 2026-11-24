use crate::model::{default_threshold, preprocess_image, preprocess_image_bytes, HarukaModel};
use anyhow::{bail, Context, Result};
use arrow_array::{ArrayRef, RecordBatch, StringArray};
use arrow_schema::{DataType, Field, Schema};
use clap::ValueEnum;
use flate2::read::GzDecoder;
use parquet::arrow::ArrowWriter;
use rusqlite::{params, Connection};
use serde::Serialize;
use std::collections::BTreeMap;
use std::fs::File;
use std::io::{BufReader, Read};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use tar::Archive;

#[derive(Copy, Clone, Debug, Eq, PartialEq, ValueEnum)]
pub enum InputFormat {
    Auto,
    Folder,
    Webdataset,
}

#[derive(Copy, Clone, Debug, Eq, PartialEq, ValueEnum)]
pub enum OutputFormat {
    Stdout,
    Txt,
    Json,
    Parquet,
    Sqlite,
}

#[derive(Clone, Debug)]
pub struct TagFilterOpts {
    pub threshold: Option<f32>,
    pub top_k: Option<usize>,
    pub save_probs: bool,
}

#[derive(Clone, Debug, Serialize)]
pub struct TagScore {
    pub tag: String,
    pub category: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub prob: Option<f32>,
}

#[derive(Clone, Debug, Serialize)]
pub struct ImagePrediction {
    pub image: String,
    pub rating: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub rating_prob: Option<f32>,
    pub tags: Vec<TagScore>,
}

const IMG_EXTS: &[&str] = &["jpg", "jpeg", "png", "webp", "bmp"];

fn is_image_name(name: &str) -> bool {
    Path::new(name)
        .extension()
        .and_then(|e| e.to_str())
        .map(|e| IMG_EXTS.contains(&e.to_ascii_lowercase().as_str()))
        .unwrap_or(false)
}

fn is_tar_path(p: &Path) -> bool {
    let s = p.to_string_lossy().to_ascii_lowercase();
    s.ends_with(".tar") || s.ends_with(".tar.gz") || s.ends_with(".tgz")
}

pub fn filter_predictions(
    image_name: &str,
    probs: &[f32],
    model: &HarukaModel,
    opts: &TagFilterOpts,
) -> ImagePrediction {
    let mut best_rating = ("rating_g".to_string(), 0.0f32);
    let mut items: Vec<(String, String, f32)> = Vec::new();

    for (i, &p) in probs.iter().enumerate() {
        let tag = &model.idx_to_tag[i];
        let cat = &model.idx_to_cat[i];
        if cat == "rating" {
            if p > best_rating.1 {
                best_rating = (tag.clone(), p);
            }
            continue;
        }
        if cat == "year" {
            continue;
        }
        let thr = opts.threshold.unwrap_or_else(|| default_threshold(cat));
        if p >= thr {
            items.push((tag.clone(), cat.clone(), p));
        }
    }

    items.sort_by(|a, b| b.2.partial_cmp(&a.2).unwrap_or(std::cmp::Ordering::Equal));
    if let Some(k) = opts.top_k {
        if k > 0 && items.len() > k {
            items.truncate(k);
        }
    }

    let tags = items
        .into_iter()
        .map(|(tag, category, p)| TagScore {
            tag,
            category,
            prob: opts
                .save_probs
                .then(|| (p * 10000.0).round() / 10000.0),
        })
        .collect();

    ImagePrediction {
        image: image_name.to_string(),
        rating: best_rating.0,
        rating_prob: opts
            .save_probs
            .then(|| (best_rating.1 * 10000.0).round() / 10000.0),
        tags,
    }
}

pub fn for_each_image<F>(
    input: &Path,
    format: InputFormat,
    img_size: usize,
    mut cb: F,
) -> Result<usize>
where
    F: FnMut(&str, Vec<f32>) -> Result<bool>,
{
    let effective = match format {
        InputFormat::Auto => {
            if input.is_file() && is_tar_path(input) {
                InputFormat::Webdataset
            } else if input.is_dir() {
                let has_tar = std::fs::read_dir(input)?
                    .filter_map(|e| e.ok())
                    .any(|e| is_tar_path(&e.path()));
                let has_img = std::fs::read_dir(input)?
                    .filter_map(|e| e.ok())
                    .any(|e| e.path().is_file() && is_image_name(&e.path().to_string_lossy()));
                if has_tar && !has_img {
                    InputFormat::Webdataset
                } else {
                    InputFormat::Folder
                }
            } else {
                InputFormat::Folder
            }
        }
        other => other,
    };

    let mut count = 0usize;
    match effective {
        InputFormat::Folder | InputFormat::Auto => {
            let paths = collect_folder_images(input)?;
            if paths.is_empty() {
                bail!("No images found at {}", input.display());
            }
            for p in paths {
                let nchw = preprocess_image(&p, img_size)?;
                let name = p.to_string_lossy().into_owned();
                let cont = cb(&name, nchw)?;
                count += 1;
                if !cont {
                    break;
                }
            }
        }
        InputFormat::Webdataset => {
            let shards = collect_tar_shards(input)?;
            if shards.is_empty() {
                bail!("No WebDataset .tar/.tar.gz shards found at {}", input.display());
            }
            let mut stopped = false;
            for shard in shards {
                let file = File::open(&shard)
                    .with_context(|| format!("Failed to open shard {}", shard.display()))?;
                let gz = shard.to_string_lossy().to_ascii_lowercase();
                let (n, cont) = if gz.ends_with(".tar.gz") || gz.ends_with(".tgz") {
                    let dec = GzDecoder::new(BufReader::new(file));
                    read_tar_stream(dec, img_size, &mut cb)?
                } else {
                    read_tar_stream(BufReader::new(file), img_size, &mut cb)?
                };
                count += n;
                if !cont {
                    stopped = true;
                    break;
                }
            }
            if count == 0 && !stopped {
                bail!("No images found inside WebDataset shard(s) at {}", input.display());
            }
        }
    }
    Ok(count)
}

fn collect_folder_images(input: &Path) -> Result<Vec<PathBuf>> {
    if input.is_file() {
        return Ok(vec![input.to_path_buf()]);
    }
    let mut out = Vec::new();
    for entry in std::fs::read_dir(input).with_context(|| format!("Failed to read {}", input.display()))? {
        let p = entry?.path();
        if p.is_file() && is_image_name(&p.to_string_lossy()) {
            out.push(p);
        }
    }
    out.sort();
    Ok(out)
}

fn collect_tar_shards(input: &Path) -> Result<Vec<PathBuf>> {
    if input.is_file() {
        return Ok(vec![input.to_path_buf()]);
    }
    let mut out = Vec::new();
    for entry in std::fs::read_dir(input).with_context(|| format!("Failed to read {}", input.display()))? {
        let p = entry?.path();
        if p.is_file() && is_tar_path(&p) {
            out.push(p);
        }
    }
    out.sort();
    Ok(out)
}

fn read_tar_stream<R: Read, F>(reader: R, img_size: usize, cb: &mut F) -> Result<(usize, bool)>
where
    F: FnMut(&str, Vec<f32>) -> Result<bool>,
{
    let mut archive = Archive::new(reader);
    let mut count = 0usize;
    let mut buf = Vec::new();
    for entry in archive.entries()? {
        let mut entry = entry?;
        if !entry.header().entry_type().is_file() {
            continue;
        }
        let path_str = entry.path()?.to_string_lossy().into_owned();
        if !is_image_name(&path_str) {
            continue;
        }
        buf.clear();
        entry.read_to_end(&mut buf)?;
        let nchw = preprocess_image_bytes(&buf, img_size)?;
        let cont = cb(&path_str, nchw)?;
        count += 1;
        if !cont {
            return Ok((count, false));
        }
    }
    Ok((count, true))
}

pub fn print_stdout(pred: &ImagePrediction) {
    println!("\n=== Predictions for {} ===", pred.image);
    if let Some(rp) = pred.rating_prob {
        println!("  [rating]    {} ({rp:.3})", pred.rating);
    } else {
        println!("  [rating]    {}", pred.rating);
    }
    let mut by_cat: BTreeMap<&str, Vec<&TagScore>> = BTreeMap::new();
    for t in &pred.tags {
        by_cat.entry(t.category.as_str()).or_default().push(t);
    }
    for c in ["character", "copyright", "artist", "general"] {
        if let Some(items) = by_cat.get(c) {
            let formatted: Vec<String> = items
                .iter()
                .take(25)
                .map(|t| match t.prob {
                    Some(p) => format!("{} ({p:.2})", t.tag),
                    None => t.tag.clone(),
                })
                .collect();
            println!("  [{c:<9}] {}", formatted.join(", "));
        }
    }
    let prompt: Vec<&str> = pred.tags.iter().map(|t| t.tag.as_str()).collect();
    println!("\nFormatted Prompt:\n{}", prompt.join(", "));
}

pub fn write_txt(pred: &ImagePrediction, out_dir: Option<&Path>) -> Result<()> {
    let p = Path::new(&pred.image);
    let stem = p.file_stem().context("Missing image file stem")?;
    let mut dst = match out_dir {
        Some(dir) => {
            std::fs::create_dir_all(dir)?;
            dir.join(stem)
        }
        None => p.with_file_name(stem),
    };
    dst.set_extension("txt");
    let line: Vec<String> = pred
        .tags
        .iter()
        .map(|t| match t.prob {
            Some(prob) => format!("{} ({prob:.3})", t.tag),
            None => t.tag.clone(),
        })
        .collect();
    std::fs::write(&dst, line.join(", "))
        .with_context(|| format!("Failed to write {}", dst.display()))?;
    Ok(())
}

pub fn write_outputs(
    preds: &[ImagePrediction],
    format: OutputFormat,
    output_path: Option<&Path>,
    save_probs: bool,
) -> Result<()> {
    match format {
        OutputFormat::Stdout | OutputFormat::Txt => Ok(()),
        OutputFormat::Json => {
            let json = serde_json::to_string_pretty(preds)?;
            if let Some(p) = output_path {
                if let Some(parent) = p.parent().filter(|d| !d.as_os_str().is_empty()) {
                    std::fs::create_dir_all(parent)?;
                }
                std::fs::write(p, json)?;
            } else {
                println!("{json}");
            }
            Ok(())
        }
        OutputFormat::Parquet => {
            let dst = output_path.unwrap_or_else(|| Path::new("predictions.parquet"));
            write_parquet(preds, dst, save_probs)
        }
        OutputFormat::Sqlite => {
            let dst = output_path.unwrap_or_else(|| Path::new("predictions.db"));
            write_sqlite(preds, dst, save_probs)
        }
    }
}

fn tag_strings(pred: &ImagePrediction, save_probs: bool) -> (String, Option<String>) {
    let tags = pred
        .tags
        .iter()
        .map(|t| t.tag.as_str())
        .collect::<Vec<_>>()
        .join(", ");
    let probs = if save_probs {
        let map: BTreeMap<&str, f32> = pred
            .tags
            .iter()
            .filter_map(|t| t.prob.map(|p| (t.tag.as_str(), p)))
            .collect();
        Some(serde_json::to_string(&map).unwrap_or_default())
    } else {
        None
    };
    (tags, probs)
}

fn write_parquet(preds: &[ImagePrediction], dst: &Path, save_probs: bool) -> Result<()> {
    if let Some(parent) = dst.parent().filter(|d| !d.as_os_str().is_empty()) {
        std::fs::create_dir_all(parent)?;
    }
    let mut fields = vec![
        Field::new("image", DataType::Utf8, false),
        Field::new("rating", DataType::Utf8, false),
        Field::new("tags", DataType::Utf8, false),
    ];
    if save_probs {
        fields.push(Field::new("probs", DataType::Utf8, true));
    }
    let schema = Arc::new(Schema::new(fields));

    let images: Vec<&str> = preds.iter().map(|p| p.image.as_str()).collect();
    let ratings: Vec<&str> = preds.iter().map(|p| p.rating.as_str()).collect();
    let pairs: Vec<(String, Option<String>)> = preds
        .iter()
        .map(|p| tag_strings(p, save_probs))
        .collect();
    let tags_col: Vec<&str> = pairs.iter().map(|(t, _)| t.as_str()).collect();

    let mut cols: Vec<ArrayRef> = vec![
        Arc::new(StringArray::from(images)),
        Arc::new(StringArray::from(ratings)),
        Arc::new(StringArray::from(tags_col)),
    ];
    if save_probs {
        let probs_col: Vec<Option<&str>> = pairs.iter().map(|(_, pr)| pr.as_deref()).collect();
        cols.push(Arc::new(StringArray::from(probs_col)));
    }

    let batch = RecordBatch::try_new(schema.clone(), cols)?;
    let file = File::create(dst).with_context(|| format!("Failed to create {}", dst.display()))?;
    let mut writer = ArrowWriter::try_new(file, schema, None)?;
    writer.write(&batch)?;
    writer.close()?;
    Ok(())
}

fn write_sqlite(preds: &[ImagePrediction], dst: &Path, save_probs: bool) -> Result<()> {
    if let Some(parent) = dst.parent().filter(|d| !d.as_os_str().is_empty()) {
        std::fs::create_dir_all(parent)?;
    }
    let mut conn = Connection::open(dst)
        .with_context(|| format!("Failed to open SQLite DB {}", dst.display()))?;
    conn.execute_batch(
        "CREATE TABLE IF NOT EXISTS predictions (
            image TEXT PRIMARY KEY,
            rating TEXT NOT NULL,
            tags TEXT NOT NULL,
            probs TEXT
        );",
    )?;
    let tx = conn.transaction()?;
    {
        let mut stmt = tx.prepare(
            "INSERT OR REPLACE INTO predictions (image, rating, tags, probs) VALUES (?1, ?2, ?3, ?4)",
        )?;
        for pred in preds {
            let (tags, probs) = tag_strings(pred, save_probs);
            stmt.execute(params![pred.image, pred.rating, tags, probs])?;
        }
    }
    tx.commit()?;
    Ok(())
}
