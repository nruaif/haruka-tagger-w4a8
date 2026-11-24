use anyhow::{bail, Context, Result};
use half::f16;
use image::{imageops::FilterType, DynamicImage, GenericImageView, Rgba, RgbaImage};
use memmap2::Mmap;
use safetensors::{Dtype, SafeTensors};
use serde::Deserialize;
use std::collections::HashMap;
use std::fs::File;
use std::path::Path;

pub const IMAGENET_MEAN: [f32; 3] = [0.485, 0.456, 0.406];
pub const IMAGENET_STD: [f32; 3] = [0.229, 0.224, 0.225];

pub fn default_threshold(category: &str) -> f32 {
    match category {
        "character" => 0.35,
        "copyright" => 0.35,
        "artist" => 0.25,
        "general" => 0.35,
        "rating" => 0.50,
        "year" => 0.25,
        _ => 0.35,
    }
}

#[derive(Debug, Deserialize)]
#[allow(dead_code)]
pub struct HarukaMetadata {
    pub backbone: String,
    pub total_tags: usize,
    #[serde(default = "default_img_size")]
    pub img_size: usize,
    #[serde(default = "default_num_queries")]
    pub num_queries: usize,
    #[serde(default = "default_pool_mode")]
    pub pool_mode: String,
    pub idx_to_tag: HashMap<String, String>,
    pub tag_to_category: HashMap<String, String>,
}

fn default_img_size() -> usize {
    448
}
fn default_num_queries() -> usize {
    18
}
fn default_pool_mode() -> String {
    "slot_sigmoid".to_string()
}

#[derive(Clone)]
pub struct LayerNormParams {
    pub weight: Vec<f32>,
    pub bias: Vec<f32>,
    pub dim: usize,
}

/// W4A8 quantized linear layer:
///   - `qweight_packed`: [N, K/2] u8, two 4-bit offset-8 nibbles per byte (low = even k, high = odd k)
///   - `qweight_u32`:    [N, K/8] u32, pre-XOR'd by 0x88888888 so each nibble is signed 2's complement [-8, 7]
///   - `scales`:         [N, num_groups] f32 (from f16)
///   - `awq_scale`:      [K] f32 (from f16)
///   - `bias`:           [N] f32 (from f16)
#[derive(Clone)]
pub struct W4Linear {
    pub out_features: usize,
    pub in_features: usize,
    pub group_size: usize,
    pub num_groups: usize,
    #[allow(dead_code)]
    pub qweight_packed: Vec<u8>,
    pub qweight_u32_signed: Vec<u32>,
    pub scales: Vec<f32>,
    pub awq_scale: Vec<f32>,
    pub bias: Option<Vec<f32>>,
}

#[derive(Clone)]
pub struct Eva02Block {
    pub norm1: LayerNormParams,
    pub q_proj: W4Linear,
    pub k_proj: W4Linear,
    pub v_proj: W4Linear,
    pub proj: W4Linear,
    pub norm2: LayerNormParams,
    pub fc1_g: W4Linear,
    pub fc1_x: W4Linear,
    pub mlp_norm: LayerNormParams,
    pub fc2: W4Linear,
}

#[derive(Clone)]
pub struct MHAttnPoolParams {
    pub num_heads: usize,
    pub head_dim: usize,
    pub num_coarse: usize,
    pub num_queries: usize,
    pub branches: usize,
    pub leaf_dim: usize,
    pub norm: LayerNormParams,
    pub kv: W4Linear,
    pub q_coarse: Vec<f32>,       // [num_coarse, dim]
    pub q_coarse_bias: Vec<f32>,  // [num_coarse]
    pub trunk_norm: LayerNormParams,
    pub q_child: Vec<f32>,        // [num_queries, dim]
    pub child_q_norm: LayerNormParams,
    pub q_child_bias: Vec<f32>,   // [num_queries]
    pub leaf_norm: LayerNormParams,
    pub leaf_proj_weight: Vec<f32>, // [num_queries, dim, leaf_dim]
    pub leaf_proj_bias: Vec<f32>,   // [num_queries, leaf_dim]
}

pub struct HarukaModel {
    pub meta: HarukaMetadata,
    pub dim: usize,
    pub num_heads: usize,
    pub head_dim: usize,
    pub num_layers: usize,
    pub grid_size: usize,
    pub patch_size: usize,
    pub patch_w: Vec<f32>, // [dim, 3 * 14 * 14]
    pub patch_b: Vec<f32>, // [dim]
    pub cls_token: Vec<f32>,
    pub pos_embed: Vec<f32>, // [1025, dim]
    pub rope_sin: Vec<f32>,  // [1024, head_dim]
    pub rope_cos: Vec<f32>,  // [1024, head_dim]
    pub blocks: Vec<Eva02Block>,
    pub norm: LayerNormParams,
    pub pool: MHAttnPoolParams,
    pub head: W4Linear,
    pub idx_to_tag: Vec<String>,
    pub idx_to_cat: Vec<String>,
}

fn tensor_to_f32(st: &SafeTensors, name: &str) -> Result<Vec<f32>> {
    let t = st.tensor(name).with_context(|| format!("Missing tensor: {name}"))?;
    match t.dtype() {
        Dtype::F32 => {
            let bytes = t.data();
            let mut out = Vec::with_capacity(bytes.len() / 4);
            for c in bytes.chunks_exact(4) {
                out.push(f32::from_le_bytes([c[0], c[1], c[2], c[3]]));
            }
            Ok(out)
        }
        Dtype::F16 => {
            let bytes = t.data();
            let mut out = Vec::with_capacity(bytes.len() / 2);
            for c in bytes.chunks_exact(2) {
                out.push(f16::from_le_bytes([c[0], c[1]]).to_f32());
            }
            Ok(out)
        }
        Dtype::BF16 => {
            let bytes = t.data();
            let mut out = Vec::with_capacity(bytes.len() / 2);
            for c in bytes.chunks_exact(2) {
                out.push(half::bf16::from_le_bytes([c[0], c[1]]).to_f32());
            }
            Ok(out)
        }
        other => bail!("Unsupported float dtype {other:?} for {name}"),
    }
}

fn load_ln(st: &SafeTensors, prefix: &str) -> Result<LayerNormParams> {
    let weight = tensor_to_f32(st, &format!("{prefix}.weight"))?;
    let bias = tensor_to_f32(st, &format!("{prefix}.bias"))?;
    let dim = weight.len();
    Ok(LayerNormParams { weight, bias, dim })
}

fn load_w4_linear(st: &SafeTensors, prefix: &str, group_size: usize) -> Result<W4Linear> {
    let qt = st
        .tensor(&format!("{prefix}.qweight_packed"))
        .with_context(|| format!("Missing {prefix}.qweight_packed"))?;
    let shape = qt.shape();
    if shape.len() != 2 {
        bail!("Expected 2D qweight_packed for {prefix}, got {shape:?}");
    }
    let out_features = shape[0];
    let k_half = shape[1];
    let in_features = k_half * 2;
    let qweight_packed = qt.data().to_vec();

    // Pack 4 bytes into u32 and XOR with 0x88888888 to convert offset-8 [0..15] -> signed 2's complement [-8..7]
    if k_half % 4 != 0 {
        bail!("Expected in_features multiple of 8 for {prefix}, got {in_features}");
    }
    let mut qweight_u32_signed = Vec::with_capacity(out_features * (k_half / 4));
    for c in qweight_packed.chunks_exact(4) {
        let word = u32::from_le_bytes([c[0], c[1], c[2], c[3]]) ^ 0x8888_8888;
        qweight_u32_signed.push(word);
    }

    let scales = tensor_to_f32(st, &format!("{prefix}.scales"))?;
    let num_groups = (in_features + group_size - 1) / group_size;

    let awq_scale = if let Ok(s) = tensor_to_f32(st, &format!("{prefix}.awq_scale")) {
        s
    } else {
        vec![1.0f32; in_features]
    };

    let bias = tensor_to_f32(st, &format!("{prefix}.bias")).ok();

    Ok(W4Linear {
        out_features,
        in_features,
        group_size,
        num_groups,
        qweight_packed,
        qweight_u32_signed,
        scales,
        awq_scale,
        bias,
    })
}

/// Computes exact EVA02 2D RoPE (sin, cos) matching `timm.layers.RotaryEmbeddingCat`
/// with `in_pixels=False`, `feat_shape=(32, 32)`, `ref_feat_shape=(16, 16)`.
fn build_eva02_rope(grid_size: usize, ref_grid: usize, head_dim: usize) -> (Vec<f32>, Vec<f32>) {
    let half_dim = head_dim / 2; // 32 per spatial axis
    let freq_count = half_dim / 2; // 16 frequencies
    let mut inv_freq = Vec::with_capacity(freq_count);
    for i in 0..freq_count {
        let exp = (2 * i) as f32 / (half_dim as f32);
        inv_freq.push(1.0f32 / 10000.0f32.powf(exp));
    }

    let n_patches = grid_size * grid_size;
    let mut rope_sin = Vec::with_capacity(n_patches * head_dim);
    let mut rope_cos = Vec::with_capacity(n_patches * head_dim);

    let scale = (ref_grid as f32) / (grid_size as f32);
    for gy in 0..grid_size {
        let y = (gy as f32) * scale;
        for gx in 0..grid_size {
            let x = (gx as f32) * scale;
            // First half: y freqs repeated twice [sin(y*f0), sin(y*f0), sin(y*f1), sin(y*f1), ...]
            for &f in &inv_freq {
                let s = (y * f).sin();
                rope_sin.push(s);
                rope_sin.push(s);
            }
            // Second half: x freqs repeated twice
            for &f in &inv_freq {
                let s = (x * f).sin();
                rope_sin.push(s);
                rope_sin.push(s);
            }

            for &f in &inv_freq {
                let c = (y * f).cos();
                rope_cos.push(c);
                rope_cos.push(c);
            }
            for &f in &inv_freq {
                let c = (x * f).cos();
                rope_cos.push(c);
                rope_cos.push(c);
            }
        }
    }
    (rope_sin, rope_cos)
}

impl HarukaModel {
    pub fn load(weights_path: &Path, metadata_path: &Path) -> Result<Self> {
        let meta_str = std::fs::read_to_string(metadata_path)
            .with_context(|| format!("Failed to read metadata: {}", metadata_path.display()))?;
        let meta: HarukaMetadata = serde_json::from_str(&meta_str)
            .with_context(|| "Failed to parse metadata JSON")?;

        let file = File::open(weights_path)
            .with_context(|| format!("Failed to open weights: {}", weights_path.display()))?;
        let mmap = unsafe { Mmap::map(&file)? };
        let st = SafeTensors::deserialize(&mmap)?;

        let patch_w = tensor_to_f32(&st, "backbone.patch_embed.proj.weight")?;
        let patch_b = tensor_to_f32(&st, "backbone.patch_embed.proj.bias")?;
        let dim = patch_b.len();
        let patch_size = 14usize;
        let grid_size = meta.img_size / patch_size;

        let cls_token = tensor_to_f32(&st, "backbone.cls_token")?;
        let pos_embed = tensor_to_f32(&st, "backbone.pos_embed")?;

        let mut num_layers = 0usize;
        while st.tensor(&format!("backbone.blocks.{num_layers}.norm1.weight")).is_ok() {
            num_layers += 1;
        }
        if num_layers == 0 {
            bail!("No backbone.blocks.* found in safetensors");
        }

        let head_dim = 64usize;
        let num_heads = dim / head_dim;
        let group_size = 64usize;

        let mut blocks = Vec::with_capacity(num_layers);
        for i in 0..num_layers {
            let bp = format!("backbone.blocks.{i}");
            blocks.push(Eva02Block {
                norm1: load_ln(&st, &format!("{bp}.norm1"))?,
                q_proj: load_w4_linear(&st, &format!("{bp}.attn.q_proj"), group_size)?,
                k_proj: load_w4_linear(&st, &format!("{bp}.attn.k_proj"), group_size)?,
                v_proj: load_w4_linear(&st, &format!("{bp}.attn.v_proj"), group_size)?,
                proj: load_w4_linear(&st, &format!("{bp}.attn.proj"), group_size)?,
                norm2: load_ln(&st, &format!("{bp}.norm2"))?,
                fc1_g: load_w4_linear(&st, &format!("{bp}.mlp.fc1_g"), group_size)?,
                fc1_x: load_w4_linear(&st, &format!("{bp}.mlp.fc1_x"), group_size)?,
                mlp_norm: load_ln(&st, &format!("{bp}.mlp.norm"))?,
                fc2: load_w4_linear(&st, &format!("{bp}.mlp.fc2"), group_size)?,
            });
        }

        let norm = load_ln(&st, "backbone.norm")?;

        let num_queries = meta.num_queries;
        let num_coarse = num_queries.min(6);
        let branches = num_queries / num_coarse;
        let pool_heads = 16usize;
        let pool = MHAttnPoolParams {
            num_heads: pool_heads,
            head_dim: dim / pool_heads,
            num_coarse,
            num_queries,
            branches,
            leaf_dim: 64,
            norm: load_ln(&st, "head_pool.norm")?,
            kv: load_w4_linear(&st, "head_pool.kv", group_size)?,
            q_coarse: tensor_to_f32(&st, "head_pool.q_coarse")?,
            q_coarse_bias: tensor_to_f32(&st, "head_pool.q_coarse_bias")?,
            trunk_norm: load_ln(&st, "head_pool.trunk_norm")?,
            q_child: tensor_to_f32(&st, "head_pool.q_child")?,
            child_q_norm: load_ln(&st, "head_pool.child_q_norm")?,
            q_child_bias: tensor_to_f32(&st, "head_pool.q_child_bias")?,
            leaf_norm: load_ln(&st, "head_pool.leaf_norm")?,
            leaf_proj_weight: tensor_to_f32(&st, "head_pool.leaf_proj_weight")?,
            leaf_proj_bias: tensor_to_f32(&st, "head_pool.leaf_proj_bias")?,
        };

        let head = load_w4_linear(&st, "head", group_size)?;
        let (rope_sin, rope_cos) = build_eva02_rope(grid_size, 16, head_dim);

        let mut idx_to_tag = vec![String::new(); meta.total_tags];
        let mut idx_to_cat = vec!["general".to_string(); meta.total_tags];
        for (k, tag) in &meta.idx_to_tag {
            if let Ok(idx) = k.parse::<usize>() {
                if idx < meta.total_tags {
                    idx_to_tag[idx] = tag.clone();
                    if let Some(cat) = meta.tag_to_category.get(tag) {
                        idx_to_cat[idx] = cat.clone();
                    }
                }
            }
        }

        Ok(Self {
            meta,
            dim,
            num_heads,
            head_dim,
            num_layers,
            grid_size,
            patch_size,
            patch_w,
            patch_b,
            cls_token,
            pos_embed,
            rope_sin,
            rope_cos,
            blocks,
            norm,
            pool,
            head,
            idx_to_tag,
            idx_to_cat,
        })
    }
}

/// Letterboxes an image onto a 448x448 white canvas (matching Haruka's PIL preprocessing)
/// and returns normalized NCHW `[3, 448, 448]` f32 tensor.
pub fn preprocess_image(path: &Path, target_size: usize) -> Result<Vec<f32>> {
    let img: DynamicImage = image::open(path)
        .with_context(|| format!("Failed to open image: {}", path.display()))?;
    Ok(preprocess_dynamic_image(&img, target_size))
}

/// Decodes an in-memory image (e.g. from a WebDataset tar entry) and preprocesses it.
pub fn preprocess_image_bytes(bytes: &[u8], target_size: usize) -> Result<Vec<f32>> {
    let img: DynamicImage = image::load_from_memory(bytes)
        .with_context(|| "Failed to decode image from memory")?;
    Ok(preprocess_dynamic_image(&img, target_size))
}

fn preprocess_dynamic_image(img: &DynamicImage, target_size: usize) -> Vec<f32> {
    let (w, h) = img.dimensions();
    let rgba = img.to_rgba8();

    // Alpha-composite onto white background
    let mut rgb_src = RgbaImage::from_pixel(w, h, Rgba([255, 255, 255, 255]));
    for y in 0..h {
        for x in 0..w {
            let p = rgba.get_pixel(x, y);
            let a = p[3] as f32 / 255.0;
            let r = (p[0] as f32 * a + 255.0 * (1.0 - a)).round() as u8;
            let g = (p[1] as f32 * a + 255.0 * (1.0 - a)).round() as u8;
            let b = (p[2] as f32 * a + 255.0 * (1.0 - a)).round() as u8;
            rgb_src.put_pixel(x, y, Rgba([r, g, b, 255]));
        }
    }

    let canvas = if w as usize != target_size || h as usize != target_size {
        let scale = (target_size as f32 / w as f32).min(target_size as f32 / h as f32);
        let nw = (w as f32 * scale).max(1.0) as u32;
        let nh = (h as f32 * scale).max(1.0) as u32;
        let resized = DynamicImage::ImageRgba8(rgb_src).resize_exact(nw, nh, FilterType::CatmullRom);
        let resized_rgba = resized.to_rgba8();
        let mut out = RgbaImage::from_pixel(target_size as u32, target_size as u32, Rgba([255, 255, 255, 255]));
        let ox = (target_size as u32 - nw) / 2;
        let oy = (target_size as u32 - nh) / 2;
        image::imageops::overlay(&mut out, &resized_rgba, ox as i64, oy as i64);
        out
    } else {
        rgb_src
    };

    let hw = target_size * target_size;
    let mut nchw = vec![0.0f32; 3 * hw];
    for y in 0..target_size {
        for x in 0..target_size {
            let p = canvas.get_pixel(x as u32, y as u32);
            let idx = y * target_size + x;
            for c in 0..3 {
                let v = (p[c] as f32) / 255.0;
                nchw[c * hw + idx] = (v - IMAGENET_MEAN[c]) / IMAGENET_STD[c];
            }
        }
    }
    nchw
}
