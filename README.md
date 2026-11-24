# haruka-w4a8 (Rust + wgpu + Rayon CPU Fallback)

Standalone, distributable Rust inference CLI for **Haruka-Tagger-12L W4A8 AWQ-QAT** (`haruka-tagger-12L-w4a8-awq-qat-packed.safetensors`).

- **GPU-accelerated via `wgpu`** (Vulkan / DirectX 12 / Metal / WebGPU-compatible compute shaders — works across AMD, NVIDIA, Intel, and Apple Silicon without CUDA).
- **Automatic multi-threaded CPU fallback** via `rayon` if no compatible GPU adapter is found, or forced via `--cpu`.
- **Zero Python / PyTorch runtime dependency**: compiles to a single native binary.

## Build

```bash
cargo build --release --bins
```

Binary outputs:
- CLI: `target/release/haruka-w4a8.exe` (Windows) or `target/release/haruka-w4a8` (Linux/macOS)
- GUI: `target/release/haruka-w4a8-gui.exe` (Windows) or `target/release/haruka-w4a8-gui` (Linux/macOS)

## Usage

```bash
# Single image (auto-selects wgpu GPU, falls back to CPU if unavailable)
haruka-w4a8 image.jpg \
  --weights haruka-tagger-12L-w4a8-awq-qat-packed.safetensors \
  --metadata haruka-tagger_metadata.json

# Directory of images -> per-image .txt captions (without probabilities)
haruka-w4a8 ./images/ \
  --weights haruka-tagger-12L-w4a8-awq-qat-packed.safetensors \
  --metadata haruka-tagger_metadata.json \
  --format txt --output ./captions/ --no-probs

# WebDataset .tar / .tar.gz shard(s) -> Parquet or SQLite
haruka-w4a8 ./shards/ \
  --weights haruka-tagger-12L-w4a8-awq-qat-packed.safetensors \
  --metadata haruka-tagger_metadata.json \
  --input-format webdataset \
  --format parquet --output tags.parquet \
  --threshold 0.35 --top-k 40

haruka-w4a8 shard_000.tar.gz \
  --weights haruka-tagger-12L-w4a8-awq-qat-packed.safetensors \
  --metadata haruka-tagger_metadata.json \
  --format sqlite --output tags.db --no-probs

# Force multi-threaded CPU fallback
haruka-w4a8 image.jpg \
  --weights haruka-tagger-12L-w4a8-awq-qat-packed.safetensors \
  --metadata haruka-tagger_metadata.json \
  --cpu

# Benchmark N iterations
haruka-w4a8 image.jpg \
  --weights haruka-tagger-12L-w4a8-awq-qat-packed.safetensors \
  --metadata haruka-tagger_metadata.json \
  --bench 5
```

## Architecture & Implementation Notes

- **Backbone**: 12-layer EVA02 (`dim=768`, `num_heads=12`, `head_dim=64`, 2D RoPE on `32x32` patch grid, SwiGLU MLP) with 4-bit AWQ-QAT packed weights (`qweight_packed`, `scales` per group of 64, `awq_scale`) and dynamic per-group INT8 activations (`act_group_size=128`).
- **Pooling & Head**: 2-level 18-query `MHAttnPool` (`slot_sigmoid`, `num_heads=16`) followed by a W4A8 `1152 -> 98782` Danbooru tag classifier head.
- **GPU Shaders**: Universal WGSL compute shaders (`extractBits` 4-bit unpacking + cooperative 64-thread online-softmax attention) batched into a single `CommandEncoder` submission for the full backbone.
