use crate::cpu_backend::forward_pool;
use crate::model::{HarukaModel, LayerNormParams, W4Linear};
use anyhow::{Context, Result};
use bytemuck::{Pod, Zeroable};
use std::borrow::Cow;
use wgpu::util::DeviceExt;

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct GemmParams {
    m: u32,
    n: u32,
    k_pack: u32,
    num_w_groups: u32,
    has_bias: u32,
    num_act_groups: u32,
    pad1: u32,
    pad2: u32,
}

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct QuantParams {
    m: u32,
    k_vec4: u32,
    num_act_groups: u32,
    pad0: u32,
}

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct LnParams {
    rows: u32,
    dim: u32,
    pad0: u32,
    pad1: u32,
}

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct AttnParams {
    seq_len: u32,
    dim: u32,
    num_heads: u32,
    head_dim: u32,
    scale: f32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct ElemParams {
    len: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct PatchParams {
    img_size: u32,
    patch_size: u32,
    grid_size: u32,
    dim: u32,
}

const QUANT_A8_WGSL: &str = r#"
struct QParams { m: u32, k_vec4: u32, num_act_groups: u32, pad0: u32 };
@group(0) @binding(0) var<storage, read> x_in: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read> inv_awq_scale: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read_write> x_q_packed: array<u32>;
@group(0) @binding(3) var<storage, read_write> x_scales: array<f32>;
@group(0) @binding(4) var<uniform> qp: QParams;

var<workgroup> sm_max: array<f32, 32>;

// 1 workgroup (32 threads) per (row, act_group=128); each thread handles 4 consecutive floats (1 vec4)
@compute @workgroup_size(32)
fn main(@builtin(workgroup_id) wgid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
    let row = wgid.x;
    let ag = wgid.y;
    let t = lid.x;
    let kv = ag * 32u + t;

    let v = x_in[row * qp.k_vec4 + kv] * inv_awq_scale[kv];
    let av = abs(v);
    sm_max[t] = max(max(av.x, av.y), max(av.z, av.w));
    workgroupBarrier();

    if (t < 16u) { sm_max[t] = max(sm_max[t], sm_max[t + 16u]); }
    workgroupBarrier();
    if (t < 8u) { sm_max[t] = max(sm_max[t], sm_max[t + 8u]); }
    workgroupBarrier();
    if (t < 4u) { sm_max[t] = max(sm_max[t], sm_max[t + 4u]); }
    workgroupBarrier();
    if (t < 2u) { sm_max[t] = max(sm_max[t], sm_max[t + 2u]); }
    workgroupBarrier();
    if (t == 0u) { sm_max[0] = max(sm_max[0], sm_max[1]); }
    workgroupBarrier();

    let sc = max(sm_max[0], 1e-6) / 127.0;
    let inv_sc = 1.0 / sc;
    let q = vec4<i32>(clamp(round(v * inv_sc), vec4<f32>(-127.0), vec4<f32>(127.0)));
    let word = (u32(q.x) & 0xFFu)
        | ((u32(q.y) & 0xFFu) << 8u)
        | ((u32(q.z) & 0xFFu) << 16u)
        | ((u32(q.w) & 0xFFu) << 24u);
    x_q_packed[row * qp.k_vec4 + kv] = word;
    if (t == 0u) {
        x_scales[row * qp.num_act_groups + ag] = sc;
    }
}
"#;

// Hardware W4A8 DP4A Linear (dot4I8Packed via wgpu 30 / naga):
// Uses coalesced pre-interleaved [K_pack, N] weights, [num_groups, N] scales, and quantized A8 activations.
const W4A8_DP4A_WGSL: &str = r#"
struct Params { m: u32, n: u32, k_pack: u32, num_w_groups: u32, has_bias: u32, num_act_groups: u32, pad1: u32, pad2: u32 };
@group(0) @binding(0) var<storage, read> x_q_packed: array<u32>;
@group(0) @binding(1) var<storage, read> x_scales: array<f32>;
@group(0) @binding(2) var<storage, read> w_buf: array<u32>;
@group(0) @binding(3) var<storage, read> w_scales: array<f32>;
@group(0) @binding(4) var<storage, read> bias_buf: array<f32>;
@group(0) @binding(5) var<storage, read_write> out_buf: array<f32>;
@group(0) @binding(6) var<uniform> params: Params;

var<workgroup> wg_x: array<u32, 128>; // 8 rows x 16 u32 (64 int8 values = 1 weight group)
var<workgroup> wg_xs: array<f32, 8>;  // 1 activation scale per row for current group

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
    let n = gid.x;
    let m_base = gid.y * 8u;
    let safe_n = min(n, params.n - 1u);
    let k_u32_stride = params.k_pack * 2u;

    var acc0: f32 = 0.0; var acc1: f32 = 0.0; var acc2: f32 = 0.0; var acc3: f32 = 0.0;
    var acc4: f32 = 0.0; var acc5: f32 = 0.0; var acc6: f32 = 0.0; var acc7: f32 = 0.0;

    var grp: u32 = 0u;
    for (var kt: u32 = 0u; kt < params.k_pack; kt = kt + 8u) {
        let t0 = lid.x;
        let valid_j = min(8u, params.k_pack - kt);
        let ag = grp >> 1u;
        if (params.m == 1u) {
            if (t0 < valid_j * 2u) {
                wg_x[t0] = x_q_packed[kt * 2u + t0];
            }
            if (t0 == 0u) {
                wg_xs[0] = x_scales[ag];
            }
        } else {
            let r0 = min(m_base + (t0 >> 4u), params.m - 1u);
            let r1 = min(m_base + (t0 >> 4u) + 4u, params.m - 1u);
            let cv = t0 & 15u;
            let k_u = kt * 2u + cv;
            wg_x[t0] = x_q_packed[r0 * k_u32_stride + k_u];
            wg_x[t0 + 64u] = x_q_packed[r1 * k_u32_stride + k_u];
            if (t0 < 8u) {
                wg_xs[t0] = x_scales[min(m_base + t0, params.m - 1u) * params.num_act_groups + ag];
            }
        }
        workgroupBarrier();

        let ws = w_scales[grp * params.n + safe_n];
        if (params.m == 1u) {
            var g0: i32 = 0;
            for (var j: u32 = 0u; j < valid_j; j = j + 1u) {
                let wp = w_buf[(kt + j) * params.n + safe_n];
                let lo_u = wp & 0x0F0F0F0Fu;
                let lo_i8 = lo_u | ((lo_u & 0x08080808u) * 30u);
                let hi_u = (wp >> 4u) & 0x0F0F0F0Fu;
                let hi_i8 = hi_u | ((hi_u & 0x08080808u) * 30u);
                let j0 = j * 2u;
                g0 += dot4I8Packed(wg_x[j0], lo_i8) + dot4I8Packed(wg_x[j0 + 1u], hi_i8);
            }
            acc0 += f32(g0) * (ws * wg_xs[0]);
        } else {
            var g0: i32 = 0; var g1: i32 = 0; var g2: i32 = 0; var g3: i32 = 0;
            var g4: i32 = 0; var g5: i32 = 0; var g6: i32 = 0; var g7: i32 = 0;
            for (var j: u32 = 0u; j < valid_j; j = j + 1u) {
                let wp = w_buf[(kt + j) * params.n + safe_n];
                let lo_u = wp & 0x0F0F0F0Fu;
                let lo_i8 = lo_u | ((lo_u & 0x08080808u) * 30u);
                let hi_u = (wp >> 4u) & 0x0F0F0F0Fu;
                let hi_i8 = hi_u | ((hi_u & 0x08080808u) * 30u);
                let j0 = j * 2u;
                let j1 = j0 + 1u;
                g0 += dot4I8Packed(wg_x[0u * 16u + j0], lo_i8) + dot4I8Packed(wg_x[0u * 16u + j1], hi_i8);
                g1 += dot4I8Packed(wg_x[1u * 16u + j0], lo_i8) + dot4I8Packed(wg_x[1u * 16u + j1], hi_i8);
                g2 += dot4I8Packed(wg_x[2u * 16u + j0], lo_i8) + dot4I8Packed(wg_x[2u * 16u + j1], hi_i8);
                g3 += dot4I8Packed(wg_x[3u * 16u + j0], lo_i8) + dot4I8Packed(wg_x[3u * 16u + j1], hi_i8);
                g4 += dot4I8Packed(wg_x[4u * 16u + j0], lo_i8) + dot4I8Packed(wg_x[4u * 16u + j1], hi_i8);
                g5 += dot4I8Packed(wg_x[5u * 16u + j0], lo_i8) + dot4I8Packed(wg_x[5u * 16u + j1], hi_i8);
                g6 += dot4I8Packed(wg_x[6u * 16u + j0], lo_i8) + dot4I8Packed(wg_x[6u * 16u + j1], hi_i8);
                g7 += dot4I8Packed(wg_x[7u * 16u + j0], lo_i8) + dot4I8Packed(wg_x[7u * 16u + j1], hi_i8);
            }
            acc0 += f32(g0) * (ws * wg_xs[0]);
            acc1 += f32(g1) * (ws * wg_xs[1]);
            acc2 += f32(g2) * (ws * wg_xs[2]);
            acc3 += f32(g3) * (ws * wg_xs[3]);
            acc4 += f32(g4) * (ws * wg_xs[4]);
            acc5 += f32(g5) * (ws * wg_xs[5]);
            acc6 += f32(g6) * (ws * wg_xs[6]);
            acc7 += f32(g7) * (ws * wg_xs[7]);
        }
        grp = grp + 1u;
        workgroupBarrier();
    }

    if (n < params.n && m_base < params.m) {
        let b = select(0.0, bias_buf[n], params.has_bias != 0u);
        out_buf[(m_base + 0u) * params.n + n] = acc0 + b;
        if (m_base + 1u < params.m) { out_buf[(m_base + 1u) * params.n + n] = acc1 + b; }
        if (m_base + 2u < params.m) { out_buf[(m_base + 2u) * params.n + n] = acc2 + b; }
        if (m_base + 3u < params.m) { out_buf[(m_base + 3u) * params.n + n] = acc3 + b; }
        if (m_base + 4u < params.m) { out_buf[(m_base + 4u) * params.n + n] = acc4 + b; }
        if (m_base + 5u < params.m) { out_buf[(m_base + 5u) * params.n + n] = acc5 + b; }
        if (m_base + 6u < params.m) { out_buf[(m_base + 6u) * params.n + n] = acc6 + b; }
        if (m_base + 7u < params.m) { out_buf[(m_base + 7u) * params.n + n] = acc7 + b; }
    }
}
"#;

const LAYERNORM_WGSL: &str = r#"
struct LnParams { rows: u32, dim: u32, pad0: u32, pad1: u32 };
@group(0) @binding(0) var<storage, read> x_in: array<f32>;
@group(0) @binding(1) var<storage, read> weight: array<f32>;
@group(0) @binding(2) var<storage, read> bias: array<f32>;
@group(0) @binding(3) var<storage, read_write> out_buf: array<f32>;
@group(0) @binding(4) var<uniform> p: LnParams;

var<workgroup> smem: array<f32, 64>;

@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
    let row = wid.x;
    if (row >= p.rows) { return; }
    let off = row * p.dim;
    let tid = lid.x;

    var sum: f32 = 0.0;
    for (var i: u32 = tid; i < p.dim; i = i + 64u) {
        sum += x_in[off + i];
    }
    smem[tid] = sum;
    workgroupBarrier();
    for (var s: u32 = 32u; s > 0u; s = s >> 1u) {
        if (tid < s) { smem[tid] += smem[tid + s]; }
        workgroupBarrier();
    }
    let mean = smem[0] / f32(p.dim);
    workgroupBarrier();

    var vsum: f32 = 0.0;
    for (var i: u32 = tid; i < p.dim; i = i + 64u) {
        let d = x_in[off + i] - mean;
        vsum += d * d;
    }
    smem[tid] = vsum;
    workgroupBarrier();
    for (var s: u32 = 32u; s > 0u; s = s >> 1u) {
        if (tid < s) { smem[tid] += smem[tid + s]; }
        workgroupBarrier();
    }
    let inv_std = inverseSqrt(smem[0] / f32(p.dim) + 1e-6);

    for (var i: u32 = tid; i < p.dim; i = i + 64u) {
        out_buf[off + i] = (x_in[off + i] - mean) * inv_std * weight[i] + bias[i];
    }
}
"#;

const ROPE_WGSL: &str = r#"
struct AttnParams { seq_len: u32, dim: u32, num_heads: u32, head_dim: u32, scale: f32, pad0: u32, pad1: u32, pad2: u32 };
@group(0) @binding(0) var<storage, read_write> q_buf: array<f32>;
@group(0) @binding(1) var<storage, read_write> k_buf: array<f32>;
@group(0) @binding(2) var<storage, read> rope_sin: array<f32>;
@group(0) @binding(3) var<storage, read> rope_cos: array<f32>;
@group(0) @binding(4) var<uniform> p: AttnParams;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let pair_idx = gid.x;
    let n_patches = p.seq_len - 1u;
    let pairs_per_tok = p.dim / 2u;
    if (pair_idx >= n_patches * pairs_per_tok) { return; }

    let patch_i = pair_idx / pairs_per_tok;
    let tok_i = patch_i + 1u; // skip CLS token
    let in_tok_pair = pair_idx % pairs_per_tok;
    let hd_pair = in_tok_pair % (p.head_dim / 2u);
    let i0 = tok_i * p.dim + in_tok_pair * 2u;
    let i1 = i0 + 1u;

    let r0 = patch_i * p.head_dim + hd_pair * 2u;
    let r1 = r0 + 1u;
    let s0 = rope_sin[r0]; let s1 = rope_sin[r1];
    let c0 = rope_cos[r0]; let c1 = rope_cos[r1];

    let q0 = q_buf[i0]; let q1 = q_buf[i1];
    q_buf[i0] = q0 * c0 - q1 * s0;
    q_buf[i1] = q1 * c1 + q0 * s1;

    let k0 = k_buf[i0]; let k1 = k_buf[i1];
    k_buf[i0] = k0 * c0 - k1 * s0;
    k_buf[i1] = k1 * c1 + k0 * s1;
}
"#;

const ATTN_WGSL: &str = r#"
struct AttnParams { seq_len: u32, dim: u32, num_heads: u32, head_dim: u32, scale: f32, pad0: u32, pad1: u32, pad2: u32 };
@group(0) @binding(0) var<storage, read> q_buf: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read> k_buf: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read> v_buf: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read_write> out_buf: array<vec4<f32>>;
@group(0) @binding(4) var<uniform> p: AttnParams;

var<workgroup> sm_k: array<vec4<f32>, 256>;
var<workgroup> sm_v: array<vec4<f32>, 256>;

// Tiled FlashAttention: BQ = 64 queries per workgroup, BK = 16 keys per tile, head_dim = 64 (16 vec4<f32>)
@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wgid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
    let qi = wgid.x * 64u + lid.x;
    let h = wgid.y;
    let valid_q = qi < p.seq_len && h < p.num_heads;
    let dim_v4 = p.dim / 4u;
    let h_v4 = h * 16u;

    var q_reg: array<vec4<f32>, 16>;
    var acc: array<vec4<f32>, 16>;
    for (var d: u32 = 0u; d < 16u; d = d + 1u) {
        q_reg[d] = select(vec4<f32>(0.0), q_buf[qi * dim_v4 + h_v4 + d], valid_q);
        acc[d] = vec4<f32>(0.0);
    }

    var m_prev: f32 = -1e30;
    var l_prev: f32 = 0.0;

    for (var k_base: u32 = 0u; k_base < p.seq_len; k_base = k_base + 16u) {
        for (var c: u32 = 0u; c < 4u; c = c + 1u) {
            let slot = c * 64u + lid.x;
            let lk = slot >> 4u;
            let lv = slot & 15u;
            let gk = k_base + lk;
            if (gk < p.seq_len) {
                let g_idx = gk * dim_v4 + h_v4 + lv;
                sm_k[slot] = k_buf[g_idx];
                sm_v[slot] = v_buf[g_idx];
            } else {
                sm_k[slot] = vec4<f32>(0.0);
                sm_v[slot] = vec4<f32>(0.0);
            }
        }
        workgroupBarrier();

        if (valid_q) {
            let tile_k = min(16u, p.seq_len - k_base);
            for (var lk: u32 = 0u; lk < tile_k; lk = lk + 1u) {
                let k_off = lk * 16u;
                var dot_v: f32 = 0.0;
                for (var d: u32 = 0u; d < 16u; d = d + 1u) {
                    dot_v += dot(q_reg[d], sm_k[k_off + d]);
                }
                let s = dot_v * p.scale;
                let m_new = max(m_prev, s);
                let alpha = exp(m_prev - m_new);
                let beta = exp(s - m_new);
                l_prev = l_prev * alpha + beta;
                for (var d: u32 = 0u; d < 16u; d = d + 1u) {
                    acc[d] = acc[d] * alpha + beta * sm_v[k_off + d];
                }
                m_prev = m_new;
            }
        }
        workgroupBarrier();
    }

    if (valid_q) {
        let inv_l = 1.0 / l_prev;
        let out_off = qi * dim_v4 + h_v4;
        for (var d: u32 = 0u; d < 16u; d = d + 1u) {
            out_buf[out_off + d] = acc[d] * inv_l;
        }
    }
}
"#;

const SWIGLU_WGSL: &str = r#"
struct ElemParams { len: u32, pad0: u32, pad1: u32, pad2: u32 };
@group(0) @binding(0) var<storage, read> g_buf: array<f32>;
@group(0) @binding(1) var<storage, read> x_buf: array<f32>;
@group(0) @binding(2) var<storage, read_write> out_buf: array<f32>;
@group(0) @binding(3) var<uniform> p: ElemParams;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= p.len) { return; }
    let g = g_buf[i];
    out_buf[i] = (g / (1.0 + exp(-g))) * x_buf[i];
}
"#;

const ADD_RESIDUAL_WGSL: &str = r#"
struct ElemParams { len: u32, pad0: u32, pad1: u32, pad2: u32 };
@group(0) @binding(0) var<storage, read_write> x_buf: array<f32>;
@group(0) @binding(1) var<storage, read> delta_buf: array<f32>;
@group(0) @binding(2) var<uniform> p: ElemParams;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= p.len) { return; }
    x_buf[i] += delta_buf[i];
}
"#;

const PATCH_EMBED_WGSL: &str = r#"
struct PatchParams { img_size: u32, patch_size: u32, grid_size: u32, dim: u32 };
@group(0) @binding(0) var<storage, read> nchw: array<f32>;
@group(0) @binding(1) var<storage, read> patch_w: array<f32>;
@group(0) @binding(2) var<storage, read> patch_b: array<f32>;
@group(0) @binding(3) var<storage, read> cls_tok: array<f32>;
@group(0) @binding(4) var<storage, read> pos_embed: array<f32>;
@group(0) @binding(5) var<storage, read_write> out_buf: array<f32>;
@group(0) @binding(6) var<uniform> p: PatchParams;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    let seq_len = 1u + p.grid_size * p.grid_size;
    if (idx >= seq_len * p.dim) { return; }

    let tok = idx / p.dim;
    let d = idx % p.dim;
    if (tok == 0u) {
        out_buf[idx] = cls_tok[d] + pos_embed[d];
        return;
    }

    let p_idx = tok - 1u;
    let py = p_idx / p.grid_size;
    let px = p_idx % p.grid_size;
    let hw = p.img_size * p.img_size;
    let patch_dim = 3u * p.patch_size * p.patch_size;
    let w_off = d * patch_dim;

    var sum: f32 = patch_b[d] + pos_embed[tok * p.dim + d];
    var k: u32 = 0u;
    for (var c: u32 = 0u; c < 3u; c = c + 1u) {
        for (var dy: u32 = 0u; dy < p.patch_size; dy = dy + 1u) {
            let y = py * p.patch_size + dy;
            let row_off = c * hw + y * p.img_size + px * p.patch_size;
            for (var dx: u32 = 0u; dx < p.patch_size; dx = dx + 1u) {
                sum += nchw[row_off + dx] * patch_w[w_off + k];
                k = k + 1u;
            }
        }
    }
    out_buf[idx] = sum;
}
"#;

#[allow(dead_code)]
struct GpuW4Linear {
    out_features: usize,
    in_features: usize,
    num_groups: usize,
    has_bias: bool,
    w_buf: wgpu::Buffer,
    w_scales_buf: wgpu::Buffer,
    awq_scale_buf: wgpu::Buffer,
    bias_buf: wgpu::Buffer,
}

#[allow(dead_code)]
struct GpuLayerNorm {
    dim: usize,
    w_buf: wgpu::Buffer,
    b_buf: wgpu::Buffer,
}

struct GpuBlock {
    norm1: GpuLayerNorm,
    q_proj: GpuW4Linear,
    k_proj: GpuW4Linear,
    v_proj: GpuW4Linear,
    proj: GpuW4Linear,
    norm2: GpuLayerNorm,
    fc1_g: GpuW4Linear,
    fc1_x: GpuW4Linear,
    mlp_norm: GpuLayerNorm,
    fc2: GpuW4Linear,
}

pub struct WgpuRunner {
    pub adapter_name: String,
    device: wgpu::Device,
    queue: wgpu::Queue,
    quant_pipe: wgpu::ComputePipeline,
    w4a8_pipe: wgpu::ComputePipeline,
    ln_pipe: wgpu::ComputePipeline,
    rope_pipe: wgpu::ComputePipeline,
    attn_pipe: wgpu::ComputePipeline,
    swiglu_pipe: wgpu::ComputePipeline,
    add_pipe: wgpu::ComputePipeline,
    patch_pipe: wgpu::ComputePipeline,

    // Persistent model buffers
    patch_w_buf: wgpu::Buffer,
    patch_b_buf: wgpu::Buffer,
    cls_buf: wgpu::Buffer,
    pos_buf: wgpu::Buffer,
    rope_sin_buf: wgpu::Buffer,
    rope_cos_buf: wgpu::Buffer,
    blocks: Vec<GpuBlock>,
    norm_final: GpuLayerNorm,
    head: GpuW4Linear,

    // Reusable scratch buffers
    nchw_buf: wgpu::Buffer,
    x_buf: wgpu::Buffer,
    h_buf: wgpu::Buffer,
    q_buf: wgpu::Buffer,
    k_buf: wgpu::Buffer,
    v_buf: wgpu::Buffer,
    attn_out_buf: wgpu::Buffer,
    fg_buf: wgpu::Buffer,
    fx_buf: wgpu::Buffer,
    delta_buf: wgpu::Buffer,
    x_q_buf: wgpu::Buffer,
    x_sc_buf: wgpu::Buffer,
    pool_in_buf: wgpu::Buffer,
    head_out_buf: wgpu::Buffer,
    readback_buf: wgpu::Buffer,

    // Pre-populated Uniform buffers (for single-CommandEncoder batching)
    u_patch: wgpu::Buffer,
    u_attn: wgpu::Buffer,
    u_ln_dim: wgpu::Buffer,
    u_ln_mlp: wgpu::Buffer,
    u_elem_tok: wgpu::Buffer,
    u_elem_mlp: wgpu::Buffer,
    u_quant_dim: wgpu::Buffer,
    u_quant_mlp: wgpu::Buffer,
    u_quant_head: wgpu::Buffer,
    u_gemm_qkv: wgpu::Buffer,
    u_gemm_k: wgpu::Buffer,
    u_gemm_fc1: wgpu::Buffer,
    u_gemm_fc2: wgpu::Buffer,
    u_gemm_head: wgpu::Buffer,
}

fn upload_f32(device: &wgpu::Device, label: &str, data: &[f32]) -> wgpu::Buffer {
    device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some(label),
        contents: bytemuck::cast_slice(data),
        usage: wgpu::BufferUsages::STORAGE,
    })
}

/// Reorders 8 packed 4-bit weights `[w0..w7]` into `[w0,w4, w1,w5, w2,w6, w3,w7]`
/// so `lo_u = wp & 0x0F0F0F0F` holds `[w0..w3]` and `hi_u = (wp >> 4) & 0x0F0F0F0F` holds `[w4..w7]` for `dot4I8Packed`.
#[inline]
fn interleave_w4_for_dp4a(word: u32) -> u32 {
    let mut out = 0u32;
    for b in 0..4u32 {
        let lo = (word >> (b * 4)) & 0xF;
        let hi = (word >> ((b + 4) * 4)) & 0xF;
        out |= (lo | (hi << 4)) << (b * 8);
    }
    out
}

fn upload_w4(device: &wgpu::Device, label: &str, lin: &W4Linear) -> GpuW4Linear {
    let n_out = lin.out_features;
    let k_pack = lin.in_features / 8;
    let mut w_t = vec![0u32; k_pack * n_out];
    for n in 0..n_out {
        for kp in 0..k_pack {
            w_t[kp * n_out + n] = interleave_w4_for_dp4a(lin.qweight_u32_signed[n * k_pack + kp]);
        }
    }
    let w_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some(label),
        contents: bytemuck::cast_slice(&w_t),
        usage: wgpu::BufferUsages::STORAGE,
    });

    let ng = lin.num_groups;
    let mut scales_t = vec![0.0f32; ng * n_out];
    for n in 0..n_out {
        for g in 0..ng {
            scales_t[g * n_out + n] = lin.scales[n * ng + g];
        }
    }
    let w_scales_buf = upload_f32(device, "w_scales", &scales_t);

    let inv_awq: Vec<f32> = lin.awq_scale.iter().map(|&s| 1.0f32 / s).collect();
    let awq_scale_buf = upload_f32(device, "awq_scale", &inv_awq);

    let dummy_bias = [0.0f32];
    let bias_slice = lin.bias.as_deref().unwrap_or(&dummy_bias);
    let bias_buf = upload_f32(device, "bias", bias_slice);
    GpuW4Linear {
        out_features: lin.out_features,
        in_features: lin.in_features,
        num_groups: lin.num_groups,
        has_bias: lin.bias.is_some(),
        w_buf,
        w_scales_buf,
        awq_scale_buf,
        bias_buf,
    }
}

fn upload_ln(device: &wgpu::Device, ln: &LayerNormParams) -> GpuLayerNorm {
    GpuLayerNorm {
        dim: ln.dim,
        w_buf: upload_f32(device, "ln_w", &ln.weight),
        b_buf: upload_f32(device, "ln_b", &ln.bias),
    }
}

impl WgpuRunner {
    pub fn try_new(model: &HarukaModel) -> Result<Self> {
        let instance = wgpu::Instance::default();
        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            compatible_surface: None,
            force_fallback_adapter: false,
            ..Default::default()
        }))
        .context("No compatible wgpu GPU adapter found")?;

        let info = adapter.get_info();
        let adapter_name = format!("{} ({:?})", info.name, info.backend);

        let mut limits = wgpu::Limits::default();
        limits.max_storage_buffer_binding_size = adapter.limits().max_storage_buffer_binding_size;
        limits.max_buffer_size = adapter.limits().max_buffer_size;

        let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
            label: Some("haruka-w4a8"),
            required_features: wgpu::Features::empty(),
            required_limits: limits,
            memory_hints: wgpu::MemoryHints::Performance,
            ..Default::default()
        }))
        .context("Failed to create wgpu device")?;

        // Validate that WGSL dot4I8Packed compiles cleanly on this GPU driver
        let err_scope = device.push_error_scope(wgpu::ErrorFilter::Validation);
        let mk_pipe = |label: &str, wgsl: &str| -> wgpu::ComputePipeline {
            let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some(label),
                source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(wgsl)),
            });
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(label),
                layout: None,
                module: &module,
                entry_point: Some("main"),
                compilation_options: Default::default(),
                cache: None,
            })
        };

        let quant_pipe = mk_pipe("quant_a8", QUANT_A8_WGSL);
        let w4a8_pipe = mk_pipe("w4a8_dp4a", W4A8_DP4A_WGSL);
        let ln_pipe = mk_pipe("layernorm", LAYERNORM_WGSL);
        let rope_pipe = mk_pipe("rope", ROPE_WGSL);
        let attn_pipe = mk_pipe("attn", ATTN_WGSL);
        let swiglu_pipe = mk_pipe("swiglu", SWIGLU_WGSL);
        let add_pipe = mk_pipe("add_res", ADD_RESIDUAL_WGSL);
        let patch_pipe = mk_pipe("patch_embed", PATCH_EMBED_WGSL);

        if let Some(err) = pollster::block_on(err_scope.pop()) {
            anyhow::bail!("wgpu shader validation failed (DP4A may not be supported on this GPU): {err}");
        }

        let mut blocks = Vec::with_capacity(model.blocks.len());
        for (i, b) in model.blocks.iter().enumerate() {
            blocks.push(GpuBlock {
                norm1: upload_ln(&device, &b.norm1),
                q_proj: upload_w4(&device, &format!("b{i}.q"), &b.q_proj),
                k_proj: upload_w4(&device, &format!("b{i}.k"), &b.k_proj),
                v_proj: upload_w4(&device, &format!("b{i}.v"), &b.v_proj),
                proj: upload_w4(&device, &format!("b{i}.proj"), &b.proj),
                norm2: upload_ln(&device, &b.norm2),
                fc1_g: upload_w4(&device, &format!("b{i}.fc1_g"), &b.fc1_g),
                fc1_x: upload_w4(&device, &format!("b{i}.fc1_x"), &b.fc1_x),
                mlp_norm: upload_ln(&device, &b.mlp_norm),
                fc2: upload_w4(&device, &format!("b{i}.fc2"), &b.fc2),
            });
        }

        let seq_len = 1 + model.grid_size * model.grid_size;
        let dim = model.dim;
        let mlp_hidden = model.blocks[0].fc1_g.out_features;

        let mk_rw = |label: &str, bytes: usize| -> wgpu::Buffer {
            device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(label),
                size: (bytes.max(64)) as u64,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            })
        };
        let mk_uni_init = |label: &str, bytes: &[u8]| -> wgpu::Buffer {
            device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some(label),
                contents: bytes,
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            })
        };

        let nchw_bytes = 3 * model.meta.img_size * model.meta.img_size * 4;
        let tok_bytes = seq_len * dim * 4;
        let mlp_bytes = seq_len * mlp_hidden * 4;
        let max_k = mlp_hidden.max(model.head.in_features);
        let head_bytes = model.head.out_features * 4;

        let pp = PatchParams {
            img_size: model.meta.img_size as u32,
            patch_size: model.patch_size as u32,
            grid_size: model.grid_size as u32,
            dim: dim as u32,
        };
        let ap = AttnParams {
            seq_len: seq_len as u32,
            dim: dim as u32,
            num_heads: model.num_heads as u32,
            head_dim: model.head_dim as u32,
            scale: 1.0f32 / (model.head_dim as f32).sqrt(),
            pad0: 0,
            pad1: 0,
            pad2: 0,
        };
        let ln_dim = LnParams { rows: seq_len as u32, dim: dim as u32, pad0: 0, pad1: 0 };
        let ln_mlp = LnParams { rows: seq_len as u32, dim: mlp_hidden as u32, pad0: 0, pad1: 0 };
        let elem_tok = ElemParams { len: (seq_len * dim) as u32, pad0: 0, pad1: 0, pad2: 0 };
        let elem_mlp = ElemParams { len: (seq_len * mlp_hidden) as u32, pad0: 0, pad1: 0, pad2: 0 };

        let mk_quant_p = |m: usize, k: usize| QuantParams {
            m: m as u32,
            k_vec4: (k / 4) as u32,
            num_act_groups: (k / 128) as u32,
            pad0: 0,
        };
        let qp_dim = mk_quant_p(seq_len, dim);
        let qp_mlp = mk_quant_p(seq_len, mlp_hidden);
        let qp_head = mk_quant_p(1, model.head.in_features);

        let mk_gemm_p = |m: usize, lin: &crate::model::W4Linear| GemmParams {
            m: m as u32,
            n: lin.out_features as u32,
            k_pack: (lin.in_features / 8) as u32,
            num_w_groups: lin.num_groups as u32,
            has_bias: if lin.bias.is_some() { 1 } else { 0 },
            num_act_groups: (lin.in_features / 128) as u32,
            pad1: 0,
            pad2: 0,
        };
        let gp_qkv = mk_gemm_p(seq_len, &model.blocks[0].q_proj);
        let gp_k = mk_gemm_p(seq_len, &model.blocks[0].k_proj);
        let gp_fc1 = mk_gemm_p(seq_len, &model.blocks[0].fc1_g);
        let gp_fc2 = mk_gemm_p(seq_len, &model.blocks[0].fc2);
        let gp_head = mk_gemm_p(1, &model.head);

        Ok(Self {
            adapter_name,
            patch_w_buf: upload_f32(&device, "patch_w", &model.patch_w),
            patch_b_buf: upload_f32(&device, "patch_b", &model.patch_b),
            cls_buf: upload_f32(&device, "cls", &model.cls_token),
            pos_buf: upload_f32(&device, "pos", &model.pos_embed),
            rope_sin_buf: upload_f32(&device, "rope_sin", &model.rope_sin),
            rope_cos_buf: upload_f32(&device, "rope_cos", &model.rope_cos),
            blocks,
            norm_final: upload_ln(&device, &model.norm),
            head: upload_w4(&device, "head", &model.head),

            nchw_buf: mk_rw("nchw", nchw_bytes),
            x_buf: mk_rw("x", tok_bytes),
            h_buf: mk_rw("h", tok_bytes),
            q_buf: mk_rw("q", tok_bytes),
            k_buf: mk_rw("k", tok_bytes),
            v_buf: mk_rw("v", tok_bytes),
            attn_out_buf: mk_rw("attn_out", tok_bytes),
            fg_buf: mk_rw("fg", mlp_bytes),
            fx_buf: mk_rw("fx", mlp_bytes),
            delta_buf: mk_rw("delta", mlp_bytes),
            x_q_buf: mk_rw("x_q", seq_len * max_k),
            x_sc_buf: mk_rw("x_sc", seq_len * (max_k / 128 + 1) * 4),
            pool_in_buf: mk_rw("pool_in", model.head.in_features * 4),
            head_out_buf: mk_rw("head_out", head_bytes),
            readback_buf: device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("readback"),
                size: tok_bytes.max(head_bytes) as u64,
                usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }),

            u_patch: mk_uni_init("u_patch", bytemuck::bytes_of(&pp)),
            u_attn: mk_uni_init("u_attn", bytemuck::bytes_of(&ap)),
            u_ln_dim: mk_uni_init("u_ln_dim", bytemuck::bytes_of(&ln_dim)),
            u_ln_mlp: mk_uni_init("u_ln_mlp", bytemuck::bytes_of(&ln_mlp)),
            u_elem_tok: mk_uni_init("u_elem_tok", bytemuck::bytes_of(&elem_tok)),
            u_elem_mlp: mk_uni_init("u_elem_mlp", bytemuck::bytes_of(&elem_mlp)),
            u_quant_dim: mk_uni_init("u_quant_dim", bytemuck::bytes_of(&qp_dim)),
            u_quant_mlp: mk_uni_init("u_quant_mlp", bytemuck::bytes_of(&qp_mlp)),
            u_quant_head: mk_uni_init("u_quant_head", bytemuck::bytes_of(&qp_head)),
            u_gemm_qkv: mk_uni_init("u_gemm_qkv", bytemuck::bytes_of(&gp_qkv)),
            u_gemm_k: mk_uni_init("u_gemm_k", bytemuck::bytes_of(&gp_k)),
            u_gemm_fc1: mk_uni_init("u_gemm_fc1", bytemuck::bytes_of(&gp_fc1)),
            u_gemm_fc2: mk_uni_init("u_gemm_fc2", bytemuck::bytes_of(&gp_fc2)),
            u_gemm_head: mk_uni_init("u_gemm_head", bytemuck::bytes_of(&gp_head)),

            device,
            queue,
            quant_pipe,
            w4a8_pipe,
            ln_pipe,
            rope_pipe,
            attn_pipe,
            swiglu_pipe,
            add_pipe,
            patch_pipe,
        })
    }

    fn encode_ln(
        &self,
        enc: &mut wgpu::CommandEncoder,
        x_buf: &wgpu::Buffer,
        ln: &GpuLayerNorm,
        out_buf: &wgpu::Buffer,
        u_ln: &wgpu::Buffer,
        rows: usize,
    ) {
        let bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &self.ln_pipe.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: x_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: ln.w_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: ln.b_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: out_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 4, resource: u_ln.as_entire_binding() },
            ],
        });
        let mut pass = enc.begin_compute_pass(&Default::default());
        pass.set_pipeline(&self.ln_pipe);
        pass.set_bind_group(0, &bg, &[]);
        pass.dispatch_workgroups(rows as u32, 1, 1);
    }

    fn encode_w4a8(
        &self,
        enc: &mut wgpu::CommandEncoder,
        x_buf: &wgpu::Buffer,
        lin: &GpuW4Linear,
        out_buf: &wgpu::Buffer,
        u_quant: &wgpu::Buffer,
        u_gemm: &wgpu::Buffer,
        rows: usize,
    ) {
        let q_bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &self.quant_pipe.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: x_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: lin.awq_scale_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: self.x_q_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: self.x_sc_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 4, resource: u_quant.as_entire_binding() },
            ],
        });
        {
            let mut pass = enc.begin_compute_pass(&Default::default());
            pass.set_pipeline(&self.quant_pipe);
            pass.set_bind_group(0, &q_bg, &[]);
            pass.dispatch_workgroups(rows as u32, (lin.in_features / 128) as u32, 1);
        }

        let g_bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &self.w4a8_pipe.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: self.x_q_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: self.x_sc_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: lin.w_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: lin.w_scales_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 4, resource: lin.bias_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 5, resource: out_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 6, resource: u_gemm.as_entire_binding() },
            ],
        });
        let mut pass = enc.begin_compute_pass(&Default::default());
        pass.set_pipeline(&self.w4a8_pipe);
        pass.set_bind_group(0, &g_bg, &[]);
        pass.dispatch_workgroups(
            ((lin.out_features + 63) / 64) as u32,
            ((rows + 7) / 8) as u32,
            1,
        );
    }

    fn readback_from_submitted(&self, count: usize) -> Vec<f32> {
        let bytes = (count * 4) as u64;
        let slice = self.readback_buf.slice(0..bytes);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |res| {
            let _ = tx.send(res);
        });
        self.device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
        rx.recv().unwrap().unwrap();
        let data = slice.get_mapped_range().unwrap();
        let out: Vec<f32> = bytemuck::cast_slice(&data).to_vec();
        drop(data);
        self.readback_buf.unmap();
        out
    }

    pub fn infer(&self, model: &HarukaModel, nchw: &[f32]) -> Vec<f32> {
        let seq_len = 1 + model.grid_size * model.grid_size;
        let dim = model.dim;
        let mlp_hidden = model.blocks[0].fc1_g.out_features;

        // 1. Upload image
        self.queue.write_buffer(&self.nchw_buf, 0, bytemuck::cast_slice(nchw));

        // 2. Single CommandEncoder for patch_embed + all 12 transformer blocks + final norm
        let mut enc = self.device.create_command_encoder(&Default::default());

        let p_bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &self.patch_pipe.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: self.nchw_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: self.patch_w_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: self.patch_b_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: self.cls_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 4, resource: self.pos_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 5, resource: self.x_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 6, resource: self.u_patch.as_entire_binding() },
            ],
        });
        {
            let mut pass = enc.begin_compute_pass(&Default::default());
            pass.set_pipeline(&self.patch_pipe);
            pass.set_bind_group(0, &p_bg, &[]);
            pass.dispatch_workgroups(((seq_len * dim + 63) / 64) as u32, 1, 1);
        }

        let rope_bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &self.rope_pipe.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: self.q_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: self.k_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: self.rope_sin_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: self.rope_cos_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 4, resource: self.u_attn.as_entire_binding() },
            ],
        });
        let attn_bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &self.attn_pipe.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: self.q_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: self.k_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: self.v_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: self.attn_out_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 4, resource: self.u_attn.as_entire_binding() },
            ],
        });
        let add_bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &self.add_pipe.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: self.x_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: self.delta_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: self.u_elem_tok.as_entire_binding() },
            ],
        });
        let sw_bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &self.swiglu_pipe.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: self.fg_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: self.fx_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: self.delta_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: self.u_elem_mlp.as_entire_binding() },
            ],
        });

        let n_pairs = (seq_len - 1) * (dim / 2);
        let tok_wg = ((seq_len * dim + 63) / 64) as u32;
        let mlp_wg = ((seq_len * mlp_hidden + 63) / 64) as u32;
        let rope_wg = ((n_pairs + 63) / 64) as u32;
        let attn_q_wg = ((seq_len + 63) / 64) as u32;
        let attn_h_wg = model.num_heads as u32;

        for blk in &self.blocks {
            self.encode_ln(&mut enc, &self.x_buf, &blk.norm1, &self.h_buf, &self.u_ln_dim, seq_len);
            self.encode_w4a8(&mut enc, &self.h_buf, &blk.q_proj, &self.q_buf, &self.u_quant_dim, &self.u_gemm_qkv, seq_len);
            self.encode_w4a8(&mut enc, &self.h_buf, &blk.k_proj, &self.k_buf, &self.u_quant_dim, &self.u_gemm_k, seq_len);
            self.encode_w4a8(&mut enc, &self.h_buf, &blk.v_proj, &self.v_buf, &self.u_quant_dim, &self.u_gemm_qkv, seq_len);

            {
                let mut pass = enc.begin_compute_pass(&Default::default());
                pass.set_pipeline(&self.rope_pipe);
                pass.set_bind_group(0, &rope_bg, &[]);
                pass.dispatch_workgroups(rope_wg, 1, 1);
            }
            {
                let mut pass = enc.begin_compute_pass(&Default::default());
                pass.set_pipeline(&self.attn_pipe);
                pass.set_bind_group(0, &attn_bg, &[]);
                pass.dispatch_workgroups(attn_q_wg, attn_h_wg, 1);
            }

            self.encode_w4a8(&mut enc, &self.attn_out_buf, &blk.proj, &self.delta_buf, &self.u_quant_dim, &self.u_gemm_qkv, seq_len);
            {
                let mut pass = enc.begin_compute_pass(&Default::default());
                pass.set_pipeline(&self.add_pipe);
                pass.set_bind_group(0, &add_bg, &[]);
                pass.dispatch_workgroups(tok_wg, 1, 1);
            }

            // SwiGLU MLP
            self.encode_ln(&mut enc, &self.x_buf, &blk.norm2, &self.h_buf, &self.u_ln_dim, seq_len);
            self.encode_w4a8(&mut enc, &self.h_buf, &blk.fc1_g, &self.fg_buf, &self.u_quant_dim, &self.u_gemm_fc1, seq_len);
            self.encode_w4a8(&mut enc, &self.h_buf, &blk.fc1_x, &self.fx_buf, &self.u_quant_dim, &self.u_gemm_fc1, seq_len);
            {
                let mut pass = enc.begin_compute_pass(&Default::default());
                pass.set_pipeline(&self.swiglu_pipe);
                pass.set_bind_group(0, &sw_bg, &[]);
                pass.dispatch_workgroups(mlp_wg, 1, 1);
            }
            self.encode_ln(&mut enc, &self.delta_buf, &blk.mlp_norm, &self.fx_buf, &self.u_ln_mlp, seq_len);
            self.encode_w4a8(&mut enc, &self.fx_buf, &blk.fc2, &self.delta_buf, &self.u_quant_mlp, &self.u_gemm_fc2, seq_len);
            {
                let mut pass = enc.begin_compute_pass(&Default::default());
                pass.set_pipeline(&self.add_pipe);
                pass.set_bind_group(0, &add_bg, &[]);
                pass.dispatch_workgroups(tok_wg, 1, 1);
            }
        }

        // 3. Final Backbone LayerNorm + readback
        self.encode_ln(&mut enc, &self.x_buf, &self.norm_final, &self.h_buf, &self.u_ln_dim, seq_len);
        enc.copy_buffer_to_buffer(&self.h_buf, 0, &self.readback_buf, 0, (seq_len * dim * 4) as u64);
        self.queue.submit(Some(enc.finish()));
        let x_final = self.readback_from_submitted(seq_len * dim);

        // 4. 2-Level 18-Query Tree Pool (tiny <0.5 ms on CPU)
        let pooled = forward_pool(&x_final, seq_len, &model.pool, dim);

        // 5. 1152 -> 98,782 Classifier Head on GPU (W4A8 DP4A)
        self.queue.write_buffer(&self.pool_in_buf, 0, bytemuck::cast_slice(&pooled));
        let mut head_enc = self.device.create_command_encoder(&Default::default());
        self.encode_w4a8(&mut head_enc, &self.pool_in_buf, &self.head, &self.head_out_buf, &self.u_quant_head, &self.u_gemm_head, 1);
        head_enc.copy_buffer_to_buffer(
            &self.head_out_buf,
            0,
            &self.readback_buf,
            0,
            (model.head.out_features * 4) as u64,
        );
        self.queue.submit(Some(head_enc.finish()));
        let logits = self.readback_from_submitted(model.head.out_features);

        logits
            .into_iter()
            .map(|l| 1.0f32 / (1.0f32 + (-l).exp()))
            .collect()
    }
}
