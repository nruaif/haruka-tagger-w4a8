use crate::cpu_backend::forward_pool_from_kv;
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
// Uses vec2<u32> activation loads, 2-op (wp << 4) signed 4-bit unpack with pre-divided (scales / 16.0).
const W4A8_DP4A_WGSL: &str = r#"
struct Params { m: u32, n: u32, k_pack: u32, num_w_groups: u32, has_bias: u32, num_act_groups: u32, pad1: u32, pad2: u32 };
@group(0) @binding(0) var<storage, read> x_q_packed: array<vec2<u32>>;
@group(0) @binding(1) var<storage, read> x_scales: array<f32>;
@group(0) @binding(2) var<storage, read> w_buf: array<u32>;
@group(0) @binding(3) var<storage, read> w_scales: array<f32>;
@group(0) @binding(4) var<storage, read> bias_buf: array<f32>;
@group(0) @binding(5) var<storage, read_write> out_buf: array<f32>;
@group(0) @binding(6) var<uniform> params: Params;

var<workgroup> wg_x: array<vec2<u32>, 64>; // 8 rows x 8 vec2<u32> (64 int8 values = 1 weight group)
var<workgroup> wg_xs: array<f32, 8>;       // 1 activation scale per row for current group

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
    let n = gid.x;
    let m_base = gid.y * 8u;
    let safe_n = min(n, params.n - 1u);

    var acc0: f32 = 0.0; var acc1: f32 = 0.0; var acc2: f32 = 0.0; var acc3: f32 = 0.0;
    var acc4: f32 = 0.0; var acc5: f32 = 0.0; var acc6: f32 = 0.0; var acc7: f32 = 0.0;

    var grp: u32 = 0u;
    for (var kt: u32 = 0u; kt < params.k_pack; kt = kt + 8u) {
        let t0 = lid.x;
        let valid_j = min(8u, params.k_pack - kt);
        let ag = grp >> 1u;
        if (params.m == 1u) {
            if (t0 < valid_j) {
                wg_x[t0] = x_q_packed[kt + t0];
            }
            if (t0 == 0u) {
                wg_xs[0] = x_scales[ag];
            }
        } else {
            let r0 = min(m_base + (t0 >> 3u), params.m - 1u);
            let c_v = t0 & 7u;
            wg_x[t0] = x_q_packed[r0 * params.k_pack + kt + c_v];
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
                let lo_i8 = (wp & 0x0F0F0F0Fu) << 4u;
                let hi_i8 = wp & 0xF0F0F0F0u;
                let x0 = wg_x[j];
                g0 += dot4I8Packed(x0.x, lo_i8) + dot4I8Packed(x0.y, hi_i8);
            }
            acc0 += f32(g0) * (ws * wg_xs[0]);
        } else {
            var g0: i32 = 0; var g1: i32 = 0; var g2: i32 = 0; var g3: i32 = 0;
            var g4: i32 = 0; var g5: i32 = 0; var g6: i32 = 0; var g7: i32 = 0;
            for (var j: u32 = 0u; j < valid_j; j = j + 1u) {
                let wp = w_buf[(kt + j) * params.n + safe_n];
                let lo_i8 = (wp & 0x0F0F0F0Fu) << 4u;
                let hi_i8 = wp & 0xF0F0F0F0u;
                let x0 = wg_x[0u * 8u + j]; let x1 = wg_x[1u * 8u + j];
                let x2 = wg_x[2u * 8u + j]; let x3 = wg_x[3u * 8u + j];
                let x4 = wg_x[4u * 8u + j]; let x5 = wg_x[5u * 8u + j];
                let x6 = wg_x[6u * 8u + j]; let x7 = wg_x[7u * 8u + j];
                g0 += dot4I8Packed(x0.x, lo_i8) + dot4I8Packed(x0.y, hi_i8);
                g1 += dot4I8Packed(x1.x, lo_i8) + dot4I8Packed(x1.y, hi_i8);
                g2 += dot4I8Packed(x2.x, lo_i8) + dot4I8Packed(x2.y, hi_i8);
                g3 += dot4I8Packed(x3.x, lo_i8) + dot4I8Packed(x3.y, hi_i8);
                g4 += dot4I8Packed(x4.x, lo_i8) + dot4I8Packed(x4.y, hi_i8);
                g5 += dot4I8Packed(x5.x, lo_i8) + dot4I8Packed(x5.y, hi_i8);
                g6 += dot4I8Packed(x6.x, lo_i8) + dot4I8Packed(x6.y, hi_i8);
                g7 += dot4I8Packed(x7.x, lo_i8) + dot4I8Packed(x7.y, hi_i8);
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

const ATTN_QK_WGSL: &str = r#"
struct AttnParams { seq_len: u32, dim: u32, num_heads: u32, head_dim: u32, scale: f32, pad0: u32, pad1: u32, pad2: u32 };
@group(0) @binding(0) var<storage, read> q_buf: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read> k_buf: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read_write> scores_buf: array<f32>;
@group(0) @binding(3) var<uniform> p: AttnParams;

var<workgroup> sm_q: array<vec4<f32>, 256>;
var<workgroup> sm_k: array<vec4<f32>, 256>;

// Tiled Q @ K^T * scale: BM=64, BN=64, BK=16 (4 outer steps for head_dim=64), 4x4 block/thread
@compute @workgroup_size(16, 16)
fn main(@builtin(workgroup_id) wgid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
    let tx = lid.x;
    let ty = lid.y;
    let tid = ty * 16u + tx;
    let m_base = wgid.y * 64u;
    let n_base = wgid.x * 64u;
    let h = wgid.z;
    let dim_v4 = p.dim / 4u;
    let h_v4 = h * 16u;

    var c0 = vec4<f32>(0.0);
    var c1 = vec4<f32>(0.0);
    var c2 = vec4<f32>(0.0);
    var c3 = vec4<f32>(0.0);

    for (var kt: u32 = 0u; kt < 4u; kt = kt + 1u) {
        let row_l = tid >> 2u;
        let kv = tid & 3u;
        let mq = m_base + row_l;
        let nk = n_base + row_l;
        let k_off = h_v4 + kt * 4u + kv;
        sm_q[tid] = select(vec4<f32>(0.0), q_buf[min(mq, p.seq_len - 1u) * dim_v4 + k_off], mq < p.seq_len);
        sm_k[tid] = select(vec4<f32>(0.0), k_buf[min(nk, p.seq_len - 1u) * dim_v4 + k_off], nk < p.seq_len);
        workgroupBarrier();

        for (var kk: u32 = 0u; kk < 4u; kk = kk + 1u) {
            let q0 = sm_q[(ty * 4u + 0u) * 4u + kk];
            let q1 = sm_q[(ty * 4u + 1u) * 4u + kk];
            let q2 = sm_q[(ty * 4u + 2u) * 4u + kk];
            let q3 = sm_q[(ty * 4u + 3u) * 4u + kk];
            let k0 = sm_k[(tx * 4u + 0u) * 4u + kk];
            let k1 = sm_k[(tx * 4u + 1u) * 4u + kk];
            let k2 = sm_k[(tx * 4u + 2u) * 4u + kk];
            let k3 = sm_k[(tx * 4u + 3u) * 4u + kk];
            c0 += vec4<f32>(dot(q0, k0), dot(q0, k1), dot(q0, k2), dot(q0, k3));
            c1 += vec4<f32>(dot(q1, k0), dot(q1, k1), dot(q1, k2), dot(q1, k3));
            c2 += vec4<f32>(dot(q2, k0), dot(q2, k1), dot(q2, k2), dot(q2, k3));
            c3 += vec4<f32>(dot(q3, k0), dot(q3, k1), dot(q3, k2), dot(q3, k3));
        }
        workgroupBarrier();
    }

    let head_off = h * p.seq_len * p.seq_len;
    let col0 = n_base + tx * 4u;
    let r0 = m_base + ty * 4u;

    if (r0 + 0u < p.seq_len) {
        let off = head_off + (r0 + 0u) * p.seq_len + col0;
        let v = c0 * p.scale;
        if (col0 + 0u < p.seq_len) { scores_buf[off + 0u] = v.x; }
        if (col0 + 1u < p.seq_len) { scores_buf[off + 1u] = v.y; }
        if (col0 + 2u < p.seq_len) { scores_buf[off + 2u] = v.z; }
        if (col0 + 3u < p.seq_len) { scores_buf[off + 3u] = v.w; }
    }
    if (r0 + 1u < p.seq_len) {
        let off = head_off + (r0 + 1u) * p.seq_len + col0;
        let v = c1 * p.scale;
        if (col0 + 0u < p.seq_len) { scores_buf[off + 0u] = v.x; }
        if (col0 + 1u < p.seq_len) { scores_buf[off + 1u] = v.y; }
        if (col0 + 2u < p.seq_len) { scores_buf[off + 2u] = v.z; }
        if (col0 + 3u < p.seq_len) { scores_buf[off + 3u] = v.w; }
    }
    if (r0 + 2u < p.seq_len) {
        let off = head_off + (r0 + 2u) * p.seq_len + col0;
        let v = c2 * p.scale;
        if (col0 + 0u < p.seq_len) { scores_buf[off + 0u] = v.x; }
        if (col0 + 1u < p.seq_len) { scores_buf[off + 1u] = v.y; }
        if (col0 + 2u < p.seq_len) { scores_buf[off + 2u] = v.z; }
        if (col0 + 3u < p.seq_len) { scores_buf[off + 3u] = v.w; }
    }
    if (r0 + 3u < p.seq_len) {
        let off = head_off + (r0 + 3u) * p.seq_len + col0;
        let v = c3 * p.scale;
        if (col0 + 0u < p.seq_len) { scores_buf[off + 0u] = v.x; }
        if (col0 + 1u < p.seq_len) { scores_buf[off + 1u] = v.y; }
        if (col0 + 2u < p.seq_len) { scores_buf[off + 2u] = v.z; }
        if (col0 + 3u < p.seq_len) { scores_buf[off + 3u] = v.w; }
    }
}
"#;

const ATTN_SOFTMAX_WGSL: &str = r#"
struct AttnParams { seq_len: u32, dim: u32, num_heads: u32, head_dim: u32, scale: f32, pad0: u32, pad1: u32, pad2: u32 };
@group(0) @binding(0) var<storage, read_write> scores_buf: array<f32>;
@group(0) @binding(1) var<uniform> p: AttnParams;

var<workgroup> row_buf: array<f32, 1088>;
var<workgroup> sm_red: array<f32, 64>;

@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wgid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
    let qi = wgid.x;
    let h = wgid.y;
    let tid = lid.x;
    let row_off = (h * p.seq_len + qi) * p.seq_len;

    var mx: f32 = -1e30;
    for (var i: u32 = tid; i < p.seq_len; i = i + 64u) {
        let v = scores_buf[row_off + i];
        row_buf[i] = v;
        mx = max(mx, v);
    }
    sm_red[tid] = mx;
    workgroupBarrier();
    if (tid < 32u) { sm_red[tid] = max(sm_red[tid], sm_red[tid + 32u]); }
    workgroupBarrier();
    if (tid < 16u) { sm_red[tid] = max(sm_red[tid], sm_red[tid + 16u]); }
    workgroupBarrier();
    if (tid < 8u)  { sm_red[tid] = max(sm_red[tid], sm_red[tid + 8u]); }
    workgroupBarrier();
    if (tid < 4u)  { sm_red[tid] = max(sm_red[tid], sm_red[tid + 4u]); }
    workgroupBarrier();
    if (tid < 2u)  { sm_red[tid] = max(sm_red[tid], sm_red[tid + 2u]); }
    workgroupBarrier();
    if (tid == 0u) { sm_red[0] = max(sm_red[0], sm_red[1]); }
    workgroupBarrier();
    let row_max = sm_red[0];

    var sum_e: f32 = 0.0;
    for (var i: u32 = tid; i < p.seq_len; i = i + 64u) {
        let e = exp(row_buf[i] - row_max);
        row_buf[i] = e;
        sum_e += e;
    }
    sm_red[tid] = sum_e;
    workgroupBarrier();
    if (tid < 32u) { sm_red[tid] += sm_red[tid + 32u]; }
    workgroupBarrier();
    if (tid < 16u) { sm_red[tid] += sm_red[tid + 16u]; }
    workgroupBarrier();
    if (tid < 8u)  { sm_red[tid] += sm_red[tid + 8u]; }
    workgroupBarrier();
    if (tid < 4u)  { sm_red[tid] += sm_red[tid + 4u]; }
    workgroupBarrier();
    if (tid < 2u)  { sm_red[tid] += sm_red[tid + 2u]; }
    workgroupBarrier();
    if (tid == 0u) { sm_red[0] += sm_red[1]; }
    workgroupBarrier();
    let inv_sum = 1.0 / sm_red[0];

    for (var i: u32 = tid; i < p.seq_len; i = i + 64u) {
        scores_buf[row_off + i] = row_buf[i] * inv_sum;
    }
}
"#;

const ATTN_PV_WGSL: &str = r#"
struct AttnParams { seq_len: u32, dim: u32, num_heads: u32, head_dim: u32, scale: f32, pad0: u32, pad1: u32, pad2: u32 };
@group(0) @binding(0) var<storage, read> scores_buf: array<f32>;
@group(0) @binding(1) var<storage, read> v_buf: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read_write> out_buf: array<vec4<f32>>;
@group(0) @binding(3) var<uniform> p: AttnParams;

var<workgroup> sm_p: array<f32, 256>;
var<workgroup> sm_v: array<vec4<f32>, 256>;

@compute @workgroup_size(16, 16)
fn main(@builtin(workgroup_id) wgid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
    let qi = wgid.x * 16u + lid.y;
    let h = wgid.y;
    let valid_q = qi < p.seq_len;
    let dim_v4 = p.dim / 4u;
    let h_v4 = h * 16u;
    let p_row_off = (h * p.seq_len + min(qi, p.seq_len - 1u)) * p.seq_len;

    var acc_v = vec4<f32>(0.0);
    for (var k_base: u32 = 0u; k_base < p.seq_len; k_base = k_base + 16u) {
        let gk_col = k_base + lid.x;
        let gk_row = k_base + lid.y;
        sm_p[lid.y * 16u + lid.x] = select(0.0, scores_buf[p_row_off + min(gk_col, p.seq_len - 1u)], valid_q && gk_col < p.seq_len);
        sm_v[lid.y * 16u + lid.x] = select(vec4<f32>(0.0), v_buf[min(gk_row, p.seq_len - 1u) * dim_v4 + h_v4 + lid.x], gk_row < p.seq_len);
        workgroupBarrier();

        let tile_k = min(16u, p.seq_len - k_base);
        for (var kk: u32 = 0u; kk < tile_k; kk = kk + 1u) {
            acc_v += sm_p[lid.y * 16u + kk] * sm_v[kk * 16u + lid.x];
        }
        workgroupBarrier();
    }

    if (valid_q) {
        out_buf[qi * dim_v4 + h_v4 + lid.x] = acc_v;
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

var<workgroup> sm_patch: array<f32, 512>; // 8 tokens x 64 floats per K-tile

// 8-token tiled patch embedding: patch_w is pre-transposed [patch_dim, dim]
@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wgid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
    let tok_base = wgid.x * 8u;
    let d = wgid.y * 64u + lid.x;
    let seq_len = 1u + p.grid_size * p.grid_size;
    let pp = p.patch_size * p.patch_size;
    let patch_dim = 3u * pp;
    let hw = p.img_size * p.img_size;

    var acc0: f32 = 0.0; var acc1: f32 = 0.0; var acc2: f32 = 0.0; var acc3: f32 = 0.0;
    var acc4: f32 = 0.0; var acc5: f32 = 0.0; var acc6: f32 = 0.0; var acc7: f32 = 0.0;

    for (var k_base: u32 = 0u; k_base < patch_dim; k_base = k_base + 64u) {
        let bk = min(64u, patch_dim - k_base);
        let k = k_base + lid.x;
        let valid_k = lid.x < bk;
        var c: u32 = 0u;
        var dy: u32 = 0u;
        var dx: u32 = 0u;
        if (valid_k) {
            c = k / pp;
            let rem = k - c * pp;
            dy = rem / p.patch_size;
            dx = rem - dy * p.patch_size;
        }
        for (var r: u32 = 0u; r < 8u; r = r + 1u) {
            let tok = tok_base + r;
            var val: f32 = 0.0;
            if (tok > 0u && tok < seq_len && valid_k) {
                let p_idx = tok - 1u;
                let py = p_idx / p.grid_size;
                let px = p_idx % p.grid_size;
                val = nchw[c * hw + (py * p.patch_size + dy) * p.img_size + (px * p.patch_size + dx)];
            }
            sm_patch[r * 64u + lid.x] = val;
        }
        workgroupBarrier();

        if (d < p.dim) {
            for (var kk: u32 = 0u; kk < bk; kk = kk + 1u) {
                let w = patch_w[(k_base + kk) * p.dim + d];
                acc0 += sm_patch[0u * 64u + kk] * w;
                acc1 += sm_patch[1u * 64u + kk] * w;
                acc2 += sm_patch[2u * 64u + kk] * w;
                acc3 += sm_patch[3u * 64u + kk] * w;
                acc4 += sm_patch[4u * 64u + kk] * w;
                acc5 += sm_patch[5u * 64u + kk] * w;
                acc6 += sm_patch[6u * 64u + kk] * w;
                acc7 += sm_patch[7u * 64u + kk] * w;
            }
        }
        workgroupBarrier();
    }

    if (d < p.dim) {
        let pb = patch_b[d];
        if (tok_base + 0u == 0u) {
            out_buf[d] = cls_tok[d] + pos_embed[d];
        } else if (tok_base + 0u < seq_len) {
            let t = tok_base + 0u;
            out_buf[t * p.dim + d] = acc0 + pb + pos_embed[t * p.dim + d];
        }
        if (tok_base + 1u < seq_len) {
            let t = tok_base + 1u;
            out_buf[t * p.dim + d] = acc1 + pb + pos_embed[t * p.dim + d];
        }
        if (tok_base + 2u < seq_len) {
            let t = tok_base + 2u;
            out_buf[t * p.dim + d] = acc2 + pb + pos_embed[t * p.dim + d];
        }
        if (tok_base + 3u < seq_len) {
            let t = tok_base + 3u;
            out_buf[t * p.dim + d] = acc3 + pb + pos_embed[t * p.dim + d];
        }
        if (tok_base + 4u < seq_len) {
            let t = tok_base + 4u;
            out_buf[t * p.dim + d] = acc4 + pb + pos_embed[t * p.dim + d];
        }
        if (tok_base + 5u < seq_len) {
            let t = tok_base + 5u;
            out_buf[t * p.dim + d] = acc5 + pb + pos_embed[t * p.dim + d];
        }
        if (tok_base + 6u < seq_len) {
            let t = tok_base + 6u;
            out_buf[t * p.dim + d] = acc6 + pb + pos_embed[t * p.dim + d];
        }
        if (tok_base + 7u < seq_len) {
            let t = tok_base + 7u;
            out_buf[t * p.dim + d] = acc7 + pb + pos_embed[t * p.dim + d];
        }
    }
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

#[allow(dead_code)]
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

    // Cached BindGroups
    norm1_bg: wgpu::BindGroup,
    q_qbg: wgpu::BindGroup,
    q_gbg: wgpu::BindGroup,
    k_qbg: wgpu::BindGroup,
    k_gbg: wgpu::BindGroup,
    v_qbg: wgpu::BindGroup,
    v_gbg: wgpu::BindGroup,
    proj_qbg: wgpu::BindGroup,
    proj_gbg: wgpu::BindGroup,
    norm2_bg: wgpu::BindGroup,
    fc1g_qbg: wgpu::BindGroup,
    fc1g_gbg: wgpu::BindGroup,
    fc1x_qbg: wgpu::BindGroup,
    fc1x_gbg: wgpu::BindGroup,
    mlp_norm_bg: wgpu::BindGroup,
    fc2_qbg: wgpu::BindGroup,
    fc2_gbg: wgpu::BindGroup,
}

#[allow(dead_code)]
pub struct WgpuRunner {
    pub adapter_name: String,
    device: wgpu::Device,
    queue: wgpu::Queue,
    quant_pipe: wgpu::ComputePipeline,
    w4a8_pipe: wgpu::ComputePipeline,
    ln_pipe: wgpu::ComputePipeline,
    rope_pipe: wgpu::ComputePipeline,
    attn_qk_pipe: wgpu::ComputePipeline,
    attn_sm_pipe: wgpu::ComputePipeline,
    attn_pv_pipe: wgpu::ComputePipeline,
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
    pool_norm: GpuLayerNorm,
    pool_kv: GpuW4Linear,
    head: GpuW4Linear,

    // Reusable scratch buffers
    nchw_buf: wgpu::Buffer,
    x_buf: wgpu::Buffer,
    h_buf: wgpu::Buffer,
    q_buf: wgpu::Buffer,
    k_buf: wgpu::Buffer,
    v_buf: wgpu::Buffer,
    attn_scores_buf: wgpu::Buffer,
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
    u_gemm_pool_kv: wgpu::Buffer,
    u_gemm_head: wgpu::Buffer,

    // Cached top-level BindGroups
    patch_bg: wgpu::BindGroup,
    rope_bg: wgpu::BindGroup,
    attn_qk_bg: wgpu::BindGroup,
    attn_sm_bg: wgpu::BindGroup,
    attn_pv_bg: wgpu::BindGroup,
    add_bg: wgpu::BindGroup,
    sw_bg: wgpu::BindGroup,
    norm_final_bg: wgpu::BindGroup,
    pool_norm_bg: wgpu::BindGroup,
    pool_kv_qbg: wgpu::BindGroup,
    pool_kv_gbg: wgpu::BindGroup,
    head_qbg: wgpu::BindGroup,
    head_gbg: wgpu::BindGroup,
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
            // Divide by 16.0 to match the 2-op (wp << 4) signed 4-bit -> i8 scale
            scales_t[g * n_out + n] = lin.scales[n * ng + g] / 16.0f32;
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
        let attn_qk_pipe = mk_pipe("attn_qk", ATTN_QK_WGSL);
        let attn_sm_pipe = mk_pipe("attn_sm", ATTN_SOFTMAX_WGSL);
        let attn_pv_pipe = mk_pipe("attn_pv", ATTN_PV_WGSL);
        let swiglu_pipe = mk_pipe("swiglu", SWIGLU_WGSL);
        let add_pipe = mk_pipe("add_res", ADD_RESIDUAL_WGSL);
        let patch_pipe = mk_pipe("patch_embed", PATCH_EMBED_WGSL);

        if let Some(err) = pollster::block_on(err_scope.pop()) {
            anyhow::bail!("wgpu shader validation failed (DP4A may not be supported on this GPU): {err}");
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
        let kv_bytes = seq_len * (2 * dim) * 4;
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
        let gp_pool_kv = mk_gemm_p(seq_len, &model.pool.kv);
        let gp_head = mk_gemm_p(1, &model.head);

        let patch_dim = 3 * model.patch_size * model.patch_size;
        let mut patch_w_t = vec![0.0f32; patch_dim * dim];
        for d in 0..dim {
            for k in 0..patch_dim {
                patch_w_t[k * dim + d] = model.patch_w[d * patch_dim + k];
            }
        }
        let patch_w_buf = upload_f32(&device, "patch_w", &patch_w_t);
        let patch_b_buf = upload_f32(&device, "patch_b", &model.patch_b);
        let cls_buf = upload_f32(&device, "cls", &model.cls_token);
        let pos_buf = upload_f32(&device, "pos", &model.pos_embed);
        let rope_sin_buf = upload_f32(&device, "rope_sin", &model.rope_sin);
        let rope_cos_buf = upload_f32(&device, "rope_cos", &model.rope_cos);
        let norm_final = upload_ln(&device, &model.norm);
        let pool_norm = upload_ln(&device, &model.pool.norm);
        let pool_kv = upload_w4(&device, "pool.kv", &model.pool.kv);
        let head = upload_w4(&device, "head", &model.head);

        let nchw_buf = mk_rw("nchw", nchw_bytes);
        let x_buf = mk_rw("x", tok_bytes);
        let h_buf = mk_rw("h", tok_bytes);
        let q_buf = mk_rw("q", tok_bytes);
        let k_buf = mk_rw("k", tok_bytes);
        let v_buf = mk_rw("v", tok_bytes);
        let attn_scores_buf = mk_rw("attn_scores", model.num_heads * seq_len * seq_len * 4);
        let attn_out_buf = mk_rw("attn_out", tok_bytes);
        let fg_buf = mk_rw("fg", mlp_bytes);
        let fx_buf = mk_rw("fx", mlp_bytes);
        let delta_buf = mk_rw("delta", mlp_bytes.max(kv_bytes));
        let x_q_buf = mk_rw("x_q", seq_len * max_k);
        let x_sc_buf = mk_rw("x_sc", seq_len * (max_k / 128 + 1) * 4);
        let pool_in_buf = mk_rw("pool_in", model.head.in_features * 4);
        let head_out_buf = mk_rw("head_out", head_bytes);
        let readback_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("readback"),
            size: tok_bytes.max(kv_bytes).max(head_bytes) as u64,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let u_patch = mk_uni_init("u_patch", bytemuck::bytes_of(&pp));
        let u_attn = mk_uni_init("u_attn", bytemuck::bytes_of(&ap));
        let u_ln_dim = mk_uni_init("u_ln_dim", bytemuck::bytes_of(&ln_dim));
        let u_ln_mlp = mk_uni_init("u_ln_mlp", bytemuck::bytes_of(&ln_mlp));
        let u_elem_tok = mk_uni_init("u_elem_tok", bytemuck::bytes_of(&elem_tok));
        let u_elem_mlp = mk_uni_init("u_elem_mlp", bytemuck::bytes_of(&elem_mlp));
        let u_quant_dim = mk_uni_init("u_quant_dim", bytemuck::bytes_of(&qp_dim));
        let u_quant_mlp = mk_uni_init("u_quant_mlp", bytemuck::bytes_of(&qp_mlp));
        let u_quant_head = mk_uni_init("u_quant_head", bytemuck::bytes_of(&qp_head));
        let u_gemm_qkv = mk_uni_init("u_gemm_qkv", bytemuck::bytes_of(&gp_qkv));
        let u_gemm_k = mk_uni_init("u_gemm_k", bytemuck::bytes_of(&gp_k));
        let u_gemm_fc1 = mk_uni_init("u_gemm_fc1", bytemuck::bytes_of(&gp_fc1));
        let u_gemm_fc2 = mk_uni_init("u_gemm_fc2", bytemuck::bytes_of(&gp_fc2));
        let u_gemm_pool_kv = mk_uni_init("u_gemm_pool_kv", bytemuck::bytes_of(&gp_pool_kv));
        let u_gemm_head = mk_uni_init("u_gemm_head", bytemuck::bytes_of(&gp_head));

        let ln_layout = ln_pipe.get_bind_group_layout(0);
        let q_layout = quant_pipe.get_bind_group_layout(0);
        let g_layout = w4a8_pipe.get_bind_group_layout(0);

        let mk_ln_bg = |x_b: &wgpu::Buffer, ln: &GpuLayerNorm, out_b: &wgpu::Buffer, u_ln: &wgpu::Buffer| {
            device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: None,
                layout: &ln_layout,
                entries: &[
                    wgpu::BindGroupEntry { binding: 0, resource: x_b.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 1, resource: ln.w_buf.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 2, resource: ln.b_buf.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 3, resource: out_b.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 4, resource: u_ln.as_entire_binding() },
                ],
            })
        };
        let mk_quant_bg = |x_b: &wgpu::Buffer, lin: &GpuW4Linear, u_q: &wgpu::Buffer| {
            device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: None,
                layout: &q_layout,
                entries: &[
                    wgpu::BindGroupEntry { binding: 0, resource: x_b.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 1, resource: lin.awq_scale_buf.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 2, resource: x_q_buf.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 3, resource: x_sc_buf.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 4, resource: u_q.as_entire_binding() },
                ],
            })
        };
        let mk_gemm_bg = |lin: &GpuW4Linear, out_b: &wgpu::Buffer, u_g: &wgpu::Buffer| {
            device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: None,
                layout: &g_layout,
                entries: &[
                    wgpu::BindGroupEntry { binding: 0, resource: x_q_buf.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 1, resource: x_sc_buf.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 2, resource: lin.w_buf.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 3, resource: lin.w_scales_buf.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 4, resource: lin.bias_buf.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 5, resource: out_b.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 6, resource: u_g.as_entire_binding() },
                ],
            })
        };

        let mut blocks = Vec::with_capacity(model.blocks.len());
        for (i, b) in model.blocks.iter().enumerate() {
            let norm1 = upload_ln(&device, &b.norm1);
            let q_proj = upload_w4(&device, &format!("b{i}.q"), &b.q_proj);
            let k_proj = upload_w4(&device, &format!("b{i}.k"), &b.k_proj);
            let v_proj = upload_w4(&device, &format!("b{i}.v"), &b.v_proj);
            let proj = upload_w4(&device, &format!("b{i}.proj"), &b.proj);
            let norm2 = upload_ln(&device, &b.norm2);
            let fc1_g = upload_w4(&device, &format!("b{i}.fc1_g"), &b.fc1_g);
            let fc1_x = upload_w4(&device, &format!("b{i}.fc1_x"), &b.fc1_x);
            let mlp_norm = upload_ln(&device, &b.mlp_norm);
            let fc2 = upload_w4(&device, &format!("b{i}.fc2"), &b.fc2);

            let norm1_bg = mk_ln_bg(&x_buf, &norm1, &h_buf, &u_ln_dim);
            let q_qbg = mk_quant_bg(&h_buf, &q_proj, &u_quant_dim);
            let q_gbg = mk_gemm_bg(&q_proj, &q_buf, &u_gemm_qkv);
            let k_qbg = mk_quant_bg(&h_buf, &k_proj, &u_quant_dim);
            let k_gbg = mk_gemm_bg(&k_proj, &k_buf, &u_gemm_k);
            let v_qbg = mk_quant_bg(&h_buf, &v_proj, &u_quant_dim);
            let v_gbg = mk_gemm_bg(&v_proj, &v_buf, &u_gemm_qkv);
            let proj_qbg = mk_quant_bg(&attn_out_buf, &proj, &u_quant_dim);
            let proj_gbg = mk_gemm_bg(&proj, &delta_buf, &u_gemm_qkv);
            let norm2_bg = mk_ln_bg(&x_buf, &norm2, &h_buf, &u_ln_dim);
            let fc1g_qbg = mk_quant_bg(&h_buf, &fc1_g, &u_quant_dim);
            let fc1g_gbg = mk_gemm_bg(&fc1_g, &fg_buf, &u_gemm_fc1);
            let fc1x_qbg = mk_quant_bg(&h_buf, &fc1_x, &u_quant_dim);
            let fc1x_gbg = mk_gemm_bg(&fc1_x, &fx_buf, &u_gemm_fc1);
            let mlp_norm_bg = mk_ln_bg(&delta_buf, &mlp_norm, &fx_buf, &u_ln_mlp);
            let fc2_qbg = mk_quant_bg(&fx_buf, &fc2, &u_quant_mlp);
            let fc2_gbg = mk_gemm_bg(&fc2, &delta_buf, &u_gemm_fc2);

            blocks.push(GpuBlock {
                norm1,
                q_proj,
                k_proj,
                v_proj,
                proj,
                norm2,
                fc1_g,
                fc1_x,
                mlp_norm,
                fc2,
                norm1_bg,
                q_qbg,
                q_gbg,
                k_qbg,
                k_gbg,
                v_qbg,
                v_gbg,
                proj_qbg,
                proj_gbg,
                norm2_bg,
                fc1g_qbg,
                fc1g_gbg,
                fc1x_qbg,
                fc1x_gbg,
                mlp_norm_bg,
                fc2_qbg,
                fc2_gbg,
            });
        }

        let patch_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &patch_pipe.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: nchw_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: patch_w_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: patch_b_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: cls_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 4, resource: pos_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 5, resource: x_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 6, resource: u_patch.as_entire_binding() },
            ],
        });
        let rope_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &rope_pipe.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: q_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: k_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: rope_sin_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: rope_cos_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 4, resource: u_attn.as_entire_binding() },
            ],
        });
        let attn_qk_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &attn_qk_pipe.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: q_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: k_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: attn_scores_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: u_attn.as_entire_binding() },
            ],
        });
        let attn_sm_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &attn_sm_pipe.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: attn_scores_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: u_attn.as_entire_binding() },
            ],
        });
        let attn_pv_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &attn_pv_pipe.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: attn_scores_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: v_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: attn_out_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: u_attn.as_entire_binding() },
            ],
        });
        let add_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &add_pipe.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: x_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: delta_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: u_elem_tok.as_entire_binding() },
            ],
        });
        let sw_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &swiglu_pipe.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: fg_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: fx_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: delta_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: u_elem_mlp.as_entire_binding() },
            ],
        });
        let norm_final_bg = mk_ln_bg(&x_buf, &norm_final, &h_buf, &u_ln_dim);
        let pool_norm_bg = mk_ln_bg(&h_buf, &pool_norm, &x_buf, &u_ln_dim);
        let pool_kv_qbg = mk_quant_bg(&x_buf, &pool_kv, &u_quant_dim);
        let pool_kv_gbg = mk_gemm_bg(&pool_kv, &delta_buf, &u_gemm_pool_kv);
        let head_qbg = mk_quant_bg(&pool_in_buf, &head, &u_quant_head);
        let head_gbg = mk_gemm_bg(&head, &head_out_buf, &u_gemm_head);

        Ok(Self {
            adapter_name,
            patch_w_buf,
            patch_b_buf,
            cls_buf,
            pos_buf,
            rope_sin_buf,
            rope_cos_buf,
            blocks,
            norm_final,
            pool_norm,
            pool_kv,
            head,

            nchw_buf,
            x_buf,
            h_buf,
            q_buf,
            k_buf,
            v_buf,
            attn_scores_buf,
            attn_out_buf,
            fg_buf,
            fx_buf,
            delta_buf,
            x_q_buf,
            x_sc_buf,
            pool_in_buf,
            head_out_buf,
            readback_buf,

            u_patch,
            u_attn,
            u_ln_dim,
            u_ln_mlp,
            u_elem_tok,
            u_elem_mlp,
            u_quant_dim,
            u_quant_mlp,
            u_quant_head,
            u_gemm_qkv,
            u_gemm_k,
            u_gemm_fc1,
            u_gemm_fc2,
            u_gemm_pool_kv,
            u_gemm_head,

            patch_bg,
            rope_bg,
            attn_qk_bg,
            attn_sm_bg,
            attn_pv_bg,
            add_bg,
            sw_bg,
            norm_final_bg,
            pool_norm_bg,
            pool_kv_qbg,
            pool_kv_gbg,
            head_qbg,
            head_gbg,

            device,
            queue,
            quant_pipe,
            w4a8_pipe,
            ln_pipe,
            rope_pipe,
            attn_qk_pipe,
            attn_sm_pipe,
            attn_pv_pipe,
            swiglu_pipe,
            add_pipe,
            patch_pipe,
        })
    }

    fn encode_ln(&self, enc: &mut wgpu::CommandEncoder, bg: &wgpu::BindGroup, rows: usize) {
        let mut pass = enc.begin_compute_pass(&Default::default());
        pass.set_pipeline(&self.ln_pipe);
        pass.set_bind_group(0, bg, &[]);
        pass.dispatch_workgroups(rows as u32, 1, 1);
    }

    fn encode_quant(
        &self,
        enc: &mut wgpu::CommandEncoder,
        q_bg: &wgpu::BindGroup,
        rows: usize,
        in_features: usize,
    ) {
        let mut pass = enc.begin_compute_pass(&Default::default());
        pass.set_pipeline(&self.quant_pipe);
        pass.set_bind_group(0, q_bg, &[]);
        pass.dispatch_workgroups(rows as u32, (in_features / 128) as u32, 1);
    }

    fn encode_gemm(
        &self,
        enc: &mut wgpu::CommandEncoder,
        g_bg: &wgpu::BindGroup,
        rows: usize,
        out_features: usize,
    ) {
        let mut pass = enc.begin_compute_pass(&Default::default());
        pass.set_pipeline(&self.w4a8_pipe);
        pass.set_bind_group(0, g_bg, &[]);
        pass.dispatch_workgroups(
            ((out_features + 63) / 64) as u32,
            ((rows + 7) / 8) as u32,
            1,
        );
    }

    fn encode_w4a8(
        &self,
        enc: &mut wgpu::CommandEncoder,
        q_bg: &wgpu::BindGroup,
        g_bg: &wgpu::BindGroup,
        lin: &GpuW4Linear,
        rows: usize,
    ) {
        self.encode_quant(enc, q_bg, rows, lin.in_features);
        self.encode_gemm(enc, g_bg, rows, lin.out_features);
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

        // 2. Single CommandEncoder for patch_embed + all 12 transformer blocks + final norm + pool.kv
        let mut enc = self.device.create_command_encoder(&Default::default());
        {
            let mut pass = enc.begin_compute_pass(&Default::default());
            pass.set_pipeline(&self.patch_pipe);
            pass.set_bind_group(0, &self.patch_bg, &[]);
            pass.dispatch_workgroups(((seq_len + 7) / 8) as u32, ((dim + 63) / 64) as u32, 1);
        }

        let n_pairs = (seq_len - 1) * (dim / 2);
        let tok_wg = ((seq_len * dim + 63) / 64) as u32;
        let mlp_wg = ((seq_len * mlp_hidden + 63) / 64) as u32;
        let rope_wg = ((n_pairs + 63) / 64) as u32;
        let qk_wg = ((seq_len + 63) / 64) as u32;
        let pv_wg = ((seq_len + 15) / 16) as u32;
        let nh = model.num_heads as u32;

        for blk in &self.blocks {
            self.encode_ln(&mut enc, &blk.norm1_bg, seq_len);
            self.encode_w4a8(&mut enc, &blk.q_qbg, &blk.q_gbg, &blk.q_proj, seq_len);
            self.encode_w4a8(&mut enc, &blk.k_qbg, &blk.k_gbg, &blk.k_proj, seq_len);
            self.encode_w4a8(&mut enc, &blk.v_qbg, &blk.v_gbg, &blk.v_proj, seq_len);

            {
                let mut pass = enc.begin_compute_pass(&Default::default());
                pass.set_pipeline(&self.rope_pipe);
                pass.set_bind_group(0, &self.rope_bg, &[]);
                pass.dispatch_workgroups(rope_wg, 1, 1);
            }
            {
                let mut pass = enc.begin_compute_pass(&Default::default());
                pass.set_pipeline(&self.attn_qk_pipe);
                pass.set_bind_group(0, &self.attn_qk_bg, &[]);
                pass.dispatch_workgroups(qk_wg, qk_wg, nh);
            }
            {
                let mut pass = enc.begin_compute_pass(&Default::default());
                pass.set_pipeline(&self.attn_sm_pipe);
                pass.set_bind_group(0, &self.attn_sm_bg, &[]);
                pass.dispatch_workgroups(seq_len as u32, nh, 1);
            }
            {
                let mut pass = enc.begin_compute_pass(&Default::default());
                pass.set_pipeline(&self.attn_pv_pipe);
                pass.set_bind_group(0, &self.attn_pv_bg, &[]);
                pass.dispatch_workgroups(pv_wg, nh, 1);
            }

            self.encode_w4a8(&mut enc, &blk.proj_qbg, &blk.proj_gbg, &blk.proj, seq_len);
            {
                let mut pass = enc.begin_compute_pass(&Default::default());
                pass.set_pipeline(&self.add_pipe);
                pass.set_bind_group(0, &self.add_bg, &[]);
                pass.dispatch_workgroups(tok_wg, 1, 1);
            }

            // SwiGLU MLP
            self.encode_ln(&mut enc, &blk.norm2_bg, seq_len);
            self.encode_w4a8(&mut enc, &blk.fc1g_qbg, &blk.fc1g_gbg, &blk.fc1_g, seq_len);
            self.encode_w4a8(&mut enc, &blk.fc1x_qbg, &blk.fc1x_gbg, &blk.fc1_x, seq_len);
            {
                let mut pass = enc.begin_compute_pass(&Default::default());
                pass.set_pipeline(&self.swiglu_pipe);
                pass.set_bind_group(0, &self.sw_bg, &[]);
                pass.dispatch_workgroups(mlp_wg, 1, 1);
            }
            self.encode_ln(&mut enc, &blk.mlp_norm_bg, seq_len);
            self.encode_w4a8(&mut enc, &blk.fc2_qbg, &blk.fc2_gbg, &blk.fc2, seq_len);
            {
                let mut pass = enc.begin_compute_pass(&Default::default());
                pass.set_pipeline(&self.add_pipe);
                pass.set_bind_group(0, &self.add_bg, &[]);
                pass.dispatch_workgroups(tok_wg, 1, 1);
            }
        }

        // 3. Final Backbone LayerNorm + pool.norm + pool.kv on GPU -> readback kv
        self.encode_ln(&mut enc, &self.norm_final_bg, seq_len);
        self.encode_ln(&mut enc, &self.pool_norm_bg, seq_len);
        self.encode_w4a8(&mut enc, &self.pool_kv_qbg, &self.pool_kv_gbg, &self.pool_kv, seq_len);
        enc.copy_buffer_to_buffer(
            &self.delta_buf,
            0,
            &self.readback_buf,
            0,
            (seq_len * 2 * dim * 4) as u64,
        );
        self.queue.submit(Some(enc.finish()));
        let kv = self.readback_from_submitted(seq_len * 2 * dim);

        // 4. 2-Level 18-Query Tree Pool (tiny <1 ms on CPU from precomputed GPU kv)
        let pooled = forward_pool_from_kv(&kv, seq_len, &model.pool, dim);

        // 5. 1152 -> 98,782 Classifier Head on GPU (W4A8 DP4A)
        self.queue.write_buffer(&self.pool_in_buf, 0, bytemuck::cast_slice(&pooled));
        let mut head_enc = self.device.create_command_encoder(&Default::default());
        self.encode_w4a8(&mut head_enc, &self.head_qbg, &self.head_gbg, &self.head, 1);
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

    pub fn profile_stages(&self, model: &HarukaModel, nchw: &[f32]) {
        let seq_len = 1 + model.grid_size * model.grid_size;
        let dim = model.dim;
        let mlp_hidden = model.blocks[0].fc1_g.out_features;
        let n_pairs = (seq_len - 1) * (dim / 2);
        let tok_wg = ((seq_len * dim + 63) / 64) as u32;
        let mlp_wg = ((seq_len * mlp_hidden + 63) / 64) as u32;
        let rope_wg = ((n_pairs + 63) / 64) as u32;
        let qk_wg = ((seq_len + 63) / 64) as u32;
        let pv_wg = ((seq_len + 15) / 16) as u32;
        let nh = model.num_heads as u32;

        // Warmup & populate valid activations in buffers
        let _ = self.infer(model, nchw);

        let sync = || {
            self.device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
        };

        // 1. patch_embed
        let t0 = std::time::Instant::now();
        {
            self.queue.write_buffer(&self.nchw_buf, 0, bytemuck::cast_slice(nchw));
            let mut enc = self.device.create_command_encoder(&Default::default());
            {
                let mut pass = enc.begin_compute_pass(&Default::default());
                pass.set_pipeline(&self.patch_pipe);
                pass.set_bind_group(0, &self.patch_bg, &[]);
                pass.dispatch_workgroups(((seq_len + 7) / 8) as u32, ((dim + 63) / 64) as u32, 1);
            }
            self.queue.submit(Some(enc.finish()));
            sync();
        }
        let dt_patch = t0.elapsed().as_secs_f64() * 1000.0;

        // 2. layernorm (38 passes including norm_final + pool_norm)
        let t0 = std::time::Instant::now();
        {
            let mut enc = self.device.create_command_encoder(&Default::default());
            for blk in &self.blocks {
                self.encode_ln(&mut enc, &blk.norm1_bg, seq_len);
                self.encode_ln(&mut enc, &blk.norm2_bg, seq_len);
                self.encode_ln(&mut enc, &blk.mlp_norm_bg, seq_len);
            }
            self.encode_ln(&mut enc, &self.norm_final_bg, seq_len);
            self.encode_ln(&mut enc, &self.pool_norm_bg, seq_len);
            self.queue.submit(Some(enc.finish()));
            sync();
        }
        let dt_ln = t0.elapsed().as_secs_f64() * 1000.0;

        // 3. quant_a8 (85 passes including pool.kv)
        let t0 = std::time::Instant::now();
        {
            let mut enc = self.device.create_command_encoder(&Default::default());
            for blk in &self.blocks {
                self.encode_quant(&mut enc, &blk.q_qbg, seq_len, blk.q_proj.in_features);
                self.encode_quant(&mut enc, &blk.k_qbg, seq_len, blk.k_proj.in_features);
                self.encode_quant(&mut enc, &blk.v_qbg, seq_len, blk.v_proj.in_features);
                self.encode_quant(&mut enc, &blk.proj_qbg, seq_len, blk.proj.in_features);
                self.encode_quant(&mut enc, &blk.fc1g_qbg, seq_len, blk.fc1_g.in_features);
                self.encode_quant(&mut enc, &blk.fc1x_qbg, seq_len, blk.fc1_x.in_features);
                self.encode_quant(&mut enc, &blk.fc2_qbg, seq_len, blk.fc2.in_features);
            }
            self.encode_quant(&mut enc, &self.pool_kv_qbg, seq_len, self.pool_kv.in_features);
            self.queue.submit(Some(enc.finish()));
            sync();
        }
        let dt_quant = t0.elapsed().as_secs_f64() * 1000.0;

        // 4. w4a8_dp4a (85 GEMMs including pool.kv)
        let t0 = std::time::Instant::now();
        {
            let mut enc = self.device.create_command_encoder(&Default::default());
            for blk in &self.blocks {
                self.encode_gemm(&mut enc, &blk.q_gbg, seq_len, blk.q_proj.out_features);
                self.encode_gemm(&mut enc, &blk.k_gbg, seq_len, blk.k_proj.out_features);
                self.encode_gemm(&mut enc, &blk.v_gbg, seq_len, blk.v_proj.out_features);
                self.encode_gemm(&mut enc, &blk.proj_gbg, seq_len, blk.proj.out_features);
                self.encode_gemm(&mut enc, &blk.fc1g_gbg, seq_len, blk.fc1_g.out_features);
                self.encode_gemm(&mut enc, &blk.fc1x_gbg, seq_len, blk.fc1_x.out_features);
                self.encode_gemm(&mut enc, &blk.fc2_gbg, seq_len, blk.fc2.out_features);
            }
            self.encode_gemm(&mut enc, &self.pool_kv_gbg, seq_len, self.pool_kv.out_features);
            self.queue.submit(Some(enc.finish()));
            sync();
        }
        let dt_gemm = t0.elapsed().as_secs_f64() * 1000.0;

        // 5. rope + attn (12 layers)
        let t0 = std::time::Instant::now();
        {
            let mut enc = self.device.create_command_encoder(&Default::default());
            for _ in &self.blocks {
                {
                    let mut pass = enc.begin_compute_pass(&Default::default());
                    pass.set_pipeline(&self.rope_pipe);
                    pass.set_bind_group(0, &self.rope_bg, &[]);
                    pass.dispatch_workgroups(rope_wg, 1, 1);
                }
                {
                    let mut pass = enc.begin_compute_pass(&Default::default());
                    pass.set_pipeline(&self.attn_qk_pipe);
                    pass.set_bind_group(0, &self.attn_qk_bg, &[]);
                    pass.dispatch_workgroups(qk_wg, qk_wg, nh);
                }
                {
                    let mut pass = enc.begin_compute_pass(&Default::default());
                    pass.set_pipeline(&self.attn_sm_pipe);
                    pass.set_bind_group(0, &self.attn_sm_bg, &[]);
                    pass.dispatch_workgroups(seq_len as u32, nh, 1);
                }
                {
                    let mut pass = enc.begin_compute_pass(&Default::default());
                    pass.set_pipeline(&self.attn_pv_pipe);
                    pass.set_bind_group(0, &self.attn_pv_bg, &[]);
                    pass.dispatch_workgroups(pv_wg, nh, 1);
                }
            }
            self.queue.submit(Some(enc.finish()));
            sync();
        }
        let dt_attn = t0.elapsed().as_secs_f64() * 1000.0;

        // 6. swiglu + add (36 passes)
        let t0 = std::time::Instant::now();
        {
            let mut enc = self.device.create_command_encoder(&Default::default());
            for _ in &self.blocks {
                {
                    let mut pass = enc.begin_compute_pass(&Default::default());
                    pass.set_pipeline(&self.swiglu_pipe);
                    pass.set_bind_group(0, &self.sw_bg, &[]);
                    pass.dispatch_workgroups(mlp_wg, 1, 1);
                }
                {
                    let mut pass = enc.begin_compute_pass(&Default::default());
                    pass.set_pipeline(&self.add_pipe);
                    pass.set_bind_group(0, &self.add_bg, &[]);
                    pass.dispatch_workgroups(tok_wg, 1, 1);
                }
                {
                    let mut pass = enc.begin_compute_pass(&Default::default());
                    pass.set_pipeline(&self.add_pipe);
                    pass.set_bind_group(0, &self.add_bg, &[]);
                    pass.dispatch_workgroups(tok_wg, 1, 1);
                }
            }
            self.queue.submit(Some(enc.finish()));
            sync();
        }
        let dt_elem = t0.elapsed().as_secs_f64() * 1000.0;

        // 7. kv readback + tree pool (CPU) + head (GPU)
        let mut kv_enc = self.device.create_command_encoder(&Default::default());
        kv_enc.copy_buffer_to_buffer(
            &self.delta_buf,
            0,
            &self.readback_buf,
            0,
            (seq_len * 2 * dim * 4) as u64,
        );
        let t0 = std::time::Instant::now();
        self.queue.submit(Some(kv_enc.finish()));
        let kv = self.readback_from_submitted(seq_len * 2 * dim);
        let dt_readback = t0.elapsed().as_secs_f64() * 1000.0;

        let t0 = std::time::Instant::now();
        let pooled = forward_pool_from_kv(&kv, seq_len, &model.pool, dim);
        let dt_pool = t0.elapsed().as_secs_f64() * 1000.0;

        let t0 = std::time::Instant::now();
        self.queue.write_buffer(&self.pool_in_buf, 0, bytemuck::cast_slice(&pooled));
        let mut head_enc = self.device.create_command_encoder(&Default::default());
        self.encode_w4a8(&mut head_enc, &self.head_qbg, &self.head_gbg, &self.head, 1);
        head_enc.copy_buffer_to_buffer(
            &self.head_out_buf,
            0,
            &self.readback_buf,
            0,
            (model.head.out_features * 4) as u64,
        );
        self.queue.submit(Some(head_enc.finish()));
        let _ = self.readback_from_submitted(model.head.out_features);
        let dt_head = t0.elapsed().as_secs_f64() * 1000.0;

        eprintln!(
            "Stage breakdown (ms): patch={dt_patch:.2}, ln={dt_ln:.2}, quant={dt_quant:.2}, dp4a_gemm={dt_gemm:.2}, rope+attn={dt_attn:.2}, swiglu+add={dt_elem:.2}, kv_read={dt_readback:.2}, pool={dt_pool:.2}, head={dt_head:.2} | sum={:.2} ms",
            dt_patch + dt_ln + dt_quant + dt_gemm + dt_attn + dt_elem + dt_readback + dt_pool + dt_head
        );
    }
}
