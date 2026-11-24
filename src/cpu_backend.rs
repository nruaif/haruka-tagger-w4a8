use crate::model::{Eva02Block, HarukaModel, LayerNormParams, MHAttnPoolParams, W4Linear};
use rayon::prelude::*;

pub fn layer_norm(x: &[f32], ln: &LayerNormParams, rows: usize) -> Vec<f32> {
    let d = ln.dim;
    let mut out = vec![0.0f32; rows * d];
    out.par_chunks_mut(d).enumerate().for_each(|(r, out_row)| {
        let row = &x[r * d..(r + 1) * d];
        let mut sum = 0.0f32;
        for &v in row {
            sum += v;
        }
        let mean = sum / (d as f32);
        let mut var = 0.0f32;
        for &v in row {
            let diff = v - mean;
            var += diff * diff;
        }
        let inv_std = 1.0f32 / (var / (d as f32) + 1e-6).sqrt();
        for i in 0..d {
            out_row[i] = (row[i] - mean) * inv_std * ln.weight[i] + ln.bias[i];
        }
    });
    out
}

/// Multi-threaded W4A8 Linear on CPU:
///   1. Smooths x by `1 / awq_scale` and dynamically quantizes to INT8 [-127, 127] per `act_group_size = 128`
///   2. Unpacks 4-bit signed weights (`[-8, 7]`) per `group_size = 64` and computes integer dot products
///      multiplied by `(x_scale * w_scale)` + bias.
pub fn w4a8_linear(x: &[f32], rows: usize, lin: &W4Linear) -> Vec<f32> {
    let k = lin.in_features;
    let n = lin.out_features;
    let act_gs = 128usize;
    let act_groups = (k + act_gs - 1) / act_gs;
    let w_gs = lin.group_size;

    // 1. Dynamic INT8 quantization of input activations (smooth by 1 / awq_scale)
    let mut x_q = vec![0i32; rows * k];
    let mut x_scales = vec![0.0f32; rows * act_groups];

    x_q.par_chunks_mut(k)
        .zip(x_scales.par_chunks_mut(act_groups))
        .enumerate()
        .for_each(|(r, (xq_row, xs_row))| {
            let x_row = &x[r * k..(r + 1) * k];
            for ag in 0..act_groups {
                let start = ag * act_gs;
                let end = (start + act_gs).min(k);
                let mut amax = 1e-6f32;
                for i in start..end {
                    let v = (x_row[i] / lin.awq_scale[i]).abs();
                    if v > amax {
                        amax = v;
                    }
                }
                let sc = amax / 127.0;
                xs_row[ag] = sc;
                let inv_sc = 1.0 / sc;
                for i in start..end {
                    let q = ((x_row[i] / lin.awq_scale[i]) * inv_sc).round().clamp(-127.0, 127.0) as i32;
                    xq_row[i] = q;
                }
            }
        });

    // 2. Parallel matmul across output rows N (or M for M > 1)
    let mut out = vec![0.0f32; rows * n];
    if rows == 1 {
        // GEMV: parallelize over output features N
        out.par_iter_mut().enumerate().for_each(|(col, dst)| {
            let w_row_u32 = &lin.qweight_u32_signed[col * (k / 8)..(col + 1) * (k / 8)];
            let ws_row = &lin.scales[col * lin.num_groups..(col + 1) * lin.num_groups];
            let mut acc_f = lin.bias.as_ref().map_or(0.0f32, |b| b[col]);

            // Step by 64 weights (which aligns with both w_gs=64 and half of act_gs=128)
            for g in 0..lin.num_groups {
                let k_start = g * w_gs;
                let k_end = (k_start + w_gs).min(k);
                let ag = k_start / act_gs;
                let xs = x_scales[ag];
                let ws = ws_row[g];
                let mut g_acc = 0i32;

                let u_start = k_start / 8;
                let u_end = (k_end + 7) / 8;
                for u_idx in u_start..u_end {
                    let word = w_row_u32[u_idx];
                    let base_k = u_idx * 8;
                    for nib in 0..8 {
                        let kk = base_k + nib;
                        if kk >= k_end {
                            break;
                        }
                        // Sign-extend 4-bit nibble: shift left to MSB of i32 then arithmetic shift right by 28
                        let w_val = ((word << (28 - nib * 4)) as i32) >> 28;
                        g_acc += x_q[kk] * w_val;
                    }
                }
                acc_f += (g_acc as f32) * (ws * xs);
            }
            *dst = acc_f;
        });
    } else {
        // GEMM: parallelize over (row_block, col) by rows of out [M, N]
        // Pre-unpack weights in L2-friendly column tiles or unpack per col in parallel
        let k_u32 = k / 8;
        let mut out_t = vec![0.0f32; n * rows];
        out_t.par_chunks_mut(rows).enumerate().for_each(|(col, dst_col)| {
            let w_row_u32 = &lin.qweight_u32_signed[col * k_u32..(col + 1) * k_u32];
            let ws_row = &lin.scales[col * lin.num_groups..(col + 1) * lin.num_groups];
            let b_val = lin.bias.as_ref().map_or(0.0f32, |b| b[col]);
            dst_col.fill(b_val);

            // Unpack this column's weights once for all M rows
            let mut w_i32 = vec![0i32; k];
            for u_idx in 0..k_u32 {
                let word = w_row_u32[u_idx];
                let base_k = u_idx * 8;
                for nib in 0..8 {
                    w_i32[base_k + nib] = ((word << (28 - nib * 4)) as i32) >> 28;
                }
            }

            for r in 0..rows {
                let xq_row = &x_q[r * k..(r + 1) * k];
                let xs_row = &x_scales[r * act_groups..(r + 1) * act_groups];
                let mut acc_f = 0.0f32;
                for g in 0..lin.num_groups {
                    let k_start = g * w_gs;
                    let k_end = (k_start + w_gs).min(k);
                    let ag = k_start / act_gs;
                    let mut g_acc = 0i32;
                    for kk in k_start..k_end {
                        g_acc += xq_row[kk] * w_i32[kk];
                    }
                    acc_f += (g_acc as f32) * (ws_row[g] * xs_row[ag]);
                }
                dst_col[r] += acc_f;
            }
        });

        // Transpose [N, M] -> [M, N]
        out.par_chunks_mut(n).enumerate().for_each(|(r, out_row)| {
            for col in 0..n {
                out_row[col] = out_t[col * rows + r];
            }
        });
    }
    out
}

fn patch_embed(model: &HarukaModel, nchw: &[f32]) -> Vec<f32> {
    let img_sz = model.meta.img_size;
    let p = model.patch_size;
    let g = model.grid_size;
    let n_patches = g * g;
    let dim = model.dim;
    let hw = img_sz * img_sz;

    let mut tokens = vec![0.0f32; (1 + n_patches) * dim];
    // CLS token + pos_embed[0]
    for d in 0..dim {
        tokens[d] = model.cls_token[d] + model.pos_embed[d];
    }

    // Patch tokens
    let patch_dim = 3 * p * p;
    tokens[dim..]
        .par_chunks_mut(dim)
        .enumerate()
        .for_each(|(p_idx, dst)| {
            let py = p_idx / g;
            let px = p_idx % g;
            let mut patch_vec = [0.0f32; 3 * 14 * 14];
            let mut idx = 0usize;
            for c in 0..3 {
                for dy in 0..p {
                    let y = py * p + dy;
                    let row_off = c * hw + y * img_sz + px * p;
                    for dx in 0..p {
                        patch_vec[idx] = nchw[row_off + dx];
                        idx += 1;
                    }
                }
            }
            let pos_off = (1 + p_idx) * dim;
            for d in 0..dim {
                let w_row = &model.patch_w[d * patch_dim..(d + 1) * patch_dim];
                let mut sum = model.patch_b[d] + model.pos_embed[pos_off + d];
                for i in 0..patch_dim {
                    sum += patch_vec[i] * w_row[i];
                }
                dst[d] = sum;
            }
        });

    tokens
}

#[inline]
fn apply_rope_to_patches(
    vec: &mut [f32],
    _seq_len: usize,
    num_heads: usize,
    head_dim: usize,
    rope_sin: &[f32],
    rope_cos: &[f32],
) {
    let d_total = num_heads * head_dim;
    // Skip token 0 (CLS), apply to tokens 1..seq_len
    vec[d_total..]
        .par_chunks_mut(d_total)
        .enumerate()
        .for_each(|(patch_i, tok)| {
            let sin_row = &rope_sin[patch_i * head_dim..(patch_i + 1) * head_dim];
            let cos_row = &rope_cos[patch_i * head_dim..(patch_i + 1) * head_dim];
            for h in 0..num_heads {
                let hs = &mut tok[h * head_dim..(h + 1) * head_dim];
                for pair in 0..(head_dim / 2) {
                    let i0 = pair * 2;
                    let i1 = i0 + 1;
                    let x0 = hs[i0];
                    let x1 = hs[i1];
                    // timm rot(x) = [-x1, x0]
                    hs[i0] = x0 * cos_row[i0] + (-x1) * sin_row[i0];
                    hs[i1] = x1 * cos_row[i1] + x0 * sin_row[i1];
                }
            }
        });
}

fn forward_block(x: &mut [f32], seq_len: usize, blk: &Eva02Block, model: &HarukaModel) {
    let dim = model.dim;
    let nh = model.num_heads;
    let hd = model.head_dim;
    let scale = 1.0f32 / (hd as f32).sqrt();

    let h1 = layer_norm(x, &blk.norm1, seq_len);
    let mut q = w4a8_linear(&h1, seq_len, &blk.q_proj);
    let mut k = w4a8_linear(&h1, seq_len, &blk.k_proj);
    let v = w4a8_linear(&h1, seq_len, &blk.v_proj);

    apply_rope_to_patches(&mut q, seq_len, nh, hd, &model.rope_sin, &model.rope_cos);
    apply_rope_to_patches(&mut k, seq_len, nh, hd, &model.rope_sin, &model.rope_cos);

    // Multi-head self-attention (parallelized over heads x query_tokens)
    let mut attn_out = vec![0.0f32; seq_len * dim];
    attn_out
        .par_chunks_mut(dim)
        .enumerate()
        .for_each(|(qi, out_tok)| {
            let q_tok = &q[qi * dim..(qi + 1) * dim];
            let mut logits = vec![0.0f32; seq_len];
            for h in 0..nh {
                let qh = &q_tok[h * hd..(h + 1) * hd];
                let mut max_l = f32::NEG_INFINITY;
                for ki in 0..seq_len {
                    let kh = &k[ki * dim + h * hd..ki * dim + (h + 1) * hd];
                    let mut dot = 0.0f32;
                    for d in 0..hd {
                        dot += qh[d] * kh[d];
                    }
                    let l = dot * scale;
                    logits[ki] = l;
                    if l > max_l {
                        max_l = l;
                    }
                }
                let mut sum_exp = 0.0f32;
                for ki in 0..seq_len {
                    let e = (logits[ki] - max_l).exp();
                    logits[ki] = e;
                    sum_exp += e;
                }
                let inv_sum = 1.0 / sum_exp;
                let out_h = &mut out_tok[h * hd..(h + 1) * hd];
                for ki in 0..seq_len {
                    let w = logits[ki] * inv_sum;
                    let vh = &v[ki * dim + h * hd..ki * dim + (h + 1) * hd];
                    for d in 0..hd {
                        out_h[d] += w * vh[d];
                    }
                }
            }
        });

    let proj_out = w4a8_linear(&attn_out, seq_len, &blk.proj);
    for i in 0..x.len() {
        x[i] += proj_out[i];
    }

    // SwiGLU MLP
    let h2 = layer_norm(x, &blk.norm2, seq_len);
    let fg = w4a8_linear(&h2, seq_len, &blk.fc1_g);
    let fx = w4a8_linear(&h2, seq_len, &blk.fc1_x);
    let mut swiglu = vec![0.0f32; fg.len()];
    swiglu.par_iter_mut().enumerate().for_each(|(i, dst)| {
        let g = fg[i];
        let silu = g / (1.0f32 + (-g).exp());
        *dst = silu * fx[i];
    });
    let mlp_n = layer_norm(&swiglu, &blk.mlp_norm, seq_len);
    let fc2_out = w4a8_linear(&mlp_n, seq_len, &blk.fc2);
    for i in 0..x.len() {
        x[i] += fc2_out[i];
    }
}

pub fn forward_pool(x_final: &[f32], seq_len: usize, pool: &MHAttnPoolParams, dim: usize) -> Vec<f32> {
    let x_norm = layer_norm(x_final, &pool.norm, seq_len);
    let kv = w4a8_linear(&x_norm, seq_len, &pool.kv);
    let nh = pool.num_heads;
    let hd = pool.head_dim;
    let scale = 1.0f32 / (hd as f32).sqrt();

    // kv is [seq_len, 2 * dim] -> k = [0..dim], v = [dim..2*dim]
    let k_at = |t: usize, h: usize, d: usize| kv[t * (2 * dim) + h * hd + d];
    let v_at = |t: usize, h: usize, d: usize| kv[t * (2 * dim) + dim + h * hd + d];

    // Level 1: Coarse trunk queries [num_coarse, dim]
    let nc = pool.num_coarse;
    let mut logits_c = vec![0.0f32; nh * nc * seq_len];
    for h in 0..nh {
        for q in 0..nc {
            for t in 0..seq_len {
                let mut dot = 0.0f32;
                for d in 0..hd {
                    dot += pool.q_coarse[q * dim + h * hd + d] * k_at(t, h, d);
                }
                logits_c[(h * nc + q) * seq_len + t] = dot * scale;
            }
        }
    }

    // slot_sigmoid: softmax over queries (dim=-2) * sigmoid(logits + q_coarse_bias)
    let mut attn_c = vec![0.0f32; nh * nc * seq_len];
    for h in 0..nh {
        for t in 0..seq_len {
            let mut max_l = f32::NEG_INFINITY;
            for q in 0..nc {
                let v = logits_c[(h * nc + q) * seq_len + t];
                if v > max_l {
                    max_l = v;
                }
            }
            let mut sum_e = 0.0f32;
            for q in 0..nc {
                sum_e += (logits_c[(h * nc + q) * seq_len + t] - max_l).exp();
            }
            for q in 0..nc {
                let l = logits_c[(h * nc + q) * seq_len + t];
                let sm = (l - max_l).exp() / sum_e;
                let sig = 1.0 / (1.0 + (-(l + pool.q_coarse_bias[q])).exp());
                attn_c[(h * nc + q) * seq_len + t] = sm * sig;
            }
        }
    }

    let mut z_coarse_raw = vec![0.0f32; nc * dim];
    for q in 0..nc {
        for h in 0..nh {
            for t in 0..seq_len {
                let a = attn_c[(h * nc + q) * seq_len + t];
                for d in 0..hd {
                    z_coarse_raw[q * dim + h * hd + d] += a * v_at(t, h, d);
                }
            }
        }
    }
    let z_coarse = layer_norm(&z_coarse_raw, &pool.trunk_norm, nc);

    // Level 2: Child leaf queries [num_queries, dim]
    let nq = pool.num_queries;
    let mut q_leaf_in = vec![0.0f32; nq * dim];
    for c in 0..nc {
        for b in 0..pool.branches {
            let q = c * pool.branches + b;
            for d in 0..dim {
                q_leaf_in[q * dim + d] = z_coarse[c * dim + d] + pool.q_child[q * dim + d];
            }
        }
    }
    let q_leaf = layer_norm(&q_leaf_in, &pool.child_q_norm, nq);

    let mut logits_l = vec![0.0f32; nh * nq * seq_len];
    for h in 0..nh {
        for q in 0..nq {
            for t in 0..seq_len {
                let mut dot = 0.0f32;
                for d in 0..hd {
                    dot += q_leaf[q * dim + h * hd + d] * k_at(t, h, d);
                }
                logits_l[(h * nq + q) * seq_len + t] = dot * scale;
            }
        }
    }

    let mut attn_l = vec![0.0f32; nh * nq * seq_len];
    for h in 0..nh {
        for t in 0..seq_len {
            let mut max_l = f32::NEG_INFINITY;
            for q in 0..nq {
                let v = logits_l[(h * nq + q) * seq_len + t];
                if v > max_l {
                    max_l = v;
                }
            }
            let mut sum_e = 0.0f32;
            for q in 0..nq {
                sum_e += (logits_l[(h * nq + q) * seq_len + t] - max_l).exp();
            }
            for q in 0..nq {
                let l = logits_l[(h * nq + q) * seq_len + t];
                let sm = (l - max_l).exp() / sum_e;
                let sig = 1.0 / (1.0 + (-(l + pool.q_child_bias[q])).exp());
                attn_l[(h * nq + q) * seq_len + t] = sm * sig;
            }
        }
    }

    let mut z_tree_in = vec![0.0f32; nq * dim];
    let mut mass_q = vec![0.0f32; nq];
    for c in 0..nc {
        for b in 0..pool.branches {
            let q = c * pool.branches + b;
            for h in 0..nh {
                let mut sum_a = 0.0f32;
                for t in 0..seq_len {
                    let a = attn_l[(h * nq + q) * seq_len + t];
                    sum_a += a;
                    for d in 0..hd {
                        z_tree_in[q * dim + h * hd + d] += a * v_at(t, h, d);
                    }
                }
                mass_q[q] += sum_a;
            }
            for d in 0..dim {
                z_tree_in[q * dim + d] += z_coarse[c * dim + d];
            }
            mass_q[q] /= nh as f32;
        }
    }

    let mut z_tree = layer_norm(&z_tree_in, &pool.leaf_norm, nq);
    for q in 0..nq {
        let m_scale = mass_q[q] / (mass_q[q] + 4.0);
        for d in 0..dim {
            z_tree[q * dim + d] *= m_scale;
        }
    }

    // Leaf projection: einsum("bqd,qdf->bqf", z_tree, leaf_proj_weight) + leaf_proj_bias
    let ld = pool.leaf_dim;
    let mut pooled = vec![0.0f32; nq * ld];
    for q in 0..nq {
        for f in 0..ld {
            let mut sum = pool.leaf_proj_bias[q * ld + f];
            for d in 0..dim {
                sum += z_tree[q * dim + d] * pool.leaf_proj_weight[(q * dim + d) * ld + f];
            }
            pooled[q * ld + f] = sum;
        }
    }
    pooled
}

pub fn infer_cpu(model: &HarukaModel, nchw: &[f32]) -> Vec<f32> {
    let seq_len = 1 + model.grid_size * model.grid_size;
    let mut x = patch_embed(model, nchw);
    for blk in &model.blocks {
        forward_block(&mut x, seq_len, blk, model);
    }
    let x_final = layer_norm(&x, &model.norm, seq_len);
    let pooled = forward_pool(&x_final, seq_len, &model.pool, model.dim);
    let logits = w4a8_linear(&pooled, 1, &model.head);
    logits
        .into_iter()
        .map(|l| 1.0f32 / (1.0f32 + (-l).exp()))
        .collect()
}
