// Gemma4 forward graph builder + step function.
//
// Architecture (from deps/llama.cpp/src/models/gemma4-iswa.cpp):
//   - Scale input embeddings by sqrt(n_embd)
//   - For each layer:
//     a. Pre-attn RMSNorm
//     b. Q/K/V projections + per-head Q/K RMSNorm + RoPE
//        (KV-sharing layers skip K/V proj, reuse source layer's KV cache)
//     c. Write K/V to cache, flash attention (full or SWA)
//     d. Post-attn RMSNorm + residual
//     e. Dense FFN (lead layer) or MoE (shared GELU-gated + routed experts)
//     f. FFN post-norm + residual
//     g. Per-layer embedding injection (gated)
//     h. Output scale
//   - Final RMSNorm + lm_head
//   - Logit softcapping: tanh(logits/cap)*cap

#include "gemma4_internal.h"
#include "dflash27b.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <vector>

#include "ggml-backend.h"
#include "ggml-cuda.h"
#include "ggml-alloc.h"

namespace dflash::common {

static constexpr float GEMMA4_EPS = 1e-6f;

static ggml_tensor * gemma4_rms_norm_mul(ggml_context * ctx, ggml_tensor * x,
                                          ggml_tensor * weight, float eps = GEMMA4_EPS) {
    ggml_tensor * n = ggml_rms_norm(ctx, x, eps);
    return ggml_mul(ctx, n, weight);
}

// Dense GELU-gated FFN (layer 0 / lead dense layers).
// Gemma4 uses GELU not SiLU: cur = down( gelu(gate(x)) * up(x) )
static ggml_tensor * build_gemma4_dense_ffn(ggml_context * ctx, ggml_tensor * cur,
                                              const Gemma4Layer & L) {
    ggml_tensor * gate = ggml_mul_mat(ctx, L.ffn_gate, cur);
    ggml_tensor * up   = ggml_mul_mat(ctx, L.ffn_up,   cur);
    ggml_tensor * gu   = ggml_mul(ctx, ggml_gelu(ctx, gate), up);
    return ggml_mul_mat(ctx, L.ffn_down, gu);
}

// MoE block: shared expert (GELU-gated) + routed experts (softmax gating).
// Gemma4-specific routing: attn_out → rms_norm → scale by 1/sqrt(n_embd) → mul ffn_gate_inp_s → ffn_gate_inp → softmax → top-k
static ggml_tensor * build_gemma4_moe_block(ggml_context * ctx, ggml_tensor * attn_out,
                                              ggml_tensor * cur_normed,
                                              const Gemma4Weights & w,
                                              const Gemma4Layer & L,
                                              int n_tokens) {
    const int n_expert = w.n_expert;
    const int n_used   = w.n_expert_used;
    const int n_embd   = w.n_embd;

    // ---- Shared expert (GELU-gated MLP) ----
    ggml_tensor * sh_gate = ggml_mul_mat(ctx, L.ffn_gate, cur_normed);
    ggml_tensor * sh_up   = ggml_mul_mat(ctx, L.ffn_up,   cur_normed);
    ggml_tensor * sh_gu   = ggml_mul(ctx, ggml_gelu(ctx, sh_gate), sh_up);
    ggml_tensor * shared  = ggml_mul_mat(ctx, L.ffn_down, sh_gu);

    if (L.ffn_post_norm_1) {
        shared = gemma4_rms_norm_mul(ctx, shared, L.ffn_post_norm_1, w.norm_eps);
    }

    // ---- Routed experts ----
    if (!L.ffn_gate_inp || n_expert == 0) {
        // No MoE on this layer, shared-only
        return shared;
    }

    // Pre-norm for routed input
    ggml_tensor * cur_moe = cur_normed;
    if (L.ffn_pre_norm_2) {
        cur_moe = gemma4_rms_norm_mul(ctx, attn_out, L.ffn_pre_norm_2, w.norm_eps);
    }

    // Router: rms_norm(attn_out) * (1/sqrt(n_embd)) * ffn_gate_inp_s → ffn_gate_inp → softmax
    ggml_tensor * router_in = ggml_rms_norm(ctx, attn_out, w.norm_eps);
    router_in = ggml_scale(ctx, router_in, 1.0f / std::sqrt((float)n_embd));
    if (L.ffn_gate_inp_s) {
        router_in = ggml_mul(ctx, router_in, L.ffn_gate_inp_s);
    }
    ggml_tensor * logits = ggml_mul_mat(ctx, L.ffn_gate_inp, router_in); // [n_expert, n_tokens]

    // Softmax over experts
    ggml_tensor * probs = ggml_soft_max(ctx, logits);

    // Top-k selection
    ggml_tensor * selected = ggml_top_k(ctx, probs, n_used);

    // Gather weights at selected indices
    ggml_tensor * probs_3d = ggml_reshape_3d(ctx, probs, 1, n_expert, n_tokens);
    ggml_tensor * weights  = ggml_get_rows(ctx, probs_3d, selected);
    weights = ggml_reshape_2d(ctx, weights, n_used, n_tokens);

    // Routed expert forward via mul_mat_id with fused gate+up
    ggml_tensor * cur_3d = ggml_reshape_3d(ctx, cur_moe, n_embd, 1, n_tokens);
    ggml_tensor * gate_up_e = ggml_mul_mat_id(ctx, L.ffn_gate_up_exps, cur_3d, selected);
    // gate_up_e is [n_ff_exp*2, n_used, n_tokens] — split and GELU-gate
    const int n_ff_exp = w.n_ff_exp;
    ggml_tensor * gate_e = ggml_view_3d(ctx, gate_up_e,
        n_ff_exp, gate_up_e->ne[1], gate_up_e->ne[2],
        gate_up_e->nb[1], gate_up_e->nb[2], 0);
    ggml_tensor * up_e = ggml_view_3d(ctx, gate_up_e,
        n_ff_exp, gate_up_e->ne[1], gate_up_e->ne[2],
        gate_up_e->nb[1], gate_up_e->nb[2],
        (size_t)n_ff_exp * ggml_element_size(gate_up_e));
    ggml_tensor * gu = ggml_mul(ctx, ggml_gelu(ctx, gate_e), up_e);
    ggml_tensor * experts = ggml_mul_mat_id(ctx, L.ffn_down_exps, gu, selected);

    // Weighted sum of expert outputs
    ggml_tensor * w_view = ggml_reshape_3d(ctx, weights, 1, n_used, n_tokens);
    experts = ggml_mul(ctx, experts, w_view);

    ggml_tensor * routed = nullptr;
    for (int i = 0; i < n_used; ++i) {
        ggml_tensor * slice = ggml_view_2d(ctx, experts,
            n_embd, n_tokens,
            experts->nb[2],
            (size_t)i * experts->nb[1]);
        routed = (i == 0) ? slice : ggml_add(ctx, routed, slice);
    }

    if (L.ffn_post_norm_2) {
        routed = gemma4_rms_norm_mul(ctx, routed, L.ffn_post_norm_2, w.norm_eps);
    }

    return ggml_add(ctx, shared, routed);
}

// Attention block for a single layer (handles both full and SWA).
static ggml_tensor * build_gemma4_attn_block(
    ggml_context * ctx,
    ggml_cgraph * gf,
    const Gemma4Weights & w,
    const Gemma4Layer & L,
    Gemma4Cache & cache,
    int il,
    ggml_tensor * cur,
    ggml_tensor * positions,
    ggml_tensor * attn_mask_full,
    ggml_tensor * attn_mask_swa,
    int kv_start,
    int n_tokens)
{
    const int head_dim   = gemma4_head_dim(w, il);
    const int n_head     = w.n_head;
    const int n_head_kv  = gemma4_n_head_kv(w, il);
    const int q_dim      = n_head * head_dim;
    const bool is_swa    = gemma4_is_swa_layer(w, il);
    const bool has_kv    = gemma4_has_kv(w, il);

    // Q projection (all layers have Q)
    ggml_tensor * Qcur = ggml_mul_mat(ctx, L.wq, cur);
    Qcur = ggml_reshape_3d(ctx, Qcur, head_dim, n_head, n_tokens);

    // Q RMSNorm per head
    if (L.q_norm) {
        Qcur = gemma4_rms_norm_mul(ctx, Qcur, L.q_norm, w.norm_eps);
    }

    // RoPE for Q
    const float rope_base = is_swa ? w.rope_freq_base_swa : w.rope_freq_base_full;
    ggml_tensor * freq_factors = is_swa ? nullptr : (L.rope_freqs ? L.rope_freqs : w.rope_freqs_global);
    Qcur = ggml_rope_ext(ctx, Qcur, positions, freq_factors,
                          head_dim, GGML_ROPE_TYPE_NEOX,
                          0, rope_base, 1.0f,
                          0.0f, 1.0f, 32.0f, 1.0f);

    // Determine which cache layer to use
    int cache_il = cache.kv_source[il];
    ggml_tensor * cache_k = cache.k[cache_il];
    ggml_tensor * cache_v = cache.v[cache_il];

    if (has_kv) {
        // K/V projection + norm + RoPE + write to cache
        ggml_tensor * Kcur = ggml_mul_mat(ctx, L.wk, cur);
        ggml_tensor * Vcur = L.wv ? ggml_mul_mat(ctx, L.wv, cur) : Kcur;

        Kcur = ggml_reshape_3d(ctx, Kcur, head_dim, n_head_kv, n_tokens);
        Vcur = ggml_reshape_3d(ctx, Vcur, head_dim, n_head_kv, n_tokens);

        if (L.k_norm) {
            Kcur = gemma4_rms_norm_mul(ctx, Kcur, L.k_norm, w.norm_eps);
        }
        // V also gets RMSNorm (gemma4 specific)
        Vcur = ggml_rms_norm(ctx, Vcur, w.norm_eps);

        Kcur = ggml_rope_ext(ctx, Kcur, positions, freq_factors,
                              head_dim, GGML_ROPE_TYPE_NEOX,
                              0, rope_base, 1.0f,
                              0.0f, 1.0f, 32.0f, 1.0f);

        // Write K/V to cache
        ggml_tensor * Kcur_T = ggml_permute(ctx, Kcur, 0, 2, 1, 3);
        ggml_tensor * Vcur_T = ggml_permute(ctx, Vcur, 0, 2, 1, 3);

        ggml_tensor * k_slot = ggml_view_3d(ctx, cache_k,
            head_dim, n_tokens, n_head_kv,
            cache_k->nb[1], cache_k->nb[2],
            cache_k->nb[1] * (size_t)kv_start);
        ggml_tensor * v_slot = ggml_view_3d(ctx, cache_v,
            head_dim, n_tokens, n_head_kv,
            cache_v->nb[1], cache_v->nb[2],
            cache_v->nb[1] * (size_t)kv_start);
        ggml_build_forward_expand(gf, ggml_cpy(ctx, Kcur_T, k_slot));
        ggml_build_forward_expand(gf, ggml_cpy(ctx, Vcur_T, v_slot));
    }
    // else: KV-sharing layer — cache already written by source layer

    // Flash attention
    // Pad kv_len to multiple of 256 for CUDA FA kernel compatibility (FATTN_KQ_STRIDE=256)
    const int kv_len_raw = kv_start + n_tokens;
    const int kv_len = (kv_len_raw + 255) & ~255;  // round up to 256

    ggml_tensor * Qfa = ggml_permute(ctx, Qcur, 0, 2, 1, 3);
    Qfa = ggml_cont(ctx, Qfa);

    ggml_tensor * Kfa = ggml_view_3d(ctx, cache_k,
        head_dim, kv_len, n_head_kv,
        cache_k->nb[1], cache_k->nb[2], 0);
    ggml_tensor * Vfa = ggml_view_3d(ctx, cache_v,
        head_dim, kv_len, n_head_kv,
        cache_v->nb[1], cache_v->nb[2], 0);

    // Gemma4 uses self.scaling = 1.0 (no QK scaling) because Q/K are already
    // RMS-normed per-head. Standard 1/sqrt(head_dim) is NOT used here.
    const float kq_scale = 1.0f;
    ggml_tensor * use_mask = is_swa ? attn_mask_swa : attn_mask_full;
    ggml_tensor * attn = ggml_flash_attn_ext(ctx, Qfa, Kfa, Vfa, use_mask,
                                              kq_scale, 0.0f, 0.0f);

    // Reshape to [q_dim, n_tokens] and output projection
    attn = ggml_reshape_2d(ctx, attn, q_dim, n_tokens);
    return ggml_mul_mat(ctx, L.wo, attn);
}

// Build one layer of the gemma4 graph.
static ggml_tensor * build_gemma4_layer(
    ggml_context * ctx,
    ggml_cgraph * gf,
    const Gemma4Weights & w,
    Gemma4Cache & cache,
    int il,
    ggml_tensor * inp,
    ggml_tensor * positions,
    ggml_tensor * attn_mask_full,
    ggml_tensor * attn_mask_swa,
    ggml_tensor * per_layer_input,  // [n_embd_per_layer, n_tokens] or nullptr
    int kv_start,
    int n_tokens)
{
    const Gemma4Layer & L = w.layers[il];

    // Pre-attn norm
    ggml_tensor * cur = gemma4_rms_norm_mul(ctx, inp, L.attn_norm, w.norm_eps);

    // Attention
    cur = build_gemma4_attn_block(ctx, gf, w, L, cache, il, cur,
                                    positions, attn_mask_full, attn_mask_swa,
                                    kv_start, n_tokens);

    // Post-attn norm
    if (L.attn_post_norm) {
        cur = gemma4_rms_norm_mul(ctx, cur, L.attn_post_norm, w.norm_eps);
    }

    // Residual
    ggml_tensor * attn_out = ggml_add(ctx, cur, inp);

    // FFN
    const bool is_moe = (L.ffn_gate_inp != nullptr && il >= w.n_layer_dense_lead);
    if (is_moe) {
        // MoE: shared expert + routed experts
        ggml_tensor * cur_normed = gemma4_rms_norm_mul(ctx, attn_out, L.ffn_norm, w.norm_eps);
        cur = build_gemma4_moe_block(ctx, attn_out, cur_normed, w, L, n_tokens);
    } else {
        // Dense FFN
        cur = gemma4_rms_norm_mul(ctx, attn_out, L.ffn_norm, w.norm_eps);
        cur = build_gemma4_dense_ffn(ctx, cur, L);
    }

    // FFN post-norm (applies to both dense and MoE paths)
    if (L.ffn_post_norm) {
        cur = gemma4_rms_norm_mul(ctx, cur, L.ffn_post_norm, w.norm_eps);
    }

    // Residual
    cur = ggml_add(ctx, cur, attn_out);

    // Per-layer embedding injection
    if (per_layer_input && L.per_layer_inp_gate && L.per_layer_proj) {
        ggml_tensor * pe_in = cur;
        // Gate: cur -> [n_embd_per_layer, n_tokens]
        ggml_tensor * gate = ggml_mul_mat(ctx, L.per_layer_inp_gate, cur);
        gate = ggml_gelu(ctx, gate);
        // Element-wise mul with per-layer input
        gate = ggml_mul(ctx, gate, per_layer_input);
        // Project back: [n_embd_per_layer, n_tokens] -> [n_embd, n_tokens]
        ggml_tensor * proj = ggml_mul_mat(ctx, L.per_layer_proj, gate);
        if (L.per_layer_post_norm) {
            proj = gemma4_rms_norm_mul(ctx, proj, L.per_layer_post_norm, w.norm_eps);
        }
        cur = ggml_add(ctx, pe_in, proj);
    }

    // Output scale
    if (L.out_scale) {
        cur = ggml_mul(ctx, cur, L.out_scale);
    }

    return cur;
}

// Helper: get a 2D slice from a 3D tensor along ne[2] (same as llama.cpp ggml_view_2d_slice).
static ggml_tensor * gemma4_view_2d_slice(ggml_context * ctx, ggml_tensor * x, int idx) {
    return ggml_view_2d(ctx, x, x->ne[0], x->ne[1],
                        ggml_row_size(x->type, x->ne[0]),
                        (size_t)idx * x->ne[0] * x->ne[1] * ggml_element_size(x));
}

bool gemma4_step(
    ggml_backend_t          backend,
    const Gemma4Weights &   w,
    Gemma4Cache &           cache,
    const float *           embed,
    const int32_t *         token_ids,
    int                     n_tokens,
    int                     kv_start,
    std::vector<float> &    out_logits)
{
    // Allocate graph context
    ggml_init_params ip{};
    ip.mem_size = ggml_tensor_overhead() * 16384 + ggml_graph_overhead() + 16 * 1024 * 1024;
    ip.no_alloc = true;
    ggml_context * ctx = ggml_init(ip);
    ggml_cgraph * gf = ggml_new_graph_custom(ctx, 16384, false);

    // Input tensors
    ggml_tensor * ie = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, w.n_embd, n_tokens);
    ggml_set_input(ie);
    ggml_tensor * pp = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, n_tokens);
    ggml_set_input(pp);

    // Token IDs input (for per-layer embedding lookup)
    ggml_tensor * tok_ids = nullptr;
    if (token_ids && w.per_layer_tok_embd && w.per_layer_model_proj && w.n_embd_per_layer > 0) {
        tok_ids = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, n_tokens);
        ggml_set_input(tok_ids);
    }

    // Attention masks (full + SWA)
    // Pad kv_len to 256 for CUDA FA kernel compatibility (FATTN_KQ_STRIDE=256)
    const int kv_len_raw = kv_start + n_tokens;
    const int kv_len_padded = (kv_len_raw + 255) & ~255;
    ggml_tensor * mk_full = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, kv_len_padded, n_tokens, 1, 1);
    ggml_set_input(mk_full);
    ggml_tensor * mk_full_f16 = ggml_cast(ctx, mk_full, GGML_TYPE_F16);
    ggml_tensor * mk_swa = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, kv_len_padded, n_tokens, 1, 1);
    ggml_set_input(mk_swa);
    ggml_tensor * mk_swa_f16 = ggml_cast(ctx, mk_swa, GGML_TYPE_F16);

    // Per-layer embedding computation (reference: gemma4-iswa.cpp build_inp_per_layer + project_per_layer_inputs)
    ggml_tensor * per_layer_all = nullptr;  // final shape: [n_embd_per_layer, n_tokens, n_layer]
    if (tok_ids) {
        const int D = w.n_embd_per_layer;
        const int L = w.n_layer;

        // 1. Token per-layer embedding lookup + scale
        //    get_rows(per_layer_tok_embd[D*L, n_vocab], tok_ids) → [D*L, n_tokens]
        ggml_tensor * inp_pl = ggml_get_rows(ctx, w.per_layer_tok_embd, tok_ids);
        inp_pl = ggml_reshape_3d(ctx, inp_pl, D, L, n_tokens);  // [D, L, n_tokens]
        inp_pl = ggml_scale(ctx, inp_pl, std::sqrt((float)D));

        // 2. Project main embedding through per_layer_model_proj
        //    mul_mat(per_layer_model_proj[n_embd, D*L], ie[n_embd, n_tokens]) → [D*L, n_tokens]
        ggml_tensor * proj = ggml_mul_mat(ctx, w.per_layer_model_proj, ie);
        proj = ggml_scale(ctx, proj, 1.0f / std::sqrt((float)w.n_embd));
        proj = ggml_reshape_3d(ctx, proj, D, L, n_tokens);  // [D, L, n_tokens]

        // 3. RMS norm on projection (normalizes over ne[0]=D for each (layer, token))
        proj = ggml_rms_norm(ctx, proj, w.norm_eps);
        // Reshape norm weight from [D*L] to [D, L] for broadcast mul over n_tokens
        ggml_tensor * norm_w = ggml_reshape_2d(ctx, w.per_layer_proj_norm, D, L);
        proj = ggml_mul(ctx, proj, norm_w);

        // 4. Add token embedding + projection, scale by 1/sqrt(2)
        per_layer_all = ggml_add(ctx, proj, inp_pl);
        per_layer_all = ggml_scale(ctx, per_layer_all, 1.0f / std::sqrt(2.0f));

        // 5. Permute to [D, n_tokens, L] for easy per-layer slicing
        per_layer_all = ggml_cont(ctx, ggml_permute(ctx, per_layer_all, 0, 2, 1, 3));
    }

    // Build the graph
    ggml_tensor * cur = ie;  // [n_embd, n_tokens] already scaled by sqrt(n_embd) in caller

    for (int il = 0; il < w.n_layer; ++il) {
        ggml_tensor * pl_input = nullptr;
        if (per_layer_all) {
            // Slice [n_embd_per_layer, n_tokens] for this layer
            pl_input = gemma4_view_2d_slice(ctx, per_layer_all, il);
        }
        cur = build_gemma4_layer(ctx, gf, w, cache, il, cur, pp,
                                   mk_full_f16, mk_swa_f16, pl_input,
                                   kv_start, n_tokens);
    }

    // Final norm
    cur = gemma4_rms_norm_mul(ctx, cur, w.out_norm, w.norm_eps);

    // Extract last token only for logits
    if (n_tokens > 1) {
        cur = ggml_view_2d(ctx, cur, w.n_embd, 1,
                            cur->nb[1],
                            (size_t)(n_tokens - 1) * cur->nb[1]);
    }

    // lm_head
    cur = ggml_mul_mat(ctx, w.output, cur);  // [n_vocab, 1]

    // Logit softcapping
    if (w.final_logit_softcap > 0.0f) {
        cur = ggml_scale(ctx, cur, 1.0f / w.final_logit_softcap);
        cur = ggml_tanh(ctx, cur);
        cur = ggml_scale(ctx, cur, w.final_logit_softcap);
    }

    ggml_set_output(cur);
    ggml_build_forward_expand(gf, cur);

    // Allocate
    static ggml_gallocr_t galloc = nullptr;
    if (!galloc) galloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
    if (!ggml_gallocr_alloc_graph(galloc, gf)) {
        std::fprintf(stderr, "gemma4_step: gallocr_alloc_graph failed\n");
        ggml_free(ctx);
        return false;
    }

    // Set input data
    ggml_backend_tensor_set(ie, embed, 0, ggml_nbytes(ie));
    std::vector<int32_t> pos((size_t)n_tokens);
    for (int i = 0; i < n_tokens; ++i) pos[i] = kv_start + i;
    ggml_backend_tensor_set(pp, pos.data(), 0, ggml_nbytes(pp));

    // Set token IDs for per-layer embedding
    if (tok_ids && token_ids) {
        ggml_backend_tensor_set(tok_ids, token_ids, 0, (size_t)n_tokens * sizeof(int32_t));
    }

    // Causal mask (full attention) — padded positions are masked with -inf
    std::vector<float> mfull((size_t)kv_len_padded * n_tokens, -INFINITY);
    for (int q = 0; q < n_tokens; ++q) {
        const int abs_q = kv_start + q;
        for (int k = 0; k <= abs_q && k < kv_len_raw; ++k) {
            mfull[(size_t)q * kv_len_padded + k] = 0.0f;
        }
    }
    ggml_backend_tensor_set(mk_full, mfull.data(), 0, ggml_nbytes(mk_full));

    // SWA mask — padded positions are masked with -inf
    std::vector<float> mswa((size_t)kv_len_padded * n_tokens, -INFINITY);
    const int W = w.sliding_window;
    for (int q = 0; q < n_tokens; ++q) {
        const int abs_q = kv_start + q;
        const int win_lo = std::max(0, abs_q - W + 1);
        for (int k = win_lo; k <= abs_q && k < kv_len_raw; ++k) {
            mswa[(size_t)q * kv_len_padded + k] = 0.0f;
        }
    }
    ggml_backend_tensor_set(mk_swa, mswa.data(), 0, ggml_nbytes(mk_swa));

    // Compute
    if (ggml_backend_graph_compute(backend, gf) != GGML_STATUS_SUCCESS) {
        std::fprintf(stderr, "gemma4_step: graph_compute failed\n");
        ggml_free(ctx);
        return false;
    }

    // Read logits
    out_logits.resize((size_t)w.n_vocab);
    ggml_backend_tensor_get(cur, out_logits.data(), 0,
                             out_logits.size() * sizeof(float));

    cache.cur_pos = kv_len_raw;
    ggml_free(ctx);
    return true;
}

// ── gemma4_verify_batch ─────────────────────────────────────────────────
// Like gemma4_step but returns argmax for ALL token positions (not just last).

bool gemma4_verify_batch(
    ggml_backend_t          backend,
    const Gemma4Weights &   w,
    Gemma4Cache &           cache,
    const float *           embed,
    const int32_t *         token_ids,
    int                     n_tokens,
    int                     kv_start,
    std::vector<int32_t> &  out_argmax)
{
    ggml_init_params ip{};
    ip.mem_size = ggml_tensor_overhead() * 16384 + ggml_graph_overhead() + 16 * 1024 * 1024;
    ip.no_alloc = true;
    ggml_context * ctx = ggml_init(ip);
    ggml_cgraph * gf = ggml_new_graph_custom(ctx, 16384, false);

    // Input tensors
    ggml_tensor * ie = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, w.n_embd, n_tokens);
    ggml_set_input(ie);
    ggml_tensor * pp = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, n_tokens);
    ggml_set_input(pp);

    // Token IDs for per-layer embedding
    ggml_tensor * tok_ids = nullptr;
    if (token_ids && w.per_layer_tok_embd && w.per_layer_model_proj && w.n_embd_per_layer > 0) {
        tok_ids = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, n_tokens);
        ggml_set_input(tok_ids);
    }

    // Attention masks (padded)
    const int kv_len_raw = kv_start + n_tokens;
    const int kv_len_padded = (kv_len_raw + 255) & ~255;
    ggml_tensor * mk_full = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, kv_len_padded, n_tokens, 1, 1);
    ggml_set_input(mk_full);
    ggml_tensor * mk_full_f16 = ggml_cast(ctx, mk_full, GGML_TYPE_F16);
    ggml_tensor * mk_swa = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, kv_len_padded, n_tokens, 1, 1);
    ggml_set_input(mk_swa);
    ggml_tensor * mk_swa_f16 = ggml_cast(ctx, mk_swa, GGML_TYPE_F16);

    // Per-layer embedding computation (same as gemma4_step)
    ggml_tensor * per_layer_all = nullptr;
    if (tok_ids) {
        const int D = w.n_embd_per_layer;
        const int L = w.n_layer;
        ggml_tensor * inp_pl = ggml_get_rows(ctx, w.per_layer_tok_embd, tok_ids);
        inp_pl = ggml_reshape_3d(ctx, inp_pl, D, L, n_tokens);
        inp_pl = ggml_scale(ctx, inp_pl, std::sqrt((float)D));
        ggml_tensor * proj = ggml_mul_mat(ctx, w.per_layer_model_proj, ie);
        proj = ggml_scale(ctx, proj, 1.0f / std::sqrt((float)w.n_embd));
        proj = ggml_reshape_3d(ctx, proj, D, L, n_tokens);
        proj = ggml_rms_norm(ctx, proj, w.norm_eps);
        ggml_tensor * norm_w = ggml_reshape_2d(ctx, w.per_layer_proj_norm, D, L);
        proj = ggml_mul(ctx, proj, norm_w);
        per_layer_all = ggml_add(ctx, proj, inp_pl);
        per_layer_all = ggml_scale(ctx, per_layer_all, 1.0f / std::sqrt(2.0f));
        per_layer_all = ggml_cont(ctx, ggml_permute(ctx, per_layer_all, 0, 2, 1, 3));
    }

    // Build graph (all layers)
    ggml_tensor * cur = ie;
    for (int il = 0; il < w.n_layer; ++il) {
        ggml_tensor * pl_input = nullptr;
        if (per_layer_all) {
            pl_input = gemma4_view_2d_slice(ctx, per_layer_all, il);
        }
        cur = build_gemma4_layer(ctx, gf, w, cache, il, cur, pp,
                                   mk_full_f16, mk_swa_f16, pl_input,
                                   kv_start, n_tokens);
    }

    // Final norm
    cur = gemma4_rms_norm_mul(ctx, cur, w.out_norm, w.norm_eps);

    // lm_head for ALL tokens (no slicing)
    cur = ggml_mul_mat(ctx, w.output, cur);  // [n_vocab, n_tokens]

    // Logit softcapping
    if (w.final_logit_softcap > 0.0f) {
        cur = ggml_scale(ctx, cur, 1.0f / w.final_logit_softcap);
        cur = ggml_tanh(ctx, cur);
        cur = ggml_scale(ctx, cur, w.final_logit_softcap);
    }

    // Argmax per token
    cur = ggml_argmax(ctx, cur);  // [n_tokens]
    ggml_set_output(cur);
    ggml_build_forward_expand(gf, cur);

    // Allocate
    static ggml_gallocr_t galloc_verify = nullptr;
    if (!galloc_verify) galloc_verify = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
    if (!ggml_gallocr_alloc_graph(galloc_verify, gf)) {
        std::fprintf(stderr, "gemma4_verify_batch: gallocr_alloc_graph failed\n");
        ggml_free(ctx);
        return false;
    }

    // Set inputs
    ggml_backend_tensor_set(ie, embed, 0, ggml_nbytes(ie));
    std::vector<int32_t> pos((size_t)n_tokens);
    for (int i = 0; i < n_tokens; ++i) pos[i] = kv_start + i;
    ggml_backend_tensor_set(pp, pos.data(), 0, ggml_nbytes(pp));

    if (tok_ids && token_ids) {
        ggml_backend_tensor_set(tok_ids, token_ids, 0, (size_t)n_tokens * sizeof(int32_t));
    }

    // Masks
    std::vector<float> mfull((size_t)kv_len_padded * n_tokens, -INFINITY);
    for (int q = 0; q < n_tokens; ++q) {
        const int abs_q = kv_start + q;
        for (int k = 0; k <= abs_q && k < kv_len_raw; ++k) {
            mfull[(size_t)q * kv_len_padded + k] = 0.0f;
        }
    }
    ggml_backend_tensor_set(mk_full, mfull.data(), 0, ggml_nbytes(mk_full));

    std::vector<float> mswa((size_t)kv_len_padded * n_tokens, -INFINITY);
    const int W = w.sliding_window;
    for (int q = 0; q < n_tokens; ++q) {
        const int abs_q = kv_start + q;
        const int win_lo = std::max(0, abs_q - W + 1);
        for (int k = win_lo; k <= abs_q && k < kv_len_raw; ++k) {
            mswa[(size_t)q * kv_len_padded + k] = 0.0f;
        }
    }
    ggml_backend_tensor_set(mk_swa, mswa.data(), 0, ggml_nbytes(mk_swa));

    // Compute
    if (ggml_backend_graph_compute(backend, gf) != GGML_STATUS_SUCCESS) {
        std::fprintf(stderr, "gemma4_verify_batch: graph_compute failed\n");
        ggml_free(ctx);
        return false;
    }

    // Read argmax
    out_argmax.resize(n_tokens);
    ggml_backend_tensor_get(cur, out_argmax.data(), 0, sizeof(int32_t) * n_tokens);

    cache.cur_pos = kv_len_raw;
    ggml_free(ctx);
    return true;
}

// ── gemma4_project_hidden ───────────────────────────────────────────────
// Runs out_norm + lm_head + softcap + argmax on external hidden states.

bool gemma4_project_hidden(
    ggml_backend_t          backend,
    const Gemma4Weights &   w,
    const float *           hidden,
    int                     n_tokens,
    std::vector<int32_t> &  out_tokens)
{
    ggml_init_params ip{};
    ip.mem_size = ggml_tensor_overhead() * 64 + ggml_graph_overhead() + 1024 * 1024;
    ip.no_alloc = true;
    ggml_context * ctx = ggml_init(ip);
    ggml_cgraph * gf = ggml_new_graph(ctx);

    // Input: hidden states [n_embd, n_tokens]
    ggml_tensor * inp = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, w.n_embd, n_tokens);
    ggml_set_input(inp);

    // out_norm + lm_head
    ggml_tensor * cur = gemma4_rms_norm_mul(ctx, inp, w.out_norm, w.norm_eps);
    cur = ggml_mul_mat(ctx, w.output, cur);  // [n_vocab, n_tokens]

    // Logit softcapping
    if (w.final_logit_softcap > 0.0f) {
        cur = ggml_scale(ctx, cur, 1.0f / w.final_logit_softcap);
        cur = ggml_tanh(ctx, cur);
        cur = ggml_scale(ctx, cur, w.final_logit_softcap);
    }

    // Argmax
    cur = ggml_argmax(ctx, cur);  // [n_tokens]
    ggml_set_output(cur);
    ggml_build_forward_expand(gf, cur);

    // Allocate
    static ggml_gallocr_t galloc_proj = nullptr;
    if (!galloc_proj) galloc_proj = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
    if (!ggml_gallocr_alloc_graph(galloc_proj, gf)) {
        std::fprintf(stderr, "gemma4_project_hidden: gallocr_alloc_graph failed\n");
        ggml_free(ctx);
        return false;
    }

    // Set input
    ggml_backend_tensor_set(inp, hidden, 0, sizeof(float) * (size_t)n_tokens * w.n_embd);

    // Compute
    if (ggml_backend_graph_compute(backend, gf) != GGML_STATUS_SUCCESS) {
        std::fprintf(stderr, "gemma4_project_hidden: graph_compute failed\n");
        ggml_free(ctx);
        return false;
    }

    // Read result
    out_tokens.resize(n_tokens);
    ggml_backend_tensor_get(cur, out_tokens.data(), 0, sizeof(int32_t) * n_tokens);

    ggml_free(ctx);
    return true;
}

}  // namespace dflash::common
