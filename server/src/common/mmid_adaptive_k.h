#pragma once
// [TAG_MMID_ADAPTIVE_K] per-token expert-count gating for small MUL_MAT_ID
// batches (speculative-verify sized, 2..16 tokens). Contract with the grouped
// MUL_MAT_ID path in ggml-cuda mmvq.cu (DFLASH_MMID_GROUPED=1): when a router
// ids tensor's ->extra points at an mmid_gate_extra, the grouped prep kernel
// keeps each token's leading experts until their cumulative combine weight
// reaches tau, sentinels the rest of the token's ids to -1 (skipped, exact
// zero contribution) and renormalizes the kept weights in place.
//   DFLASH_ADAPTIVE_K_TAU=<0..1>   enables (0/unset = off)
//   DFLASH_ADAPTIVE_K_DENSE=<csv>  layers kept at full top-k. DFlash capture
//                                  layers MUST stay dense so drafter
//                                  conditioning features are unchanged.
#include "ggml.h"

#include <cstdint>
#include <cstdlib>
#include <string>

struct mmid_gate_extra {
    uint32_t            magic;   // MMID_GATE_MAGIC
    float               tau;
    const ggml_tensor * weights; // [n_used, n_tokens] f32 combine weights
};
#define MMID_GATE_MAGIC 0x4D474154u

// il < 0 = layer index unknown for this family: the dense-layer list cannot
// be applied, every MoE layer is gated. Only builds run this; the extra must
// outlive the graph, so it is deliberately never freed.
inline void mmid_adaptive_k_attach(ggml_tensor * ids, const ggml_tensor * weights,
                                   int n_tokens, int il, const char * dense_default) {
    static const float tau = []() {
        const char * e = std::getenv("DFLASH_ADAPTIVE_K_TAU");
        return e ? (float) std::atof(e) : 0.0f;
    }();
    if (tau <= 0.0f || n_tokens < 2 || n_tokens > 16 || ids == nullptr || weights == nullptr) {
        return;
    }
    if (il >= 0) {
        const char * e = std::getenv("DFLASH_ADAPTIVE_K_DENSE");
        const std::string str = e ? e : (dense_default ? dense_default : "");
        size_t pos = 0;
        while (pos < str.size()) {
            const size_t q = str.find(',', pos);
            const std::string tok = str.substr(pos, q == std::string::npos ? std::string::npos : q - pos);
            if (!tok.empty() && std::atoi(tok.c_str()) == il) {
                return;
            }
            if (q == std::string::npos) break;
            pos = q + 1;
        }
    }
    ids->extra = new mmid_gate_extra{MMID_GATE_MAGIC, tau, weights};
}
