#pragma once
// Admission gate: reject oversized requests with HTTP 400.
// When compression is enabled, lets oversized requests through — post-compress check is the real gate.

inline bool should_reject_oversized(int prompt_tokens, int max_output,
                                    int max_ctx, bool compression_enabled)
{
    if (prompt_tokens + max_output <= max_ctx) {
        return false;  // fits — accept regardless of compression
    }
    // Oversized: only reject if compression cannot help.
    return !compression_enabled;
}
