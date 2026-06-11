// Unit tests for should_reject_oversized — pure, GPU-free.
// Reject iff prompt+max_output > max_ctx AND compression is NOT enabled.
// Build: /usr/bin/g++-11 -std=gnu++17 -O0 -I server/src -o /tmp/test_admission server/test/test_admission.cpp && /tmp/test_admission
#include "server/admission.h"

#include <cstdio>

static int test_failures = 0;
static int test_count    = 0;

#define TEST_ASSERT(expr) do {                                  \
    test_count++;                                               \
    if (!(expr)) {                                              \
        test_failures++;                                        \
        std::fprintf(stderr, "  FAIL: %s:%d: %s\n",            \
                     __FILE__, __LINE__, #expr);                \
    }                                                           \
} while (0)

#define RUN_TEST(fn) do {                                       \
    std::fprintf(stderr, "  %s ...", #fn);                      \
    int before = test_failures;                                 \
    fn();                                                       \
    std::fprintf(stderr, (test_failures == before) ? " ok\n" : "\n"); \
} while (0)

// 100+100 <= 1024, no compression -> accept
static void test_small_prompt_no_compression_accepts() {
    TEST_ASSERT(!should_reject_oversized(100, 100, 1024, false));
}

// 900+200 > 1024, no compression -> reject (hard gate preserved)
static void test_oversized_no_compression_rejects() {
    TEST_ASSERT(should_reject_oversized(900, 200, 1024, false));
}

// 167000+2048 > 65536, compression enabled -> accept (post-compress check is gate)
static void test_oversized_with_compression_accepts() {
    TEST_ASSERT(!should_reject_oversized(167000, 2048, 65536, true));
}

// prompt+max_output == max_ctx is not oversized -> accept
static void test_exactly_at_limit_accepts() {
    TEST_ASSERT(!should_reject_oversized(1024, 0, 1024, false));
    TEST_ASSERT(!should_reject_oversized(512, 512, 1024, false));
}

// 1025 > 1024, no compression -> reject
static void test_one_over_limit_no_compression_rejects() {
    TEST_ASSERT(should_reject_oversized(1025, 0, 1024, false));
}

// 1025 > 1024, compression enabled -> accept
static void test_one_over_limit_with_compression_accepts() {
    TEST_ASSERT(!should_reject_oversized(1025, 0, 1024, true));
}

int main() {
    std::fprintf(stderr, "=== test_admission ===\n");
    RUN_TEST(test_small_prompt_no_compression_accepts);
    RUN_TEST(test_oversized_no_compression_rejects);
    RUN_TEST(test_oversized_with_compression_accepts);
    RUN_TEST(test_exactly_at_limit_accepts);
    RUN_TEST(test_one_over_limit_no_compression_rejects);
    RUN_TEST(test_one_over_limit_with_compression_accepts);
    std::fprintf(stderr, "\n%d tests, %d failures\n", test_count, test_failures);
    return (test_failures == 0) ? 0 : 1;
}
