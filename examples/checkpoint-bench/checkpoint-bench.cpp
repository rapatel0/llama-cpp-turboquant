// Sprint 004 Phase 2: snapshot save/restore wallclock benchmark.
//
// Loads a model, prefills N tokens, then times:
//   - PARTIAL_ONLY checkpoint (recurrent state only, used by speculative)
//   - full checkpoint (all state)
// Each timing reports save (get_data_ext) and restore (set_data_ext) round-trips.
//
// Output is a single JSON object on stdout for easy aggregation.
//
// Usage:
//   llama-checkpoint-bench -m model.gguf -c 65536 -ngl 99 -fa 1 \
//     --cache-type-k iso3 --cache-type-v iso3 --ckpt-tokens 65000

#include "arg.h"
#include "common.h"
#include "llama.h"

#include "../../src/llama-vram-checkpoint.h"

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

using clk = std::chrono::high_resolution_clock;

static double us_since(const clk::time_point & t0) {
    auto dt = clk::now() - t0;
    return std::chrono::duration_cast<std::chrono::microseconds>(dt).count();
}

int main(int argc, char ** argv) {
    common_params params;
    params.prompt = "Hello";
    params.n_predict = 0;
    params.n_batch = 2048;

    // Custom flag: --ckpt-tokens N — number of synthetic tokens to prefill.
    // Parsed manually to avoid wrestling with arg.cpp.
    int ckpt_tokens = 1024;
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--ckpt-tokens") == 0 && i + 1 < argc) {
            ckpt_tokens = atoi(argv[++i]);
            // remove these two args so common_params_parse doesn't choke
            for (int j = i - 1; j + 2 < argc; j++) argv[j] = argv[j + 2];
            argc -= 2;
            i -= 2;
        }
    }

    common_init();

    if (!common_params_parse(argc, argv, params, LLAMA_EXAMPLE_COMMON)) {
        return 1;
    }
    params.kv_unified = true;

    auto llama_init = common_init_from_params(params);
    auto * model = llama_init->model();
    auto * ctx   = llama_init->context();
    if (!model || !ctx) {
        fprintf(stderr, "failed to init\n");
        return 1;
    }

    const auto * vocab = llama_model_get_vocab(model);
    const llama_token bos = llama_vocab_bos(vocab);

    // Build a synthetic prompt of `ckpt_tokens` tokens. Using BOS plus a sequence of
    // small token IDs is acceptable: PPL is irrelevant; we only care about KV occupancy.
    std::vector<llama_token> prompt_toks;
    prompt_toks.reserve(ckpt_tokens);
    if (bos >= 0) prompt_toks.push_back(bos);
    while ((int) prompt_toks.size() < ckpt_tokens) {
        // Use small token IDs in [10, 200] to avoid embedding/special-token issues
        prompt_toks.push_back(10 + ((int) prompt_toks.size() % 191));
    }

    const int n_ctx = llama_n_ctx(ctx);
    if (ckpt_tokens > n_ctx) {
        fprintf(stderr, "ckpt_tokens %d exceeds model n_ctx %d\n", ckpt_tokens, n_ctx);
        return 1;
    }

    // Prefill in batches
    fprintf(stderr, "prefilling %d tokens (ctx=%d)\n", ckpt_tokens, n_ctx);
    auto t_pre = clk::now();
    int n_past = 0;
    while (n_past < (int) prompt_toks.size()) {
        const int n_eval = std::min((int) prompt_toks.size() - n_past, params.n_batch);
        llama_batch batch = llama_batch_init(n_eval, 0, 1);
        for (int j = 0; j < n_eval; j++) {
            common_batch_add(batch, prompt_toks[n_past + j], n_past + j, {0}, j == n_eval - 1);
        }
        if (llama_decode(ctx, batch)) {
            fprintf(stderr, "decode failed at n_past=%d\n", n_past);
            llama_batch_free(batch);
            return 1;
        }
        llama_batch_free(batch);
        n_past += n_eval;
    }
    double prefill_ms = us_since(t_pre) / 1000.0;
    fprintf(stderr, "prefill complete in %.1f ms\n", prefill_ms);

    auto bench_one = [&](uint32_t flags, const char * label) {
        const int seq_id = 0;
        // Size query
        auto t0 = clk::now();
        size_t sz = llama_state_seq_get_size_ext(ctx, seq_id, flags);
        double size_us = us_since(t0);
        if (sz == 0) {
            fprintf(stderr, "%s: get_size_ext returned 0 — no data for this flag combo\n", label);
            return;
        }
        std::vector<uint8_t> buf(sz);

        // Save (get_data_ext) — 5 trials, take min
        double save_us_min = 1e18;
        for (int trial = 0; trial < 5; trial++) {
            auto t = clk::now();
            size_t n = llama_state_seq_get_data_ext(ctx, buf.data(), sz, seq_id, flags);
            double dt = us_since(t);
            if (n != sz) {
                fprintf(stderr, "%s: get_data_ext returned %zu, expected %zu\n", label, n, sz);
                return;
            }
            if (dt < save_us_min) save_us_min = dt;
        }

        // Restore (set_data_ext) — 5 trials, take min. Note: set_data_ext doesn't unconsume the seq,
        // so we don't need a clear between trials; the same data is just re-applied.
        double restore_us_min = 1e18;
        for (int trial = 0; trial < 5; trial++) {
            auto t = clk::now();
            size_t n = llama_state_seq_set_data_ext(ctx, buf.data(), sz, seq_id, flags);
            double dt = us_since(t);
            if (n != sz) {
                fprintf(stderr, "%s: set_data_ext returned %zu, expected %zu\n", label, n, sz);
                return;
            }
            if (dt < restore_us_min) restore_us_min = dt;
        }

        printf("\"%s\":{\"size_bytes\":%zu,\"size_query_us\":%.1f,\"save_us_min\":%.1f,\"restore_us_min\":%.1f,\"save_plus_restore_ms\":%.3f},",
               label, sz, size_us, save_us_min, restore_us_min, (save_us_min + restore_us_min) / 1000.0);
        fflush(stdout);
    };

    printf("{\"ckpt_tokens\":%d,\"prefill_ms\":%.1f,", ckpt_tokens, prefill_ms);
    bench_one(LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY, "partial");
    bench_one(0, "full");

    // VRAM-resident shadow snapshot (D->D copies via cudaMemcpy).
    {
        vram_seq_checkpoint vc(ctx);
        if (!vc.is_valid()) {
            fprintf(stderr, "vram_seq_checkpoint: not a hybrid model or CUDA disabled — skipping\n");
        } else {
            const size_t sz = vc.size_bytes();
            // Save: 5 trials, take min
            double save_us_min = 1e18;
            for (int trial = 0; trial < 5; ++trial) {
                auto t = clk::now();
                size_t n = vc.save();
                double dt = us_since(t);
                if (n != sz) {
                    fprintf(stderr, "vram save: returned %zu, expected %zu\n", n, sz);
                }
                if (dt < save_us_min) save_us_min = dt;
            }
            // Restore: 5 trials, take min
            double restore_us_min = 1e18;
            for (int trial = 0; trial < 5; ++trial) {
                auto t = clk::now();
                size_t n = vc.restore();
                double dt = us_since(t);
                if (n != sz) {
                    fprintf(stderr, "vram restore: returned %zu, expected %zu\n", n, sz);
                }
                if (dt < restore_us_min) restore_us_min = dt;
            }
            printf("\"vram_partial\":{\"size_bytes\":%zu,\"save_us_min\":%.1f,\"restore_us_min\":%.1f,\"save_plus_restore_ms\":%.4f},",
                   sz, save_us_min, restore_us_min, (save_us_min + restore_us_min) / 1000.0);
            fflush(stdout);

            // Bit-exactness check: prove that save->mutate->restore reproduces
            // the pre-mutation tensor bytes byte-for-byte. We use the upstream
            // host-RAM PARTIAL_ONLY path as our ground-truth byte stream.
            const int seq_id = 0;
            const size_t pre_sz = llama_state_seq_get_size_ext(ctx, seq_id, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY);
            std::vector<uint8_t> pre_bytes(pre_sz);
            llama_state_seq_get_data_ext(ctx, pre_bytes.data(), pre_sz, seq_id, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY);

            // 1) Snapshot tensors into VRAM shadow
            vc.save();

            // 2) Mutate the recurrent state by decoding 16 more tokens
            //    (simulates a draft block in speculative verification).
            const int n_mut = 16;
            llama_batch mb = llama_batch_init(n_mut, 0, 1);
            for (int j = 0; j < n_mut; ++j) {
                common_batch_add(mb, prompt_toks[j % prompt_toks.size()], n_past + j, {0}, j == n_mut - 1);
            }
            if (llama_decode(ctx, mb)) {
                fprintf(stderr, "correctness: decode-mutate failed\n");
                llama_batch_free(mb);
                return 1;
            }
            llama_batch_free(mb);

            // 3) Verify the mutation actually changed the state
            std::vector<uint8_t> post_bytes(pre_sz);
            llama_state_seq_get_data_ext(ctx, post_bytes.data(), pre_sz, seq_id, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY);
            const bool mutated = (post_bytes != pre_bytes);

            // 4) Restore from VRAM shadow
            vc.restore();
            // Roll back the cell metadata that decode advanced. The vram shadow
            // restore only handles tensor data; head/used live in CPU bookkeeping.
            llama_memory_seq_rm(llama_get_memory(ctx), seq_id, n_past, -1);

            // 5) Compare restored state to pre-mutation reference
            std::vector<uint8_t> after_bytes(pre_sz);
            llama_state_seq_get_data_ext(ctx, after_bytes.data(), pre_sz, seq_id, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY);

            // Bit-exactness check on tensor data only (skip the metadata header
            // which may differ in head/used after seq_rm). The format is:
            //   [meta header] then tensor data in order.
            // For a strict tensor-only compare, we'd need to know the header
            // length. Instead, we check whether the after_bytes match pre_bytes
            // exactly; if they do, both metadata and tensors are bit-equal.
            const bool exact = (after_bytes == pre_bytes);

            // Also do a tail-byte compare which excludes the small metadata
            // header at the start (typically <100 bytes) and isolates the
            // tensor data which is what vram_seq_checkpoint actually preserves.
            const size_t tail_offset = pre_sz / 100;  // skip first 1% (covers metadata)
            const bool tail_exact = (pre_sz > tail_offset) &&
                std::equal(pre_bytes.begin() + tail_offset, pre_bytes.end(),
                           after_bytes.begin() + tail_offset);

            printf("\"vram_correctness\":{\"mutated_state_diverged\":%s,\"after_restore_full_match\":%s,\"after_restore_tail_match\":%s},",
                   mutated ? "true" : "false",
                   exact ? "true" : "false",
                   tail_exact ? "true" : "false");
            fflush(stdout);
        }
    }

    printf("\"_end\":1}\n");

    return 0;
}
