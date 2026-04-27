// Sprint 004 Phase 2: VRAM-resident shadow buffer for hybrid recurrent state.
//
// Replaces the host-RAM byte-copy path of llama_state_seq_*_ext for the
// PARTIAL_ONLY (recurrent state) snapshot used by speculative decoding.
// Shadow tensors are allocated as raw CUDA device memory and snapshot/restore
// uses cudaMemcpyDeviceToDevice — bandwidth-bounded by HBM (~3 TB/s on RTX 5090)
// rather than PCIe (~14 GB/s host pageable).
//
// Use is restricted to:
//   * hybrid models with a recurrent memory submodule
//   * single-sequence usage (the checkpoint copies entire layer tensors,
//     including all cells; on multi-seq caches this would clobber other seqs)
//
// On non-hybrid models or non-CUDA builds, is_valid() returns false and
// save/restore are no-ops returning 0.

#pragma once

#include <cstddef>
#include <vector>

struct llama_context;
class llama_memory_recurrent;

class vram_seq_checkpoint {
public:
    // Discover the recurrent memory submodule and allocate device shadow buffers.
    // After construction, check is_valid().
    explicit vram_seq_checkpoint(llama_context * ctx);
    ~vram_seq_checkpoint();

    vram_seq_checkpoint(const vram_seq_checkpoint &) = delete;
    vram_seq_checkpoint & operator=(const vram_seq_checkpoint &) = delete;

    // True if shadow buffers were allocated. False on non-hybrid models or
    // on builds without GGML_USE_CUDA.
    bool is_valid() const { return mem_recr != nullptr && total_bytes > 0; }

    // Total bytes allocated across all shadow buffers.
    size_t size_bytes() const { return total_bytes; }

    // Save the current recurrent state into the shadow buffers (D->D copy).
    // Returns total bytes copied. Calls cudaDeviceSynchronize() before returning.
    size_t save();

    // Restore the recurrent state from the shadow buffers (D->D copy).
    // Returns total bytes copied. Calls cudaDeviceSynchronize() before returning.
    size_t restore();

private:
    llama_memory_recurrent * mem_recr = nullptr;
    std::vector<void *>      r_shadows;
    std::vector<void *>      s_shadows;
    std::vector<size_t>      r_sizes;
    std::vector<size_t>      s_sizes;
    size_t                   total_bytes = 0;
};
