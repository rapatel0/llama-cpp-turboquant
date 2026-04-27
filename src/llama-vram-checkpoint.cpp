#include "llama-vram-checkpoint.h"

#include "llama.h"
#include "llama-memory-recurrent.h"
#include "llama-memory-hybrid.h"

#include "ggml.h"

#include <cstdio>

#ifdef GGML_USE_CUDA
#  include <cuda_runtime.h>
#endif

vram_seq_checkpoint::vram_seq_checkpoint(llama_context * ctx) {
#ifdef GGML_USE_CUDA
    auto * mem = llama_get_memory(ctx);
    if (!mem) {
        return;
    }
    auto * hybrid = dynamic_cast<llama_memory_hybrid *>(mem);
    if (!hybrid) {
        // Not a hybrid model — nothing to checkpoint at the recurrent layer.
        return;
    }
    mem_recr = hybrid->get_mem_recr();
    if (!mem_recr) {
        return;
    }

    r_shadows.assign(mem_recr->r_l.size(), nullptr);
    s_shadows.assign(mem_recr->s_l.size(), nullptr);
    r_sizes  .assign(mem_recr->r_l.size(), 0);
    s_sizes  .assign(mem_recr->s_l.size(), 0);

    for (size_t il = 0; il < mem_recr->r_l.size(); ++il) {
        if (!mem_recr->r_l[il]) continue;
        const size_t sz = ggml_nbytes(mem_recr->r_l[il]);
        cudaError_t err = cudaMalloc(&r_shadows[il], sz);
        if (err != cudaSuccess) {
            fprintf(stderr, "vram_seq_checkpoint: cudaMalloc(r_l[%zu], %zu) failed: %s\n",
                    il, sz, cudaGetErrorString(err));
            r_shadows[il] = nullptr;
            mem_recr = nullptr;
            return;
        }
        r_sizes[il] = sz;
        total_bytes += sz;
    }
    for (size_t il = 0; il < mem_recr->s_l.size(); ++il) {
        if (!mem_recr->s_l[il]) continue;
        const size_t sz = ggml_nbytes(mem_recr->s_l[il]);
        cudaError_t err = cudaMalloc(&s_shadows[il], sz);
        if (err != cudaSuccess) {
            fprintf(stderr, "vram_seq_checkpoint: cudaMalloc(s_l[%zu], %zu) failed: %s\n",
                    il, sz, cudaGetErrorString(err));
            s_shadows[il] = nullptr;
            mem_recr = nullptr;
            return;
        }
        s_sizes[il] = sz;
        total_bytes += sz;
    }
#else
    (void) ctx;
#endif
}

vram_seq_checkpoint::~vram_seq_checkpoint() {
#ifdef GGML_USE_CUDA
    for (auto * p : r_shadows) if (p) cudaFree(p);
    for (auto * p : s_shadows) if (p) cudaFree(p);
#endif
}

size_t vram_seq_checkpoint::save() {
#ifdef GGML_USE_CUDA
    if (!is_valid()) return 0;
    size_t total = 0;
    for (size_t il = 0; il < r_shadows.size(); ++il) {
        if (!r_shadows[il]) continue;
        cudaMemcpyAsync(r_shadows[il], mem_recr->r_l[il]->data,
                        r_sizes[il], cudaMemcpyDeviceToDevice);
        total += r_sizes[il];
    }
    for (size_t il = 0; il < s_shadows.size(); ++il) {
        if (!s_shadows[il]) continue;
        cudaMemcpyAsync(s_shadows[il], mem_recr->s_l[il]->data,
                        s_sizes[il], cudaMemcpyDeviceToDevice);
        total += s_sizes[il];
    }
    cudaDeviceSynchronize();
    return total;
#else
    return 0;
#endif
}

size_t vram_seq_checkpoint::restore() {
#ifdef GGML_USE_CUDA
    if (!is_valid()) return 0;
    size_t total = 0;
    for (size_t il = 0; il < r_shadows.size(); ++il) {
        if (!r_shadows[il]) continue;
        cudaMemcpyAsync(mem_recr->r_l[il]->data, r_shadows[il],
                        r_sizes[il], cudaMemcpyDeviceToDevice);
        total += r_sizes[il];
    }
    for (size_t il = 0; il < s_shadows.size(); ++il) {
        if (!s_shadows[il]) continue;
        cudaMemcpyAsync(mem_recr->s_l[il]->data, s_shadows[il],
                        s_sizes[il], cudaMemcpyDeviceToDevice);
        total += s_sizes[il];
    }
    cudaDeviceSynchronize();
    return total;
#else
    return 0;
#endif
}
