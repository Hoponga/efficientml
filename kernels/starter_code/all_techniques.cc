#include <assert.h>
#include <pthread.h>
#include <stdio.h>

#include <cmath>
#include <cstdlib>

#include "../matmul.h"
#include "common.h"

#ifdef QM_ARM
#include <arm_neon.h>
#endif
#ifdef QM_x86
#include <immintrin.h>
#endif
// Cache blocking values 
static constexpr int KC = 128; 
static constexpr int NC = 32; 
static constexpr int MC = 8; 

struct thread_arg {
    int start_j, end_j;
    const struct matmul_params *params;
    int8_t *scratch; // Scratch space for this thread, lives in L1? 
};

struct w4a8_thread_args {
    int start_j, end_j;
    const struct matmul_params *params;
};



// pack B matrix into the scratch for this thread 
static inline void pack_B_panel(int8_t *dst, const uint8_t *src, int k, int n_cols) {
    const uint8x16_t mask_low4 = vdupq_n_u8(0x0f); 
    const int8x16_t offs = vdupq_n_s8(8); 

    for (int j = 0; j < n_cols; j++) {
        const uint8_t *col = src + (size_t) j * k/2; 
        int8_t* out = dst + (size_t)j * KC * 2; 
        for (int b = 0; b < KC; b += 32) {
            uint8x16_t v0 = vld1q_u8(col + b); 
            uint8x16_t v1 = vld1q_u8(col + b + 16); // unroll by 2 

            int8x16_t lo0 = vsubq_s8(vreinterpretq_s8_u8(vandq_u8(v0, mask_low4)), offs); 
            int8x16_t lo1 = vsubq_s8(vreinterpretq_s8_u8(vandq_u8(v1, mask_low4)), offs); 

            vst1q_s8(out, lo0); 
            vst1q_s8(out + 16, lo1); 

            v0 = vshrq_n_u8(v0, 4); 
            v1 = vshrq_n_u8(v1, 4); 
            int8x16_t hi0 = vsubq_s8(vreinterpretq_s8_u8(v0), offs); 
            int8x16_t hi1 = vsubq_s8(vreinterpretq_s8_u8(v1), offs); 
            vst1q_s8(out + 32, hi0); 
            vst1q_s8(out + 48, hi1); 
            out += 64; 



        }

    }

}

static inline void kernel_8x4(float *Cptr, int ldc, const int8_t *Aptr, int lda, const int8_t *Bptr, int ldb, const float *Sa, const float *Sw, int k_left) {
    float32x4_t acc[MC]; 
    for (int i = 0; i < MC; i++) acc[i] = vdupq_n_f32(0.f); 

    // k loop -- k multiple of 32, treat two 16-byte dot products as 32 multiplies per iteration 
    for (int k = 0, blk = 0; k < k_left; k+= 32, ++blk) {
        const int8_t *B0 = Bptr + k*4; // 4 columns contiguous, already packed 
        int8x16_t b0 = vld1q_s8(B0); 
        int8x16_t b1 = vld1q_s8(B0 + 16); 
        const float scl = Sa[blk]*Sw[blk]; 

        for (int i = 0; i < MC; i++) {
            const int8_t *Arow = Aptr + (size_t)i*lda + k; 
            int8x16_t a0 = vld1q_s8(Arow); 
            int8x16_t a1 = vld1q_s8(Arow + 16); 

            int32x4_t dot = vdotq_s32(vdupq_n_s32(0), a0, b0); 
            dot = vdotq_s32(dot, a1, b1); 

            acc[i] = vfmaq_n_f32(acc[i], vcvtq_f32_s32(dot), scl); 
            

        }
    }
    for (int i = 0; i < MC; i++) {
        vst1q_f32(Cptr + (size_t)i*ldc, vaddq_f32(vld1q_f32(Cptr + (size_t)i*ldc), acc[i])); 


    }
}


// worker function 

static void *worker(void *arg_) {
    thread_arg *arg = static_cast<thread_arg*>(arg_); 

    const struct matmul_params *params = arg->params;
    const struct matrix *A = &params->A, *B = &params->B, *C = &params->C;
    const int block_size = params->block_size;

    const int m = C->row, n = C->column, k = A->column;

    int8_t *packedB = arg->scratch; 
    for (int jc = arg->start_j; jc < arg->end_j; jc += NC) {
        const int jlim = std::min(NC, arg->end_j - jc); 
        // inner dimension of matmul 
        for (int pc = 0; pc < k; pc += KC) {
            const int klim = std::min(KC, k - pc); 
            pack_B_panel(packedB, &B->int4_data_ptr[(size_t)jc * k / 2 + pc / 2], klim, jlim); 
            for (int ic = 0; ic < m; ic += MC) {
                const int ilim = std::min(MC, m - ic); 

                // potential TODO: memset the C-tile to zero? 
                const int8_t *Aptr = &A->int8_data_ptr[(size_t)ic * k + pc]; 
                const float *Sa = &params->A_scales[(size_t)ic * k / 32 + pc / 32]; 
                const float *Sw = &params->scales[(size_t) (jc) * k/32 + pc / 32]; 

                // micro tiles: 4 columns each  (over the N dimension of the output matrix)
                for (int j = 0; j < jlim; j += 4) {
                    kernel_8x4(&C->data_ptr[(size_t)ic * n + jc + j], n, Aptr, k, packedB + (size_t)j *KC*2, jlim*KC*2, Sa, Sw + j*k/32, klim); 

                }
            }
        }
    }


    return nullptr; 

}

namespace matmul {

// CACHE BLOCKED IMPLEMENTATION INCLUDING PER-THREAD SCRATCH BUFFER 
void MatmulOperator::mat_mul_all_techniques(struct matmul_params *params) {
    int i, j, k;
    const struct matrix *A = &params->A, *B = &params->B, *C = &params->C;
    const int block_size = params->block_size;
    float *scale = params->scales, *offset = params->offset;

    assert(params->block_size % 32 == 0);  // support block size to be multiples of 32
    assert(A->row == C->row);              // support block size to be multiples of 32

    quantize_fp32_to_int8(A->data_ptr, A->int8_data_ptr, params->A_scales, A->row * A->column, block_size);

    const int num_thread = 8;
    pthread_t thread_pool[num_thread];
    struct thread_arg targs[num_thread];

    int8_t *scratch[num_thread]; 
    for (int t = 0; t < num_thread; ++t) {
        posix_memalign(reinterpret_cast<void**>(&scratch[t]), 64, NC * KC * 2); 
    }
    assert(params->block_size == 32);  // support block size 32 for now

     int thread_block_size = C->column / num_thread; 

    // TODO: Thread creation
    for (int i = 0; i < num_thread; ++i) {
        targs[i] = {i*thread_block_size, std::min(C->column, (i+1)*thread_block_size), params, scratch[i]}; 
        pthread_create(&thread_pool[i], nullptr, worker, &targs[i]); 

    }

    // TODO: Join threads

    for (int i = 0; i < num_thread; ++i) {
        pthread_join(thread_pool[i], nullptr); 
        free(scratch[i]); 


    }
};
}  // namespace matmul
