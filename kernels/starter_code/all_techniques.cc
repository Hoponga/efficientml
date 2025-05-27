#include <assert.h>
#include <pthread.h>
#include <stdio.h>

#include <cmath>
#include <cstdlib>
#include <cstring>

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
static constexpr int NC = 1; 
static constexpr int MC = 1; 

struct thread_arg {
    int start_j, end_j;
    const struct matmul_params *params;
};

static void* matmul_int8_int4_no_offset_optimized(void* arg_) {
    const struct thread_arg* args = (struct thread_arg*) arg_; 

    const struct matmul_params *params = args->params;
    int n = params->C.column, m = params->C.row, k = params->A.column, block_size = params->block_size;
    assert(params->block_size == 32);

    const int num_block = k / block_size;
    const uint8x16_t mask_low4bit = vdupq_n_u8(0xf);
    const int8x16_t offsets = vdupq_n_s8(8);
    
    
    
    for (int jc = args->start_j; jc < args->end_j; jc += NC) {
        for (int pc = 0; pc < k; pc += KC) {
            
                const int blk_lim = KC / params->block_size; 
                for (int ic = 0; ic < m; ic += MC) {
                    if (pc == 0) {
                        for (int ii = 0; ii < MC; ii++) {
                            memset(&params->C.data_ptr[(ic + ii)*n + jc], 0, NC * sizeof(float)); 

                        }

                    }

                    for (int j = 0; j < NC; j++) {
                        const uint8_t* w_col = &params->B.int4_data_ptr[(jc + j)*k / 2 + pc / 2]; 
                        const float* sw = &params->scales[(jc + j)*k / 32 + pc / 32]; 
                        for (int ii = 0; ii < MC; ++ii) {
                            const int8_t* a_row = &params->A.int8_data_ptr[(ic + ii)*k + pc]; 
                            const float* sa = &params->A_scales[(ic + ii)*k/32 + pc / 32]; 
                            float32x4_t sumv0 = vdupq_n_f32(0.0f);
                            float32x4_t sumv1 = vdupq_n_f32(0.0f);
                            float32x4_t sumv2 = vdupq_n_f32(0.0f);
                            float32x4_t sumv3 = vdupq_n_f32(0.0f);

                            const uint8_t* w_ptr = w_col; 
                            const int8_t* a_ptr = a_row; 
                            for (int blk = 0; blk < blk_lim; blk += 4) {
                                int32x4_t int_sum0 = vdupq_n_s32(0);
                                int32x4_t int_sum1 = vdupq_n_s32(0);
                                int32x4_t int_sum2 = vdupq_n_s32(0);
                                int32x4_t int_sum3 = vdupq_n_s32(0);
                                float s_0 = *sa++ * *sw++;
                                float s_1 = *sa++ * *sw++;
                                float s_2 = *sa++ * *sw++;
                                float s_3 = *sa++ * *sw++;

                                const uint8x16_t w0 = vld1q_u8(w_ptr);       // 32 4bit weight
                                const uint8x16_t w1 = vld1q_u8(w_ptr + 16);  // 32 4bit weight
                                const uint8x16_t w2 = vld1q_u8(w_ptr + 32);  // 32 4bit weight
                                const uint8x16_t w3 = vld1q_u8(w_ptr + 48);  // 32 4bit weight
                                w_ptr += 64;

                                // Quantization Method QM_ARM, convert 64 4-bit to 64 8-bit
                                // sequential: (0, 1), (2, 3), (4, 5), (6, 7)... : 128 bit
                                // expected layout of inB: (0, 16), (1, 17), (2, 18), (3, 19)...
                                // low; (0, 0), (1, 0), (2, 0), (3, 0) ...
                                // high: (16, 0), (17, 0), (18, 0), (19, 0) ...
                                int8x16_t w0_low = vreinterpretq_s8_u8(vandq_u8(w0, mask_low4bit));
                                int8x16_t w0_high = vreinterpretq_s8_u8(vshrq_n_u8(w0, 4));
                                int8x16_t w1_low = vreinterpretq_s8_u8(vandq_u8(w1, mask_low4bit));
                                int8x16_t w1_high = vreinterpretq_s8_u8(vshrq_n_u8(w1, 4));
                                int8x16_t w2_low = vreinterpretq_s8_u8(vandq_u8(w2, mask_low4bit));
                                int8x16_t w2_high = vreinterpretq_s8_u8(vshrq_n_u8(w2, 4));
                                int8x16_t w3_low = vreinterpretq_s8_u8(vandq_u8(w3, mask_low4bit));
                                int8x16_t w3_high = vreinterpretq_s8_u8(vshrq_n_u8(w3, 4));

                                // apply offset
                                w0_low = vsubq_s8(w0_low, offsets);
                                w0_high = vsubq_s8(w0_high, offsets);
                                w1_low = vsubq_s8(w1_low, offsets);
                                w1_high = vsubq_s8(w1_high, offsets);
                                w2_low = vsubq_s8(w2_low, offsets);
                                w2_high = vsubq_s8(w2_high, offsets);
                                w3_low = vsubq_s8(w3_low, offsets);
                                w3_high = vsubq_s8(w3_high, offsets);

                                // load 64 8-bit activation
                                const int8x16_t a0 = vld1q_s8(a_ptr);
                                const int8x16_t a1 = vld1q_s8(a_ptr + 16);
                                const int8x16_t a2 = vld1q_s8(a_ptr + 32);
                                const int8x16_t a3 = vld1q_s8(a_ptr + 48);
                                const int8x16_t a4 = vld1q_s8(a_ptr + 64);
                                const int8x16_t a5 = vld1q_s8(a_ptr + 80);
                                const int8x16_t a6 = vld1q_s8(a_ptr + 96);
                                const int8x16_t a7 = vld1q_s8(a_ptr + 112);
                                a_ptr += 128;

                                // dot product into int32x4_t
                                int_sum0 = vdotq_s32(int_sum0, w0_low, a0);
                                int_sum0 = vdotq_s32(int_sum0, w0_high, a1);
                                int_sum1 = vdotq_s32(int_sum1, w1_low, a2);
                                int_sum1 = vdotq_s32(int_sum1, w1_high, a3);
                                int_sum2 = vdotq_s32(int_sum2, w2_low, a4);
                                int_sum2 = vdotq_s32(int_sum2, w2_high, a5);
                                int_sum3 = vdotq_s32(int_sum3, w3_low, a6);
                                int_sum3 = vdotq_s32(int_sum3, w3_high, a7);

                                sumv0 = vmlaq_n_f32(sumv0, vcvtq_f32_s32(int_sum0), s_0);
                                sumv1 = vmlaq_n_f32(sumv1, vcvtq_f32_s32(int_sum1), s_1);
                                sumv2 = vmlaq_n_f32(sumv2, vcvtq_f32_s32(int_sum2), s_2);
                                sumv3 = vmlaq_n_f32(sumv3, vcvtq_f32_s32(int_sum3), s_3);

                            }
                            params->C.data_ptr[(ic + ii)* n + (jc + j)] +=
                                vaddvq_f32(sumv0) + vaddvq_f32(sumv1) + vaddvq_f32(sumv2) + vaddvq_f32(sumv3);


                        }
                    }
                }

            }
        
    }
    return nullptr; 
}


// static void* matmul_int8_int4_no_offset(void* arg_) {
//     const struct thread_arg* args = (struct thread_arg*) arg_; 

//     const struct matmul_params *params = args->params;
//     int n = params->C.column, m = params->C.row, k = params->A.column, block_size = params->block_size;
//     assert(params->block_size == 32);

//     const int num_block = k / block_size;
//     for (int i = 0; i < m; i++) {
//         for (int j = args->start_j; j < args->end_j; j++) {
//             float32x4_t sumv0 = vdupq_n_f32(0.0f);
//             float32x4_t sumv1 = vdupq_n_f32(0.0f);
//             float32x4_t sumv2 = vdupq_n_f32(0.0f);
//             float32x4_t sumv3 = vdupq_n_f32(0.0f);
//             const unsigned char* w_start = &params->B.int4_data_ptr[j * k / 2];
//             const signed char* a_start = &params->A.int8_data_ptr[i * k];
//             float* s_a = &params->A_scales[i * k / 32];
//             float* s_w = &params->scales[j * k / 32];

//             const uint8x16_t mask_low4bit = vdupq_n_u8(0xf);
//             const int8x16_t offsets = vdupq_n_s8(8);
//             for (int q = 0; q < num_block; q += 4) {
//                 int32x4_t int_sum0 = vdupq_n_s32(0);
//                 int32x4_t int_sum1 = vdupq_n_s32(0);
//                 int32x4_t int_sum2 = vdupq_n_s32(0);
//                 int32x4_t int_sum3 = vdupq_n_s32(0);
//                 float s_0 = *s_a++ * *s_w++;
//                 float s_1 = *s_a++ * *s_w++;
//                 float s_2 = *s_a++ * *s_w++;
//                 float s_3 = *s_a++ * *s_w++;

//                 const uint8x16_t w0 = vld1q_u8(w_start);       // 32 4bit weight
//                 const uint8x16_t w1 = vld1q_u8(w_start + 16);  // 32 4bit weight
//                 const uint8x16_t w2 = vld1q_u8(w_start + 32);  // 32 4bit weight
//                 const uint8x16_t w3 = vld1q_u8(w_start + 48);  // 32 4bit weight
//                 w_start += 64;

//                 // Quantization Method QM_ARM, convert 64 4-bit to 64 8-bit
//                 // sequential: (0, 1), (2, 3), (4, 5), (6, 7)... : 128 bit
//                 // expected layout of inB: (0, 16), (1, 17), (2, 18), (3, 19)...
//                 // low; (0, 0), (1, 0), (2, 0), (3, 0) ...
//                 // high: (16, 0), (17, 0), (18, 0), (19, 0) ...
//                 int8x16_t w0_low = vreinterpretq_s8_u8(vandq_u8(w0, mask_low4bit));
//                 int8x16_t w0_high = vreinterpretq_s8_u8(vshrq_n_u8(w0, 4));
//                 int8x16_t w1_low = vreinterpretq_s8_u8(vandq_u8(w1, mask_low4bit));
//                 int8x16_t w1_high = vreinterpretq_s8_u8(vshrq_n_u8(w1, 4));
//                 int8x16_t w2_low = vreinterpretq_s8_u8(vandq_u8(w2, mask_low4bit));
//                 int8x16_t w2_high = vreinterpretq_s8_u8(vshrq_n_u8(w2, 4));
//                 int8x16_t w3_low = vreinterpretq_s8_u8(vandq_u8(w3, mask_low4bit));
//                 int8x16_t w3_high = vreinterpretq_s8_u8(vshrq_n_u8(w3, 4));

//                 // apply offset
//                 w0_low = vsubq_s8(w0_low, offsets);
//                 w0_high = vsubq_s8(w0_high, offsets);
//                 w1_low = vsubq_s8(w1_low, offsets);
//                 w1_high = vsubq_s8(w1_high, offsets);
//                 w2_low = vsubq_s8(w2_low, offsets);
//                 w2_high = vsubq_s8(w2_high, offsets);
//                 w3_low = vsubq_s8(w3_low, offsets);
//                 w3_high = vsubq_s8(w3_high, offsets);

//                 // load 64 8-bit activation
//                 const int8x16_t a0 = vld1q_s8(a_start);
//                 const int8x16_t a1 = vld1q_s8(a_start + 16);
//                 const int8x16_t a2 = vld1q_s8(a_start + 32);
//                 const int8x16_t a3 = vld1q_s8(a_start + 48);
//                 const int8x16_t a4 = vld1q_s8(a_start + 64);
//                 const int8x16_t a5 = vld1q_s8(a_start + 80);
//                 const int8x16_t a6 = vld1q_s8(a_start + 96);
//                 const int8x16_t a7 = vld1q_s8(a_start + 112);
//                 a_start += 128;

//                 // dot product into int32x4_t
//                 int_sum0 = vdotq_s32(int_sum0, w0_low, a0);
//                 int_sum0 = vdotq_s32(int_sum0, w0_high, a1);
//                 int_sum1 = vdotq_s32(int_sum1, w1_low, a2);
//                 int_sum1 = vdotq_s32(int_sum1, w1_high, a3);
//                 int_sum2 = vdotq_s32(int_sum2, w2_low, a4);
//                 int_sum2 = vdotq_s32(int_sum2, w2_high, a5);
//                 int_sum3 = vdotq_s32(int_sum3, w3_low, a6);
//                 int_sum3 = vdotq_s32(int_sum3, w3_high, a7);

//                 sumv0 = vmlaq_n_f32(sumv0, vcvtq_f32_s32(int_sum0), s_0);
//                 sumv1 = vmlaq_n_f32(sumv1, vcvtq_f32_s32(int_sum1), s_1);
//                 sumv2 = vmlaq_n_f32(sumv2, vcvtq_f32_s32(int_sum2), s_2);
//                 sumv3 = vmlaq_n_f32(sumv3, vcvtq_f32_s32(int_sum3), s_3);
//             }
//             params->C.data_ptr[i * n + j] =
//                 vaddvq_f32(sumv0) + vaddvq_f32(sumv1) + vaddvq_f32(sumv2) + vaddvq_f32(sumv3);
//         }
//     }
//     return nullptr; 
// }

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

    const int num_thread = 32;
    pthread_t thread_pool[num_thread];
    struct thread_arg targs[num_thread];

    assert(params->block_size == 32);  // support block size 32 for now

     int thread_block_size = C->column / num_thread; 

    // TODO: Thread creation
    for (int i = 0; i < num_thread; ++i) {
        targs[i] = {i*thread_block_size, std::min(C->column, (i+1)*thread_block_size), params}; 
        pthread_create(&thread_pool[i], nullptr, matmul_int8_int4_no_offset_optimized, &targs[i]); 

    }

    // TODO: Join threads

    for (int i = 0; i < num_thread; ++i) {
        pthread_join(thread_pool[i], nullptr); 

    }
};
}  // namespace matmul
