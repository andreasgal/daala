/*Daala video codec
Copyright (c) 2002-2013 Daala project contributors.  All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

- Redistributions of source code must retain the above copyright notice, this
  list of conditions and the following disclaimer.

- Redistributions in binary form must reproduce the above copyright notice,
  this list of conditions and the following disclaimer in the documentation
  and/or other materials provided with the distribution.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.*/

#if !defined(_x86_avx2simd_H)
# define _x86_avx2simd_H (1)
# include "../state.h"

# if OD_GNUC_PREREQ(3, 0, 0)
#  define OD_SIMD_INLINE static __inline __attribute__((always_inline))
# else
#  define OD_SIMD_INLINE static
# endif

#include <immintrin.h>

typedef __m256i od_coeff_vec_t;

OD_SIMD_INLINE od_coeff_vec_t od_coeff_vec_add(od_coeff_vec_t a,
 od_coeff_vec_t b) {
  return _mm256_add_epi32(a, b);
}

OD_SIMD_INLINE od_coeff_vec_t od_coeff_vec_sub(od_coeff_vec_t a,
 od_coeff_vec_t b) {
  return _mm256_sub_epi32(a, b);
}

OD_SIMD_INLINE od_coeff_vec_t od_coeff_vec_mul(od_coeff_vec_t a, od_coeff_vec_t b) {
  return _mm256_mullo_epi32(a, b);
}

OD_SIMD_INLINE od_coeff_vec_t od_coeff_vec_srai(od_coeff_vec_t a, int c) {
  return _mm256_srai_epi32(a, c);
}

OD_SIMD_INLINE od_coeff_vec_t od_coeff_vec_sli(od_coeff_vec_t a, int c) {
  return _mm256_slli_epi32(a, c);
}

OD_SIMD_INLINE od_coeff_vec_t od_coeff_vec_neg(od_coeff_vec_t a) {
  return od_coeff_vec_sub(_mm256_setzero_si256(), a);
}

OD_SIMD_INLINE od_coeff_vec_t od_coeff_vec_const(int a) {
  return _mm256_set1_epi32(a);
}

OD_SIMD_INLINE od_coeff_vec_t od_coeff_vec_load(const od_coeff_vec_t *x) {
  return _mm256_load_si256(x);
}

OD_SIMD_INLINE void od_coeff_vec_store(od_coeff_vec_t *x, od_coeff_vec_t a) {
  return _mm256_store_si256(x, a);
}

#endif
