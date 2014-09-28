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

#if !defined(_x86_sse2simd_H)
# define _x86_sse2simd_H (1)
# include "../state.h"

# if OD_GNUC_PREREQ(3, 0, 0)
#  define OD_SIMD_INLINE static __inline __attribute__((always_inline))
# else
#  define OD_SIMD_INLINE static
# endif

#include <xmmintrin.h>

typedef __m128i od_coeff_vec_t;

OD_SIMD_INLINE od_coeff_vec_t od_coeff_vec_add(od_coeff_vec_t a,
 od_coeff_vec_t b) {
  return _mm_add_epi32(a, b);
}

OD_SIMD_INLINE od_coeff_vec_t od_coeff_vec_sub(od_coeff_vec_t a,
 od_coeff_vec_t b) {
  return _mm_sub_epi32(a, b);
}

/*This is overridden by the SSE4.1 version.*/
#if !defined(OD_MULLO_EPI32)
OD_SIMD_INLINE __m128i od_mullo_epi32_sse2(__m128i a, __m128i b) {
  __m128i lo = _mm_mul_epu32(a, b);
  __m128i hi = _mm_mul_epu32(_mm_srli_si128(a, 4), _mm_srli_si128(b, 4));
  return _mm_unpacklo_epi32(_mm_shuffle_epi32(lo, _MM_SHUFFLE(0, 0, 2, 0)),
			    _mm_shuffle_epi32(hi, _MM_SHUFFLE(0, 0, 2, 0)));
}
#endif

# define OD_MULLO_EPI32 od_mullo_epi32_sse2

OD_SIMD_INLINE od_coeff_vec_t od_coeff_vec_mul(od_coeff_vec_t a, od_coeff_vec_t b) {
  return OD_MULLO_EPI32(a, b);
}

OD_SIMD_INLINE od_coeff_vec_t od_coeff_vec_srai(od_coeff_vec_t a, int c) {
  return _mm_srai_epi32(a, c);
}

OD_SIMD_INLINE od_coeff_vec_t od_coeff_vec_sli(od_coeff_vec_t a, int c) {
  return _mm_slli_epi32(a, c);
}

OD_SIMD_INLINE od_coeff_vec_t od_coeff_vec_neg(od_coeff_vec_t a) {
  return od_coeff_vec_sub(_mm_setzero_si128(), a);
}

OD_SIMD_INLINE od_coeff_vec_t od_coeff_vec_const(int a) {
  return _mm_set1_epi32(a);
}

OD_SIMD_INLINE od_coeff_vec_t od_coeff_vec_load(const od_coeff_vec_t *x) {
  return _mm_load_si128(x);
}

OD_SIMD_INLINE void od_coeff_vec_store(od_coeff_vec_t *x, od_coeff_vec_t a) {
  return _mm_store_si128(x, a);
}

#endif
