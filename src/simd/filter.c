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

#if defined(HAVE_CONFIG_H)
# include "config.h"
#endif

#ifdef __AVX2__
#include "../x86/avx2simd.h"
#else
#include "../x86/sse2simd.h"
#endif

#include "../filter.h"

OD_SIMD_INLINE void od_pre_filter4_vec1(const od_coeff_vec_t x[4],
 od_coeff_vec_t y[4], const od_coeff_vec_t *p, const od_coeff_vec_t *r) {
  od_coeff_vec_t t[4];
  /*+1/-1 butterflies (required for FIR, PR, LP).*/
  t[3] = od_coeff_vec_sub(x[0], x[3]);
  t[2] = od_coeff_vec_sub(x[1], x[2]);
  t[1] = od_coeff_vec_sub(x[1], od_coeff_vec_srai(t[2], 1));
  t[0] = od_coeff_vec_sub(x[0], od_coeff_vec_srai(t[3], 1));
  /*U filter (arbitrary invertible, omitted).*/
  /*V filter (arbitrary invertible).*/
  /*Scaling factors: the biorthogonal part.*/
  /*Note: t[i]+=t[i]>>(OD_COEFF_BITS-1)&1 is equivalent to: if(t[i]>0)t[i]++
    This step ensures that the scaling is trivially invertible on the decoder's
    side, with perfect reconstruction.*/
  /*s0*/
  t[2] = od_coeff_vec_srai(od_coeff_vec_mul(t[2], p[0]), 6);
  t[2] = od_coeff_vec_add(t[2], od_coeff_vec_srai(od_coeff_vec_neg(t[2]), (OD_COEFF_BITS-1)&1));
  /*s1*/
  t[3] = od_coeff_vec_srai(od_coeff_vec_mul(t[3], p[1]), 6);
  t[3] = od_coeff_vec_add(t[3], od_coeff_vec_srai(od_coeff_vec_neg(t[3]), (OD_COEFF_BITS-1)&1));
  /*Rotation:*/
  /*p0*/
  t[3] = od_coeff_vec_add(t[3], od_coeff_vec_srai(od_coeff_vec_add(od_coeff_vec_mul(t[2], p[2]), od_coeff_vec_const(32)), 6));
  /*u0*/
  t[2] = od_coeff_vec_add(t[2], od_coeff_vec_srai(od_coeff_vec_add(od_coeff_vec_mul(t[3], p[3]), od_coeff_vec_const(32)), 6));
  /*More +1/-1 butterflies (required for FIR, PR, LP).*/
  t[0] = od_coeff_vec_add(t[0], od_coeff_vec_srai(t[3], 1));
  y[0] = t[0];
  t[1] = od_coeff_vec_add(t[1], od_coeff_vec_srai(t[2], 1));
  y[1] = t[1];
  y[2] = od_coeff_vec_sub(t[1], t[2]);
  y[3] = od_coeff_vec_sub(t[0], t[3]);
}

OD_SIMD_INLINE void od_post_filter4_vec1(const od_coeff_vec_t y[4],
 od_coeff_vec_t x[4], const od_coeff_vec_t *p, const od_coeff_vec_t *r) {
  od_coeff_vec_t t[4];
  t[3] = od_coeff_vec_sub(y[0], y[3]);
  t[2] = od_coeff_vec_sub(y[1], y[2]);
  t[1] = od_coeff_vec_sub(y[1], od_coeff_vec_srai(t[2], 1));
  t[0] = od_coeff_vec_sub(y[0], od_coeff_vec_srai(t[3], 1));
  t[2] = od_coeff_vec_sub(t[2], od_coeff_vec_srai(od_coeff_vec_add(od_coeff_vec_mul(t[3], p[3]), od_coeff_vec_const(32)), 6));
  t[3] = od_coeff_vec_sub(t[3], od_coeff_vec_srai(od_coeff_vec_add(od_coeff_vec_mul(t[2], p[2]), od_coeff_vec_const(32)), 6));
  t[3] = od_coeff_vec_mul(od_coeff_vec_sli(t[3], 6), r[1]);
  t[2] = od_coeff_vec_mul(od_coeff_vec_sli(t[2], 6), r[0]);
  t[0] = od_coeff_vec_add(t[0], od_coeff_vec_srai(t[3], 1));
  x[0] = t[0];
  t[1] = od_coeff_vec_add(t[1], od_coeff_vec_srai(t[2], 1));
  x[1] = t[1];
  x[2] = od_coeff_vec_sub(t[1], t[2]);
  x[3] = od_coeff_vec_sub(t[0], t[3]);
}

#define UNROLL(l, op) do { int i; for (i = 0; i < l; ++i) { op; }; } while (0)
#define DEFINE_FILTER(f, l, op) \
  void f(const od_coeff_vec_t *_x, od_coeff_vec_t *_y, int *_n,   \
	 const int *_p, const int *_r) {                          \
    od_coeff_vec_t p[l], r[l];                                    \
    UNROLL(l, p[i] = od_coeff_vec_const(_p[i]));                  \
    UNROLL(l, r[i] = od_coeff_vec_const(_r[i]));                  \
    static const int w = sizeof(od_coeff_vec_t)/sizeof(od_coeff); \
    int n = *_n;                                                  \
    while (n > w) {                                               \
      n -= w;                                                     \
      od_coeff_vec_t x[l];                                        \
      od_coeff_vec_t y[l];                                        \
      UNROLL(l, x[i] = od_coeff_vec_load(_x++));                  \
      op(x, y, p, r);						  \
      UNROLL(l, od_coeff_vec_store(_y++, y[i]));                  \
    }                                                             \
    *_n = n;                                                      \
  }

DEFINE_FILTER(op_pre_filter4_vec, 4, od_pre_filter4_vec1)
DEFINE_FILTER(op_post_filter4_vec, 4, od_post_filter4_vec1)
