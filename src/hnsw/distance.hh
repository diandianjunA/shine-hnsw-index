#pragma once

#include "common/types.hh"

#ifdef __AVX__
#include <x86intrin.h>
#endif

#ifdef __AVX__
// taken from https://github.com/nmslib/hnswlib/blob/master/hnswlib/space_l2.h#L61
static f32 L2SqrSIMD16ExtAVX(const void* pVect1v, const void* pVect2v, const void* qty_ptr) {
  f32* pVect1 = (f32*)pVect1v;
  f32* pVect2 = (f32*)pVect2v;
  size_t qty = *((size_t*)qty_ptr);
  f32 __attribute__((aligned(32))) TmpRes[8];
  size_t qty16 = qty >> 4;

  const f32* pEnd1 = pVect1 + (qty16 << 4);

  __m256 diff, v1, v2;
  __m256 sum = _mm256_set1_ps(0);

  while (pVect1 < pEnd1) {
    v1 = _mm256_loadu_ps(pVect1);
    pVect1 += 8;
    v2 = _mm256_loadu_ps(pVect2);
    pVect2 += 8;
    diff = _mm256_sub_ps(v1, v2);
    sum = _mm256_add_ps(sum, _mm256_mul_ps(diff, diff));

    v1 = _mm256_loadu_ps(pVect1);
    pVect1 += 8;
    v2 = _mm256_loadu_ps(pVect2);
    pVect2 += 8;
    diff = _mm256_sub_ps(v1, v2);
    sum = _mm256_add_ps(sum, _mm256_mul_ps(diff, diff));
  }

  _mm256_store_ps(TmpRes, sum);
  return TmpRes[0] + TmpRes[1] + TmpRes[2] + TmpRes[3] + TmpRes[4] + TmpRes[5] + TmpRes[6] + TmpRes[7];
}

// taken from https://github.com/nmslib/hnswlib/blob/c1b9b79af3d10c6ee7b5d0afa1ce851ae975254c/hnswlib/space_ip.h#L210
static f32 InnerProductSIMD16ExtAVX(const void* pVect1v, const void* pVect2v, const void* qty_ptr) {
  f32 __attribute__((aligned(32))) TmpRes[8];
  f32* pVect1 = (f32*)pVect1v;
  f32* pVect2 = (f32*)pVect2v;
  size_t qty = *((size_t*)qty_ptr);

  size_t qty16 = qty / 16;

  const f32* pEnd1 = pVect1 + 16 * qty16;

  __m256 sum256 = _mm256_set1_ps(0);

  while (pVect1 < pEnd1) {
    //_mm_prefetch((char*)(pVect2 + 16), _MM_HINT_T0);

    __m256 v1 = _mm256_loadu_ps(pVect1);
    pVect1 += 8;
    __m256 v2 = _mm256_loadu_ps(pVect2);
    pVect2 += 8;
    sum256 = _mm256_add_ps(sum256, _mm256_mul_ps(v1, v2));

    v1 = _mm256_loadu_ps(pVect1);
    pVect1 += 8;
    v2 = _mm256_loadu_ps(pVect2);
    pVect2 += 8;
    sum256 = _mm256_add_ps(sum256, _mm256_mul_ps(v1, v2));
  }

  _mm256_store_ps(TmpRes, sum256);
  f32 sum = TmpRes[0] + TmpRes[1] + TmpRes[2] + TmpRes[3] + TmpRes[4] + TmpRes[5] + TmpRes[6] + TmpRes[7];

  return sum;
}

#endif

static f32 l2(const span<const f32>& lhs, const span<const f32>& rhs, size_t dim) {
  const f32* a = lhs.data();
  const f32* b = rhs.data();

  const f32* last = a + dim;
  f32 result = 0.;

#ifdef __AVX__
  f32 diff0;
  const size_t qty16 = dim >> 4 << 4;
  result = L2SqrSIMD16ExtAVX(a, b, &qty16);

  a += qty16;
  b += qty16;

#else
  // taken from https://github.com/flann-lib/flann/blob/master/src/cpp/flann/algorithms/dist.h
  f32 diff0, diff1, diff2, diff3;
  const f32* unroll_group = last - 3;

  /* Process 4 items with each loop for efficiency. */
  while (a < unroll_group) {
    diff0 = a[0] - b[0];
    diff1 = a[1] - b[1];
    diff2 = a[2] - b[2];
    diff3 = a[3] - b[3];
    result += diff0 * diff0 + diff1 * diff1 + diff2 * diff2 + diff3 * diff3;
    a += 4;
    b += 4;
  }
#endif
  /* Process last 0-3 pixels.  Not needed for standard vector lengths. */
  while (a < last) {
    diff0 = *a++ - *b++;
    result += diff0 * diff0;
  }

  return result;
}

static f32 ip_distance(const span<const f32>& lhs, const span<const f32>& rhs, size_t dim) {
  const f32* a = lhs.data();
  const f32* b = rhs.data();

  f32 result = 0.;

#ifdef __AVX__
  const size_t qty16 = dim >> 4 << 4;
  const f32 res = InnerProductSIMD16ExtAVX(a, b, &qty16);

  a += qty16;
  b += qty16;
  const size_t qty_left = dim - qty16;

  f32 res_tail = 0;
  for (idx_t i = 0; i < qty_left; i++) {
    res_tail += a[i] * b[i];
  }

  result = 1.0f - (res + res_tail);

#else
  f32 res = 0;
  for (idx_t i = 0; i < dim; i++) {
    res += a[i] * b[i];
  }

  result = 1.0f - res;
#endif

  return result;
}

struct L2Distance {
  static f32 dist(const span<const f32>& lhs, const span<const f32>& rhs, size_t dim) { return l2(lhs, rhs, dim); }
};

struct IPDistance {
  static f32 dist(const span<const f32>& lhs, const span<const f32>& rhs, size_t dim) {
    return ip_distance(lhs, rhs, dim);
  }
};
