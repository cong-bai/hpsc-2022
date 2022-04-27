#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <immintrin.h>

float reduce(__m256 avec) {
  float a[8];
  __m256 bvec = _mm256_permute2f128_ps(avec, avec, 1);
  bvec = _mm256_add_ps(bvec, avec);
  bvec = _mm256_hadd_ps(bvec, bvec);
  bvec = _mm256_hadd_ps(bvec, bvec);
  _mm256_store_ps(a, bvec);
  return a[0];
}


int main() {
  const int N = 8;
  float x[N], y[N], m[N], fx[N], fy[N];
  float ramp[N] = {0, 1, 2, 3, 4, 5, 6, 7};
  __m256 rampvec = _mm256_load_ps(ramp);
  for(int i=0; i<N; i++) {
    x[i] = drand48();
    y[i] = drand48();
    m[i] = drand48();
    fx[i] = fy[i] = 0;
  }
  __m256 xvec = _mm256_load_ps(x);
  __m256 yvec = _mm256_load_ps(y);
  __m256 mvec = _mm256_load_ps(m);
  __m256 mask, rxvec, ryvec, mivec, rsqrtvec, mrvec, fxvec, fyvec;
  for(int i=0; i<N; i++) {
    mask = _mm256_cmp_ps(_mm256_set1_ps(i), rampvec, _CMP_EQ_OQ);
    rxvec = _mm256_sub_ps(_mm256_set1_ps(x[i]), xvec);
    ryvec = _mm256_sub_ps(_mm256_set1_ps(y[i]), yvec);
    // 1 / sqrt(rx * rx + ry * ry)
    // Seems rsqrt is not accurate?
    // rsqrtvec = _mm256_rsqrt_ps(_mm256_add_ps(_mm256_mul_ps(rxvec, rxvec), _mm256_mul_ps(ryvec, ryvec)));
    rsqrtvec = _mm256_div_ps(_mm256_set1_ps(1), _mm256_sqrt_ps(_mm256_add_ps(_mm256_mul_ps(rxvec, rxvec), _mm256_mul_ps(ryvec, ryvec))));
    // m / r^3
    mrvec = _mm256_mul_ps(_mm256_mul_ps(_mm256_mul_ps(rsqrtvec, rsqrtvec), rsqrtvec), mvec);
    fxvec = _mm256_blendv_ps(_mm256_mul_ps(mrvec, rxvec), _mm256_set1_ps(0), mask);
    fyvec = _mm256_blendv_ps(_mm256_mul_ps(mrvec, ryvec), _mm256_set1_ps(0), mask);

    fx[i] -= reduce(fxvec);
    fy[i] -= reduce(fyvec);

    printf("%d %g %g\n",i,fx[i],fy[i]);
  }
}
