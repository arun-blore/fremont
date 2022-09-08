#include <iostream>
#include <cstdlib>
#include "test_utils.h"
#include <cassert>
#include <string>
#include <cstring>
#include <immintrin.h>
#include <papi.h>

const int l = 32;

template <class T>
void init (T *vec, T *mat, T *out) {
    std::cout << "vec\n";
    for (int i = 0; i < l; i++) {
        out[i] = 0;
        vec[i] = rand()%10;
        std::cout << vec[i] << ", ";
    }
    std::cout << "\n";

    std::cout << "mat\n";
    for (int i = 0; i < l; i++) {
        for (int j = 0; j < l; j++) {
            mat[i*l+j] = rand()%10;
            std::cout << mat[i*l+j] << ", ";
        }
        std::cout << "\n";
    }
    std::cout << "\n";
}

template <class T>
void test1 (T *vec, T *mat, T *out, int niter) {
    START_CLOCK;
    for (int iter = 0; iter < niter; iter++) {
        for (int vi = 0; vi < l; vi++) {
            #pragma GCC unroll 16
            #pragma GCC ivdep
            for (int mc = 0; mc < l; mc++) {
                out[mc] += (vec[vi] * mat[vi*l+mc]);
            }
        }
    }
    STOP_CLOCK;
    std::cout << "test1 Time elapsed = " << TIME_ELAPSED << "\n";
}

template <class T>
void test2 (T *vec, T *mat, T *out, int niter) {
    START_CLOCK;
    for (int iter = 0; iter < niter; iter++) {
        for (int vi = 0; vi < l; vi++) {
            for (int mc = 0; mc < l;mc+=8) {
                int i = 0, j = 0, k = 0;
                T out00 = out[mc+i]; i++;
                T out01 = out[mc+i]; i++;
                T out02 = out[mc+i]; i++;
                T out03 = out[mc+i]; i++;
                T out04 = out[mc+i]; i++;
                T out05 = out[mc+i]; i++;
                T out06 = out[mc+i]; i++;
                T out07 = out[mc+i]; i++;
                T mat00 = mat[vi*l+mc+j]; j++;
                T mat01 = mat[vi*l+mc+j]; j++;
                T mat02 = mat[vi*l+mc+j]; j++;
                T mat03 = mat[vi*l+mc+j]; j++;
                T mat04 = mat[vi*l+mc+j]; j++;
                T mat05 = mat[vi*l+mc+j]; j++;
                T mat06 = mat[vi*l+mc+j]; j++;
                T mat07 = mat[vi*l+mc+j]; j++;
                T v = vec[vi];
                out00 += (v * mat00);
                out01 += (v * mat01);
                out02 += (v * mat02);
                out03 += (v * mat03);
                out04 += (v * mat04);
                out05 += (v * mat05);
                out06 += (v * mat06);
                out07 += (v * mat07);
                out[mc+k] = out00; k++;
                out[mc+k] = out01; k++;
                out[mc+k] = out02; k++;
                out[mc+k] = out03; k++;
                out[mc+k] = out04; k++;
                out[mc+k] = out05; k++;
                out[mc+k] = out06; k++;
                out[mc+k] = out07; k++;
            }
        }
    }
    STOP_CLOCK;
    std::cout << "test2 Time elapsed = " << TIME_ELAPSED << "\n";
}

void matmul_32x32 () {
    const int w = 32;
    const int h = 32;
    const int p = 64;
    const int size = p*h;
    double m1[size], m2[size], out[size], out_ref[size];
    for (int i = 0; i < size; i++) {
        m1[i] = rand()%10;
        m2[i] = rand()%10;
        out[i] = 0.0f;
    }

    int niter = 524288;

    for (int i = 0; i < niter; i++) {
        for (int m1_row = 0; m1_row < h; m1_row++) {
            for (int m2_col = 0; m2_col < w; m2_col++) {
                double tmp = 0;
                for (int k = 0; k < w; k++) {
                    int m1_ind = m1_row*p+k;
                    int m2_ind = k*p+m2_col;
                    int out_ind = m1_row*w+m2_col;
                    out_ref[out_ind] += m1[m1_ind] * m2[m2_ind];
                }
            }
        }
    }

    int zmm_w = 8;
    START_CLOCK;
    PAPI_hl_region_begin("fmadd_perf_matrix_multiply");
    for (int i = 0; i < niter; i++) {
        for (int m1_row = 0; m1_row < h; m1_row++) {
            for (int m1_col = 0; m1_col < w; m1_col+=4) {
                int m1_ind = m1_row*p+m1_col;
                int m2_ind_0 = (m1_col+0)*p;
                int m2_ind_1 = (m1_col+1)*p;
                int m2_ind_2 = (m1_col+2)*p;
                int m2_ind_3 = (m1_col+3)*p;
                int out_ind = m1_row*w;
                // Load output
                __m512d   out_0 = _mm512_load_pd(&out[out_ind+0*zmm_w]);
                __m512d   out_1 = _mm512_load_pd(&out[out_ind+1*zmm_w]);
                __m512d   out_2 = _mm512_load_pd(&out[out_ind+2*zmm_w]);
                __m512d   out_3 = _mm512_load_pd(&out[out_ind+3*zmm_w]);

                // Load 4 m1 values
                 __m512d m1_0 = _mm512_set1_pd(m1[m1_ind+0]);
                 __m512d m1_1 = _mm512_set1_pd(m1[m1_ind+1]);
                 __m512d m1_2 = _mm512_set1_pd(m1[m1_ind+2]);
                 __m512d m1_3 = _mm512_set1_pd(m1[m1_ind+3]);
                 // __m512 m1_4 = _mm512_set1_pd(m1[m1_ind+4]);
                 // __m512 m1_5 = _mm512_set1_pd(m1[m1_ind+5]);
                 // __m512 m1_6 = _mm512_set1_pd(m1[m1_ind+6]);
                 // __m512 m1_7 = _mm512_set1_pd(m1[m1_ind+7]);

                 // Load 4 rows (16) of m2
                 __m512d m2_00 = _mm512_load_pd(&m2[m2_ind_0+zmm_w*0]);
                 __m512d m2_01 = _mm512_load_pd(&m2[m2_ind_0+zmm_w*1]);
                 __m512d m2_02 = _mm512_load_pd(&m2[m2_ind_0+zmm_w*2]);
                 __m512d m2_03 = _mm512_load_pd(&m2[m2_ind_0+zmm_w*3]);
                 __m512d m2_10 = _mm512_load_pd(&m2[m2_ind_1+zmm_w*0]);
                 __m512d m2_11 = _mm512_load_pd(&m2[m2_ind_1+zmm_w*1]);
                 __m512d m2_12 = _mm512_load_pd(&m2[m2_ind_1+zmm_w*2]);
                 __m512d m2_13 = _mm512_load_pd(&m2[m2_ind_1+zmm_w*3]);
                 __m512d m2_20 = _mm512_load_pd(&m2[m2_ind_2+zmm_w*0]);
                 __m512d m2_21 = _mm512_load_pd(&m2[m2_ind_2+zmm_w*1]);
                 __m512d m2_22 = _mm512_load_pd(&m2[m2_ind_2+zmm_w*2]);
                 __m512d m2_23 = _mm512_load_pd(&m2[m2_ind_2+zmm_w*3]);
                 __m512d m2_30 = _mm512_load_pd(&m2[m2_ind_3+zmm_w*0]);
                 __m512d m2_31 = _mm512_load_pd(&m2[m2_ind_3+zmm_w*1]);
                 __m512d m2_32 = _mm512_load_pd(&m2[m2_ind_3+zmm_w*2]);
                 __m512d m2_33 = _mm512_load_pd(&m2[m2_ind_3+zmm_w*3]);

                 // out_0 += m1_0 * m2_00;
                 // out_1 += m1_0 * m2_01;
                 // out_2 += m1_0 * m2_02;
                 // out_3 += m1_0 * m2_03;
                 out_0 = _mm512_fmadd_pd(m1_0, m2_00, out_0);
                 out_1 = _mm512_fmadd_pd(m1_0, m2_01, out_1);
                 out_2 = _mm512_fmadd_pd(m1_0, m2_02, out_2);
                 out_3 = _mm512_fmadd_pd(m1_0, m2_03, out_3);
                 // out_0 += m1_1 * m2_10;
                 // out_1 += m1_1 * m2_11;
                 // out_2 += m1_1 * m2_12;
                 // out_3 += m1_1 * m2_13;
                 out_0 = _mm512_fmadd_pd(m1_1, m2_10, out_0);
                 out_1 = _mm512_fmadd_pd(m1_1, m2_11, out_1);
                 out_2 = _mm512_fmadd_pd(m1_1, m2_12, out_2);
                 out_3 = _mm512_fmadd_pd(m1_1, m2_13, out_3);
                 // out_0 += m1_2 * m2_20;
                 // out_1 += m1_2 * m2_21;
                 // out_2 += m1_2 * m2_22;
                 // out_3 += m1_2 * m2_23;
                 out_0 = _mm512_fmadd_pd(m1_2, m2_20, out_0);
                 out_1 = _mm512_fmadd_pd(m1_2, m2_21, out_1);
                 out_2 = _mm512_fmadd_pd(m1_2, m2_22, out_2);
                 out_3 = _mm512_fmadd_pd(m1_2, m2_23, out_3);
                 // out_0 += m1_3 * m2_30;
                 // out_1 += m1_3 * m2_31;
                 // out_2 += m1_3 * m2_32;
                 // out_3 += m1_3 * m2_33;
                 out_0 = _mm512_fmadd_pd(m1_3, m2_30, out_0);
                 out_1 = _mm512_fmadd_pd(m1_3, m2_31, out_1);
                 out_2 = _mm512_fmadd_pd(m1_3, m2_32, out_2);
                 out_3 = _mm512_fmadd_pd(m1_3, m2_33, out_3);

                // Store output
                _mm512_store_pd(&out[out_ind+0*zmm_w], out_0);
                _mm512_store_pd(&out[out_ind+1*zmm_w], out_1);
                _mm512_store_pd(&out[out_ind+2*zmm_w], out_2);
                _mm512_store_pd(&out[out_ind+3*zmm_w], out_3);
            }
        }
    }
    PAPI_hl_region_end("fmadd_perf_matrix_multiply");
    STOP_CLOCK;
    std::cout << "matmul_32x32 Time elapsed = " << TIME_ELAPSED << "\n";

    if (memcmp(out, out_ref, sizeof(double)*w*w))
        std::cout << "Test Failed\n";
    else
        std::cout << "Test Passed\n";

}

int main (int argc, char **argv) {
    assert(argc == 2);
    int niter = std::stoi(argv[1]);

    alignas(64) double vec[l];
    alignas(64) double out[l];
    alignas(64) double mat[l*l];

    init  <double>(vec, mat, out);
    test1 <double>(vec, mat, out, niter);
    test2 <double>(vec, mat, out, niter);

    std::cout << "out\n";
    for (int i = 0; i < l; i++)
        std::cout << out[i] << ", ";
    std::cout << "\n";

    double sum = 0;
    for (int i = 0; i < l; i++)
        sum+=out[i];

    std::cout << sum << "\n";

    matmul_32x32();
    return 0;
}
