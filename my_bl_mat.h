#ifndef __MY_BL_MAT_H__
#define __MY_BL_MAT_H__

#include <vector>
#include <immintrin.h>
#include "my_mat.h"

namespace my {

    void bl_prefetch (void *base, int blk_w, int blk_h, int pitch);

    template <class T>
    class Mat;

    template <class T>
    class bl_mat {
        public:
        std::vector<Mat<T>> m_vec{};
        int m_cols{0};
        int m_rows{0};

        bl_mat() = default;
        bl_mat(Mat<T> &m, int w, int h);
        Mat<T>& operator()(int row, int col);
        template <int R, int C>
        bl_mat<T> mult(bl_mat<T>& other);
        Mat<T> to_mat();
    };

    template <class T>
    bl_mat<T>::bl_mat(Mat<T> &m, int w, int h) {
        assert(m.rows()%h == 0);
        assert(m.cols()%w == 0);
        
        m_cols = m.cols()/w;
        m_rows = m.rows()/h;
        for (int row = 0; row < m.rows(); row+=h) {
            for (int col = 0; col < m.cols(); col+=w) {
                Mat<T> sub = m.submat(row, col, h, w);
                m_vec.emplace_back(sub.contiguous_copy());
            }
        }
    }

    template <class T>
    Mat<T>& bl_mat<T>::operator()(int row, int col) {
        int ind = row * m_cols + col;
        return m_vec[ind];
    }

    template <class T>
    template <int R, int C>
    bl_mat<T> bl_mat<T>::mult(bl_mat<T>& other) {
        assert (m_cols == other.m_rows);
        assert (m_vec[0].rows() == R);
        assert (m_vec[0].cols() == C);
        assert (other(0, 0).rows() == C);
        assert (other(0, 0).cols() == R);
        bl_mat<T> out;
        out.m_rows = m_rows;
        out.m_cols = other.m_cols;

        for (int m1_row = 0; m1_row < m_rows; m1_row++) {
            for (int m2_col = 0; m2_col < other.m_cols; m2_col++) {
                Mat<T> block_out (R,R);
                for (auto iter = block_out.begin(); iter != block_out.end(); ++iter)
                    *iter = static_cast<T>(0);
                for (int k = 0; k < m_cols; k++) {
                    // if (k < m_cols-1)
                    //     bl_prefetch(&(other(k+1, m2_col)(0, 0)), 32, 32, 32);
                    if (R == 32 && C == 32) {
                        ((*this)(m1_row, k)).madd_32x32_zmm(other(k, m2_col), block_out);
                    } else {
                        ((*this)(m1_row, k)).template madd_RxC_zmm<R,C>(other(k, m2_col), block_out);
                    }
                }
                out.m_vec.push_back(std::move(block_out));
            }
        }
        return out;
    }

    template <class T>
    Mat<T> bl_mat<T>::to_mat() {
        int block_rows = m_vec[0].rows();
        int block_cols = m_vec[0].cols();
        int mat_rows = m_rows * block_rows;
        int mat_cols = m_cols * block_cols;
        Mat<T> m(mat_rows, mat_cols);
        for (int row = 0, block_row_ind = 0; row < mat_rows; row+=block_rows, block_row_ind++) {
            for (int col = 0, block_col_ind = 0; col < mat_cols; col+=block_cols, block_col_ind++) {
                Mat<T> sub = m.submat(row, col, block_rows, block_cols);
                auto iter = ((*this)(block_row_ind, block_col_ind)).begin();
                for (auto &el : sub) {
                    el = *iter;
                    ++iter;
                }
            }
        }
        return m;
    }

    void bl_prefetch (void *base, int blk_w, int blk_h, int pitch) {
        const int cacheline_bytes = 64;
        const int doubles_per_cacheline = cacheline_bytes/sizeof(double);
        // const int zmm_bytes = 64;
        // const int doubles_per_zmm_reg = zmm_bytes/sizeof(double);
        // assert(doubles_per_zmm_reg == 8);
        assert(doubles_per_cacheline == 8);
        assert(blk_w == 32);
        assert(blk_h == 32);

        // alignas(64) long vindex_base[doubles_per_zmm_reg] = {0, 8, 16, 24, 32, 40, 48, 56};
        // __m512i vindex = _mm512_load_epi64(vindex_base);

        for (int row = 0; row < blk_h; row++) {
            for (int col = 0; col < blk_w; col+=doubles_per_cacheline) {
                // _mm512_prefetch_i64gather_pd (vindex, base, 1, _MM_HINT_T1);
                // base+=zmm_bytes;
                _mm_prefetch(base, _MM_HINT_T1);
                base+=cacheline_bytes;
            }
        }
    }
}

#endif
