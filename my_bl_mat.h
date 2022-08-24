#ifndef __MY_BL_MAT_H__
#define __MY_BL_MAT_H__

#include <vector>

namespace my {

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
        bl_mat<T> mult(bl_mat<T>& other);
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
    bl_mat<T> bl_mat<T>::mult(bl_mat<T>& other) {
        assert (m_cols == other.m_rows);
        bl_mat<T> out;
        out.m_rows = m_rows;
        out.m_cols = other.m_cols;

        for (int m1_row = 0; m1_row < m_rows; m1_row++) {
            for (int m2_col = 0; m2_col < other.m_cols; m2_col++) {
                Mat<T> block_out (32,32);
                for (auto iter = block_out.begin(); iter != block_out.end(); ++iter)
                    *iter = static_cast<T>(0);
                for (int k = 0; k < m_cols; k++) {
                    ((*this)(m1_row, k)).madd_32x32_zmm(other(k, m2_col), block_out);
                }
                out.m_vec.push_back(std::move(block_out));
            }
        }
        return out;
    }
}

#endif
