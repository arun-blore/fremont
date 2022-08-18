#ifndef __MY_BL_MAT_H__
#define __MY_BL_MAT_H__

#include <vector>

namespace my {

    template <class T>
    class Mat;

    template <class T>
    class bl_mat {
        public:
        std::vector<Mat<T>> m_vec;
        int m_cols;

        bl_mat(Mat<T> &m, int w, int h);
        Mat<T>& operator()(int row, int col);
    };

    template <class T>
    bl_mat<T>::bl_mat(Mat<T> &m, int w, int h) {
        assert(m.rows()%h == 0);
        assert(m.cols()%w == 0);
        
        m_cols = m.cols()/w;
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
}

#endif
