#ifndef __MY_MAT_H__
#define __MY_MAT_H__

#include <memory>
#include <cstring>
#include <ostream>
#include <cstdlib>
#include <cassert>
#include <initializer_list>
#include <iostream>
#include <new>

const int naive_block_multiply = 0, cacheline_col_multiply = 1, cacheline_block_multiply = 2, transp_block_multiply = 3;
const int bytes_per_cacheline = 64;

namespace my {
    template <class T>
    class Mat_iterator : public std::iterator<std::forward_iterator_tag, T> {
        T *m_ptr{nullptr};
        int m_mat_pitch{0};
        int m_mat_cols{0};
        int m_cur_col{0};

        public:
        Mat_iterator(T *ptr, int mat_cols, int mat_pitch) : m_ptr(ptr), m_mat_cols(mat_cols), m_mat_pitch(mat_pitch) {}

        T& operator*() {
            return *m_ptr;
        }

        T& operator++() {
            if (m_cur_col == (m_mat_cols-1)) {
                m_cur_col = 0;
                m_ptr+=(m_mat_pitch-m_mat_cols+1);
            } else {
                ++m_cur_col;
                ++m_ptr;
            }
            return *m_ptr;
        }

        bool operator==(Mat_iterator<T> const& other) {
            return (other.m_ptr == m_ptr);
        }

        bool operator!=(Mat_iterator<T> const& other) {
            return (other.m_ptr != m_ptr);
        }
    };

    template <class T>
    class Mat_row_col_iterator {
        T *m_ptr{nullptr};
        int m_incr;

        public:
        Mat_row_col_iterator(T *ptr, int incr) : m_ptr(ptr), m_incr(incr) {}
        T& operator*() {
            return *m_ptr;
        }
        T& operator++() {
            m_ptr+=m_incr;
            return *m_ptr;
        }
        bool operator==(Mat_row_col_iterator<T> const& other) {
            return (other.m_ptr == m_ptr);
        }
        bool operator!=(Mat_row_col_iterator<T> const& other) {
            return (other.m_ptr != m_ptr);
        }
    };

    template <class T>
    class Mat {
        int m_rows{0};
        int m_cols{0};
        int m_pitch{0};
        std::unique_ptr<T> m_data{nullptr};
        T *m_ptr{nullptr};

        public:
        Mat() = default;
        Mat(int rows, int cols); // construct Mat with specified size; pitch is assumed to be equal to width
        Mat(int rows, int cols, int pitch); // construct Mat with specified size
        Mat(Mat<T> const&); // copy constructor
        Mat(Mat<T> &&); // move constructor
        Mat(int rows, int cols, std::initializer_list<T> const&); // constructor with initializer list
        Mat(int rows, int cols, int pitch, std::initializer_list<T> const&); // constructor with initializer list
        Mat(T *array, int rows, int cols, int pitch);

        void operator=(Mat<T> const&); // copy assignment operator
        void operator=(Mat<T> &&); // move assignment operator
        void operator=(std::initializer_list<T> const&); // assign from initializer list
        T& operator()(int row, int col) const; // access element
        int rows() const;
        int cols() const;
        int pitch() const;
        // Mat<T> add(Mat<T> const& other) const;
        Mat<T> operator+(Mat<T> const& other) const;
        Mat<T> sub(Mat<T> const& other) const;
        // Mat<T> mul(Mat<T> const& other) const;
        Mat<T> operator*(Mat<T> const& other) const;
        bool operator==(Mat<T> const&) const;
        bool operator!=(Mat<T> const&) const;
        friend std::ostream& operator<<(std::ostream& stream, Mat<T> const& m) {
            for (int row = 0; row < m.rows(); row++) {
                for (int col = 0; col < m.cols(); col++) {
                    stream << m(row, col) << ", ";
                }
                stream << std::endl;
            }
            return stream;
        }

        Mat_iterator<T> begin() const;
        Mat_iterator<T> end() const;
        Mat<T> submat (int start_row, int start_col, int rows, int cols) const;
        Mat_row_col_iterator<T> row_begin(int row) const;
        Mat_row_col_iterator<T> row_end(int row) const;
        Mat_row_col_iterator<T> col_begin(int col) const;
        Mat_row_col_iterator<T> col_end(int col) const;

        void naive_multiply(Mat<T> const& other, Mat<T> &out) const;
        template <int block_mult_option> Mat<T> block_multiply(Mat<T> const& other) const;
        void row_scalar_multiply_add(T scalar, Mat<T> &out) const;
        void opt_small_multiply(Mat<T> const& other, Mat<T> &out) const;
        void opt_small_transpose_multiply(Mat<T> &other, Mat<T> &out) const;
        void opt_8x8_block_multiply_add(Mat<T> const& other, Mat<T> &out) const;
        void opt_32x32_block_multiply_add(Mat<T> const& other, Mat<T> &out) const;
        Mat<T> block_transpose_multiply(Mat<T> const &other) const;
        Mat<T> transpose() const;
        void opt_transp_multiply_add(Mat<T> const& other, Mat<T> &out) const;
    };

    template <class T>
    Mat<T>::Mat(int rows, int cols) : m_rows(rows), m_cols(cols), m_pitch(m_cols), m_data(new(std::align_val_t(bytes_per_cacheline)) T[m_rows*m_pitch]), m_ptr(m_data.get()) {
    }

    template <class T>
    Mat<T>::Mat(int rows, int cols, int pitch) : m_rows(rows), m_cols(cols), m_pitch(pitch), m_data(new(std::align_val_t(bytes_per_cacheline)) T[m_rows*m_pitch]), m_ptr(m_data.get()) {
        assert(m_pitch >= m_cols);
    }

    template <class T>
    Mat<T>::Mat(Mat<T> const& other) : m_rows(other.m_rows), m_cols(other.m_cols), m_pitch(other.m_pitch), m_data(new(std::align_val_t(bytes_per_cacheline)) T[m_rows*m_pitch]), m_ptr(m_data.get()) {
        void *dst_array = (void*)m_ptr;
        void *src_array = (void*)other.m_ptr;
        std::memcpy(dst_array, src_array, sizeof(T)*m_rows*m_pitch); // should not throw if src_array has been allocated correctly
    }
    
    template <class T>
    Mat<T>::Mat(Mat<T> && other) : m_rows(other.m_rows), m_cols(other.m_cols), m_pitch(other.m_pitch), m_data(std::move(other.m_data)), m_ptr(m_data.get()) {
    }

    template <class T>
    Mat<T>::Mat(int rows, int cols, int pitch, std::initializer_list<T> const& l) : m_rows(rows), m_cols(cols), m_pitch(pitch), m_data(new(std::align_val_t(bytes_per_cacheline)) T[m_rows*m_pitch]), m_ptr(m_data.get()) {
        assert(m_rows*m_cols == l.size());

        auto iter = begin();
        auto list_iter = l.begin();
        for (; iter != end(); ++iter, ++list_iter)
            *iter = *list_iter;                     // TODO: Could trigger an exception since assignment operator is begin called. Is this a problem? Mat memory allocated will be destroyed since unique_ptr will be destroyed.
    }

    template <class T>
    Mat<T>::Mat(int rows, int cols, std::initializer_list<T> const& l) : Mat<T>(rows, cols, cols, l) {}

    template <class T>
    Mat<T>::Mat(T *array, int rows, int cols, int pitch) : m_rows(rows), m_cols(cols), m_pitch(pitch), m_ptr(array) {}

    template <class T>
    void Mat<T>::operator=(Mat<T> const& other) {
        // Check for self assignment
        if (this == &other)
            return;
    
        Mat<T> new_mat(other); // could throw, but will leave "this" Mat unchanged
        *this = std::move(new_mat); // Mat move assignment operator
    }
    
    template <class T>
    void Mat<T>::operator=(Mat<T> && other) {
        // Check for self assignment
        if (this == &other)
            return;
    
        m_rows = other.m_rows;
        m_cols = other.m_cols;
        m_pitch = other.m_pitch;
        m_data = std::move(other.m_data); // will deallocate Mat's previously allocated memory
        m_ptr = m_data.get();
    }

    template <class T>
    void Mat<T>::operator=(std::initializer_list<T> const& l) {
        assert(m_rows*m_cols == l.size());

        auto iter = begin();
        auto list_iter = l.begin();
        for (; iter != end(); ++iter, ++list_iter)
            *iter = *list_iter;                     // TODO: Could trigger an exception since assignment operator is begin called. Is this a problem? Mat memory allocated will be destroyed since unique_ptr will be destroyed.
    }
    
    template <class T>
    T& Mat<T>::operator()(int row, int col) const {
        return m_ptr[row*m_pitch+col];
    }
    
    template <class T>
    int Mat<T>::rows() const {
        return m_rows;
    }
    
    template <class T>
    int Mat<T>::cols() const {
        return m_cols;
    }

    template <class T>
    int Mat<T>::pitch() const {
        return m_pitch;
    }

    template <class T>
    Mat_iterator<T> Mat<T>::begin() const {
        return Mat_iterator<T>(m_ptr, m_cols, m_pitch);
    }

    template <class T>
    Mat_iterator<T> Mat<T>::end() const {
        return Mat_iterator<T>(&m_ptr[m_rows*m_pitch], m_cols, m_pitch);
    }

    template <class T>
    Mat_row_col_iterator<T> Mat<T>::row_begin(int row) const {
        return Mat_row_col_iterator<T>(&m_ptr[row*m_pitch], 1);
    }

    template <class T>
    Mat_row_col_iterator<T> Mat<T>::row_end(int row) const {
        return Mat_row_col_iterator<T>(&m_ptr[row*m_pitch+m_cols], 1);
    }

    template <class T>
    Mat_row_col_iterator<T> Mat<T>::col_begin(int col) const {
        return Mat_row_col_iterator<T>(&m_ptr[col], m_pitch);
    }

    template <class T>
    Mat_row_col_iterator<T> Mat<T>::col_end(int col) const {
        return Mat_row_col_iterator<T>(&m_ptr[m_pitch*m_rows+col], m_pitch);
    }

    template <class T>
    bool Mat<T>::operator==(Mat<T> const& other) const {
        if ((m_rows != other.m_rows) || (m_cols != other.m_cols))
            return false;

        auto iter = begin();
        auto other_iter = other.begin();

        for (; iter != end(); ++iter, ++other_iter) {
            if (*iter != *other_iter)
                return false;
        }

        return true;
    }

    template <class T>
    bool Mat<T>::operator!=(Mat<T> const& other) const {
        return !(*this == other);
    }

    template <class T>
    Mat<T> Mat<T>::operator+(Mat<T> const& other) const {
        assert(m_rows == other.m_rows);
        assert(m_cols == other.m_cols);

        Mat<T> out(m_rows, m_cols);
        auto in1_iter = begin();
        auto in2_iter = other.begin();
        auto out_iter = out.begin();

        for (; in1_iter != end(); ++in1_iter, ++in2_iter, ++out_iter) {
            *out_iter = *in1_iter + *in2_iter;
        }
        return out;
    }

    template <class T>
    Mat<T> Mat<T>::operator*(Mat<T> const& other) const {
        // assert(m_cols == other.m_rows);

        Mat<T> out(m_rows, other.m_cols);
        for (int i = 0; i < m_rows; i++) {
            for (int j = 0; j < other.m_cols; j++) {
                T pp = 0;
                for (int k = 0; k < m_cols; k++) {
                    pp += (*this)(i,k) * other(k,j);
                }
                out(i,j) = pp;
            }
        }
        return out;
    }

    template <class T>
    void Mat<T>::row_scalar_multiply_add(T scalar, Mat<T> &out) const {
        assert(rows() == 1);
        assert(out.rows() == 1);

        auto iter = row_begin(0);
        auto out_iter = out.row_begin(0);
        #pragma omp unroll partial(8)
        for(; iter != row_end(0); ++iter, ++out_iter) {
            *out_iter += ((*iter) * scalar);
        }
    }

    template <class T>
    void Mat<T>::opt_small_multiply(Mat<T> const& other, Mat<T> &out) const {
        assert(m_cols == other.m_rows);

        constexpr int nums_per_cacheline = bytes_per_cacheline/sizeof(T);

        assert(cols() % nums_per_cacheline == 0);
        assert(rows() % nums_per_cacheline == 0);
        assert(other.cols() % nums_per_cacheline == 0);
        assert(other.rows() % nums_per_cacheline == 0);

        for (int m1_row = 0; m1_row < rows(); m1_row++) {
            for (int m2_col = 0; m2_col < other.cols(); m2_col+=nums_per_cacheline) {
                auto m1_iter = row_begin(m1_row);
                Mat<T> out_row = out.submat(m1_row, m2_col, 1, nums_per_cacheline);
                for (int m2_row = 0; m2_row < other.rows(); m2_row++) {
                    Mat<T> m2_row_mat = other.submat(m2_row, m2_col, 1, nums_per_cacheline);
                    // out_row += (*m1_iter) * m2_row;
                    m2_row_mat.row_scalar_multiply_add(*m1_iter, out_row);
                    ++m1_iter;
                }
            }
        }
    }

    template <class T>
    void Mat<T>::naive_multiply(Mat<T> const& other, Mat<T> &out) const {
        assert(m_cols == other.m_rows);

        auto out_iter = out.begin();
        for (int i = 0; i < m_rows; i++) {
            for (int j = 0; j < other.m_cols; j++, ++out_iter) {
                auto iter_m1 = row_begin(i);
                auto iter_m2 = other.col_begin(j);
                T pp = 0;
                for (; iter_m1 != row_end(i); ++iter_m1, ++iter_m2) {
                    pp += (*iter_m1) * (*iter_m2);
                }
                *out_iter += pp;
            }
        }
    }

    template <class T>
    void Mat<T>::opt_8x8_block_multiply_add(Mat<T> const& other, Mat<T> &out) const {
        assert(m_cols == 8);
        assert(m_rows == 8);
        assert(other.m_cols == 8);
        assert(other.m_rows == 8);

        for (int m1_row = 0; m1_row < 8; m1_row++) {
            auto iter_m1 = row_begin(m1_row);
            Mat<T> out_row = out.submat(m1_row, 0, 1, 8);
            for (int m2_row = 0; m2_row < 8; m2_row++, ++iter_m1) {
                Mat<T> m2_row_mat = other.submat(m2_row, 0, 1, 8);
                m2_row_mat.row_scalar_multiply_add(*iter_m1, out_row);
            }
        }
    }

    template <class T>
    void Mat<T>::opt_32x32_block_multiply_add(Mat<T> const& other, Mat<T> &out) const {
        assert(m_cols == 32);
        assert(m_rows == 32);
        assert(other.m_cols == 32);
        assert(other.m_rows == 32);

        for (int out_row = 0; out_row < 32; out_row+=8) {
            for (int out_col = 0; out_col < 32; out_col+=8) {
                Mat<T> submat_out = out.submat(out_row, out_col, 8, 8);
                for (int m1_col = 0; m1_col < 32; m1_col+=8) {
                    Mat<T> submat1 = submat(out_row, m1_col, 8, 8);
                    Mat<T> submat2 = other.submat(m1_col, out_col, 8, 8);
                    // if (mult_option == cacheline_block_multiply)
                       submat1.opt_8x8_block_multiply_add(submat2, submat_out);
                    // else if (mult_option == transp_block_multiply)
                    //    submat1.opt_transp_multiply_add(submat2, submat_out);
                    // std::cout << submat1 << "\n" << submat2 << "\n" << submat_out << "\n";
                }
            }
        }
    }

    template <class T>
    Mat<T> Mat<T>::transpose() const {
        my::Mat<T> transp(m_cols, m_rows);
        for (int row = 0; row < m_rows; row++) {
            auto t_col_iter = transp.col_begin(row);
            auto iter = row_begin(row);
            for (; iter != row_end(row); ++iter, ++t_col_iter) {
                *t_col_iter = *iter;
            }
        }
        return transp;
    }

    template <class T>
    void Mat<T>::opt_transp_multiply_add(Mat<T> const& other, Mat<T> &out) const {
        assert(m_rows == other.m_cols);
        assert(m_cols == other.m_rows);
        Mat<T> tr(other.transpose());

        for (int m1_row = 0; m1_row < m_rows; m1_row++) {
            auto out_row_iter = out.row_begin(m1_row);
            for (int m2_row = 0; m2_row < tr.rows(); m2_row++) {
                auto m1_row_iter = row_begin(m1_row);
                auto m2_row_iter = tr.row_begin(m2_row);
                T sum = *out_row_iter;
                #pragma omp unroll partial(8)
                for (; m1_row_iter != row_end(m1_row); ++m1_row_iter, ++m2_row_iter) {
                    sum += (*m1_row_iter)*(*m2_row_iter);
                }
                *out_row_iter = sum;
                ++out_row_iter;
            }
        }
    }

    template <class T>
    template <int block_mult_option>
    Mat<T> Mat<T>::block_multiply(Mat<T> const& other) const {
        assert(m_cols == other.m_rows);

        Mat<T> out(m_rows, other.m_cols);

        for (auto iter = out.begin(); iter != out.end(); ++iter)
            *iter = static_cast<T>(0);
        
        const int d1 = 32; const int d2 = 32; // use constexpr
        for (int m1_row = 0; m1_row < rows(); m1_row+=d2) {
            for (int m2_col = 0; m2_col < other.cols(); m2_col+=d2) {
                for (int k = 0; k < cols(); k+=d1) {
                    Mat<T> submat1 = submat(m1_row, k, d2, d1);
                    Mat<T> submat2 = other.submat(k, m2_col, d1, d2);
                    Mat<T> submat_out = out.submat(m1_row, m2_col, d2, d2);
                    if (block_mult_option == naive_block_multiply)
                        submat1.naive_multiply(submat2, submat_out);
                    else if (block_mult_option == cacheline_col_multiply)
                        submat1.opt_small_multiply(submat2, submat_out);
                    else if (block_mult_option == cacheline_block_multiply)
                        submat1.opt_32x32_block_multiply_add(submat2, submat_out);
                    else if (block_mult_option == transp_block_multiply)
                        submat1.opt_32x32_block_multiply_add(submat2, submat_out);
                }
            }
        }
        return out;
    }

    template <class T>
    Mat<T> Mat<T>::block_transpose_multiply(Mat<T> const &other) const {
        assert(m_cols == other.m_cols);
        Mat<T> out(m_rows, other.m_rows);
        for (auto iter = out.begin(); iter != out.end(); ++iter) {
            *iter = static_cast<T>(0);
        }

        const int d1 = 32; const int d2 = 32;
        for (int m1_row = 0; m1_row < rows(); m1_row+=d2) {
            for (int m2_row = 0; m2_row < other.rows(); m2_row+=d2) {
                for (int col = 0; col < cols(); col+=d1) {
                    Mat<T> submat1 = submat(m1_row, col, d2, d1);
                    Mat<T> submat2 = other.submat(m2_row, col, d2, d1);
                    Mat<T> submat_out = out.submat(m1_row, m2_row, d2, d2);
                    submat1.opt_small_transpose_multiply(submat2, submat_out);
                }
            }
        }
        return out;
    }

    template <class T>
    void Mat<T>::opt_small_transpose_multiply(Mat<T> &other, Mat<T> &out) const {
        assert(m_cols == other.m_cols);

        for (int m1_row = 0; m1_row < rows(); m1_row++) {
            auto out_iter = out.row_begin(m1_row);
            for (int m2_row = 0; m2_row < other.rows(); m2_row++) {
                auto m1_iter = row_begin(m1_row);
                auto m2_iter = other.row_begin(m2_row);
                T pp = static_cast<T>(0);
                for (; m1_iter != row_end(m1_row); ++m1_iter, ++m2_iter) {
                    pp += (*m1_iter) * (*m2_iter);
                }
                (*out_iter)+=pp;
                ++out_iter;
            }
        }
    }

    template <class T>
    Mat<T> Mat<T>::submat (int start_row, int start_col, int rows, int cols) const {
        return Mat(&m_ptr[start_row*m_pitch+start_col], rows, cols, m_pitch);
    }
}

#endif
