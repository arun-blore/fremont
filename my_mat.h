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
#include <immintrin.h>
#include "my_bl_mat.h"
#include <cstdio>
#include <papi.h>

const int naive_block_multiply = 0;
const int cacheline_col_multiply = 1;
const int cacheline_block_multiply = 2;
const int transp_block_multiply = 3;
const int zmm_32x32_multiply = 4;
const int bytes_per_cacheline = 64;

namespace my {
    void prefetch (void *base, int blk_w, int blk_h, int pitch);

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
        void operator+=(int inc) {
            m_ptr+=inc;
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
        void madd_32x32_zmm(Mat<T> &other, Mat<T> &out);
        Mat<T> contiguous_copy();
        Mat<T> bl_block_multiply(bl_mat<T>& other);
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
    void Mat<T>::madd_32x32_zmm(Mat<T> &other, Mat<T> &out) {
        assert(rows() == 32);
        assert(cols() == 32);
        assert(other.rows() == 32);
        assert(other.cols() == 32);
        assert(out.rows() == 32);
        assert(out.cols() == 32);

        const int zmm_w = 64/sizeof(T);

        const int blk_w = 32;
        const int blk_h = 32*other.m_pitch;
        // prefetch(m_ptr+blk_w      , 32, 32, m_pitch);
        // prefetch(other.m_ptr+blk_h, 32, 32, other.m_pitch);
        // prefetch(m_ptr      , 32, 32, m_pitch);
        // prefetch(other.m_ptr, 32, 32, other.m_pitch);
        // prefetch(out.m_ptr  , 32, 32, out.m_pitch);

        // printf ("m2 start address %0x\n", other.m_ptr);
        // printf ("Prefetching addr %0x\n", other.m_ptr+blk_h);

        const int row_incr = (other.m_pitch-other.cols()+zmm_w);
        // const int row_incr = zmm_w;

        for(int m1_row = 0; m1_row < rows(); m1_row++) {
            auto m1_iter = row_begin(m1_row);
            // Load outputs
            auto out_iter = out.row_begin(m1_row);
            __m512d   out_0 = _mm512_load_pd(&(*out_iter));
            out_iter+=zmm_w;
            __m512d   out_1 = _mm512_load_pd(&(*out_iter));
            out_iter+=zmm_w;
            __m512d   out_2 = _mm512_load_pd(&(*out_iter));
            out_iter+=zmm_w;
            __m512d   out_3 = _mm512_load_pd(&(*out_iter));
            out_iter+=zmm_w;

            auto m2_iter_0 = other.row_begin(0);

            for(int m2_row = 0; m2_row < other.rows(); m2_row+=4) {
                // auto m2_iter_1 = other.row_begin(m2_row+1);
                // auto m2_iter_2 = other.row_begin(m2_row+2);
                // auto m2_iter_3 = other.row_begin(m2_row+3);

                // Load 4 m1 values
                __m512d m1_0 = _mm512_set1_pd(*m1_iter);
                ++m1_iter;
                __m512d m1_1 = _mm512_set1_pd(*m1_iter);
                ++m1_iter;
                __m512d m1_2 = _mm512_set1_pd(*m1_iter);
                ++m1_iter;
                __m512d m1_3 = _mm512_set1_pd(*m1_iter);
                ++m1_iter;

                // Load 4 rows of m2 (16 zmms)
                __m512d m2_00 = _mm512_load_pd(&(*m2_iter_0));
                m2_iter_0+=zmm_w;
                __m512d m2_01 = _mm512_load_pd(&(*m2_iter_0));
                m2_iter_0+=zmm_w;
                __m512d m2_02 = _mm512_load_pd(&(*m2_iter_0));
                m2_iter_0+=zmm_w;
                __m512d m2_03 = _mm512_load_pd(&(*m2_iter_0));
                m2_iter_0+=row_incr;

                __m512d m2_10 = _mm512_load_pd(&(*m2_iter_0));
                m2_iter_0+=zmm_w;
                __m512d m2_11 = _mm512_load_pd(&(*m2_iter_0));
                m2_iter_0+=zmm_w;
                __m512d m2_12 = _mm512_load_pd(&(*m2_iter_0));
                m2_iter_0+=zmm_w;
                __m512d m2_13 = _mm512_load_pd(&(*m2_iter_0));
                m2_iter_0+=row_incr;

                __m512d m2_20 = _mm512_load_pd(&(*m2_iter_0));
                m2_iter_0+=zmm_w;
                __m512d m2_21 = _mm512_load_pd(&(*m2_iter_0));
                m2_iter_0+=zmm_w;
                __m512d m2_22 = _mm512_load_pd(&(*m2_iter_0));
                m2_iter_0+=zmm_w;
                __m512d m2_23 = _mm512_load_pd(&(*m2_iter_0));
                m2_iter_0+=row_incr;

                __m512d m2_30 = _mm512_load_pd(&(*m2_iter_0));
                m2_iter_0+=zmm_w;
                __m512d m2_31 = _mm512_load_pd(&(*m2_iter_0));
                m2_iter_0+=zmm_w;
                __m512d m2_32 = _mm512_load_pd(&(*m2_iter_0));
                m2_iter_0+=zmm_w;
                __m512d m2_33 = _mm512_load_pd(&(*m2_iter_0));
                m2_iter_0+=row_incr;

                out_0 = _mm512_fmadd_pd(m1_0, m2_00, out_0);
                out_1 = _mm512_fmadd_pd(m1_0, m2_01, out_1);
                out_2 = _mm512_fmadd_pd(m1_0, m2_02, out_2);
                out_3 = _mm512_fmadd_pd(m1_0, m2_03, out_3);

                out_0 = _mm512_fmadd_pd(m1_1, m2_10, out_0);
                out_1 = _mm512_fmadd_pd(m1_1, m2_11, out_1);
                out_2 = _mm512_fmadd_pd(m1_1, m2_12, out_2);
                out_3 = _mm512_fmadd_pd(m1_1, m2_13, out_3);

                out_0 = _mm512_fmadd_pd(m1_2, m2_20, out_0);
                out_1 = _mm512_fmadd_pd(m1_2, m2_21, out_1);
                out_2 = _mm512_fmadd_pd(m1_2, m2_22, out_2);
                out_3 = _mm512_fmadd_pd(m1_2, m2_23, out_3);

                out_0 = _mm512_fmadd_pd(m1_3, m2_30, out_0);
                out_1 = _mm512_fmadd_pd(m1_3, m2_31, out_1);
                out_2 = _mm512_fmadd_pd(m1_3, m2_32, out_2);
                out_3 = _mm512_fmadd_pd(m1_3, m2_33, out_3);

            }
            out_iter = out.row_begin(m1_row);
            _mm512_store_pd(&(*out_iter), out_0);
            out_iter+=zmm_w;
            _mm512_store_pd(&(*out_iter), out_1);
            out_iter+=zmm_w;
            _mm512_store_pd(&(*out_iter), out_2);
            out_iter+=zmm_w;
            _mm512_store_pd(&(*out_iter), out_3);
            out_iter+=zmm_w;
        }
    }

#if 0
    template <class T>
    // __attribute__((optimize("unroll-loops")))
    void Mat<T>::row_scalar_multiply_add(T scalar, Mat<T> &out) const {
        assert(rows() == 1);
        assert(out.rows() == 1);

        auto iter = row_begin(0);
        auto out_iter = out.row_begin(0);
        #pragma GCC unroll 32
        #pragma GCC ivdep
        for(; iter != row_end(0); ++iter, ++out_iter) {
            *out_iter += ((*iter) * scalar);
        }
    }
#else
    template <class T>
    void Mat<T>::row_scalar_multiply_add(T scalar, Mat<T> &out) const {
        const int unroll_amt = 16;
        assert(rows() == 1);
        assert(out.rows() == 1);
        assert(cols()%unroll_amt == 0);
        assert(out.cols()%unroll_amt == 0);

        auto iter = row_begin(0);
        auto out_iter_read = out.row_begin(0);
        auto out_iter_write = out.row_begin(0);
        auto out_iter = out.row_begin(0);
        #pragma ivdep
        /*
        for(; iter != row_end(0);) {
            T v0 = *iter; ++iter;
            T v1 = *iter; ++iter;
            T v2 = *iter; ++iter;
            T v3 = *iter; ++iter;
            T v4 = *iter; ++iter;
            T v5 = *iter; ++iter;
            T v6 = *iter; ++iter;
            T v7 = *iter; ++iter;
            T o0 = *out_iter_read; ++out_iter_read;
            T o1 = *out_iter_read; ++out_iter_read;
            T o2 = *out_iter_read; ++out_iter_read;
            T o3 = *out_iter_read; ++out_iter_read;
            T o4 = *out_iter_read; ++out_iter_read;
            T o5 = *out_iter_read; ++out_iter_read;
            T o6 = *out_iter_read; ++out_iter_read;
            T o7 = *out_iter_read; ++out_iter_read;
            o0 += (v0 * scalar);
            o1 += (v1 * scalar);
            o2 += (v2 * scalar);
            o3 += (v3 * scalar);
            o4 += (v4 * scalar);
            o5 += (v5 * scalar);
            o6 += (v6 * scalar);
            o7 += (v7 * scalar);
            *out_iter_write = o0; ++out_iter_write;
            *out_iter_write = o1; ++out_iter_write;
            *out_iter_write = o2; ++out_iter_write;
            *out_iter_write = o3; ++out_iter_write;
            *out_iter_write = o4; ++out_iter_write;
            *out_iter_write = o5; ++out_iter_write;
            *out_iter_write = o6; ++out_iter_write;
            *out_iter_write = o7; ++out_iter_write;
        }
        */
        for (; iter != row_end(0);) {
            alignas(64) T in[unroll_amt], out[unroll_amt];
            for (int i = 0; i < unroll_amt; i++) {
                in[i] = *iter; ++iter;
                // out[i] = *out_iter_read; ++out_iter_read;
            }
            // for (int i = 0; i < 8; i++) {
            //     out[i] += in[i] * scalar;
            // }
            for (int i = 0; i < unroll_amt; i++) {
                *out_iter_write += in[i]*scalar; ++out_iter_write;
            }
            // *out_iter_write = in[0] * scalar + out[0]; ++out_iter_write;
            // *out_iter_write = in[1] * scalar + out[1]; ++out_iter_write;
            // *out_iter_write = in[2] * scalar + out[2]; ++out_iter_write;
            // *out_iter_write = in[3] * scalar + out[3]; ++out_iter_write;
            // *out_iter_write = in[4] * scalar + out[4]; ++out_iter_write;
            // *out_iter_write = in[5] * scalar + out[5]; ++out_iter_write;
            // *out_iter_write = in[6] * scalar + out[6]; ++out_iter_write;
            // *out_iter_write = in[7] * scalar + out[7]; ++out_iter_write;
        }
    }
#endif

    /*
    template <class T>
    template <int U>
    void Mat<T>::row_scalar_multiply_add(T scalar, Mat<T> &out) const {
        assert(rows() == 1);
        assert(out.rows() == 1);

        auto iter = row_begin(0);
        auto out_iter_read  = out.row_begin(0);
        auto out_iter_write = out.row_begin(0);
        // #pragma omp unroll partial(8)
        #pragma ivdep

        T in[U], out[U];
        for (int i = 0; i < U; ++iter, ++out_iter_read, ++i) {
            in[i]  = *iter;
            out[i] = *out_iter_read;
        }

        for (int i = 0; i < U; ++i, ++out_iter_write) {
            T tmp = out[i] + in[i] * scalar;
            *out_iter_write = tmp;
        }
    }
    */

    template <class T>
    void Mat<T>::opt_small_multiply(Mat<T> const& other, Mat<T> &out) const {
        assert(m_cols == other.m_rows);

        // constexpr int nums_per_cacheline = bytes_per_cacheline/sizeof(T);
        constexpr int nums_per_cacheline = 32;

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
                    // Mat<T> submat2_copy = submat2.contiguous_copy();
                    if (block_mult_option == naive_block_multiply)
                        submat1.naive_multiply(submat2, submat_out);
                    else if (block_mult_option == cacheline_col_multiply)
                        submat1.opt_small_multiply(submat2, submat_out);
                    else if (block_mult_option == cacheline_block_multiply)
                        submat1.opt_32x32_block_multiply_add(submat2, submat_out);
                    else if (block_mult_option == transp_block_multiply)
                        submat1.opt_32x32_block_multiply_add(submat2, submat_out);
                    else if (block_mult_option == zmm_32x32_multiply) {
                        submat1.madd_32x32_zmm(submat2, submat_out);
                        // submat1.madd_32x32_zmm(submat2_copy, submat_out);
                    }
                }
            }
        }
        return out;
    }

    template <class T>
    Mat<T> Mat<T>::bl_block_multiply(bl_mat<T>& other) {
        // assert(m_cols == other.m_rows);

        Mat<T> out(m_rows, other.m_cols*32);

        for (auto iter = out.begin(); iter != out.end(); ++iter)
            *iter = static_cast<T>(0);
        
        const int d1 = 32; const int d2 = 32; // use constexpr
        for (int m1_row = 0; m1_row < rows(); m1_row+=d2) {
            int m2_blk_col = 0;
            for (int m2_col = 0; m2_col < other.m_cols*32; m2_col+=d2, m2_blk_col++) {
                int m2_blk_row = 0;
                for (int k = 0; k < cols(); k+=d1) {
                    Mat<T> submat_out = out.submat(m1_row, m2_col, d2, d2);
                    Mat<T> submat1 = submat(m1_row, k, d2, d1);
                    submat1.madd_32x32_zmm(other(m2_blk_row, m2_blk_col), submat_out);
                    m2_blk_row++;
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

    void prefetch (void *base, int blk_w, int blk_h, int pitch) {
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
                _mm_prefetch(base, _MM_HINT_T0);
                base+=cacheline_bytes;
            }
            base+=((pitch-blk_w)*sizeof(double));
        }
    }

    template <class T>
    Mat<T> Mat<T>::contiguous_copy() {
        Mat<T> new_mat(m_rows, m_cols);
        auto iter = begin();
        auto new_iter = new_mat.begin();

        for (; iter != end(); ++iter, ++new_iter)
            *new_iter = *iter;

        // for (int row = 0; row < m_rows; row++) {
        //     T *src = &((*this)(row, 0));
        //     T *dst = &new_mat(row, 0);
        //     memcpy(src, dst, m_cols*sizeof(T));
        // }

        return new_mat;
    }
}

#endif
