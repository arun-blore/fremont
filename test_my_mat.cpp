#include "my_mat.h"
#include "test_utils.h"
#include <Eigen/Dense>
#include <cstdlib>

template <class T>
T random_mat(int rows, int cols) {
    T m(rows, cols);
    for (int row = 0; row < rows; row++) {
        for (int col = 0; col < cols; col++) {
            m(row, col) = rand()%10;
        }
    }
    return m;
}

#ifdef ALL_TESTS
bool test_constructors() {
    my::Mat<int> m1;
    my::Mat<int> m2(10,10);
    return true;
}

bool test_set_element() {
    my::Mat<int> m1(5,5);
    int value = 0;
    for (int row = 0; row < m1.rows(); row++) {
        for (int col = 0; col < m1.cols(); col++) {
            m1(row, col) = value++;
        }
    }
    std::cout << m1;

    return true;
}

bool test_pitch() {
    my::Mat<int> m1(5,5,6);
    int i = 0;
    for (auto &el:m1) {
        el = i++;
    }
    std::cout << m1 << std::endl;

    my::Mat<int> m2(3,3,4,{1,2,3,4,5,6,7,8,9});
    my::Mat<int> m3(3,3,4,{9,8,7,6,5,4,3,2,1});
    my::Mat<int> m4(m2*m3);

    std::cout << m2 << m3 << m4;

    my::Mat<int> m5(3,3,{1,2,3,4,5,6,7,8,9});
    my::Mat<int> m6(3,3,{9,8,7,6,5,4,3,2,1});
    my::Mat<int> m7(m5*m6);

    if (m4 == m7)
        return true;
    else
        return false;
}

bool test_row_col_iter() {
    my::Mat<int> m1(3,3,10,{1,2,3,4,5,6,7,8,9});

    std::cout << m1;

    for (int row = 0; row < m1.rows(); row++) {
        std::cout << "Row " << row << ": ";
        for (auto iter = m1.row_begin(row); iter != m1.row_end(row); ++iter){
            std::cout << *iter << ",";
        }
        std::cout << "\n";
    }

    for (int col = 0; col < m1.cols(); col++) {
        std::cout << "col " << col << ": ";
        for (auto iter = m1.col_begin(col); iter != m1.col_end(col); ++iter) {
            std::cout << *iter << ",";
        }
        std::cout << "\n";
    }
    return true;
}

bool test_copy() {
    my::Mat<int> m1(random_mat<my::Mat<int>>(5,5));
    my::Mat<int> m2;
    m2 = m1;

    std::cout << m1 << std::endl << m2;

    if (m1 != m2)
        return false;

    return true;
}

bool test_equality() {
    my::Mat<int> m1(random_mat<my::Mat<int>>(5,5));
    my::Mat<int> m2(random_mat<my::Mat<int>>(3,3));

    if (m1 == m2)
        return false;

    m2 = m1;
    m2(0,0) = m2(0,0)+1;

    if (m1 == m2)
        return false;

    return true;
}

bool test_add() {
    my::Mat<int> m1(random_mat<my::Mat<int>>(5,5));
    my::Mat<int> m2(random_mat<my::Mat<int>>(5,5));

    my::Mat<int> m3 = m2+m1;

    auto m1_iter = m1.begin();
    auto m2_iter = m2.begin();
    auto m3_iter = m3.begin();

    for (; m1_iter != m1.end(); ++m1_iter, ++m2_iter, ++m3_iter) {
        if (*m3_iter != *m1_iter+*m2_iter)
            return false;
    }

    return true;
}

bool test_init_list() {
    my::Mat<int> m1(3,3,{1,2,3,4,5,6,7,8,9});
    std::cout << m1;
    return true;
}

bool test_mul() {
    my::Mat<int> m1(1,3,{1,2,3}), m2(3,1,{4,5,6});
    my::Mat<int> m3 = m1*m2;
    if (m3 != my::Mat<int>(1,1,{32}))
        return false;

    m1 = my::Mat<int>(2,2,{1,2,3,4});
    m2 = m1*m1;
    if (m2 != my::Mat<int>(2,2,{7,10,15,22}))
        return false;

    return true;
}
#endif

template <class T>
bool test_mult_perf(int rows, int cols) {
    T m1(random_mat<T>(rows, cols)), m2(random_mat<T>(cols, rows));
    T m3;
    START_CLOCK;
    m3 = m1*m2;
    STOP_CLOCK;
    int duration = TIME_ELAPSED;
    std::cout << "Time elapsed = " << duration << " us.\n";
    std::cout << "m3(0,0) = " << m3(0,0) << " m3(rows-1,cols-1) = " << m3(rows-1,rows-1) << std::endl;
    return true;
}

#ifdef ALL_TESTS
template <class T>
bool test_add_perf(int rows, int cols) {
    T m1(random_mat<T>(rows, cols)), m2(random_mat<T>(cols, rows));
    T m3;
    START_CLOCK;
    m3 = m1+m2;
    STOP_CLOCK;
    int duration = TIME_ELAPSED;
    std::cout << "Time elapsed = " << duration << " us.\n";
    return true;
}

bool test_submat() {
    my::Mat<int> m(random_mat<my::Mat<int>>(5,5));
    std::cout << m;

    std::cout << m.submat(1,1,3,3) << std::endl;
    std::cout << m.submat(0,0,5,1) << std::endl;

    my::Mat<int> m1 = m.submat(0,0,5,1);
    m1(0,0) = 2;

    std::cout << m;
    std::cout << m * m1;
    return true;
}

bool test_naive_mult() {
    my::Mat<int> m(random_mat<my::Mat<int>>(5,5));
    my::Mat<int> m1(m*m);
    my::Mat<int> m2(5,5);
    m.naive_multiply(m, m2);

    std::cout << m << std::endl << m1 << std::endl << m2;

    if (m2 == m1)
        return true;
    else
        return false;
}

template <int block_mult_option>
bool test_block_mult(int rows, int cols) {
    using T = int;
    my::Mat<T> m1(random_mat<my::Mat<T>>(rows, cols));
    my::Mat<T> m2(random_mat<my::Mat<T>>(cols, rows));
    my::Mat<T> m3 = m1*m2;
    my::Mat<T> m4 = m1.block_multiply<block_mult_option>(m2);

    // std::cout << m3 << std::endl << m4 << std::endl;

    if (m3 == m4)
        return true;
    else
        return false;
}
#endif

template <int block_mult_option>
bool test_block_mult_perf(int rows, int cols) {
    using T = double;
    my::Mat<T> m1(random_mat<my::Mat<T>>(rows,cols));
    my::Mat<T> m2(random_mat<my::Mat<T>>(cols,rows));

    START_CLOCK;
    my::Mat<T> m3 = m1.block_multiply<block_mult_option>(m2);
    STOP_CLOCK;
    std::cout << "Time elapsed (block multiply) = " << TIME_ELAPSED << " us.\n";
    std::cout << "m3(0,0) = " << m3(0,0) << " m3(rows-1,cols-1) = " << m3(rows-1,rows-1) << std::endl;

    // my::Mat<int> m4 = m1*m2;

    // if (m3 == m4)
    //     return true;
    // else
    //     return false;
}

#ifdef ALL_TESTS
template <class T>
bool test_naive_mult_perf() {
    my::Mat<T> m1(random_mat<my::Mat<T>>(1980,3960));
    my::Mat<T> m2(random_mat<my::Mat<T>>(3960,1980));

    START_CLOCK;
    my::Mat<T> m4 = m1*m2;
    STOP_CLOCK;
    std::cout << "Time elapsed (naive multiply) = " << TIME_ELAPSED << " us.\n";
}

bool test_opt_small_multiply() {
    my::Mat<int> m1(random_mat<my::Mat<int>>(16,16));
    my::Mat<int> m2(random_mat<my::Mat<int>>(16,16));

    my::Mat<int> m3 = m1*m2;
    my::Mat<int> m4(16,16);
    m1.opt_small_multiply(m2, m4);

    std::cout << m3 << std::endl << m4 << std::endl;

    if (m3 == m4)
        return true;
    else
        return false;
}

template <class T>
bool test_block_transp_mult_perf(int rows, int cols) {
    my::Mat<T> m1(random_mat<my::Mat<T>>(rows, cols));
    my::Mat<T> m2(random_mat<my::Mat<T>>(rows, cols));

    START_CLOCK;
    my::Mat<T> m3 = m1.block_transpose_multiply(m2);
    STOP_CLOCK;

    std::cout << "Time elapsed (block transpose multiply) = " << TIME_ELAPSED << " us.\n";
}

bool test_8x8_block_multiply_add () {
    my::Mat<int> m1(random_mat<my::Mat<int>>(8,8)), m2(random_mat<my::Mat<int>>(8,8));
    my::Mat<int> m3(m1*m2);
    my::Mat<int> m4(8,8);

    for (auto iter = m4.begin(); iter != m4.end(); ++iter)
        *iter = 1;

    m3 = m3+m4;

    m1.opt_8x8_block_multiply_add(m2, m4);
    std::cout << m1 << "\n" << m2 << "\n" << m3 << "\n" << m4 << std::endl;

    return (m3 == m4);
};

bool test_32x32_block_multiply_add () {
    my::Mat<int> m1(random_mat<my::Mat<int>>(32,32)), m2(random_mat<my::Mat<int>>(32,32));
    my::Mat<int> m3(m1*m2);
    my::Mat<int> m4(32,32);

    for (auto iter = m4.begin(); iter != m4.end(); ++iter)
        *iter = 0;

    m1.opt_32x32_block_multiply_add(m2, m4);
    std::cout << m1 << "\n" << m2 << "\n" << m3 << "\n" << m4 << std::endl;

    return (m3 == m4);
};

bool test_transpose() {
    my::Mat<int> m1(random_mat<my::Mat<int>>(5,5));
    my::Mat<int> m2 = m1.transpose();

    std::cout << m1 << "\n" << m2;

    for (int row = 0; row < 5; row++) {
        for (int col = 0; col < 5; col++) {
            if (m1(row,col) != m2(col, row)) {
                return false;
            }
        }
    }

    return true;
}

bool test_transp_multiply_add() {
    my::Mat<int> m1(random_mat<my::Mat<int>>(3,3)), m2(random_mat<my::Mat<int>>(3,3));
    my::Mat<int> m3(m1*m2);
    my::Mat<int> m4(3,3);

    for (auto iter = m4.begin(); iter != m4.end(); ++iter)
        *iter = 0;

    m1.opt_transp_multiply_add(m2, m4);

    return (m3 == m4);
}
#endif

int main(int argc, char **argv) {
    int test_num = 0;
    if (argc > 1) {
        test_num = std::stoi(argv[1]);
    }

    switch(test_num) {
#ifdef ALL_TESTS
        case  0:
        case  1: RUN_TEST(test_constructors); if(test_num) break;
        case  2: RUN_TEST(test_set_element); if(test_num) break;
        case  3: RUN_TEST(test_copy); if(test_num) break;
        case  4: RUN_TEST(test_equality); if (test_num) break;
        case  5: RUN_TEST(test_add); if (test_num) break;
        case  6: RUN_TEST(test_init_list); if(test_num) break;
        case  7: RUN_TEST(test_mul); if(test_num) break;
        case  8: RUN_TEST(test_pitch); if (test_num) break;
        case  9: RUN_TEST(test_row_col_iter); if(test_num) break;
        case 91: RUN_TEST(test_submat); if(test_num) break;
        case 92: RUN_TEST(test_naive_mult); if(test_num) break;
        case 94: RUN_TEST(test_8x8_block_multiply_add); if(test_num) break;
        case 95: RUN_TEST(test_32x32_block_multiply_add); if(test_num) break;
        case 93: RUN_TEST1((test_block_mult<naive_block_multiply>(64,64))); if(test_num) break;
        case 96: RUN_TEST1((test_block_mult<cacheline_col_multiply>(64,64))); if(test_num) break;
        case 97: RUN_TEST1((test_block_mult<cacheline_block_multiply>(64,64))); if(test_num) break;
        case 98: RUN_TEST(test_transpose); if(test_num) break;
        case 99: RUN_TEST(test_transp_multiply_add); if(test_num) break;

        // Vector dot product
        case 10: RUN_TEST1(test_mult_perf<my::Mat<double>>(1,100000000)); if(test_num) break;
        case 11: RUN_TEST1(test_mult_perf<Eigen::MatrixXd>(1,100000000)); if(test_num) break;

        // Matrix matrix multiply
        case 20: RUN_TEST1(test_mult_perf<my::Mat<double>>(1000,1000)); if(test_num) break;
        case 21: RUN_TEST1(test_mult_perf<Eigen::MatrixXd>(1000,1000)); if(test_num) break;
        case 22: RUN_TEST1(test_mult_perf<my::Mat<double>>(100,100)); if(test_num) break;
        case 23: RUN_TEST1(test_mult_perf<Eigen::MatrixXd>(100,100)); if(test_num) break;
        case 24: RUN_TEST1(test_mult_perf<my::Mat<double>>(10,10)); if(test_num) break;
        case 25: RUN_TEST1(test_mult_perf<Eigen::MatrixXd>(10,10)); if(test_num) break;
        case 26: RUN_TEST1(test_mult_perf<my::Mat<double>>(1000,10000)); if(test_num) break;
        case 27: RUN_TEST1(test_mult_perf<Eigen::MatrixXd>(1000,10000)); if(test_num) break;
        //case 28: RUN_TEST (test_block_mult_perf<double>); if(test_num) break;
        case 29: RUN_TEST (test_naive_mult_perf<double>); if(test_num) break;
#endif

        // case 100: RUN_TEST1(test_mult_perf<Eigen::MatrixXd>(2048,4096)); if(test_num) break;
#ifdef ALL_TESTS
        case 101: RUN_TEST1(test_mult_perf<my::Mat<double>>(2048,4096)); if(test_num) break;
        case 102: RUN_TEST1(test_block_mult_perf<naive_block_multiply>(2048,4096)); if(test_num) break;
#endif
        // case 103: RUN_TEST1(test_block_mult_perf<cacheline_col_multiply>(2048,4096)); if(test_num) break;
        case 107: RUN_TEST1(test_block_mult_perf<zmm_32x32_multiply>(2048,4096)); if(test_num) break;
#ifdef ALL_TESTS
        case 104: RUN_TEST1(test_block_mult_perf<cacheline_block_multiply>(2048,4096)); if(test_num) break;
        case 106: RUN_TEST1(test_block_mult_perf<cacheline_block_multiply>(128,128)); if(test_num) break;
        case 105: RUN_TEST1(test_block_mult_perf<transp_block_multiply>(2048,4096)); if(test_num) break;
        case 110: RUN_TEST1(test_block_transp_mult_perf<double>(2048,4096)); if(test_num) break;

        case 53: RUN_TEST1(test_mult_perf<Eigen::MatrixXd>(1024,1024)); if(test_num) break;
        case 54: RUN_TEST1(test_mult_perf<my::Mat<double>>(1024,1024)); if(test_num) break;
        case 55: RUN_TEST1(test_block_mult_perf<naive_block_multiply>(1024,1024)); if(test_num) break;
        case 71: RUN_TEST1(test_block_transp_mult_perf<double>(1024,1024)); if(test_num) break;

        case 56: RUN_TEST1(test_mult_perf<Eigen::MatrixXd>(2048,1024)); if(test_num) break;
        case 57: RUN_TEST1(test_mult_perf<my::Mat<double>>(2048,1024)); if(test_num) break;
        case 58: RUN_TEST1(test_block_mult_perf<naive_block_multiply>(2048,1024)); if(test_num) break;
        case 72: RUN_TEST1(test_block_transp_mult_perf<double>(2048,1024)); if(test_num) break;

        case 59: RUN_TEST1(test_mult_perf<Eigen::MatrixXd>(2048,2048)); if(test_num) break;
        case 60: RUN_TEST1(test_mult_perf<my::Mat<double>>(2048,2048)); if(test_num) break;
        case 61: RUN_TEST1(test_block_mult_perf<naive_block_multiply>(2048,2048)); if(test_num) break;
        case 73: RUN_TEST1(test_block_transp_mult_perf<double>(2048,2048)); if(test_num) break;

        case 62: RUN_TEST1(test_opt_small_multiply()); if(test_num) break;

        // vector outer product
        case 30: RUN_TEST1(test_mult_perf<my::Mat<double>>(10000,1)); if(test_num) break;
        case 31: RUN_TEST1(test_mult_perf<Eigen::MatrixXd>(10000,1)); if(test_num) break;

        // vector, matrix add
        case 40: RUN_TEST1(test_add_perf<my::Mat<double>>(1,100000000)); if (test_num) break;
        case 41: RUN_TEST1(test_add_perf<Eigen::MatrixXd>(1,100000000)); if (test_num) break;
        case 42: RUN_TEST1(test_add_perf<my::Mat<double>>(1000,1000)); if (test_num) break;
        case 43: RUN_TEST1(test_add_perf<Eigen::MatrixXd>(1000,1000)); if (test_num) break;
#endif
    }

    return 0;
}
