#include <papi.h>
#include <vector>
#include <array>
#include <iostream>

using std::vector;
using std::array;
using std::cout;
using std::endl;

constexpr int array_size = 32*32;
constexpr int narrays = 2048/32 * 4096/32;
constexpr int size = array_size * narrays;

void test_mem () {
    double *mem = new double[size];
    double d = 0;

    PAPI_hl_region_begin("write_mem");
    for (int i = 0; i < size; i++) {
        mem[i] = d;
        d++;
    }
    PAPI_hl_region_end("write_mem");

    PAPI_hl_region_begin("incr_mem");
    for (int i = 0; i < size; i++) {
        mem[i]++;
    }
    PAPI_hl_region_end("incr_mem");

    delete mem;
}

void test_vec_array() {
    vector<array<double, array_size>> v(narrays);

    double d = 0;
    PAPI_hl_region_begin("write_vec_array");
    for (auto &a : v) {
        for (auto &el : a) {
            el = d;
            d++;
        }
    }
    PAPI_hl_region_end("write_vec_array");

    PAPI_hl_region_begin("incr_vec_array");
    for (auto &a : v) {
        for (auto &el : a) {
            el++;
        }
    }
    PAPI_hl_region_end("incr_vec_array");
}

void test_vector() {
    vector<double> mem_vector(size);
    double d = 0;

    PAPI_hl_region_begin("write_vector");
    for (auto &el : mem_vector) {
        el = d;
        d++;
    }
    PAPI_hl_region_end("write_vector");

    PAPI_hl_region_begin("write_vector");
    for (auto &el : mem_vector) {
        el++;
    }
    PAPI_hl_region_end("write_vector");
}

void test_stack_mem() {
    const int stack_size = 1000000;
    const int mem_size = stack_size/sizeof(double);
    double *mem = new double[mem_size];
    // double mem[mem_size];
    double d = 0;

    PAPI_hl_region_begin("write_stack_mem");
    for (int i = 0; i < mem_size; i++) {
        mem[i] = d;
        d++;
    }
    PAPI_hl_region_end("write_stack_mem");

    PAPI_hl_region_begin("incr_stack_mem");
    for (int i = 0; i < mem_size; i++) {
        mem[i]++;
    }
    PAPI_hl_region_end("incr_stack_mem");

    delete mem;
}

int main () {
    test_vec_array();
    test_vector();
    test_mem();
    test_stack_mem();
    return 0;
}
