#include "my_vector.h"
#include <iostream>
#include <chrono>
#include <vector>
#include <utility>
#include <algorithm>
#include <cstdlib>
#include <cassert>
#include <string>
#include <iterator>
#include <typeinfo>
#include <type_traits>
#include "test_utils.h"
#include "my_list.h"
#include <list>

using namespace std::chrono;

template <class T>
void fill_container (T& container, int len) {
    for (int i = 0; i < len; i++) {
        container.push_back(i);
    }
}

void print(const my::vector<int> &v) {
    for (int i = 0; i < v.size(); i++) {
        std::cout << v.at(i) << ", ";
    }
    std::cout<<std::endl;
}

void print(const my::list<int> &l) {
    for (auto &el: l) {
        std::cout << el << ", ";
    }
    std::cout << std::endl;
}

void test0 () {
    my::vector<int> v;
    for (int i = 0; i < 10; i++) {
        v.push_back(i);
    }

    print(v);

    my::vector<int> v_copy(v);

    print(v_copy);
}

template <class T>
bool test_push_back () {
    long nelem = 100000000;

    T container;

    auto start = high_resolution_clock::now();
    for (int i = 0; i < nelem; i++) {
        container.push_back(i);
    }
    auto stop = high_resolution_clock::now();
    auto duration = duration_cast<microseconds>(stop-start);

    typename T::iterator iter_container = container.begin();
    for (int i = 0; i < nelem; i++) {
        if (*iter_container != i) {
            return false;
        }
        iter_container++;
    }

    std::cout << "  push_back with container took " << duration.count() << " us." << std::endl;

    return true;
}

/*
bool test_push_back () {
    long nvec = 100000000;
    long nelem = 1;
    my::vector<my::vector<int> > v;
    auto start = high_resolution_clock::now();
    for (int i = 0; i < nvec; i++) {
        my::vector<int> v1;
        for (int j = 0; j < nelem; j++) {
            v1.push_back(j);
        }
        v.push_back(v1);
    }
    auto stop = high_resolution_clock::now();
    auto duration = duration_cast<microseconds>(stop - start);
    std::cout << "  push_back with  my::vector: " << duration.count() << " us. ";

    std::vector<std::vector<int> > v_std;
    start = high_resolution_clock::now();
    for (int i = 0; i < nvec; i++) {
        std::vector<int> v1;
        for (int j = 0; j < nelem; j++) {
            v1.push_back(j);
        }
        v_std.push_back(v1);
    }
    stop = high_resolution_clock::now();
    duration = duration_cast<microseconds>(stop - start);
    std::cout << "std::vector: " << duration.count() << " us." << std::endl;

    return true;
}
*/

void test2 () {
    // Test vector of vector of vector
    my::vector<int> v1, v2, v3{7,8,9};
    v1.push_back(1);
    v1.push_back(2);
    v1.push_back(3);

    v2.push_back(4);
    v2.push_back(5);
    v2.push_back(6);

    print(v1);
    print(v2);
    print(v3);

    my::vector<my::vector<int> > vv1;
    vv1.push_back(v1);
    vv1.push_back(v2);

    print(vv1.at(0));
    print(vv1.at(1));

    my::vector<my::vector<my::vector<int> > > vvv1, vvv2;
    vvv1.push_back(vv1);

    vvv2 = vvv1;

    print(vvv2.at(0).at(0));
    print(vvv2.at(0).at(1));
}

int tmp () {
    return 1;
}

my::vector<int> tmp1 () {
    my::vector<int> x{1,2,3};
    return x;
}

void test3 () {
    my::vector<int> v;
    std::cout << "Should call push_back(int&&) - which inturn calls the move assignment operator" << std::endl;
    v.push_back(tmp()); // why isnt move assignment operator getting called?
    print(v);

    std::cout << "Should call the move assignment operator" << std::endl;
    my::vector<int> &&v1 = tmp1(); // Why didnt this call the move assignment operator?
    print(v1);

    my::vector<int> v2{4,5,6}, v3;
    std::cout << "Should call the copy assignment operator" << std::endl;
    v3 = v2; // Calls the copy assignment operator as expected
    print(v3);
}

template <class T>
bool test_sort () {
    // my::vector<int> v_my;
    std::vector<int> v_std;
    T container;

    for (int i = 0; i < 100000000; i++) {
        int r = rand();
        // v_my.push_back(r); // Swap these 2 and see if it changes perf
        v_std.push_back(r);
        container.push_back(r);
    }

    // auto start = high_resolution_clock::now();
    std::sort(v_std.begin(), v_std.end());
    // auto stop = high_resolution_clock::now();
    // auto duration_std = duration_cast<microseconds>(stop - start);

    auto start = high_resolution_clock::now();
    // std::sort(v_my.begin(), v_my.end());
    std::sort(container.begin(), container.end());
    auto stop = high_resolution_clock::now();
    auto duration_cont = duration_cast<microseconds>(stop - start);

    auto iter_container = container.begin();
    auto iter_std = v_std.begin();
    assert(container.size() == v_std.size());
    for (; iter_container != container.end(); iter_container++, iter_std++) {
        if (*iter_container != *iter_std)
            return false;
    }

    std::cout << "  Sort with container took " << duration_cont.count() << " us." << std::endl;

    return true;
}

bool test_swap () {
    bool result = true;
    my::vector<int> v1{1,2,3}, v2{4,5,6}, v1_copy = v1, v2_copy = v2;
    std::swap(v1, v2);
    
    result &= (v2 == v1_copy);
    result &= (v1 == v2_copy);

    return result;
}

bool test_equality () {
    // Vector of ints
    bool result = true;
    my::vector<int> v1{1,2,3}, v2{4,5,6}, v3{1,2,3};

    result &= (v1 != v2);
    result &= (v1 == v3);

    // Vector of vector of ints
    my::vector<my::vector<int> > vv1 = {{1,2,3},{4,5,6}}, vv2 = {{1,2,3},{1,2,3}}, vv3 = {{1,2,3},{4,5,6}};
    result &= (vv1 != vv2);
    result &= (vv1 == vv3);

    // Different lengths
    v1 = {1,2,3}, v2 = {1,2};
    result &= (v1 != v2);

    return result;
}

bool test_iterator () {
    my::vector<int> v1{0,1,2,3,4,5,6,7,8,9};

    int i = 0;
    for (auto &val:v1) {
        if (val != i)
            return false;
        i++;
    }

    i = 0;
    for (my::vector<int>::iterator iter = v1.begin(); iter < v1.end(); iter++) {
        if (*iter != i)
            return false;
        i++;
    }
    return true;
}

template <class T>
bool test_back_inserter () {
    T v1 = {0,1,2,3,4,5}, v2 = {6,7,8,9,10}, v3 = {0,1,2,3,4,5,6,7,8,9,10};
    std::copy(v2.begin(), v2.end(), std::back_inserter(v1));

    if (v1 == v3)
        return true;
    else
        return false;
}

template <class T>
bool test_reverse () {
    T v1, v2;
    for (int i = 0; i < 10; i++) {
        v1.push_back(i);
        v2.push_back(9-i);
    }

    std::reverse(v1.begin(), v1.end());
    if (v1 == v2)
        return true;
    else
        return false;
}

bool test_iterator_traits () {
    std::iterator_traits<my::vector<int>::iterator>::value_type tmp;
    std::iterator_traits<my::vector<int>::iterator>::iterator_category category;

    std::cout << "  Category: " << typeid(category).name() << std::endl;

    if (typeid(tmp).name() != typeid(int).name())
        return false;

    return true;
}

/*
bool test_vec_assign () {
    std::vector<int> v_std{1,2,3};
    my::vector<int> v_my;

    v_my = v_std;
    return true;
}
*/

bool test_type_traits () {
    bool result = true;
    result &= (std::has_virtual_destructor<my::vector<int> >::value == false);
    result &= (std::is_assignable<my::vector<int>, my::vector<int> >::value == true);
    result &= (std::is_assignable<my::vector<int>, std::vector<int> >::value == false);
    result &= (std::is_constructible<my::vector<int> >::value == true);
    result &= (std::is_copy_constructible<my::vector<int> >::value == true);
    result &= (std::is_move_constructible<my::vector<int> >::value == true);
    result &= (std::is_move_assignable<my::vector<int> >::value == true);
    result &= (std::is_copy_assignable<my::vector<int> >::value == true);
    return result;
}

template <class T>
bool test_erase() {
    T l1;
    std::list<int> l2;
    for (int i = 0; i < 100000; i++) {
        l1.push_back(i);
        l2.push_back(i);
    }

    srand(0);
    auto start = high_resolution_clock::now();
    for (int i = 0; i < 100000; i++) {
        auto l1_iter = l1.begin();
        int size = l1.size();
        int n = rand() % size;
        for (int j = 0; j < n; j++) {
            ++l1_iter;
        }
        l1.erase(l1_iter);
    }
    auto stop = high_resolution_clock::now();
    auto duration = duration_cast<microseconds>(stop-start);

    for (int i = 0; i < 100000; i++) {
        auto l2_iter = l2.begin();
        int size = l2.size();
        int n = rand() % size;
        for (int j = 0; j < n; j++) {
            ++l2_iter;
        }
        l2.erase(l2_iter);
    }

    std::cout << "  erase with container took " << duration.count() << " us." << std::endl;

    if (l1 != l2)
        return false;
    else
        return true;
}

int main (int argc, char **argv) {
    int test_num = 0;
    if (argc > 1) {
        test_num = std::stoi(argv[1]);
    }

    // test0 ();
    // test1 ();
    // test2 ();
    // test3 ();
    switch(test_num) {
        case 0:
        case 1: RUN_TEST(test_equality); if (test_num) break;
        case 2: RUN_TEST(test_swap);     if (test_num) break;
        case 3: RUN_TEST(test_sort<my::vector<int> >); if (test_num) break;
        case 4: RUN_TEST(test_sort<std::vector<int> >); if (test_num) break;
        case 5: RUN_TEST(test_iterator); if (test_num) break;
        case 6: RUN_TEST(test_iterator_traits); if (test_num) break;
        case 7: RUN_TEST(test_back_inserter<my::vector<int> >); if (test_num) break;
        case 8: RUN_TEST(test_push_back<my::vector<int> >); if (test_num) break;
        case 9: RUN_TEST(test_push_back<std::vector<int> >); if (test_num) break;
        case 10: RUN_TEST(test_reverse<my::vector<int> >); if (test_num) break;
        case 11: RUN_TEST(test_type_traits); if (test_num); if (test_num) break;

        // list tests
        case 50: RUN_TEST(test_push_back<my::list<int> >); if (test_num) break;
        case 51: RUN_TEST(test_push_back<std::list<int> >); if (test_num) break;
        case 52: RUN_TEST(test_reverse<my::list<int> >); if (test_num) break; // TODO: seg fault
        case 53: RUN_TEST(test_erase<my::list<int> >); if (test_num) break;
        case 54: RUN_TEST(test_erase<std::list<int> >); if (test_num) break;
    }

    return 0;
}
