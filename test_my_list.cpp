#include "my_list.h"
#include <iostream>
#include "test_utils.h"
#include <list>
#include <memory>

void print_list (my::list<int> &l) {
    for (auto &el: l) {
        std::cout << el << ", ";
    }
    std::cout << std::endl;
}

bool test_constructor () {
    my::list<int> l;
    if (l.size() == 0)
        return true;
    else
        return false;
}

bool test_push_back () {
    my::list<int> l;
    for (int i = 0; i < 10; i++) {
        l.push_back(i);
    }

    for (auto &el : l) {
        std::cout << el << " ";
    }
    std::cout<<std::endl;

    for (auto iter = l.begin(); iter != l.end(); ++iter) {
        std::cout << *iter << " ";
    }
    std::cout<<std::endl;

    return true;
}

/*
class A {
    int _i;
    public:
    A(int i) {
        _i = i;
        std::cout << __PRETTY_FUNCTION__ << std::endl;
    }
    A(A const& val) {
        _i = val._i;
        std::cout << __PRETTY_FUNCTION__ << "_i = " << _i << std::endl;
    }
    A(A&& val) {
        _i = val._i;
        std::cout << __PRETTY_FUNCTION__ << "_i = " << _i << std::endl;
    }
    ~A() {
        std::cout << __PRETTY_FUNCTION__ << "_i = " << _i << std::endl;
    }
};

bool test_destructor() {
    my::list<A> l;
    for (int i = 0; i < 1; i++) {
        l.push_back(A(i));
    }
    std::cout << "push_back complete\n";
}
*/

bool test_insert () {
    my::list<int> l;
    auto iter = l.begin();
    l.insert(iter, 0);
    l.insert(iter, 1);
    l.insert(iter, 2);
    return true;
}

bool test_push_front () {
    my::list<int> l_my;
    std::list<int> l_std;

    for (int i = 0; i < 10; i++) {
        l_my.push_front(i);
        l_std.push_front(i);
    }

    bool result = true;

    auto iter1 = l_my.begin();
    auto iter2 = l_std.begin();
    for (; iter1 != l_my.end(); ++iter1, ++iter2) {
        if (*iter1 != *iter2) {
            result = false;
        }
    }

    return result;
}

bool test_copy_constructor () {
    my::list<int> l;
    for (int i = 0; i < 10; i++) {
        l.push_front(i);
    }

    std::cout << "Finished creating l\n";

    try {
        // my::list<int> l_copy = l;
        std::unique_ptr<my::list<int> > l_copy(new my::list<int>(l));
        // new my::list<int>(l);
        std::cout << "Finished copying\n";
    } catch(int v) {
    }
}

int main (int argc, char **argv) {
    int test_num = 0;
    if (argc > 1) {
        test_num = std::stoi(argv[1]);
    }
    switch(test_num) {
        case 0:
        case 1: RUN_TEST(test_constructor); if (test_num) break;
        case 2: RUN_TEST(test_push_back);if (test_num) break;
        case 3: RUN_TEST(test_insert);if (test_num) break;
        // case 4: RUN_TEST(test_destructor);if (test_num) break;
        case 6: RUN_TEST(test_push_front);if (test_num) break;
        case 7: RUN_TEST(test_copy_constructor);if (test_num) break;
    }
    return 0;
}
