// Notes:
// 1. Delay malloc as much as possible
//  add size based constructor
// Run valgrind, address sanitizer, undefined sanitizer
// clang-tidy (linter)
// sw-dl-tensorrt

#include <initializer_list>
#include <iostream>

// #define PRETTY_FUNCTION

namespace my {
    template <class T>
    class vector {
        int capacity_{0};
        int len_{0};
        T *first{nullptr};
        T *beyond_last{nullptr};

        void expand();

        public:
        vector() = default;
        vector(int);
        vector(my::vector<T>const&); // Copy constructor
        vector(my::vector<T>&&); // Move constructor
        vector(const std::initializer_list<T>&); // Constructor with initializer list
        void operator=(const my::vector<T>&); // Copy assignment
        vector<T>& operator=(vector<T>&&); // Move assignment
        virtual ~vector(); // Desctructor

        void push_back(const T&);
        void push_back(T&&);
        T& at(int) const;
        int size() const;
        int capacity() const;
        bool operator==(my::vector<T>const&) const;
        inline bool operator!=(my::vector<T>const&) const;

        typedef T* iterator;
        typedef T value_type;
        typedef std::input_iterator_tag iterator_category;
        iterator begin();
        iterator end();
    };

    /*
    // Default constructor
    template<class T> vector<T>::vector() {
        capacity = 4;
        first = (T*)malloc(len_of(T)*capacity);
        beyond_last = first;
        len_ = 0;
    }
    */

    template<class T> vector<T>::vector(int count) : capacity_(count), len_(0), first(new T[count]), beyond_last(first+count)
    {
    }

    // Copy constructor
    template<class T> vector<T>::vector(const my::vector<T> &v) : capacity_{v.size()}, len_{v.size()}, first{new T[len_]}, beyond_last{first+len_}
    {
        for (int i = 0; i < v.size(); i++) { // range based for loop
            first[i] = v.at(i); // use std copy, iterators, back_inserter
        }
    }

    // Move constructor
    // Dont forget to assign pointers of object moved from to null
    template<class T> vector<T>::vector(my::vector<T> &&v) : capacity_(v.capacity_), len_(v.len_), first(v.first), beyond_last(v.beyond_last) {
        v.capacity_ = 0;
        v.len_ = 0;
        v.first = nullptr;
        v.beyond_last = nullptr;
    }

    // Constructor with initializer list
    template<class T> vector<T>::vector(const std::initializer_list<T> &l) {
        for (auto &el: l) {
            this->push_back(el);
        }
    }

    template<class T> vector<T>::~vector () {
        if (capacity_)
            delete[] first;
    }

    // Copy Assignment operator
    template<class T> void vector<T>::operator=(const vector<T> &src) {
        // std::cout << "Calling copy assignment operator" << std::endl;
        len_ = 0;
        beyond_last = first;
        /*
        len_ = src.len_;
        first = src.first;
        beyond_last = src.beyond_last;
        */
        for (int i = 0; i < src.len_; i++) {
            this->push_back(src.first[i]);
        }
    }

    // Move assignment operator
    template<class T> vector<T>& vector<T>::operator=(vector<T> &&src) {
#ifdef PRETTY_FUNCTION
        std::cout << __PRETTY_FUNCTION__ << std::endl;
#endif
        capacity_ = src.capacity_;
        len_ = src.len_;
        first = src.first;
        beyond_last = src.beyond_last;

        src.capacity_ = 0;
        src.len_ = 0;
        src.first = nullptr;
        src.beyond_last = nullptr;

        return *this;
    }

    template<class T> void vector<T>::expand() {
        // Allocate new array with twice the capcaity and copy current array
        int new_capacity;
        if (capacity_ == 0) {
            new_capacity = 4;
        } else {
            new_capacity = 2*capacity_;
        }
        T *new_first = new T[new_capacity]; // (T*)malloc(len_of(T)*new_capacity);
        T *ptr = first;
        T *new_ptr = new_first;
        for (int i = 0; i < len_; i++) {
            *new_ptr = *ptr;
            new_ptr++;
            ptr++;
        }
        capacity_ = new_capacity;
        first = new_first;
        beyond_last = new_ptr;
    }

    template<class T> void vector<T>::push_back(const T &value) {
#ifdef PRETTY_FUNCTION
        std::cout << __PRETTY_FUNCTION__ << std::endl;
#endif
        if(capacity_ <= len_) {
            expand();
        }
        *beyond_last = value;
        beyond_last++;
        len_++;
    }

    // implementation is same as push_back(T&).
    // except that *beyond_last = value should called the move assignment operator
    //      instead of the copy assignment operator
    template<class T> void vector<T>::push_back(T&& value) {
#ifdef PRETTY_FUNCTION
        std::cout << __PRETTY_FUNCTION__ << std::endl;
#endif
        if(capacity_ <= len_) {
            expand();
        }
        *beyond_last = value; // should call the move assignment operator
        beyond_last++;
        len_++;
    }

    template<class T>
    T& vector<T>::at(int pos) const {
        return first[pos];
    }

    template<class T>
    int vector<T>::size() const {
        return len_;
    }

    template<class T>
    int vector<T>::capacity() const {
        return capacity_;
    }

    template<class T>
    bool vector<T>::operator==(vector<T> const &b) const {
        if (size() != b.size())
            return false;

        for (int i = 0; i < size(); i++) {
            if (first[i] != b.first[i]) {
                return false;
            }
        }

        return true;
    }

    template<class T>
    inline bool vector<T>::operator!=(vector<T> const &b) const {
        return (!(*this == b));
    }

    template<class T>
    typename vector<T>::iterator vector<T>::begin() {
        return first;
    }

    template<class T>
    typename vector<T>::iterator vector<T>::end() {
        return beyond_last;
    }

    my::vector<int> random_vector (int len) {
        my::vector<int> v;
        for (int i = 0; i < len; i++) {
            v.push_back(rand());
        }
        return v;
    }
}
