#ifndef __MY_LIST_H__
#define __MY_LIST_H__

#include <initializer_list>
#include <iostream>
#include <iterator>
#include <memory>
#include <utility>
#include <list>

// #define PRETTY_FUNCTION

namespace my {

    class list_node {
        public:
        std::unique_ptr<list_node> _next{nullptr};
        list_node *_prev{nullptr};
        virtual ~list_node() {
#ifdef PRETTY_FUNCTION
            std::cout << __PRETTY_FUNCTION__ << std::endl;
#endif
        };
    };

    template <class T>
    class list_node_with_data : public list_node {
        public:
        T _data;

        list_node_with_data(T const&);
        list_node_with_data(T &&);
        ~list_node_with_data () {
#ifdef PRETTY_FUNCTION
            std::cout << __PRETTY_FUNCTION__ << ", _data = " << _data << std::endl;
#endif
         }
    };

    template <class T>
    list_node_with_data<T>::list_node_with_data(T const& value) : _data(value) {}

    template <class T>
    list_node_with_data<T>::list_node_with_data(T && value) : _data(std::move(value)) {}

    template <class T>
    class list_iterator : public std::iterator<std::bidirectional_iterator_tag, list_node> {
        public:
        list_node *_ptr{nullptr}; // ok to make this public? insert function needs to access the pointer

        list_iterator() = default;
        list_iterator(list_node *ptr) : _ptr(ptr) {}
        T& operator*() {
            return (reinterpret_cast<list_node_with_data<T>*>(_ptr))->_data;
        }
        list_iterator& operator++() {
            _ptr = _ptr->_next.get();
            return *this;
        }
        list_iterator& operator--() {
            _ptr = _ptr->_prev;
            return *this;
        }
        list_iterator operator++(int) {
            list_iterator out_iter = *this;
            ++(*this);
            return out_iter;
        }
        list_iterator operator--(int) {
            list_iterator out_iter = *this;
            --(*this);
            return out_iter;
        }
        bool operator==(list_iterator const& other_iter) {
            return other_iter._ptr == _ptr;
        }
        bool operator!=(list_iterator const& other_iter) {
            return other_iter._ptr != _ptr;
        }
    };

    template <class T>
    class list {
        int _len{0};
        std::unique_ptr<list_node> _first{new list_node};
        list_node *_beyond_last{_first.get()};

        public:
        list() = default;
        list(list<T> const&);
        list(list<T> &&);
        virtual ~list();

        typedef T value_type;
        typedef list_iterator<T> iterator;
        iterator begin() const;
        iterator end() const;

        int size() const;
        void push_back(T const& value);
        void push_back(T && value);
        void push_front(T const& value);
        void push_front(T && value);
        void insert(iterator pos, T const& value);
        void insert(iterator pos, T && value);
        void erase(iterator pos);
        void erase(iterator first, iterator last); // TODO
        void pop_back(); // TODO
        void pop_front(); // TODO
        template <class U> bool operator==(U const&) const;
        template <class U> bool operator!=(U const&) const;
    };

    template <class T>
    list<T>::~list () {
        // for (int i = 0; i < size(); i++) {
        //     erase(begin());
        // }
        list_node *ptr = _beyond_last->_prev;
        while (ptr != nullptr) {
            ptr->_next = nullptr;
            ptr = ptr->_prev;
        }
        _first = nullptr;
    }

    template <class T>
    int list<T>::size() const {
        return _len;
    }

    /*
    template <class T>
    void list<T>::erase(iterator pos) {
        // Move the unique_ptr only at the end
        list_node<T> *parent_node = (*pos)._prev;
    }
    */

    template <class T>
    void list<T>::push_back(T const& value) {
#ifdef PRETTY_FUNCTION
        std::cout << __PRETTY_FUNCTION__ << std::endl;
#endif
        // static int count = 5;
        // if (count == 0)
        //     throw 0;
        insert(end(), value);
        // count--;
    }

    template <class T>
    void list<T>::push_back(T&& value) {
#ifdef PRETTY_FUNCTION
        std::cout << __PRETTY_FUNCTION__ << std::endl;
#endif
        insert(begin(), value);
    }

    template <class T>
    void list<T>::push_front(T const& value) {
        insert(begin(), value);
    }

    template <class T>
    void list<T>::push_front(T && value) {
        insert(++(begin()), std::move(value));
    }

    template <class T>
    list<T>::list(list<T> const& value) {
        auto value_iter = value.begin();
        for (; value_iter != value.end(); ++value_iter) {
            // *value_iter returns by reference => constructor is not called => no exception
            // push_back takes the argument by reference => constructor not called while passing the argument
            // push_back function can throw since it calls the constructor of value inside. What if it throws after pushing back not all elements of the list?
            // in that case _first is going to be destroyed, destroying the other nodes that have been created in a "chain reaction"
            push_back(*value_iter); 
        }
    }

    template <class T>
    void list<T>::insert(iterator pos, T const& value) {
        // TODO: check that pos is not a null ptr
#ifdef PRETTY_FUNCTION
        std::cout << __PRETTY_FUNCTION__ << std::endl;
#endif
        // If new throws/fails, we are fine since we have not modified our list.
        // Any memory allocated will be freed.
        std::unique_ptr<list_node> new_node(new list_node_with_data<T>(value)); // Note: base class pointer pointing to derived class object
        // Does *pos have a parent node? If this is the first node in the list, there will not be a parent node
        bool first_node = (pos._ptr == _first.get());
        // This block (both if and else) do not call any constructors or functions. So it cannot throw an exception
        list_node *cur_node = pos._ptr;
        if (first_node) {
            cur_node->_prev = new_node.get();
            new_node->_next = std::move(_first);
            _first = std::move(new_node);
        } else {
            list_node *parent_node = cur_node->_prev;
            new_node->_prev = parent_node;
            cur_node->_prev = new_node.get();
            new_node->_next = std::move(parent_node->_next);
            parent_node->_next = std::move(new_node);
        }
        // Safe to increment _len since we are guaranteed to have inserted if we reached this point
        _len++;
    }

    template <class T>
    void list<T>::insert(iterator pos, T && value) {
#ifdef PRETTY_FUNCTION
        std::cout << __PRETTY_FUNCTION__ << std::endl;
#endif
        std::unique_ptr<list_node> new_node(new list_node_with_data<T>(std::move(value)));
        bool first_node = (pos._ptr == _first.get());
        list_node *cur_node = pos._ptr;
        if (first_node) {
            cur_node->_prev = new_node.get();
            new_node->_next = std::move(_first);
            _first = std::move(new_node);
        } else {
            list_node *parent_node = cur_node->_prev;
            new_node->_prev = parent_node;
            cur_node->_prev = new_node.get();
            new_node->_next = std::move(parent_node->_next);
            parent_node->_next = std::move(new_node);
        }
        _len++;
    }

    template <class T>
    list_iterator<T> list<T>::begin() const {
        return iterator(_first.get());
    }

    template <class T>
    list_iterator<T> list<T>::end() const {
        return iterator(_beyond_last);
    }

    template <class T>
    void list<T>::erase(iterator pos) {
        bool is_first_node = (pos._ptr == _first.get());
        list_node* cur_node = pos._ptr;
        std::unique_ptr<list_node> child_node = std::move(cur_node->_next);
        if (is_first_node) {
            child_node.get()->_prev = nullptr;
            _first = std::move(child_node);
        } else {
            list_node* parent_node = cur_node->_prev;
            child_node.get()->_prev = parent_node;
            parent_node->_next = std::move(child_node);
        }
        _len--;
    }

    template <class T>
    template <class U>
    bool list<T>::operator==(U const& other_list) const {
        if (other_list.size() != size())
            return false;

        auto other_list_iter = other_list.begin();
        auto iter = begin();
        for (; iter != end(); ++iter, ++other_list_iter) {
            if (*iter != *other_list_iter) {
                return false;
            }
        }
        return true;
    }

    template <class T>
    template <class U>
    bool list<T>::operator!=(U const& other_list) const {
        return !(*this == other_list);
    }
}

#endif
