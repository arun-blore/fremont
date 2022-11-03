#include <utility>

using std::pair;

namespace my {
    template <class I1, class I2, class T1, class T2>
    class pair_iter {
        private:
        I1 m_iter1;
        I2 m_iter2;

        public:
        pair_iter (I1 iter1, I2 iter2) : m_iter1(iter1), m_iter2(iter2) {}
        void operator++() {
            ++m_iter1;
            ++m_iter2;
        }

        bool operator==(pair_iter<I1, I2, T1, T2> &iter_other) {
            return (m_iter1 == iter_other.m_iter1);
        }

        bool operator!=(pair_iter<I1, I2, T1, T2> &iter_other) {
            return !(*this == iter_other);
        }

        pair<T1, T2> operator*() {
            return pair<T1, T2>(*m_iter1, *m_iter2);
        }
    };
}
