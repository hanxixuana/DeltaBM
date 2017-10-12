//
// Created by xixuan on 10/10/16.
//

#ifndef DBM_CODE_DATA_SET_H
#define DBM_CODE_DATA_SET_H

#include "matrix.h"

//#ifndef _DEBUG_DATA_SET
//#define _DEBUG_DATA_SET
//#endif

namespace dbm {

    template<typename T>
    class Data_set {
    private:
        T portion_for_test;

        int no_samples, no_features, no_train_samples, no_test_samples;

        Matrix<T> *train_x = nullptr;
        Matrix<T> *train_y = nullptr;
        Matrix<T> *test_x = nullptr;
        Matrix<T> *test_y = nullptr;

    public:

        int random_seed;

        Data_set(const Matrix<T> &data_x, const Matrix<T> &data_y, T test_portion, int random_seed = -1);

        ~Data_set();

        const Matrix<T> &get_train_x() const;

        const Matrix<T> &get_train_y() const;

        const Matrix<T> &get_test_x() const;

        const Matrix<T> &get_test_y() const;

        void shuffle_all(int random_seed = -1);

    };

}

#endif //DBM_CODE_DATA_SET_H



