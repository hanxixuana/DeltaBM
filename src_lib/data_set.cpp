//
// Created by Xixuan Han on 10/10/2016.
//

#include "data_set.h"
#include "tools.h"

#include <cassert>

namespace dbm {

    template
    class Data_set<double>;

    template
    class Data_set<float>;

}

namespace dbm {

    template<typename T>
    Data_set<T>::Data_set(const Matrix<T> &data_x,
                          const Matrix<T> &data_y,
                          T test_portion,
                          int random_seed):
            portion_for_test(test_portion),
            random_seed(random_seed) {

        no_samples = data_x.get_height();
        no_features = data_x.get_width();

        #ifdef _DEBUG_DATA_SET
            assert(no_samples == data_y.get_height());
        #endif

        no_test_samples = (int)(no_samples * portion_for_test);
        no_train_samples = no_samples - no_test_samples;

        int *row_inds = new int[no_samples],
                *train_row_inds = new int[no_train_samples],
                *test_row_inds = new int[no_test_samples];
        for (int i = 0; i < no_samples; ++i) {
            row_inds[i] = i;
        }

        if(random_seed < 0)
            shuffle(row_inds, no_samples);
        else
            shuffle(row_inds, no_samples, (unsigned int) random_seed);

        for (int i = 0; i < no_train_samples; ++i) {
            train_row_inds[i] = row_inds[i];
        }
        for (int i = 0; i < no_test_samples; ++i) {
            test_row_inds[i] = row_inds[no_train_samples + i];
        }

        Matrix<T> temp_1 = data_x.rows(train_row_inds, no_train_samples);
        train_x = new Matrix<T>(no_train_samples, no_features, 0);
        copy(temp_1, *train_x);

        Matrix<T> temp_3 = data_y.rows(train_row_inds, no_train_samples);
        train_y = new Matrix<T>(no_train_samples, 1, 0);
        copy(temp_3, *train_y);

        Matrix<T> temp_2 = data_x.rows(test_row_inds, no_test_samples);
        test_x = new Matrix<T>(no_test_samples, no_features, 0);
        copy(temp_2, *test_x);

        Matrix<T> temp_4 = data_y.rows(test_row_inds, no_test_samples);
        test_y = new Matrix<T>(no_test_samples, 1, 0);
        copy(temp_4, *test_y);

        delete[] row_inds;
        delete[] train_row_inds;
        delete[] test_row_inds;

    }

    template<typename T>
    Data_set<T>::~Data_set() {
        delete train_x;
        delete train_y;
        delete test_x;
        delete test_y;
    }

    template<typename T>
    const Matrix<T> &Data_set<T>::get_train_x() const {
        const Matrix<T> &ref_to_train_x = *train_x;
        return ref_to_train_x;
    }

    template<typename T>
    const Matrix<T> &Data_set<T>::get_train_y() const {
        const Matrix<T> &ref_to_train_y = *train_y;
        return ref_to_train_y;
    }

    template<typename T>
    const Matrix<T> &Data_set<T>::get_test_x() const {
        const Matrix<T> &ref_to_test_x = *test_x;
        return ref_to_test_x;
    }

    template<typename T>
    const Matrix<T> &Data_set<T>::get_test_y() const {
        const Matrix<T> &ref_to_test_y = *test_y;
        return ref_to_test_y;
    }

    template<typename T>
    void Data_set<T>::shuffle_all(int random_seed) {

        Matrix<T> data_x = vert_merge(*train_x, *test_x);
        Matrix<T> data_y = vert_merge(*train_y, *test_y);

        delete train_x;
        delete train_y;
        delete test_x;
        delete test_y;

        int *row_inds = new int[no_samples],
                *train_row_inds = new int[no_train_samples],
                *test_row_inds = new int[no_test_samples];
        for (int i = 0; i < no_samples; ++i) {
            row_inds[i] = i;
        }

        if(random_seed < 0)
            shuffle(row_inds, no_samples);
        else
            shuffle(row_inds, no_samples, (unsigned int) random_seed);

        for (int i = 0; i < no_train_samples; ++i) {
            train_row_inds[i] = row_inds[i];
        }
        for (int i = 0; i < no_test_samples; ++i) {
            test_row_inds[i] = row_inds[no_train_samples + i];
        }

        Matrix<T> temp_1 = data_x.rows(train_row_inds, no_train_samples);
        train_x = new Matrix<T>(no_train_samples, no_features, 0);
        copy(temp_1, *train_x);

        Matrix<T> temp_3 = data_y.rows(train_row_inds, no_train_samples);
        train_y = new Matrix<T>(no_train_samples, 1, 0);
        copy(temp_3, *train_y);

        Matrix<T> temp_2 = data_x.rows(test_row_inds, no_test_samples);
        test_x = new Matrix<T>(no_test_samples, no_features, 0);
        copy(temp_2, *test_x);

        Matrix<T> temp_4 = data_y.rows(test_row_inds, no_test_samples);
        test_y = new Matrix<T>(no_test_samples, 1, 0);
        copy(temp_4, *test_y);

        delete[] row_inds;
        delete[] train_row_inds;
        delete[] test_row_inds;

    }

}



