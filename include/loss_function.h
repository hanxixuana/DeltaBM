//
// Created by xixuan on 10/10/16.
//

#ifndef DBM_CODE_LOSS_FUNCTION_H
#define DBM_CODE_LOSS_FUNCTION_H

//#ifndef _DEBUG_LOSS_FUNCTION
//#define _DEBUG_LOSS_FUNCTION
//#endif

#include "matrix.h"
#include "tools.h"

#define MAX_PROB_BERNOULLI 0.9999999999999
#define MIN_PROB_BERNOULLI 0.0000000000001

#define MAX_IND_DELTA 9999999999999
#define MIN_IND_DELTA -9999999999999

#define MIN_NUMERATOR_TWEEDIE 1e-20

#define TOLERANCE_INVERSE_LINK_FUNC 1e-20

namespace dbm {

    template<typename T>
    class Loss_function {
    private:

        Params params;

    public:


        Loss_function(const Params &params);

        T loss(const Matrix<T> &train_y,
               const Matrix<T> &prediction,
               const char &dist,
               const T beta = 0,
               const int *row_inds = nullptr,
               int no_rows = 0) const;

        T estimate_mean(const Matrix<T> &ind_delta,
                        const char &dist,
                        const int *row_inds = nullptr,
                        int no_rows = 0) const;

        void link_function(Matrix<T> &in_and_out,
                           char &dist);

        T inversed_link_function(T value, const char &dist);

        void calculate_ind_delta(const Matrix<T> &train_y,
                                 const Matrix<T> &prediction,
                                 Matrix<T> &ind_delta,
                                 const char &dist,
                                 const int *row_inds = nullptr,
                                 int no_rows = 0);

        Matrix<T> first_comp(const Matrix<T> &train_y,
                             const Matrix<T> &prediction,
                             const char loss_function_type,
                             const int *row_inds = nullptr,
                             int no_rows = 0);

        Matrix<T> second_comp(const Matrix<T> &train_y,
                             const Matrix<T> &prediction,
                              const char loss_function_type,
                              const int *row_inds = nullptr,
                             int no_rows = 0);

        T loss_reduction(const T first_comp_in_loss,
                         const T second_comp_in_loss,
                         const T beta,
                         const char loss_function_type);

    };

}

#endif //DBM_CODE_LOSS_FUNCTION_H





