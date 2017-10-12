//
// Created by xixuan on 10/10/16.
//

#include "base_learner_trainer.h"

#include <cassert>
#include <iostream>
#include <cmath>
#include <algorithm>
#include <chrono>

namespace dbm {

    template
    class Mean_trainer<double>;

    template
    class Mean_trainer<float>;

    template
    class Neural_network_trainer<float>;

    template
    class Neural_network_trainer<double>;

    template
    class Linear_regression_trainer<double>;

    template
    class Linear_regression_trainer<float>;

    template
    class DPC_stairs_trainer<double>;

    template
    class DPC_stairs_trainer<float>;

    template
    class Kmeans2d_trainer<double>;

    template
    class Kmeans2d_trainer<float>;

    template
    class Splines_trainer<double>;

    template
    class Splines_trainer<float>;

    template
    class Fast_tree_trainer<double>;

    template
    class Fast_tree_trainer<float>;

}

// for means
namespace dbm {

    template <typename T>
    Mean_trainer<T>::Mean_trainer(const Params &params) :
            display_training_progress(params.dbm_display_training_progress),
            loss_function(Loss_function<T>(params)) {};

    template <typename T>
    Mean_trainer<T>::~Mean_trainer() {}

    template <typename T>
    void Mean_trainer<T>::train(Global_mean<T> *mean,
                                const Matrix<T> &train_x,
                                const Matrix<T> &ind_delta,
                                const Matrix<T> &prediction,
                                char loss_function_type,
                                const int *row_inds,
                                int no_rows) {
        if(display_training_progress)
            std::cout << "Training Global Mean at "
                      << mean
                      << " number of samples: "
                      << (no_rows != 0 ? std::to_string(no_rows) : "all")
                      << " ... "
                      << std::endl;

        if(row_inds == nullptr) {
            mean->mean = loss_function.estimate_mean(ind_delta,
                                                     loss_function_type);
//
//            mean->mean = 0;

        }
        else {
            #ifdef _DEBUG_BASE_LEARNER_TRAINER
                assert(no_rows > 0);
            #endif
            mean->mean = loss_function.estimate_mean(ind_delta,
                                                     loss_function_type,
                                                     row_inds,
                                                     no_rows);
//
//            mean->mean = 0;
        }

    }

}

// for neural networks
namespace dbm {

    template <typename T>
    Neural_network_trainer<T>::Neural_network_trainer(const Params &params) :
            dbm_remove_rows_containing_nans(params.dbm_remove_rows_containing_nans),
            dbm_min_no_samples_per_bl(params.dbm_min_no_samples_per_bl),
            batch_size(params.nn_batch_size),
            nn_max_iteration(params.nn_max_iteration),
            step_size(params.nn_step_size),
            validate_portion(params.nn_validate_portion),
            no_rise_of_loss_on_validate(params.nn_no_rise_of_loss_on_validate),
            loss_function(Loss_function<T>(params)) {}

    template <typename T>
    Neural_network_trainer<T>::~Neural_network_trainer() {}

    template <typename T>
    inline T Neural_network_trainer<T>::activation_derivative(const T &input) {
        return input * (1 - input);
    }

    template <typename T>
    void Neural_network_trainer<T>::backward(Neural_network<T> *neural_network,
                                             const Matrix<T> &input_output,
                                             Matrix<T> &hidden_output,
                                             T &output_output,
                                             Matrix<T> &hidden_delta,
                                             Matrix<T> &input_delta,
                                             T ind_delta,
                                             T weight) {
        T temp = step_size * weight * (output_output - ind_delta);

        for(int i = 0; i < neural_network->no_hidden_neurons + 1; ++i)
            hidden_delta.assign(0, i,
                                hidden_delta.get(0, i) - temp * hidden_output.get(i, 0));

        for(int i = 0; i < neural_network->no_hidden_neurons; ++i)
            for(int j = 0; j < neural_network->no_predictors + 1; ++j)
                input_delta.assign(i, j,
                                   input_delta.get(i, j) -
                                           temp * neural_network->hidden_weight->get(0, i) *
                                                   activation_derivative(hidden_output.get(i, 0))*
                                                   input_output.get(j, 0));
    }

    template <typename T>
    void Neural_network_trainer<T>::train(Neural_network<T> *neural_network,
                                          const Matrix<T> &train_x,
                                          const Matrix<T> &ind_delta,
                                          int *row_inds,
                                          int no_rows,
                                          const int *col_inds,
                                          int no_cols) {

        if(row_inds == nullptr) {



        }
        else {

            #ifdef _DEBUG_BASE_LEARNER_TRAINER
                assert(no_rows > 0 && neural_network->no_predictors == no_cols);
            #endif

            if(dbm_remove_rows_containing_nans)
                remove_nan_row_inds(row_inds,
                                    no_rows,
                                    col_inds,
                                    no_cols,
                                    train_x);

            if(no_rows < dbm_min_no_samples_per_bl) {
                std::cout << "Too few training samples (" << no_rows << ") without NaNs and Omit the BL." << std::endl;
                for(int i = 0; i < neural_network->no_predictors; ++i)
                    neural_network->col_inds[i] = 0;
                for(int i = 0; i < neural_network->no_hidden_neurons; ++i)
                    for(int j = 0; j < neural_network->no_predictors + 1; ++j)
                        neural_network->input_weight->assign(i, j, NAN);
                for(int i = 0; i < neural_network->no_hidden_neurons + 1; ++i)
                    neural_network->hidden_weight->assign(0, i, NAN);
                return;
            }

            Matrix<T> input_output(neural_network->no_predictors + 1, 1, 0);
            Matrix<T> hidden_output(neural_network->no_hidden_neurons + 1, 1, 0);
            T output_output;

            // no_hidden_neurons * (no_predictors + 1)
            Matrix<T> input_delta(neural_network->no_hidden_neurons,
                                                   neural_network->no_predictors + 1, 0);
            // 1 * (no_hidden_neurons + 1)
            Matrix<T> hidden_delta(1, neural_network->no_hidden_neurons + 1, 0);

            for(int i = 0; i < no_cols; ++i)
                neural_network->col_inds[i] = col_inds[i];

            int no_validate = int(no_rows * validate_portion),
                    no_train = no_rows - no_validate;
            int *validate_row_inds = new int[no_validate],
                    *train_row_inds = new int[no_train];

            unsigned int *seeds = new unsigned int[nn_max_iteration];
            for(int i = 0; i < nn_max_iteration; ++i)
                seeds[i] = (unsigned int)(std::rand() / (RAND_MAX / 1e5));

            T weight_sum_ino_batch = 0,
                    validate_weight_sum = 0;
            for(int i = 0; i < no_validate; ++i) {
                validate_row_inds[i] = row_inds[i];
                validate_weight_sum += ind_delta.get(validate_row_inds[i], 1);
            }

            for(int i = no_validate; i < no_rows; ++i)
                train_row_inds[i - no_validate] = row_inds[i];

            int no_batch = no_train / batch_size, row_index, count_to_break = 0;
            T last_mse = std::numeric_limits<T>::max(), mse;
//            T first_mse;

            for(int i = 0; i < nn_max_iteration; ++i) {
                shuffle(train_row_inds,
                        no_train,
                        seeds[i]);
                for(int j = 0; j < no_batch / 3; ++j) {

                    input_delta.clear();
                    hidden_delta.clear();

                    weight_sum_ino_batch = 0;
                    for(int k = 0; k < batch_size; ++k) {
                        weight_sum_ino_batch += ind_delta.get(train_row_inds[batch_size * j + k], 1);
                    }

                    for(int k = 0; k < batch_size; ++k) {

                        row_index = train_row_inds[batch_size * j + k];

                        for(int l = 0; l < neural_network->no_predictors; ++l)
                            input_output.assign(l, 0,
                                                 train_x.get(row_index, neural_network->col_inds[l]));

                        input_output.assign(neural_network->no_predictors, 0, 1);

                        neural_network->forward(input_output, hidden_output, output_output);
                        backward(neural_network,
                                 input_output,
                                 hidden_output,
                                 output_output,
                                 hidden_delta,
                                 input_delta,
                                 ind_delta.get(row_index, 0) / ind_delta.get(row_index, 1),
                                 ind_delta.get(row_index, 1) / weight_sum_ino_batch);
                    }

                    Matrix<T> temp_input_delta = plus(*neural_network->input_weight,
                                                      input_delta);
                    Matrix<T> temp_hidden_delta = plus(*neural_network->hidden_weight,
                                                       hidden_delta);

                    copy(temp_input_delta,
                         *neural_network->input_weight);
                    copy(temp_hidden_delta,
                         *neural_network->hidden_weight);
                }

                mse = 0;
                for(int j = 0; j < no_validate; ++j) {

                    for(int k = 0; k < neural_network->no_predictors; ++k)
                        input_output.assign(k, 0,
                                            train_x.get(validate_row_inds[j], neural_network->col_inds[k]));

                    input_output.assign(neural_network->no_predictors, 0, 1);
                    neural_network->forward(input_output, hidden_output, output_output);

                    mse += std::pow(ind_delta.get(validate_row_inds[j], 0) / ind_delta.get(validate_row_inds[j], 1) - output_output, 2.0) *
                           ind_delta.get(validate_row_inds[j], 1) / validate_weight_sum;

                }

//                std::cout << "( " << i << " ) MSE: "<< mse << std::endl;

//                if(i == 0)
//                    first_mse = mse;

                if(mse > last_mse) {
                    count_to_break += 1;
                    if(count_to_break > no_rise_of_loss_on_validate) {
//                        std::cout << "Training of Neural Network at " << neural_network
//                                  << " was early stoped after " << i << "/" << nn_max_iteration
//                                  << " iterations with starting MSE: " << first_mse
//                                  << " and ending MSE: " << mse
//                                  << " !"
//                                  << std::endl;
                        break;
                    }
                }
                last_mse = mse;
            }

            delete[] validate_row_inds;
            delete[] train_row_inds;
            delete[] seeds;
            validate_row_inds = nullptr, train_row_inds = nullptr, seeds = nullptr;

        }

    }

}

// for splines
namespace dbm {

    template <typename T>
    Splines_trainer<T>::Splines_trainer(const Params &params) :
            dbm_remove_rows_containing_nans(params.dbm_remove_rows_containing_nans),
            dbm_min_no_samples_per_bl(params.dbm_min_no_samples_per_bl),
            regularization(params.splines_regularization),
            fraction_of_pairs(params.splines_portion_of_pairs){

        no_pairs = params.dbm_no_candidate_feature * (params.dbm_no_candidate_feature - 1) / 2;

        predictor_pairs_inds = new int *[no_pairs];
        for(int i = 0; i < no_pairs; ++i) {
            predictor_pairs_inds[i] = new int[2];
        }

        for(int i = 0, count = 0; i < params.dbm_no_candidate_feature - 1; ++i)
            for(int j = i + 1; j < params.dbm_no_candidate_feature; ++j) {
                predictor_pairs_inds[count][0] = i;
                predictor_pairs_inds[count][1] = j;
                count += 1;
            }

    }

    template <typename T>
    Splines_trainer<T>::~Splines_trainer() {

        for(int i = 0; i < no_pairs; ++i) {
            delete[] predictor_pairs_inds[i];
        }
        delete[] predictor_pairs_inds;

    }

    template <typename T>
    void Splines_trainer<T>::train(Splines<T> *splines,
                                   const Matrix<T> &train_x,
                                   const Matrix<T> &ind_delta,
                                   const Matrix<T> &monotonic_constraints,
                                   int *row_inds,
                                   int no_rows,
                                   const int *col_inds,
                                   int no_cols) {

        if(row_inds == nullptr) {

        }
        else {

            #ifdef _DEBUG_BASE_LEARNER_TRAINER
                assert(no_rows > 0 && no_cols >= splines->no_predictors);
            #endif

            if(dbm_remove_rows_containing_nans)
                remove_nan_row_inds(row_inds,
                                    no_rows,
                                    col_inds,
                                    no_cols,
                                    train_x);

            if(no_rows < dbm_min_no_samples_per_bl) {
                std::cout << "Too few training samples (" << no_rows << ") without NaNs and Omit the BL." << std::endl;
                for(int i = 0; i < splines->no_predictors; ++i)
                    splines->col_inds[i] = 0;
                for(int i = 0; i < splines->no_knots; ++i) {

                    splines->x_knots[i] = 0;
                    splines->x_left_coefs[i] = NAN;
                    splines->x_right_coefs[i] = NAN;

                    splines->y_knots[i] = 0;
                    splines->y_left_coefs[i] = NAN;
                    splines->y_right_coefs[i] = NAN;

                }
                return;
            }

            if(monotonic_constraints.frobenius_norm() > 0) {
                std::cout << "This BL does not support Monotonic Constraints." << std::endl;
                for(int i = 0; i < splines->no_predictors; ++i)
                    splines->col_inds[i] = 0;
                for(int i = 0; i < splines->no_knots; ++i) {

                    splines->x_knots[i] = 0;
                    splines->x_left_coefs[i] = NAN;
                    splines->x_right_coefs[i] = NAN;

                    splines->y_knots[i] = 0;
                    splines->y_left_coefs[i] = NAN;
                    splines->y_right_coefs[i] = NAN;

                }
                return;
            }

            Matrix<T> w(1, no_rows, 0);
            for(int i = 0; i < no_rows; ++i)
                w.assign(0, i, ind_delta.get(row_inds[i], 1));

            T scaling = 0.7;
            T **knots, min, max;
            knots = new T*[no_cols];
            for(int i = 0; i < no_cols; ++i) {
                knots[i] = new T[splines->no_knots];
                min = train_x.get_col_min(col_inds[i], row_inds, no_rows);
                max = train_x.get_col_max(col_inds[i], row_inds, no_rows);
                range(min, max, splines->no_knots, knots[i], scaling);
            }

            double mse, lowest_mse = std::numeric_limits<T>::max();
            T knot;

            int lowest_mse_pair_ind = 0;

            T x, y, prediction;

            int no_splines = splines->no_knots * 4;

            Matrix<T> lowest_mse_coefs(no_splines, 1, 0);
            Matrix<T> design_matrix(no_rows, no_splines, 0);
            Matrix<T> left_in_left;
            Matrix<T> left;
            Matrix<T> Y(no_rows, 1, 0);
            for(int i = 0; i < no_rows; ++i)
                Y.assign(i, 0, ind_delta.get(row_inds[i], 0) / ind_delta.get(row_inds[i], 1));
            Matrix<T> coefs;

            Matrix<T> eye(no_splines, no_splines, 0);

            int x_col_no, y_col_no;
//            int row_ind;

            for(int i = 0; i < no_splines; ++i)
                eye.assign(i, i, regularization);

            for(int i = 0; i < std::max((int)(no_pairs * fraction_of_pairs), 1); ++i) {

                x_col_no = predictor_pairs_inds[i][0];
                y_col_no = predictor_pairs_inds[i][1];

                splines->col_inds[0] = col_inds[ x_col_no ];
                splines->col_inds[1] = col_inds[ y_col_no ];

                design_matrix.clear();

                int j = 0;
                std::for_each(row_inds, row_inds + no_rows,
                              [&j, &x, &y, &knot, &x_col_no, &y_col_no, &train_x,
                                      &splines, &knots, &design_matrix]
                                      (int index) {

                    x = train_x.get(index,
                                    splines->col_inds[0]);
                    y = train_x.get(index,
                                    splines->col_inds[1]);

                    for(int k = 0; k < splines->no_knots; ++k) {

                        knot = knots[ x_col_no ][k];
                        design_matrix.assign(j,
                                             4 * k,
                                             splines->x_left_hinge(x, y, knot) );
                        design_matrix.assign(j,
                                             4 * k + 1,
                                             splines->x_right_hinge(x, y, knot) );

                        knot = knots[ y_col_no ][k];
                        design_matrix.assign(j,
                                             4 * k + 2,
                                             splines->y_left_hinge(x, y, knot) );
                        design_matrix.assign(j,
                                             4 * k + 3,
                                             splines->y_right_hinge(x, y, knot) );

                    }

                    ++j;
                });

//                for(int j = 0; j < no_rows; ++j) {
//
//                    row_ind = row_inds[j];
//
//                    x = train_x.get(row_ind,
//                                    splines->col_inds[0]);
//                    y = train_x.get(row_ind,
//                                    splines->col_inds[1]);
//
//                    for(int k = 0; k < splines->no_knots; ++k) {
//
//                        knot = knots[ x_col_no ][k];
//                        design_matrix.assign(j,
//                                             4 * k,
//                                             splines->x_left_hinge(x, y, knot) );
//                        design_matrix.assign(j,
//                                             4 * k + 1,
//                                             splines->x_right_hinge(x, y, knot) );
//
//                        knot = knots[ y_col_no ][k];
//                        design_matrix.assign(j,
//                                             4 * k + 2,
//                                             splines->y_left_hinge(x, y, knot) );
//                        design_matrix.assign(j,
//                                             4 * k + 3,
//                                             splines->y_right_hinge(x, y, knot) );
//
//                    }
//                }

                left_in_left = transpose(design_matrix);
                left_in_left.inplace_elewise_prod_mat_with_row_vec(w);
                left = inner_product(left_in_left, design_matrix);
                left = inverse(plus(left, eye));

                coefs = inner_product(left, inner_product(left_in_left, Y));

                // check this part, should I use predict_for_row?
                mse = 0;
                j = 0;
                std::for_each(row_inds, row_inds + no_rows,
                              [&j, &prediction, &mse, &x, &y, &knot, &knots, &train_x,
                                      &Y, &splines, &x_col_no, &y_col_no, &coefs]
                                      (int index){
                    prediction = 0;

                    x = train_x.get(index,
                                    splines->col_inds[0]);
                    y = train_x.get(index,
                                    splines->col_inds[1]);

                    for(int k = 0; k < splines->no_knots; ++k) {

                        knot = knots[ x_col_no ][k];
                        prediction += splines->x_left_hinge(x, y, knot) * coefs.get(4 * k, 0);
                        prediction += splines->x_right_hinge(x, y, knot) * coefs.get(4 * k + 1, 0);

                        knot = knots[ y_col_no ][k];
                        prediction += splines->y_left_hinge(x, y, knot) * coefs.get(4 * k + 2, 0);
                        prediction += splines->y_right_hinge(x, y, knot) * coefs.get(4 * k + 3, 0);

                    }
                    mse += (Y.get(j, 0) - prediction) * (Y.get(j, 0) - prediction);
                    ++j;
                });

//                for(j = 0; j < no_rows; ++j) {
//
//                    prediction = 0;
//
//                    x = train_x.get(row_inds[j],
//                                    splines->col_inds[0]);
//                    y = train_x.get(row_inds[j],
//                                    splines->col_inds[1]);
//
//                    for(int k = 0; k < splines->no_knots; ++k) {
//
//                        knot = knots[ predictor_pairs_inds[i][0] ][k];
//                        prediction += splines->x_left_hinge(x, y, knot) * coefs.get(4 * k, 0);
//                        prediction += splines->x_right_hinge(x, y, knot) * coefs.get(4 * k + 1, 0);
//
//                        knot = knots[ predictor_pairs_inds[i][1] ][k];
//                        prediction += splines->y_left_hinge(x, y, knot) * coefs.get(4 * k + 2, 0);
//                        prediction += splines->y_right_hinge(x, y, knot) * coefs.get(4 * k + 3, 0);
//
//                    }
//                    mse += (Y.get(j, 0) - prediction) * (Y.get(j, 0) - prediction);
//                }

                mse /= no_rows;

                if(mse < lowest_mse) {
                    lowest_mse = mse;
                    lowest_mse_pair_ind = i;
                    copy(coefs, lowest_mse_coefs);
                }

            }

            splines->col_inds[0] = col_inds[ predictor_pairs_inds[lowest_mse_pair_ind][0] ];
            splines->col_inds[1] = col_inds[ predictor_pairs_inds[lowest_mse_pair_ind][1] ];

            for(int j = 0; j < splines->no_knots; ++j) {

                splines->x_knots[j] = knots[ predictor_pairs_inds[lowest_mse_pair_ind][0] ][j];

                splines->y_knots[j] = knots[ predictor_pairs_inds[lowest_mse_pair_ind][1] ][j];

            }

            for(int j = 0; j < splines->no_knots; ++j) {

                splines->x_left_coefs[j] = lowest_mse_coefs.get(4 * j, 0);
                splines->x_right_coefs[j] = lowest_mse_coefs.get(4 * j + 1, 0);

                splines->y_left_coefs[j] = lowest_mse_coefs.get(4 * j + 2, 0);
                splines->y_right_coefs[j] = lowest_mse_coefs.get(4 * j + 3, 0);

            }


            for(int i = 0; i < no_cols; ++i)
                delete[] knots[i];
            delete[] knots;
        }
    }

}

// for k-means-2d
namespace dbm {

    template <typename T>
    Kmeans2d_trainer<T>::Kmeans2d_trainer(const Params &params) :
            random_seed(params.dbm_random_seed),
            dbm_remove_rows_containing_nans(params.dbm_remove_rows_containing_nans),
            dbm_min_no_samples_per_bl(params.dbm_min_no_samples_per_bl),
            loss_type(params.dbm_loss_function),
            no_centroids(params.kmeans_no_centroids),
            no_candidate_feature(params.dbm_no_candidate_feature),
            fraction_of_pairs(params.kmeans_fraction_of_pairs),
            kmeans_max_iteration(params.kmeans_max_iteration),
            kmeans_tolerance(params.kmeans_tolerance),
            loss_function(Loss_function<T>(params)) {

        no_pairs = params.dbm_no_candidate_feature * (params.dbm_no_candidate_feature - 1) / 2;

        predictor_pairs_inds = new int *[no_pairs];
        for(int i = 0; i < no_pairs; ++i) {
            predictor_pairs_inds[i] = new int[2];
        }

        for(int i = 0, count = 0; i < params.dbm_no_candidate_feature - 1; ++i)
            for(int j = i + 1; j < params.dbm_no_candidate_feature; ++j) {
                predictor_pairs_inds[count][0] = i;
                predictor_pairs_inds[count][1] = j;
                count += 1;
            }

    }

    template <typename T>
    Kmeans2d_trainer<T>::~Kmeans2d_trainer() {

        for(int i = 0; i < no_pairs; ++i)
            delete[] predictor_pairs_inds[i];
        delete[] predictor_pairs_inds;

    }

    template <typename T>
    void Kmeans2d_trainer<T>::train(Kmeans2d<T> *kmeans2d,
                                    const Matrix<T> &train_x,
                                    const Matrix<T> &ind_delta,
                                    const Matrix<T> &monotonic_constraints,
                                    int *row_inds,
                                    int no_rows,
                                    const int *col_inds,
                                    int no_cols) {

        if(row_inds == nullptr) {

        }
        else {

            #ifdef _DEBUG_BASE_LEARNER_TRAINER
                assert(no_rows > 0 && no_cols == no_candidate_feature);
            #endif

            if(dbm_remove_rows_containing_nans)
                remove_nan_row_inds(row_inds,
                                    no_rows,
                                    col_inds,
                                    no_cols,
                                    train_x);

            if(no_rows < dbm_min_no_samples_per_bl) {
                std::cout << "Too few training samples (" << no_rows << ") without NaNs and Omit the BL." << std::endl;
                for(int i = 0; i < kmeans2d->no_predictors; ++i)
                    kmeans2d->col_inds[i] = 0;
                for(int i = 0; i < kmeans2d->no_centroids; ++i) {
                    for(int j = 0; j < kmeans2d->no_predictors; ++j)
                        kmeans2d->centroids[i][j] = 0;
                    kmeans2d->predictions[i] = NAN;
                }
                return;
            }

            if(monotonic_constraints.frobenius_norm() > 0) {
                std::cout << "This BL does not support Monotonic Constraints." << std::endl;
                for(int i = 0; i < kmeans2d->no_predictors; ++i)
                    kmeans2d->col_inds[i] = 0;
                for(int i = 0; i < kmeans2d->no_centroids; ++i) {
                    for(int j = 0; j < kmeans2d->no_predictors; ++j)
                        kmeans2d->centroids[i][j] = 0;
                    kmeans2d->predictions[i] = NAN;
                }
                return;
            }

            T **start_centroids = new T*[no_centroids],
                    *start_prediction = new T[no_centroids],
                    **next_centroids = new T*[no_centroids],
                    *best_prediction = new T[no_centroids];

            for(int i = 0; i < no_centroids; ++i) {
                start_centroids[i] = new T[kmeans2d->no_predictors];
                next_centroids[i] = new T[kmeans2d->no_predictors];
            }

            int *no_samples_in_each_centroid = new int[no_centroids],
                    predictor_col_inds[kmeans2d->no_predictors];

            int closest_centroid_ind = 0;

            T feature_mins[kmeans2d->no_predictors],
                    feature_maxes[kmeans2d->no_predictors];

            T dist = 0,
                    lowest_dist = 0,
                    standard_dev = 0,
                    lowest_standard_dev = std::numeric_limits<T>::max();

            int **sample_inds_for_each_centroid = new int*[no_centroids];
            for(int i = 0; i < no_centroids; ++i)
                sample_inds_for_each_centroid[i] = nullptr;

//            T mse, lowest_mse = std::numeric_limits<T>::max();

            Matrix<T> prediction(no_centroids, 1, 0);

            T denominator_in_prediction;

            unsigned int *seeds = new unsigned int[no_pairs];

            if(random_seed < 0) {
                std::random_device rd;
                std::mt19937 mt(rd());
                std::uniform_real_distribution<T> distri(0, 1000);
                for(int i = 0; i < no_pairs; ++i)
                    seeds[i] = (unsigned int) distri(mt);
            }
            else {
                std::mt19937 mt(random_seed);
                std::uniform_real_distribution<T> distri(0, 1000);
                for(int i = 0; i < no_pairs; ++i)
                    seeds[i] = (unsigned int) distri(mt);
            }

            for(int l = 0; l < no_pairs * fraction_of_pairs; ++l) {

                for(int i = 0; i < kmeans2d->no_predictors; ++i) {

                    predictor_col_inds[i] = col_inds[ predictor_pairs_inds[l][i] ];

                    feature_mins[i] = train_x.get_col_min(col_inds[i], row_inds, no_rows);
                    feature_maxes[i] = train_x.get_col_max(col_inds[i], row_inds, no_rows);

                }

//                std::srand((unsigned int)
//                                   std::chrono::duration_cast< std::chrono::milliseconds >
//                                           (std::chrono::system_clock::now().time_since_epoch()).count() );

                std::srand(seeds[l]);

                for(int i = 0; i < no_centroids; ++i)
                    for(int j = 0; j < kmeans2d->no_predictors; ++j) {

                        start_centroids[i][j] =
                                std::rand() / double(RAND_MAX) *
                                                    (feature_maxes[j] - feature_mins[j]) + feature_mins[j];

                    }

                for(int iter = 0; iter < kmeans_max_iteration; ++iter) {

                    for(int i = 0; i < no_centroids; ++i) {

                        no_samples_in_each_centroid[i] = 0;

                        for(int j = 0; j < kmeans2d->no_predictors; ++j) {

                            next_centroids[i][j] = 0;

                        }
                    }

                    for(int i = 0; i < no_rows; ++i) {

                        lowest_dist = std::numeric_limits<T>::max();

                        for(int j = 0; j < no_centroids; ++j) {

                            dist = 0;

                            for(int k = 0; k < kmeans2d->no_predictors; ++k) {
                                dist += (start_centroids[j][k] - train_x.get(row_inds[i], predictor_col_inds[k])) *
                                        (start_centroids[j][k] - train_x.get(row_inds[i], predictor_col_inds[k]));
                            }

                            dist = std::sqrt(dist);

                            if(dist < lowest_dist) {
                                lowest_dist = dist;
                                closest_centroid_ind = j;
                            }

                        }

                        no_samples_in_each_centroid[closest_centroid_ind] += 1;

                        for(int j = 0; j < kmeans2d->no_predictors; ++j)

                            next_centroids[closest_centroid_ind][j] +=
                                    train_x.get(row_inds[i], predictor_col_inds[j]);

                    }

                    dist = 0;
                    for(int i = 0; i < no_centroids; ++i)
                        for(int j = 0; j < kmeans2d->no_predictors; ++j) {

                            next_centroids[i][j] /= (no_samples_in_each_centroid[i] != 0) ? no_samples_in_each_centroid[i] : 1;

                            dist += (start_centroids[i][j] - next_centroids[i][j]) *
                                    (start_centroids[i][j] - next_centroids[i][j]);

                        }

                    dist = std::sqrt(dist);

                    if(dist < kmeans_tolerance) {

//                        std::cout << "Training of Kmeans(" << predictor_col_inds[0]
//                                  <<", " << predictor_col_inds[1]
//                                  << ") at " << kmeans2d
//                                  << " was early stoped after " << iter << "/" << kmeans_max_iteration
//                                  << " with the ending distance change: " << dist
//                                  << " !"
//                                  << std::endl;

                        break;

                    }
                    else if(iter < kmeans_max_iteration - 1) {

                        for(int i = 0; i < no_centroids; ++i)
                            for(int j = 0; j < kmeans2d->no_predictors; ++j)
                                start_centroids[i][j] = next_centroids[i][j];

                    }

                }

                for(int i = 0; i < no_centroids; ++i) {

                    sample_inds_for_each_centroid[i] = new int[no_samples_in_each_centroid[i]];

                    no_samples_in_each_centroid[i] = 0;

                }

                for(int i = 0; i < no_rows; ++i) {

                    lowest_dist = std::numeric_limits<T>::max();

                    for(int j = 0; j < no_centroids; ++j) {

                        dist = 0;

                        for(int k = 0; k < kmeans2d->no_predictors; ++k) {
                            dist += (start_centroids[j][k] - train_x.get(row_inds[i], predictor_col_inds[k])) *
                                    (start_centroids[j][k] - train_x.get(row_inds[i], predictor_col_inds[k]));
                        }

                        dist = std::sqrt(dist);

                        if(dist < lowest_dist) {
                            lowest_dist = dist;
                            closest_centroid_ind = j;
                        }

                    }

                    sample_inds_for_each_centroid[closest_centroid_ind][no_samples_in_each_centroid[closest_centroid_ind]] = row_inds[i];

                    no_samples_in_each_centroid[closest_centroid_ind] += 1;

                }

                for(int i = 0; i < no_centroids; ++i) {

                    start_prediction[i] = 0, denominator_in_prediction = 0;

                    for(int j = 0; j < no_samples_in_each_centroid[i]; ++j) {
                        start_prediction[i] += ind_delta.get(sample_inds_for_each_centroid[i][j], 0);
                        denominator_in_prediction += ind_delta.get(sample_inds_for_each_centroid[i][j], 1);
                    }

                    if(denominator_in_prediction)
                        start_prediction[i] /= denominator_in_prediction;
                    else {
                        if(loss_type == 'p' || loss_type == 't')
                            start_prediction[i] = 1;
                        else if(loss_type == 'n' || loss_type == 'b')
                            start_prediction[i] = 0;
                    }

//                    prediction.assign(i, 0, start_prediction[i]);

                    delete[] sample_inds_for_each_centroid[i];

                    sample_inds_for_each_centroid[i] = nullptr;

                }

                for(int i = 0; i < no_rows; ++i) {

                    lowest_dist = std::numeric_limits<T>::max();

                    for(int j = 0; j < no_centroids; ++j) {

                        dist = 0;

                        for(int k = 0; k < kmeans2d->no_predictors; ++k) {
                            dist += (start_centroids[j][k] - train_x.get(row_inds[i], predictor_col_inds[k])) *
                                    (start_centroids[j][k] - train_x.get(row_inds[i], predictor_col_inds[k]));
                        }

                        dist = std::sqrt(dist);

                        if(dist < lowest_dist) {
                            lowest_dist = dist;
                            closest_centroid_ind = j;
                        }

                    }

                    standard_dev += std::pow(ind_delta.get(row_inds[i], 0) / ind_delta.get(row_inds[i], 1) - start_prediction[closest_centroid_ind], 2.0);

                }

                standard_dev = std::sqrt(standard_dev / (no_rows - 1));

//                standard_dev = prediction.col_std(0);
                if(standard_dev < lowest_standard_dev) {
                    lowest_standard_dev = standard_dev;

                    kmeans2d->col_inds[0] = predictor_col_inds[0];
                    kmeans2d->col_inds[1] = predictor_col_inds[1];

                    for(int i = 0; i < no_centroids; ++i) {

                        kmeans2d->predictions[i] = start_prediction[i];

                        for(int j = 0; j < kmeans2d->no_predictors; ++j)
                            kmeans2d->centroids[i][j] = start_centroids[i][j];

                    }

                }

            }

            delete[] start_prediction;
            delete[] best_prediction;
            delete[] sample_inds_for_each_centroid;

            for(int i = 0; i < no_centroids; ++i) {
                delete[] start_centroids[i];
                delete[] next_centroids[i];
            }
            delete[] start_centroids;
            delete[] next_centroids;

            delete[] no_samples_in_each_centroid;

        }

    }
}

// for linear regression
namespace dbm {

    template <typename T>
    Linear_regression_trainer<T>::Linear_regression_trainer(const Params &params) :
            dbm_remove_rows_containing_nans(params.dbm_remove_rows_containing_nans),
            dbm_min_no_samples_per_bl(params.dbm_min_no_samples_per_bl),
            regularization(params.lr_regularization) {}

    template <typename T>
    Linear_regression_trainer<T>::~Linear_regression_trainer() {}

    template <typename T>
    void Linear_regression_trainer<T>::train(Linear_regression<T> *linear_regression,
                                             const Matrix<T> &train_x,
                                             const Matrix<T> &ind_delta,
                                             const Matrix<T> &monotonic_constraints,
                                             int *row_inds,
                                             int no_rows,
                                             const int *col_inds,
                                             int no_cols) {

        if(row_inds == nullptr) {


        } // if
        else {

            #ifdef _DEBUG_BASE_LEARNER_TRAINER
                assert(no_rows > 0 && no_cols == linear_regression->no_predictors);
            #endif

            if(dbm_remove_rows_containing_nans)
                remove_nan_row_inds(row_inds,
                                    no_rows,
                                    col_inds,
                                    no_cols,
                                    train_x);

            if(no_rows < dbm_min_no_samples_per_bl) {
                std::cout << "Too few training samples (" << no_rows << ") without NaNs and Omit the BL." << std::endl;
                for(int i = 0; i < no_cols; ++i)
                    linear_regression->col_inds[i] = col_inds[i];
                for(int i = 0; i < linear_regression->no_predictors; ++i)
                    linear_regression->coefs_no_intercept[i] = NAN;
                linear_regression->intercept = NAN;
                return;
            }

            Matrix<T> w(1, no_rows, 0);
            for(int i = 0; i < no_rows; ++i)
                w.assign(0, i, ind_delta.get(row_inds[i], 1));

            for(int j = 0; j < no_cols; ++j)
                linear_regression->col_inds[j] = col_inds[j];

            Matrix<T> intercept(no_rows, 1, 1);

            Matrix<T> eye(2, 2, 0);
            eye.assign(0, 0, regularization);
            eye.assign(1, 1, regularization);
            Matrix<T> y(no_rows, 1, 0);
            for(int i = 0; i < no_rows; ++i)
                y.assign(i, 0, ind_delta.get(row_inds[i], 0) / ind_delta.get(row_inds[i], 1));
            Matrix<T> coefs(linear_regression->no_predictors + 1, 1, 0.);

            #ifdef _DEBUG_BASE_LEARNER_TRAINER
                assert(coefs.get_width() == 1 && coefs.get_height() == no_cols + 1);
            #endif

            int no_valid_predictor = linear_regression->no_predictors;

            for (int j = 0; j < linear_regression->no_predictors; j ++) {
                /** Loop over all features, fit \beta_{j} based on x_{j} */
                /* Now x has only intercept and x_{j} */
                Matrix<T> x = hori_merge(intercept, train_x.submatrix(row_inds, no_rows,
                                                            col_inds + j, 1));

                Matrix<T> left_in_left = transpose(x);
                left_in_left.inplace_elewise_prod_mat_with_row_vec(w);
                Matrix<T> left = inner_product(left_in_left, x);

                left = inverse(plus(left, eye));

                /** coefs = [x^{T} w x + \lambda]^{-1} x^{T} w y */
                // (2, 2) * (2, n_observe) * (n_observe, 1) -> (2, 1)
                Matrix<T> coef = inner_product(left, inner_product(left_in_left, y));

                // check if the coef is consistent with the monotonic constraint
                // otherwise set the coef to 0
                if ( coef.get(1, 0) * monotonic_constraints.get(col_inds[j], 0) >= 0 ) {
                    coefs[0][0] += coef[0][0];
                    coefs.assign(j + 1, 0, coef.get(1, 0));
                }
                else {
                    --no_valid_predictor;
                }
            } // j

            if(no_valid_predictor) {
                linear_regression->intercept = coefs.get(0, 0) / (T) no_valid_predictor;
                for(int j = 1; j < no_cols + 1; j ++)
                    linear_regression->coefs_no_intercept[j - 1] = coefs.get(j, 0) / (T) no_valid_predictor;
            }
            else {
                std::cout << "Warning: all coefs are not consistent with the monotonic constraints." << std::endl;
            }

/////////////////////////////
//            Matrix<T> x = hori_merge(intercept, train_x.submatrix(row_inds,
//                                                                  no_rows,
//                                                                  col_inds,
//                                                                  no_cols));
//
//            Matrix<T> left_in_left = transpose(x);
//            left_in_left.inplace_elewise_prod_mat_with_row_vec(w);
//            Matrix<T> left = inner_product(left_in_left, x);
//
//            int y_ind[] = {0}, no_y_ind = 1;
//            Matrix<T> y = ind_delta.submatrix(row_inds,
//                                              no_rows,
//                                              y_ind,
//                                              no_y_ind);
//
//            Matrix<T> eye(no_cols + 1, no_cols + 1, 0);
//
//            for(int i = 0; i < no_cols + 1; ++i)
//                eye.assign(i, i, regularization);
//
//            left = inverse(plus(left, eye));
//
//            Matrix<T> coefs = inner_product(left, inner_product(left_in_left, y));
//
//            #ifdef _DEBUG_BASE_LEARNER_TRAINER
//                assert(coefs.get_width() == 1 && coefs.get_height() == no_cols + 1);
//            #endif
//            linear_regression->intercept = coefs.get(0, 0);
//            for(int j = 1; j < no_cols + 1; ++j)
//                linear_regression->coefs_no_intercept[j - 1] = coefs.get(j, 0);
        } // else row_ind != nullptr
    } // END of void Linear_regression_trainer<T>::train()

}

// for dpc stairs
namespace dbm {

    template <typename T>
    DPC_stairs_trainer<T>::DPC_stairs_trainer(const Params &params) :
            dbm_remove_rows_containing_nans(params.dbm_remove_rows_containing_nans),
            dbm_min_no_samples_per_bl(params.dbm_min_no_samples_per_bl),
            loss_type(params.dbm_loss_function),
            range_shrinkage_of_ticks(params.dpcs_range_shrinkage_of_ticks) {}

    template <typename T>
    DPC_stairs_trainer<T>::~DPC_stairs_trainer() {}

    template <typename T>
    void DPC_stairs_trainer<T>::train(DPC_stairs<T> *dpc_stairs,
                                      const Matrix<T> &train_x,
                                      const Matrix<T> &ind_delta,
                                      const Matrix<T> &monotonic_constraints,
                                      int *row_inds,
                                      int no_rows,
                                      const int *col_inds,
                                      int no_cols) {

        if(row_inds == nullptr) {



        }
        else {

            #ifdef _DEBUG_BASE_LEARNER_TRAINER
                assert(no_cols == dpc_stairs->no_predictors);
            #endif

            if(dbm_remove_rows_containing_nans)
                remove_nan_row_inds(row_inds,
                                    no_rows,
                                    col_inds,
                                    no_cols,
                                    train_x);

            if(no_rows < dbm_min_no_samples_per_bl) {
                std::cout << "Too few training samples (" << no_rows << ") without NaNs and Omit the BL." << std::endl;
                for(int i = 0; i < no_cols; ++i)
                    dpc_stairs->col_inds[i] = col_inds[i];
                for(int i = 0; i < dpc_stairs->no_predictors; ++i)
                    dpc_stairs->coefs[i] = 0;
                for(int i = 0; i < dpc_stairs->no_ticks; ++i)
                    dpc_stairs->ticks[i] = 0;
                for(int i = 0; i < dpc_stairs->no_ticks + 1; ++i)
                    dpc_stairs->predictions[i] = NAN;
                return;
            }

            if(monotonic_constraints.frobenius_norm() > 0) {
                std::cout << "This BL does not support Monotonic Constraints." << std::endl;
                for(int i = 0; i < no_cols; ++i)
                    dpc_stairs->col_inds[i] = col_inds[i];
                for(int i = 0; i < dpc_stairs->no_predictors; ++i)
                    dpc_stairs->coefs[i] = 0;
                for(int i = 0; i < dpc_stairs->no_ticks; ++i)
                    dpc_stairs->ticks[i] = 0;
                for(int i = 0; i < dpc_stairs->no_ticks + 1; ++i)
                    dpc_stairs->predictions[i] = NAN;
                return;
            }

            Matrix<T> w(1, no_rows, 0);
            for(int i = 0; i < no_rows; ++i)
                w.assign(0, i, ind_delta.get(row_inds[i], 1));

            for(int i = 0; i < no_cols; ++i)
                dpc_stairs->col_inds[i] = col_inds[i];

            Matrix<T> data_x = train_x.submatrix(row_inds, no_rows, col_inds, no_cols);
            Matrix<T> centered_data_x = data_x;
            centered_data_x.columnwise_centering();
            Matrix<T> covariance_mat = inner_product(transpose(data_x), data_x);
            covariance_mat.scaling(1.0 / (no_rows - 1.0));

            Matrix<T> eigen_vec;
            covariance_mat.dominant_eigen_decomp(eigen_vec);
            for(int i = 0; i < no_cols; ++i)
                dpc_stairs->coefs[i] = eigen_vec.get(i, 0);

            T *pc_scores = new T[no_rows];
            T min_pc_score = std::numeric_limits<T>::max(),
                    max_pc_score = std::numeric_limits<T>::lowest(),
                    spread_shrinkage;

            for(int i = 0; i < no_rows; ++i) {

                pc_scores[i] = inner_product(data_x.row(i), eigen_vec).get(0, 0);

                if(pc_scores[i] < min_pc_score)
                    min_pc_score = pc_scores[i];

                if(pc_scores[i] > max_pc_score)
                    max_pc_score = pc_scores[i];

            } // i < no_rows

            spread_shrinkage = range_shrinkage_of_ticks * (max_pc_score - min_pc_score) / 2.0;
            min_pc_score += spread_shrinkage;
            max_pc_score -= spread_shrinkage;

            for(int i = 0; i < dpc_stairs->no_ticks; ++i)
                dpc_stairs->ticks[i] = min_pc_score + (max_pc_score - min_pc_score) * i / (dpc_stairs->no_ticks - 1.0);

            T *denominators = new T[dpc_stairs->no_ticks + 1];
            for(int i = 0; i < dpc_stairs->no_ticks + 1; ++i) {
                dpc_stairs->predictions[i] = 0;
                denominators[i] = 0;
            }

            for(int i = 0; i < no_rows; ++i) {

//                int j = 0;
//                while(j < dpc_stairs->no_ticks && pc_scores[i] > dpc_stairs->ticks[j]) {
//                    ++j;
//                }

                int j = 0;
                for(; j < dpc_stairs->no_ticks + 1; ++j) {
                    if(j == dpc_stairs->no_ticks || pc_scores[i] < dpc_stairs->ticks[j])
                        break;
                }

                dpc_stairs->predictions[j] += ind_delta.get(row_inds[i], 0);
                denominators[j] += ind_delta.get(row_inds[i], 1);

            } // i < no_rows

            for(int i = 0; i < dpc_stairs->no_ticks + 1; ++i) {

                if(denominators[i])
                    dpc_stairs->predictions[i] /= denominators[i];
                else {
                    if(loss_type == 'p' || loss_type == 't')
                        dpc_stairs->predictions[i] = 1;
                    else if(loss_type == 'n' || loss_type == 'b')
                        dpc_stairs->predictions[i] = 0;
                }

            }

            delete[] pc_scores;
            delete[] denominators;

        }
    }

}

// for trees
namespace dbm {

    template<typename T>
    Fast_tree_trainer<T>::Fast_tree_trainer(const Params &params) :
            dbm_remove_rows_containing_nans(params.dbm_remove_rows_containing_nans),
            min_samples_in_a_node(params.cart_min_samples_in_a_node),
            max_depth(params.cart_max_depth),
            loss_function(Loss_function<T>(params)) {};

    template<typename T>
    Fast_tree_trainer<T>::~Fast_tree_trainer() {};

    template<typename T>
    void Fast_tree_trainer<T>::train(Tree_node<T> *tree,

                                     const Matrix<T> &train_x,
                                     const Matrix<T> &sorted_train_x_from,

                                     const Matrix<T> &train_y,
                                     const Matrix<T> &ind_delta,
                                     const Matrix<T> &prediction,

                                     const Matrix<T> &first_comp_in_loss,
                                     const Matrix<T> &second_comp_in_loss,

                                     const Matrix<T> &monotonic_constraints,

                                     char loss_function_type,

                                     int *row_inds,
                                     int no_rows,
                                     const int *col_inds,
                                     int no_cols) {

        if(row_inds == nullptr) {


        }
        else {

            /*
             *  @TODO two ways to deal with min number of samples in a node
             *      1. split at the best point and then check if child nodes have enough samples
             *      2. do not split until first cart_min_samples_in_a_node samples and stop splitting
             *          after last cart_min_samples_in_a_node samples
             */

            tree->no_training_samples = no_rows;

            #ifdef _DEBUG_BASE_LEARNER_TRAINER
                assert(no_rows > 0 && no_cols > 0);
            #endif

            if(tree->depth == 0) {

                if(dbm_remove_rows_containing_nans)
                    remove_nan_row_inds(row_inds,
                                        no_rows,
                                        col_inds,
                                        no_cols,
                                        train_x);
                tree->prediction = loss_function.estimate_mean(ind_delta,
                                                               loss_function_type,
                                                               row_inds,
                                                               no_rows);
            }
            if (tree->depth == max_depth || no_rows < min_samples_in_a_node) {
                tree->last_node = true;
                return;
            }

            int *left_inds = new int[no_rows],
                    *right_inds = new int[no_rows];

            int train_x_height = train_x.get_height();
            unsigned char *choose_labels = new unsigned char[train_x_height];
            std::fill(choose_labels, choose_labels + train_x_height, 0);
            for(int i = 0; i < no_rows; ++i)
                choose_labels[row_inds[i]] = 1;

            double left_beta, right_beta, loss_reduction = 0;
            tree->loss_reduction = 0;

            double left_numerator = 0,
                    left_denominator = 0,
                    right_numerator = 0,
                    right_denominator = 0;
            double left_1st_comp_in_loss = 0,
                    left_2nd_comp_in_loss = 0,
                    right_1st_comp_in_loss = 0,
                    right_2nd_comp_in_loss = 0;

            double sum_of_numerators = 0, sum_of_denominators = 0, sum_of_first_comp = 0, sum_of_second_comp = 0;
            std::for_each(row_inds, row_inds + no_rows,
                          [&sum_of_numerators,
                                  &sum_of_denominators,
                                  &sum_of_first_comp,
                                  &sum_of_second_comp,
                                  &ind_delta,
                                  &first_comp_in_loss,
                                  &second_comp_in_loss] (int index)
                          {
                              sum_of_numerators += ind_delta.get(index, 0);
                              sum_of_denominators += ind_delta.get(index, 1);
                              sum_of_first_comp += first_comp_in_loss.get(index, 0);
                              sum_of_second_comp += second_comp_in_loss.get(index, 0);
                          }
            );

            int original_col_ind, original_row_ind;
            int last_chosen_ind_in_sorted_array, best_last_chosen_ind_in_sorted_array = -1;
            int next_chosen_row_ind = -1, best_split_row_ind_in_sorted_array = -1;
            double best_left_beta = 0, best_right_beta = 0;

            for(int i = 0; i < no_cols; ++i) {

//                Time_measurer second;

                original_col_ind = col_inds[i];

                for(last_chosen_ind_in_sorted_array = train_x_height - 1;
                    last_chosen_ind_in_sorted_array >= 0;
                    --last_chosen_ind_in_sorted_array) {
                    original_row_ind = (int) sorted_train_x_from.get(last_chosen_ind_in_sorted_array, original_col_ind);
                    if(choose_labels[original_row_ind])
                        break;
                }

                left_numerator = 0;
                left_denominator = 0;
                left_1st_comp_in_loss = 0;
                left_2nd_comp_in_loss = 0;

                for(int j = 0; j < last_chosen_ind_in_sorted_array; ++j) {

                    original_row_ind = (int) sorted_train_x_from.get(j, original_col_ind);

                    if(choose_labels[original_row_ind]) {

                        left_numerator += ind_delta.get(original_row_ind, 0);
                        left_denominator += ind_delta.get(original_row_ind, 1);
                        left_1st_comp_in_loss += first_comp_in_loss.get(original_row_ind, 0);
                        left_2nd_comp_in_loss += second_comp_in_loss.get(original_row_ind, 0);

                        right_numerator = sum_of_numerators - left_numerator;
                        right_denominator = sum_of_denominators - left_denominator;
                        right_1st_comp_in_loss = sum_of_first_comp - left_1st_comp_in_loss;
                        right_2nd_comp_in_loss = sum_of_second_comp - left_2nd_comp_in_loss;

//                        left_beta = left_numerator / left_denominator;
//                        right_beta = right_numerator / right_denominator;

                        left_beta = loss_function.inversed_link_function(left_numerator / left_denominator,
                                                                         loss_function_type);
                        right_beta = loss_function.inversed_link_function(right_numerator / right_denominator,
                                                                          loss_function_type);

                        if ( (right_beta - left_beta) * monotonic_constraints.get(original_col_ind, 0) < 0 )
                            continue;

//                        loss_reduction = left_1st_comp_in_loss * left_beta * left_beta +
//                                right_1st_comp_in_loss * right_beta * right_beta;

                        loss_reduction =
                                loss_function.loss_reduction(left_1st_comp_in_loss,
                                                                      left_2nd_comp_in_loss,
                                                                      left_beta,
                                                                      loss_function_type) +
                                loss_function.loss_reduction(right_1st_comp_in_loss,
                                                             right_2nd_comp_in_loss,
                                                             right_beta,
                                                             loss_function_type);

                        if((loss_reduction < tree->loss_reduction) ||
                                ((loss_reduction == tree->loss_reduction) &&
                                        (std::abs(best_split_row_ind_in_sorted_array - train_x_height / 2) >
                                                std::abs(j - train_x_height / 2)))) {
                            tree->loss_reduction = loss_reduction;
                            tree->column = original_col_ind;

                            best_split_row_ind_in_sorted_array = j;
                            best_last_chosen_ind_in_sorted_array = last_chosen_ind_in_sorted_array;

                            best_left_beta = left_beta;
                            best_right_beta = right_beta;
                        }

                    }
                } // j < train_x_height
            } // i < no_cols

            if(tree->loss_reduction < 0) {

                for(int k = best_split_row_ind_in_sorted_array + 1; k <= best_last_chosen_ind_in_sorted_array; ++k) {
                    next_chosen_row_ind = (int) sorted_train_x_from.get(k, tree->column);
                    if(choose_labels[next_chosen_row_ind])
                        break;
                }

                tree->split_value = (train_x.get((int)sorted_train_x_from.get(best_split_row_ind_in_sorted_array, tree->column), tree->column) +
                                     train_x.get(next_chosen_row_ind, tree->column)) / 2;

                int left_count = 0, right_count = 0;
                for(int j = 0; j <= best_split_row_ind_in_sorted_array; ++j) {
                    original_row_ind = (int)sorted_train_x_from.get(j, tree->column);
                    if(choose_labels[original_row_ind]) {
                        left_inds[left_count] = original_row_ind;
                        ++left_count;
                    }
                }
                for(int j = best_split_row_ind_in_sorted_array + 1; j < train_x_height; ++j) {
                    original_row_ind = (int)sorted_train_x_from.get(j, tree->column);
                    if(choose_labels[original_row_ind]) {
                        right_inds[right_count] = original_row_ind;
                        ++right_count;
                    }
                }

                if (tree->right != nullptr)
                    delete tree->right;
                if (tree->left != nullptr)
                    delete tree->left;

                tree->right = new Tree_node<T>(tree->depth + 1);
                tree->right->prediction = best_right_beta;
                tree->left = new Tree_node<T>(tree->depth + 1);
                tree->left->prediction = best_left_beta;

                train(tree->right,
                      train_x,
                      sorted_train_x_from,
                      train_y,
                      ind_delta,
                      prediction,
                      first_comp_in_loss,
                      second_comp_in_loss,
                      monotonic_constraints,
                      loss_function_type,
                      right_inds,
                      right_count,
                      col_inds,
                      no_cols);
                train(tree->left,
                      train_x,
                      sorted_train_x_from,
                      train_y,
                      ind_delta,
                      prediction,
                      first_comp_in_loss,
                      second_comp_in_loss,
                      monotonic_constraints,
                      loss_function_type,
                      left_inds,
                      left_count,
                      col_inds,
                      no_cols);
                delete[] left_inds;
                delete[] right_inds;
                delete[] choose_labels;

            }
            else {

                std::cout << "Warning: this tree is not finished at level " << tree->depth << ". " << std::endl;

                if (tree->right != nullptr)
                    delete tree->right;
                if (tree->left != nullptr)
                    delete tree->left;

                tree->last_node = true;
                delete[] left_inds;
                delete[] right_inds;
                delete[] choose_labels;
            }

        }

    }

    template <typename T>
    T Fast_tree_trainer<T>::update_loss_reduction(Tree_node<T> *tree) {

        if(tree->last_node || tree->left->last_node || tree->right->last_node)
            return tree->loss_reduction;

        T left_loss_reduction, right_loss_reduction;
        right_loss_reduction = update_loss_reduction(tree->right);
        left_loss_reduction = update_loss_reduction(tree->left);

        tree->loss_reduction = left_loss_reduction + right_loss_reduction;

        return tree->loss_reduction;

    }

    template<typename T>
    void Fast_tree_trainer<T>::prune(Tree_node<T> *tree) {

        if (tree->last_node) return;
        #ifdef _DEBUG_BASE_LEARNER_TRAINER
            assert(tree->right != NULL && tree->left != NULL);
        #endif
        if (tree->right->loss_reduction > tree->left->loss_reduction) {
            tree->right->last_node = true;

            delete_tree(tree->right->right);
            delete_tree(tree->right->left);

            tree->right->right = nullptr,
                    tree->right->left = nullptr;

            prune(tree->left);
        } else {
            tree->left->last_node = true;

            delete_tree(tree->left->right);
            delete_tree(tree->left->left);

            tree->left->right = nullptr,
                    tree->left->left = nullptr;

            prune(tree->right);
        }
    }

}



