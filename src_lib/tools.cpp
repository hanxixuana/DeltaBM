//
// Created by xixuan on 10/10/16.
//

#include "tools.h"

#include <iostream>
#include <cassert>
#include <algorithm>
#include <chrono>

namespace dbm {

    void pause_at_end(std::string words) {
        std::cout << std::endl << std::endl << words << std::endl;
        std::string stop_at_end;
        std::getline(std::cin, stop_at_end);
    }

    Time_measurer::Time_measurer(int no_cores):
            no_cores(no_cores) {
        begin_time = std::clock();
        std::cout << std::endl
                  << "Timer at " << this
                  << " ---> Start recording time..."
                  << std::endl << std::endl;
    }

    Time_measurer::~Time_measurer() {
        end_time = std::clock();
        std::cout << std::endl
                  << "Timer at " << this
                  << " ---> " << "Elapsed Time: "
                  << double(end_time - begin_time) / CLOCKS_PER_SEC / no_cores
                  << " seconds"
                  << std::endl << std::endl;
    }

    int split_into_words(const std::string &line,
                         std::string *words,
                         const char sep) {
        size_t prev = 0, next = 0;
        int n_words = 0;
        while ((next = line.find_first_of(sep, prev)) != std::string::npos) {
            if (next - prev != 0) {
                words[n_words] = line.substr(prev, next - prev);
                n_words += 1;
            }
            prev = next + 1;
        }
        if (prev < line.size()) {
            words[n_words] = line.substr(prev);
            n_words += 1;
        }
        return n_words;
    }

    template <typename T>
    void remove_nan_row_inds(int *row_inds,
                             int &no_rows,
                             const int *col_inds,
                             const int &no_cols,
                             const Matrix<T> &train_x) {
        int no_nan_in_row, original_no_rows = no_rows;
        int temp;
        for(int i = 0; i < no_rows; ++i) {
            no_nan_in_row = 0;
            for(int j = 0; j < no_cols; ++j)
                no_nan_in_row += std::isnan(train_x.get(row_inds[i], col_inds[j]));
//            std::cout << row_inds[i] << ':' << no_nan_in_row << ' ';
            if(no_nan_in_row > 0) {
                temp = row_inds[i];
                row_inds[i] = row_inds[no_rows - 1];
                row_inds[no_rows - 1] = temp;
                --no_rows;
                --i;
            }
        }
//        std::cout << std::endl;
        if(original_no_rows > no_rows)
            std::cout << "Removed " << original_no_rows - no_rows << " rows containing NaNs." << std::endl;

    }

    template <typename T>
    void add_nans_to_mat(Matrix<T> &mat, int max_no_nan, int random_seed) {

        if(random_seed < 0)
            std::srand((unsigned int)(std::time(nullptr)));
        else
            std::srand((unsigned int) random_seed);

        int height = mat.get_height(), width = mat.get_width() - 1;

        int *mat_col_inds = new int[width];
        int *no_nan = new int[height];

        for(int i = 0; i < width; ++i)
            mat_col_inds[i] = i;

        for(int i = 0; i < height; ++i)
            no_nan[i] = rand() / (RAND_MAX / max_no_nan);

        for(int i = 0; i < height; ++i) {

            std::random_shuffle(mat_col_inds, mat_col_inds + width);

            for(int j = 0; j < no_nan[i]; ++j) {

                mat.assign(i, mat_col_inds[j], NAN);

            }

        }

        delete[] mat_col_inds;
        delete[] no_nan;
    }

    template <typename T>
    void range(const T &start, const T &end,
               const int & number,
               T *result,
               const T &scaling) {
        #ifdef _DEBUG_TOOLS
            assert(end > start);
        #endif
        T length = scaling * (end - start) / (number - 1);
        result[0] = start + (1.0 - scaling) * (end - start) / 2.0;
        for(int i = 1; i < number; ++i)
            result[i] = result[0] + i * length;
    }

    template<typename T>
    inline int middles(T *uniques,
                       int no_uniques) {
        for (int i = 0; i < no_uniques - 1; ++i) uniques[i] = (uniques[i] + uniques[i + 1]) / 2.0;
        return no_uniques - 1;
    }

    template<typename T>
    void shuffle(T *values,
                int no_values,
                int seed) {
        if(seed < 0)
            std::srand((unsigned int)
                               std::chrono::duration_cast< std::chrono::milliseconds >
                                       (std::chrono::system_clock::now().time_since_epoch()).count());
        else
            std::srand((unsigned int) seed);
        std::random_shuffle(values, values + no_values);
    }

    template<typename T>
    void make_data(const std::string &file_name,
                   int n_samples,
                   int n_features,
                   char data_type,
                   const int *sig_lin_inds,
                   const T *coef_sig_lin,
                   int n_sig_lin_feats,
                   const int *sig_quad_inds,
                   const T *coef_sig_quad,
                   int n_sig_quad_feats) {

        if (sig_lin_inds == NULL ||
                coef_sig_lin == NULL ||
                sig_quad_inds == NULL ||
                coef_sig_quad == NULL) {

            n_sig_lin_feats = 8, n_sig_quad_feats = 8;
            int lin_inds[] = {int(n_features * 0.1), int(n_features * 0.2),
                              int(n_features * 0.3), int(n_features * 0.4),
                              int(n_features * 0.5), int(n_features * 0.6),
                              int(n_features * 0.7), int(n_features * 0.8)};
            T coef_lin[] = {-10, 10, 1, 2, 5, -5, 10, -10};
            int quad_inds[] = {int(n_features * 0.15), int(n_features * 0.25),
                               int(n_features * 0.35), int(n_features * 0.45),
                               int(n_features * 0.55), int(n_features * 0.65),
                               int(n_features * 0.75), int(n_features * 0.85)};
            T coef_quad[] = {5, -3, -10, 4, 10, -5, 1, -2};


            dbm::Matrix<T> train_data(n_samples, n_features + 1);

            for (int i = 0; i < n_samples; ++i) {
                for (int j = 0; j < n_sig_lin_feats; ++j)
                    train_data[i][n_features] += coef_lin[j] * train_data[i][lin_inds[j]];
                for (int j = 0; j < n_sig_quad_feats; ++j)
                    train_data[i][n_features] +=
                            coef_quad[j] * std::pow(train_data[i][quad_inds[j]], T(2.0));
                switch (data_type) {

                    case 'n' : {
                        break;
                    }

                    case 'p' : {
                        train_data[i][n_features] =
                                std::round(std::max(train_data[i][n_features], T(0.001)));
                        break;
                    }

                    case 'b' : {
                        train_data[i][n_features] = train_data[i][n_features] < 0 ? 0 : 1;
                        break;
                    }

                    case 't': {
                        train_data[i][n_features] = std::max(train_data[i][n_features], T(0));
                        break;
                    }

                    default: {
                        throw std::invalid_argument("Specified data type does not exist.");
                    }

                }
            }

            train_data.print_to_file(file_name);
        } else {
            dbm::Matrix<T> train_data(n_samples, n_features + 1);

            for (int i = 0; i < n_samples; ++i) {
                for (int j = 0; j < n_sig_lin_feats; ++j)
                    train_data[i][n_features] +=
                            coef_sig_lin[j] * train_data[i][sig_lin_inds[j]];
                for (int j = 0; j < n_sig_quad_feats; ++j)
                    train_data[i][n_features] +=
                            coef_sig_quad[j] * std::pow(train_data[i][sig_quad_inds[j]], T(2.0));
                switch (data_type) {

                    case 'n' : {
                        break;
                    }

                    case 'p' : {
                        train_data[i][n_features] =
                                std::round(std::max(train_data[i][n_features], T(0.001)));
                        break;
                    }

                    case 'b' : {
                        train_data[i][n_features] = train_data[i][n_features] < 0 ? 0 : 1;
                        break;
                    }

                    case 't': {
                        train_data[i][n_features] = std::max(train_data[i][n_features], T(0));
                        break;
                    }

                    default: {
                        throw std::invalid_argument("Specified data type does not exist.");
                    }

                }
            }

            train_data.print_to_file(file_name);
        }

    }

    Params set_params(const std::string &param_string,
                      const char delimiter) {

        std::string words[100];

        size_t prev = 0, next = 0;
        int count = 0;
        while ((next = param_string.find_first_of(delimiter, prev)) != std::string::npos) {
            if (next - prev != 0) {
                words[count] = param_string.substr(prev, next - prev);
                count += 1;
            }
            prev = next + 1;
        }

        if (prev < param_string.size()) {
            words[count] = param_string.substr(prev);
            count += 1;
        }

        #ifdef _DEBUG_TOOLS
            assert(count % 2 == 0);
        #endif

        Params params;

        for (int i = 0; i < count / 2; ++i) {

            // DBM
            if (words[2 * i] == "dbm_no_bunches_of_learners")
                params.dbm_no_bunches_of_learners = std::stoi(words[2 * i + 1]);
            else if (words[2 * i] == "dbm_no_candidate_feature")
                params.dbm_no_candidate_feature = std::stoi(words[2 * i + 1]);
            else if (words[2 * i] == "dbm_portion_train_sample")
                params.dbm_portion_train_sample = std::stod(words[2 * i + 1]);
            else if (words[2 * i] == "dbm_no_cores")
                params.dbm_no_cores = std::stoi(words[2 * i + 1]);

            else if (words[2 * i] == "dbm_loss_function")
                params.dbm_loss_function = words[2 * i + 1].front();

            else if (words[2 * i] == "dbm_display_training_progress")
                params.dbm_display_training_progress = std::stoi(words[2 * i + 1]) > 0;
            else if (words[2 * i] == "dbm_record_every_tree")
                params.dbm_record_every_tree = std::stoi(words[2 * i + 1]) > 0;
            else if (words[2 * i] == "dbm_freq_showing_loss_on_test")
                params.dbm_freq_showing_loss_on_test = std::stoi(words[2 * i + 1]);

            else if (words[2 * i] == "dbm_shrinkage")
                params.dbm_shrinkage = std::stod(words[2 * i + 1]);

            else if (words[2 * i] == "dbm_nonoverlapping_training")
                params.dbm_nonoverlapping_training = std::stoi(words[2 * i + 1]);

            else if (words[2 * i] == "dbm_remove_rows_containing_nans")
                params.dbm_remove_rows_containing_nans = std::stoi(words[2 * i + 1]);
            else if (words[2 * i] == "dbm_min_no_samples_per_bl")
                params.dbm_min_no_samples_per_bl = std::stoi(words[2 * i + 1]);

            else if (words[2 * i] == "dbm_random_seed")
                params.dbm_random_seed = std::stoi(words[2 * i + 1]);

            else if (words[2 * i] == "dbm_portion_for_trees")
                params.dbm_portion_for_trees = std::stod(words[2 * i + 1]);
            else if (words[2 * i] == "dbm_portion_for_lr")
                params.dbm_portion_for_lr = std::stod(words[2 * i + 1]);
            else if (words[2 * i] == "dbm_portion_for_nn")
                params.dbm_portion_for_nn = std::stod(words[2 * i + 1]);
            else if (words[2 * i] == "dbm_portion_for_s")
                params.dbm_portion_for_s = std::stod(words[2 * i + 1]);
            else if (words[2 * i] == "dbm_portion_for_k")
                params.dbm_portion_for_k = std::stod(words[2 * i + 1]);
            else if (words[2 * i] == "dbm_portion_for_d")
                params.dbm_portion_for_d = std::stod(words[2 * i + 1]);

            else if (words[2 * i] == "dbm_accumulated_portion_shrinkage_for_selected_bl")
                params.dbm_accumulated_portion_shrinkage_for_selected_bl = std::stod(words[2 * i + 1]);
            else if (words[2 * i] == "dbm_portion_shrinkage_for_unselected_bl")
                params.dbm_portion_shrinkage_for_unselected_bl = std::stod(words[2 * i + 1]);

            // tweedie
            else if (words[2 * i] == "tweedie_p")
                params.tweedie_p = std::stod(words[2 * i + 1]);

            // splines
            else if (words[2 * i] == "splines_no_knot")
                params.splines_no_knot = std::stoi(words[2 * i + 1]);
            else if (words[2 * i] == "splines_regularization")
                params.splines_regularization = std::stod(words[2 * i + 1]);
            else if (words[2 * i] == "splines_hinge_coefficient")
                params.splines_hinge_coefficient = std::stod(words[2 * i + 1]);

            // kmeans
            else if (words[2 * i] == "kmeans_no_centroids")
                params.kmeans_no_centroids = std::stoi(words[2 * i + 1]);
            else if (words[2 * i] == "kmeans_max_iteration")
                params.kmeans_max_iteration = std::stoi(words[2 * i + 1]);
            else if (words[2 * i] == "kmeans_tolerance")
                params.kmeans_tolerance = std::stod(words[2 * i + 1]);
            else if (words[2 * i] == "kmeans_fraction_of_pairs")
                params.kmeans_fraction_of_pairs = std::stod(words[2 * i + 1]);

            // neural networks
            else if (words[2 * i] == "nn_no_hidden_neurons")
                params.nn_no_hidden_neurons = std::stoi(words[2 * i + 1]);
            else if (words[2 * i] == "nn_step_size")
                params.nn_step_size = std::stod(words[2 * i + 1]);
            else if (words[2 * i] == "nn_validate_portion")
                params.nn_validate_portion = std::stod(words[2 * i + 1]);
            else if (words[2 * i] == "nn_batch_size")
                params.nn_batch_size = std::stoi(words[2 * i + 1]);
            else if (words[2 * i] == "nn_max_iteration")
                params.nn_max_iteration = std::stoi(words[2 * i + 1]);

            // CART
            else if (words[2 * i] == "cart_min_samples_in_a_node")
                params.cart_min_samples_in_a_node = std::stoi(words[2 * i + 1]);
            else if (words[2 * i] == "cart_max_depth")
                params.cart_max_depth = std::stoi(words[2 * i + 1]);
            else if (words[2 * i] == "cart_prune")
                params.cart_prune = std::stoi(words[2 * i + 1]);

            // linear regression
            else if (words[2 * i] == "lr_regularization")
                params.lr_regularization = std::stod(words[2 * i + 1]);

            // dpc stairs
            else if (words[2 * i] == "dpcs_no_ticks")
                params.dpcs_no_ticks = std::stoi(words[2 * i + 1]);
            else if (words[2 * i] == "dpcs_range_shrinkage_of_ticks")
                params.dpcs_range_shrinkage_of_ticks = std::stod(words[2 * i + 1]);

            else if (words[2 * i] == "dbm_do_perf") {
                std::string opt = words[2 * i + 1];
                std::transform(opt.begin(), opt.end(), opt.begin(), ::tolower);
                if (opt == "yes") params.dbm_do_perf = true;
            }

            // partial dependence plot
            else if (words[2 * i] == "pdp_no_x_ticks")
                params.pdp_no_x_ticks = std::stoi(words[2 * i + 1]);
            else if (words[2 * i] == "pdp_no_resamplings")
                params.pdp_no_resamplings = std::stoi(words[2 * i + 1]);
            else if (words[2 * i] == "pdp_resampling_portion")
                params.pdp_resampling_portion = std::stod(words[2 * i + 1]);
            else if (words[2 * i] == "pdp_ci_bandwidth")
                params.pdp_ci_bandwidth = std::stod(words[2 * i + 1]);
            else if (words[2 * i] == "pdp_save_files")
                params.pdp_save_files = std::stoi(words[2 * i + 1]);

            // two-way models
            else if (words[2 * i] == "twm_no_x_ticks")
                params.twm_no_x_ticks = std::stoi(words[2 * i + 1]);
            else if (words[2 * i] == "twm_no_resamplings")
                params.twm_no_resamplings = std::stoi(words[2 * i + 1]);
            else if (words[2 * i] == "twm_resampling_portion")
                params.twm_resampling_portion = std::stod(words[2 * i + 1]);
            else if (words[2 * i] == "twm_ci_bandwidth")
                params.twm_ci_bandwidth = std::stod(words[2 * i + 1]);

            // throw an exception
            else {
                throw std::invalid_argument("Specified parameter does not exist.");
            }

        }

        return params;

    }

}

// explicit instantiation of templated functions
namespace dbm {

    template void remove_nan_row_inds<double>(int *row_inds, int &no_rows, const int *col_inds, const int &no_cols, const Matrix<double> &train_x);

    template void remove_nan_row_inds<float>(int *row_inds, int &no_rows, const int *col_inds, const int &no_cols, const Matrix<float> &train_x);

    template void add_nans_to_mat<double>(Matrix<double> &mat, int max_no_nan, int random_seed);

    template void add_nans_to_mat<float>(Matrix<float> &mat, int max_no_nan, int random_seed);

    template void range<double>(const double &start, const double &end, const int & number, double *result, const double &scaling);

    template void range<float>(const float &start, const float &end, const int & number, float *result, const float &scaling);

    template int middles<float>(float *uniqes, int no_uniques);

    template int middles<double>(double *uniqes, int no_uniques);

    template void shuffle<int>(int *values, int no_values, int seed);

    template void shuffle<float>(float *values, int no_values, int seed);

    template void shuffle<double>(double *values, int no_values, int seed);

    template void make_data<double>(const std::string &file_name, int n_samples, int n_features, char data_type,
                                    const int *sig_lin_inds, const double *coef_sig_lin, int n_sig_lin_feats,
                                    const int *sig_quad_inds, const double *coef_sig_quad, int n_sig_quad_feats);

    template void make_data<float>(const std::string &file_name, int n_samples, int n_features, char data_type,
                                   const int *sig_lin_inds, const float *coef_sig_lin, int n_sig_lin_feats,
                                   const int *sig_quad_inds, const float *coef_sig_quad, int n_sig_quad_feats);


}





