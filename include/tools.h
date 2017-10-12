//
// Created by xixuan on 10/10/16.
//

#ifndef DBM_CODE_TOOLS_H
#define DBM_CODE_TOOLS_H

//#ifndef _DEBUG_TOOLS
//#define _DEBUG_TOOLS
//#endif

#include "matrix.h"

#include <string>
#include <ctime>
#include <limits>

/*
 * tools for parameters
 */

namespace dbm {

    struct Params {

        // DBM
        int dbm_no_bunches_of_learners = 101;
        int dbm_no_candidate_feature = 5;
        double dbm_portion_train_sample = 0.75;

        int dbm_no_cores = 0;

        char dbm_loss_function = 'n';

        bool dbm_display_training_progress = true;
        bool dbm_record_every_tree = false;
        int dbm_freq_showing_loss_on_test = 1;

        double dbm_shrinkage = 0.01;

        int dbm_nonoverlapping_training = 1;

        int dbm_remove_rows_containing_nans = 1;
        int dbm_min_no_samples_per_bl = 50;

        int dbm_random_seed = -1;

        // portions should be summed to 1
        double dbm_portion_for_trees = 0.2;
        double dbm_portion_for_lr = 0.2;
        double dbm_portion_for_s = 0.2;
        double dbm_portion_for_k = 0.2;
        double dbm_portion_for_nn = 0;
        double dbm_portion_for_d = 0.2;

        double dbm_accumulated_portion_shrinkage_for_selected_bl = 1.05;
        double dbm_portion_shrinkage_for_unselected_bl = 3;

        // tweedie: p should in (1, 2)
        double tweedie_p = 1.6;

        // splines
        int splines_no_knot = 4;
        double splines_portion_of_pairs = 0.01;
        double splines_regularization = 3;
        double splines_hinge_coefficient = 1;

        // kmeans
        int kmeans_no_centroids = 5;
        int kmeans_max_iteration = 50;
        double kmeans_tolerance = 1e-2;
        double kmeans_fraction_of_pairs = 0.01;

        // neural networks
        int nn_no_hidden_neurons = 10;
        double nn_step_size = 0.1;
        double nn_validate_portion = 0.25;
        int nn_batch_size = 100;
        int nn_max_iteration = 100;
        int nn_no_rise_of_loss_on_validate = 20;

        // CART
        int cart_min_samples_in_a_node = 50;
        int cart_max_depth = 3;
        int cart_prune = 1;

        // linear regression
        double lr_regularization = 0.1;

        // dpc stairs
        int dpcs_no_ticks = 10;
        double dpcs_range_shrinkage_of_ticks = 0.1;

        /* Switch for output */
        // performance
        bool dbm_do_perf = true;

        // partial dependence plot
        int pdp_no_x_ticks = 10;
        int pdp_no_resamplings = 3;
        double pdp_resampling_portion = 0.1;
        double pdp_ci_bandwidth = 4;
        int pdp_save_files = 1;

        // two-way models
        int twm_no_x_ticks = 5;
        int twm_no_resamplings = 1;
        double twm_resampling_portion = 0.1;
        double twm_ci_bandwidth = 4;

    };

    Params set_params(const std::string &param_string, const char delimiter = ' ');

}

/*
 * other tools
 */

namespace dbm {

    // pause at end
    void pause_at_end(std::string words = "=============== \nEND OF PROGRAM. \n=============== \n ");

    // elapsed time measurer
    class Time_measurer {
    private:
        int no_cores;
        std::time_t begin_time;
        std::time_t end_time;
    public:
        Time_measurer(int no_cores = 1);

        ~Time_measurer();
    };

    // split a line into words
    int split_into_words(const std::string &line, std::string *words, const char sep = ' ');

    // remove row indices with nans in selected columns
    template <typename T>
    void remove_nan_row_inds(int *row_inds, int &no_rows, const int *col_inds, const int &no_cols, const Matrix<T> &train_x);

    // add nans to a matrix
    template <typename T>
    void add_nans_to_mat(Matrix<T> &mat, int max_no_nan, int random_seed = -1);

}

/*
 * tools for matrices
 */

namespace dbm {

    // return n-dimensional array of the form:
    // [start, start + (end - start) / (n - 1), start + 2 * (end - start) / (n - 1), ..., end]
    template <typename T>
    void range(const T &start, const T &end, const int & number, T *result, const T &scaling = 1);

    // find middle values in sorted uniques
    template<typename T>
    int middles(T *uniques, int no_uniques);

    // shuffle an array
    template<typename T>
    void shuffle(T *values, int no_values, int seed = -1);

    template<typename T>
    void make_data(const std::string &file_name, int n_samples = 100000, int n_features = 30, char data_type = 'n',
                   const int *sig_lin_inds = NULL, const T *coef_sig_lin = NULL, int n_sig_lin_feats = 0,
                   const int *sig_quad_inds = NULL, const T *coef_sig_quad = NULL, int n_sig_quad_feats = 0);

}

/*
 * tools for calibrate.plot
 * used to store averaged observe value
 */
namespace dbm {
    template<typename T>
    struct AveragedObservation{
        T sum = (T)0;
        T sum2 = (T)0;
        int N;
    };
} // namespace dbm::calibrate.plot

#endif //DBM_CODE_TOOLS_H




