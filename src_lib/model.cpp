//
// Created by xixuan on 10/10/16.
//

#include "model.h"

#include <cassert>
#include <iostream>

#ifdef _OMP
#include <omp.h>
#endif

namespace dbm {

    template
    class DBM<double>;

    template
    class DBM<float>;

    template
    class AUTO_DBM<double>;

    template
    class AUTO_DBM<float>;

}

// DBM
namespace dbm {

    template<typename T>
    DBM<T>::DBM(int no_bunches_of_learners,
                int no_cores,
                int no_candidate_feature,
                int no_train_sample,
                int total_no_feature):
            no_bunches_of_learners(no_bunches_of_learners),
            no_cores(no_cores),
            total_no_feature(total_no_feature),
            no_candidate_feature(no_candidate_feature),
            no_train_sample(no_train_sample),
            loss_function(Loss_function<T>(params)) {

        learners = new Base_learner<T> *[(no_bunches_of_learners - 1) * no_cores + 1];

        for (int i = 0; i < (no_bunches_of_learners - 1) * no_cores + 1; ++i) {
            learners[i] = nullptr;
        }

        tree_trainer = nullptr;
        mean_trainer = nullptr;
        linear_regression_trainer = nullptr;
        neural_network_trainer = nullptr;
        splines_trainer = nullptr;
        kmeans2d_trainer = nullptr;
        dpc_stairs_trainer = nullptr;
    }

    template<typename T>
    DBM<T>::DBM(const Params &params) :
            params(params),
            loss_function(Loss_function<T>(params)) {

        no_cores = params.dbm_no_cores;
        no_bunches_of_learners = params.dbm_no_bunches_of_learners;
        no_candidate_feature = params.dbm_no_candidate_feature;

        std::uniform_real_distribution<double> dist(0.0, 1.0);

        double *type_choices = new double[no_bunches_of_learners];

        if(params.dbm_random_seed < 0) {
            std::random_device rd;
            std::mt19937 mt(rd());
            for(int i = 0; i < no_bunches_of_learners; ++i)
                type_choices[i] = dist(mt);
        }
        else {
            std::mt19937 mt(params.dbm_random_seed);
            for(int i = 0; i < no_bunches_of_learners; ++i)
                type_choices[i] = dist(mt);
        }

        #ifdef _OMP
        if(no_cores == 0 || no_cores > omp_get_max_threads()) {
            std::cout << std::endl
                      << "================================="
                      << std::endl
                      << "no_cores: " << no_cores
                      << " is ignored, using " << omp_get_max_threads() << " !"
                      << std::endl
                      << "================================="
                      << std::endl;
            no_cores = omp_get_max_threads();
        }
        else if(no_cores < 0)
            throw std::invalid_argument("Specified no_cores is negative.");
        #else
        std::cout << std::endl
                  << "================================="
                  << std::endl
                  << "OpenMP is disabled!"
                  << std::endl
                  << "no_cores: " << no_cores
                  << " is ignored, using " << 1 << " !"
                  << std::endl
                  << "================================="
                  << std::endl;
        no_cores = 1;
        #endif

        learners = new Base_learner<T> *[(no_bunches_of_learners - 1) * no_cores + 1];

        learners[0] = new Global_mean<T>;

        #ifdef _DEBUG_MODEL
            assert((params.dbm_portion_for_trees +
                           params.dbm_portion_for_lr +
                           params.dbm_portion_for_s +
                           params.dbm_portion_for_k +
                           params.dbm_portion_for_nn+
                           params.dbm_portion_for_d) >= 1.0);
        #endif

        double type_choose;
        for (int i = 1; i < no_bunches_of_learners; ++i) {

            type_choose = type_choices[i];

            if(type_choose < params.dbm_portion_for_trees)
                for(int j = 0; j < no_cores; ++j)
                    learners[no_cores * (i - 1) + j + 1] = new Tree_node<T>(0);

            else if(type_choose < (params.dbm_portion_for_trees + params.dbm_portion_for_lr))
                for(int j = 0; j < no_cores; ++j)
                    learners[no_cores * (i - 1) + j + 1] = new Linear_regression<T>(no_candidate_feature,
                                                                                    params.dbm_loss_function);

            else if(type_choose < (params.dbm_portion_for_trees + params.dbm_portion_for_lr + params.dbm_portion_for_s))
                for(int j = 0; j < no_cores; ++j)
                    learners[no_cores * (i - 1) + j + 1] = new Splines<T>(params.splines_no_knot,
                                                                          params.dbm_loss_function,
                                                                          params.splines_hinge_coefficient);

            else if(type_choose < (params.dbm_portion_for_trees + params.dbm_portion_for_lr +
                                    params.dbm_portion_for_s + params.dbm_portion_for_k))
                for(int j = 0; j < no_cores; ++j)
                    learners[no_cores * (i - 1) + j + 1] = new Kmeans2d<T>(params.kmeans_no_centroids,
                                                                           params.dbm_loss_function);

            else if(type_choose < (params.dbm_portion_for_trees + params.dbm_portion_for_lr +
                                    params.dbm_portion_for_s + params.dbm_portion_for_k +
                                    params.dbm_portion_for_nn))
                for(int j = 0; j < no_cores; ++j)
                    learners[no_cores * (i - 1) + j + 1] = new Neural_network<T>(no_candidate_feature,
                                                                                 params.nn_no_hidden_neurons,
                                                                                 params.dbm_loss_function);

            else if(type_choose < (params.dbm_portion_for_trees + params.dbm_portion_for_lr +
                                   params.dbm_portion_for_s + params.dbm_portion_for_k +
                                   params.dbm_portion_for_nn + params.dbm_portion_for_d))
                for(int j = 0; j < no_cores; ++j)
                    learners[no_cores * (i - 1) + j + 1] = new DPC_stairs<T>(no_candidate_feature,
                                                                             params.dbm_loss_function,
                                                                             params.dpcs_no_ticks);
            /* END of choose base_learners */
        } // i, 1, no_bunches_of_learners

        tree_trainer = new Fast_tree_trainer<T>(params);
        mean_trainer = new Mean_trainer<T>(params);
        linear_regression_trainer = new Linear_regression_trainer<T>(params);
        neural_network_trainer = new Neural_network_trainer<T>(params);
        splines_trainer = new Splines_trainer<T>(params);
        kmeans2d_trainer = new Kmeans2d_trainer<T>(params);
        dpc_stairs_trainer = new DPC_stairs_trainer<T>(params);

        delete[] type_choices;

    }

    template<typename T>
    DBM<T>::~DBM() {

        for (int i = 0; i < (no_bunches_of_learners - 1) * no_cores + 1; ++i) {

            delete learners[i];

        }
        delete[] learners;

        delete tree_trainer;
        delete mean_trainer;
        delete linear_regression_trainer;
        delete neural_network_trainer;
        delete splines_trainer;
        delete kmeans2d_trainer;
        delete dpc_stairs_trainer;

        delete prediction_train_data;

        delete test_loss_record;
        if (train_loss_record != nullptr)
            delete train_loss_record;

        delete pdp_result;
        delete ss_result;

        delete prediction;

        if(vec_of_two_way_predictions != nullptr) {

            for(int i = 0; i < no_two_way_models; ++i) {

                delete vec_of_two_way_predictions[i];

            }
            delete[] vec_of_two_way_predictions;

        }
        delete prediction_two_way;
        delete predictor_x_ticks;

        delete mat_plot_dat;

    }

    template<typename T>
    void DBM<T>::train(const Data_set<T> &data_set,
                       const Matrix<T> &input_monotonic_constraints) {

        std::cout << "dbm_no_bunches_of_learners: " << params.dbm_no_bunches_of_learners << std::endl
                  << "dbm_no_cores: " << params.dbm_no_cores << std::endl
                  << "dbm_portion_train_sample: " << params.dbm_portion_train_sample << std::endl
                  << "dbm_no_candidate_feature: " << params.dbm_no_candidate_feature << std::endl
                  << "dbm_shrinkage: " << params.dbm_shrinkage << std::endl
                  << "random_seed in Parameters: " << params.dbm_random_seed << std::endl
                  << "random_seed in Data_set: " << data_set.random_seed << std::endl;


        Time_measurer timer(no_cores);

        Matrix<T> const &train_x = data_set.get_train_x();
        Matrix<T> const &train_y = data_set.get_train_y();
        Matrix<T> const &test_x = data_set.get_test_x();
        Matrix<T> const &test_y = data_set.get_test_y();

        Matrix<T> sorted_train_x = copy(train_x);
        const Matrix<T> sorted_train_x_from = col_sort(sorted_train_x);
//        const Matrix<T> train_x_sorted_to = col_sorted_to(sorted_train_x_from);

        int n_samples = train_x.get_height(), n_features = train_x.get_width();
        int n_test_samples = test_x.get_height();

        no_train_sample = (int)(params.dbm_portion_train_sample * n_samples);
        total_no_feature = n_features;

        int no_samples_in_nonoverlapping_batch = no_train_sample / no_cores - 1;
        int *whole_row_inds = new int[n_samples];
        int **thread_row_inds_vec = new int*[no_cores];
        for(int i = 0; i < no_cores; ++i) {
            thread_row_inds_vec[i] = whole_row_inds + i * no_samples_in_nonoverlapping_batch;
        }

        #ifdef _DEBUG_MODEL
            assert(no_train_sample <= n_samples &&
                           no_candidate_feature <= total_no_feature &&
                           input_monotonic_constraints.get_height() == total_no_feature);

            for (int i = 0; i < n_features; ++i) {
                // serves as a check of whether the length of monotonic_constraints
                // is equal to the length of features in some sense

                assert(input_monotonic_constraints.get(i, 0) == 0 ||
                       input_monotonic_constraints.get(i, 0) == -1 ||
                       input_monotonic_constraints.get(i, 0) == 1);
            }
        #endif

        int *row_inds = new int[n_samples], *col_inds = new int[n_features];

        for (int i = 0; i < n_features; ++i)
            col_inds[i] = i;
        for (int i = 0; i < n_samples; ++i)
            row_inds[i] = i;

        if (prediction_train_data != nullptr)
            delete prediction_train_data;

        if (test_loss_record != nullptr)
            delete[] test_loss_record;
        prediction_train_data = new Matrix<T>(n_samples, 1, 0);
        test_loss_record = new T[no_bunches_of_learners / params.dbm_freq_showing_loss_on_test + 1];
        if (params.dbm_do_perf) {
            if (train_loss_record != nullptr)
                delete[] train_loss_record;
            train_loss_record = new T[no_bunches_of_learners / params.dbm_freq_showing_loss_on_test + 1];
        }

        T lowest_test_loss = std::numeric_limits<T>::max();

        /*
         * ind_delta:
         * col 0: individual delta
         * col 1: denominator of individual delta
         */
        Matrix<T> ind_delta(n_samples, 2, 0);

        Matrix<T> prediction_test_data(n_test_samples, 1, 0);

        #ifdef _OMP
            omp_set_num_threads(no_cores);
        #endif

        unsigned int *seeds = new unsigned int[(no_bunches_of_learners - 1) * no_cores];

        if(params.dbm_random_seed < 0) {
            std::random_device rd;
            std::mt19937 mt(rd());
            std::uniform_real_distribution<T> dist(0, 1000);
            for(int i = 0; i < (no_bunches_of_learners - 1) * no_cores; ++i)
                seeds[i] = (unsigned int) dist(mt);
        }
        else {
            std::mt19937 mt(params.dbm_random_seed);
            std::uniform_real_distribution<T> dist(0, 1000);
            for(int i = 0; i < (no_bunches_of_learners - 1) * no_cores; ++i)
                seeds[i] = (unsigned int) dist(mt);
        }

        std::cout << "Learner " << "(" << learners[0]->get_type() << ") "
                  << " No. " << 0 << " -> ";

        loss_function.calculate_ind_delta(train_y,
                                          *prediction_train_data,
                                          ind_delta,
                                          params.dbm_loss_function);
        mean_trainer->train(dynamic_cast<Global_mean<T> *>(learners[0]),
                            train_x,
                            ind_delta,
                            *prediction_train_data,
                            params.dbm_loss_function);
        learners[0]->predict(train_x,
                             *prediction_train_data);
        learners[0]->predict(test_x,
                             prediction_test_data);

        test_loss_record[0] = loss_function.loss(test_y,
                                                 prediction_test_data,
                                                 params.dbm_loss_function);
        train_loss_record[0] = loss_function.loss(train_y,
                                                  *prediction_train_data,
                                                  params.dbm_loss_function);

        if(test_loss_record[0] < lowest_test_loss)
            lowest_test_loss = test_loss_record[0];

        if (params.dbm_display_training_progress) {
            std::cout << std::endl
                      << '(' << 0 << ')'
                      << " \tLowest loss on test set: "
                      << lowest_test_loss
                      << std::endl
                      << " \t\tLoss on test set: "
                      << test_loss_record[0]
                      << std::endl
                      << " \t\tLoss on train set: "
                      << train_loss_record[0]
                      << std::endl << std::endl;
        }

        if (params.dbm_do_perf) {
            train_loss_record[0] = loss_function.loss(train_y,
                                                      *prediction_train_data,
                                                      params.dbm_loss_function);
        }

        char type;
        for (int i = 1; i < no_bunches_of_learners; ++i) {

            if((!params.dbm_display_training_progress) && i % 10 == 0)
                printf("\n");

            loss_function.calculate_ind_delta(train_y,
                                              *prediction_train_data,
                                              ind_delta,
                                              params.dbm_loss_function);

            if(params.dbm_nonoverlapping_training) {
                std::copy(row_inds, row_inds + n_samples, whole_row_inds);
                shuffle(whole_row_inds,
                        n_samples,
                        seeds[no_cores * (i - 1) + 1]);
//                std::cout << seeds[no_cores * (i - 1) + 1] << std::endl;
            }

            type = learners[(i - 1) * no_cores + 1]->get_type();
            switch (type) {
                case 't': {
                    #ifdef _OMP
                    #pragma omp parallel default(shared)
                    {
                        int thread_id = omp_get_thread_num();
                        int learner_id = no_cores * (i - 1) + thread_id + 1;
                    #else
                    {
                        int thread_id = 0, learner_id = i;
                    #endif
                        if (params.dbm_display_training_progress) {
                            #pragma omp critical
                            if(params.dbm_nonoverlapping_training)
                                std::printf("Learner (%c) No. %d -> "
                                                    "Training Tree at %p "
                                                    "number of samples: %d "
                                                    "max_depth: %d %d...\n",
                                            type,
                                            learner_id,
                                            learners[learner_id],
                                            no_samples_in_nonoverlapping_batch,
                                            params.cart_max_depth,
                                            seeds[learner_id - 1]);
                            else
                                std::printf("Learner (%c) No. %d -> "
                                                    "Training Tree at %p "
                                                    "number of samples: %d "
                                                    "max_depth: %d %d...\n",
                                            type,
                                            learner_id,
                                            learners[learner_id],
                                            no_train_sample,
                                            params.cart_max_depth,
                                            seeds[learner_id - 1]);
                        }
                        else {
                            printf(".");
                        }

                        if(params.dbm_nonoverlapping_training) {
                            int *thread_col_inds = new int[n_features];
                            std::copy(col_inds,
                                      col_inds + n_features,
                                      thread_col_inds);

                            #ifdef _OMP
                            #pragma omp critical
                            #endif
                            shuffle(thread_col_inds,
                                    n_features,
                                    seeds[learner_id - 1]);

                            Matrix<T> first_comp_in_loss = loss_function.first_comp(train_y,
                                                                                    *prediction_train_data,
                                                                                    params.dbm_loss_function);
                            Matrix<T> second_comp_in_loss = loss_function.second_comp(train_y,
                                                                                      *prediction_train_data,
                                                                                      params.dbm_loss_function);

                            tree_trainer->train(dynamic_cast<Tree_node<T> *>(learners[learner_id]),
                                                train_x,
                                                sorted_train_x_from,
                                                train_y,
                                                ind_delta,
                                                *prediction_train_data,
                                                first_comp_in_loss,
                                                second_comp_in_loss,
                                                input_monotonic_constraints,
                                                params.dbm_loss_function,
                                                thread_row_inds_vec[thread_id],
                                                no_samples_in_nonoverlapping_batch,
                                                thread_col_inds,
                                                no_candidate_feature);

                            delete[] thread_col_inds;

                        }
                        else {
                            int *thread_row_inds = new int[n_samples];
                            int *thread_col_inds = new int[n_features];
                            std::copy(row_inds,
                                      row_inds + n_samples,
                                      thread_row_inds);
                            std::copy(col_inds,
                                      col_inds + n_features,
                                      thread_col_inds);

                            #ifdef _OMP
                            #pragma omp critical
                            #endif
                            {
                                shuffle(thread_row_inds,
                                        n_samples,
                                        seeds[learner_id - 1]);
                                shuffle(thread_col_inds,
                                        n_features,
                                        seeds[learner_id - 1]);
                            }

                            Matrix<T> first_comp_in_loss = loss_function.first_comp(train_y,
                                                                                    *prediction_train_data,
                                                                                    params.dbm_loss_function);
                            Matrix<T> second_comp_in_loss = loss_function.second_comp(train_y,
                                                                                      *prediction_train_data,
                                                                                      params.dbm_loss_function);

                            tree_trainer->train(dynamic_cast<Tree_node<T> *>(learners[learner_id]),
                                                train_x,
                                                sorted_train_x_from,
                                                train_y,
                                                ind_delta,
                                                *prediction_train_data,
                                                first_comp_in_loss,
                                                second_comp_in_loss,
                                                input_monotonic_constraints,
                                                params.dbm_loss_function,
                                                thread_row_inds,
                                                no_train_sample,
                                                thread_col_inds,
                                                no_candidate_feature);

                            delete[] thread_row_inds;
                            delete[] thread_col_inds;

                        }

                        tree_trainer->update_loss_reduction(dynamic_cast<Tree_node<T> *>(learners[learner_id]));

                        if(params.cart_prune) {
                            tree_trainer->prune(dynamic_cast<Tree_node<T> *>(learners[learner_id]));
                        }

                        #ifdef _OMP
                        #pragma omp barrier
                        #endif
                        {
                            #ifdef _OMP
                            #pragma omp critical
                            #endif
                            learners[learner_id]->predict(train_x,
                                                          *prediction_train_data,
                                                          params.dbm_shrinkage);
                            #ifdef _OMP
                            #pragma omp critical
                            #endif
                            learners[learner_id]->predict(test_x,
                                                          prediction_test_data,
                                                          params.dbm_shrinkage);

                            if(params.dbm_record_every_tree) {
                                #ifdef _OMP
                                #pragma omp critical
                                #endif
                                {
                                    Tree_info<T> tree_info(dynamic_cast<Tree_node<T> *>
                                                           (learners[learner_id]));
                                    tree_info.print_to_file("trees.txt",
                                                            learner_id);
                                }
                            }
                        }

                    }

                    break;

                }
                case 'l': {
                    #ifdef _OMP
                    #pragma omp parallel default(shared)
                    {
                        int thread_id = omp_get_thread_num();
                        int learner_id = no_cores * (i - 1) + thread_id + 1;
                    #else
                    {
                        int thread_id = 0, learner_id = i;
                    #endif
                        if (params.dbm_display_training_progress) {
                            #pragma omp critical
                            if(params.dbm_nonoverlapping_training) {
                                std::printf("Learner (%c) No. %d -> "
                                                    "Training Linear Regression at %p "
                                                    "number of samples: %d "
                                                    "number of predictors: %d ...\n",
                                            type,
                                            learner_id,
                                            learners[learner_id],
                                            no_samples_in_nonoverlapping_batch,
                                            no_candidate_feature);
                            }
                            else {
                                std::printf("Learner (%c) No. %d -> "
                                                    "Training Linear Regression at %p "
                                                    "number of samples: %d "
                                                    "number of predictors: %d ...\n",
                                            type,
                                            learner_id,
                                            learners[learner_id],
                                            no_train_sample,
                                            no_candidate_feature);
                            }
                        }
                        else {
                            printf(".");
                        }

                        if(params.dbm_nonoverlapping_training) {
                            int *thread_col_inds = new int[n_features];
                            std::copy(col_inds,
                                      col_inds + n_features,
                                      thread_col_inds);

                            #ifdef _OMP
                            #pragma omp critical
                            #endif
                            shuffle(thread_col_inds,
                                    n_features,
                                    seeds[learner_id - 1]);

                            linear_regression_trainer->train(dynamic_cast<Linear_regression<T> *>
                                                             (learners[learner_id]),
                                                             train_x,
                                                             ind_delta,
                                                             input_monotonic_constraints,
                                                             thread_row_inds_vec[thread_id],
                                                             no_samples_in_nonoverlapping_batch,
                                                             thread_col_inds,
                                                             no_candidate_feature);

                            delete[] thread_col_inds;

                        }
                        else {
                            int *thread_row_inds = new int[n_samples];
                            int *thread_col_inds = new int[n_features];
                            std::copy(row_inds,
                                      row_inds + n_samples,
                                      thread_row_inds);
                            std::copy(col_inds,
                                      col_inds + n_features,
                                      thread_col_inds);

                            #ifdef _OMP
                            #pragma omp critical
                            #endif
                            {
                                shuffle(thread_row_inds,
                                        n_samples,
                                        seeds[learner_id - 1]);
                                shuffle(thread_col_inds,
                                        n_features,
                                        seeds[learner_id - 1]);
                            }

                            linear_regression_trainer->train(dynamic_cast<Linear_regression<T> *>
                                                             (learners[learner_id]),
                                                             train_x,
                                                             ind_delta,
                                                             input_monotonic_constraints,
                                                             thread_row_inds,
                                                             no_train_sample,
                                                             thread_col_inds,
                                                             no_candidate_feature);

                            delete[] thread_row_inds;
                            delete[] thread_col_inds;

                        }

                        #ifdef _OMP
                        #pragma omp barrier
                        #endif
                        {
                            #ifdef _OMP
                            #pragma omp critical
                            #endif
                            learners[learner_id]->predict(train_x,
                                                          *prediction_train_data,
                                                          params.dbm_shrinkage);
                            #ifdef _OMP
                            #pragma omp critical
                            #endif
                            learners[learner_id]->predict(test_x,
                                                          prediction_test_data,
                                                          params.dbm_shrinkage);
                        }

                    }

                    break;

                }
                case 'k': {
                    #ifdef _OMP
                    #pragma omp parallel default(shared)
                    {
                        int thread_id = omp_get_thread_num();
                        int learner_id = no_cores * (i - 1) + thread_id + 1;
                    #else
                    {
                        int thread_id = 0, learner_id = i;
                    #endif
                        if (params.dbm_display_training_progress) {
                            #pragma omp critical
                            if(params.dbm_nonoverlapping_training) {
                                std::printf("Learner (%c) No. %d -> "
                                                    "Kmeans2d at %p "
                                                    "number of samples: %d "
                                                    "number of predictors: %d "
                                                    "number of centroids: %d ...\n",
                                            type,
                                            learner_id,
                                            learners[learner_id],
                                            no_samples_in_nonoverlapping_batch,
                                            no_candidate_feature,
                                            params.kmeans_no_centroids);
                            }
                            else {
                                std::printf("Learner (%c) No. %d -> "
                                                    "Kmeans2d at %p "
                                                    "number of samples: %d "
                                                    "number of predictors: %d "
                                                    "number of centroids: %d ...\n",
                                            type,
                                            learner_id,
                                            learners[learner_id],
                                            no_train_sample,
                                            no_candidate_feature,
                                            params.kmeans_no_centroids);
                            }
                        }
                        else {
                            printf(".");
                        }

                        if(params.dbm_nonoverlapping_training) {
                            int *thread_col_inds = new int[n_features];
                            std::copy(col_inds,
                                      col_inds + n_features,
                                      thread_col_inds);

                            #ifdef _OMP
                            #pragma omp critical
                            #endif
                            shuffle(thread_col_inds,
                                    n_features,
                                    seeds[learner_id - 1]);

                            kmeans2d_trainer->train(dynamic_cast<Kmeans2d<T> *> (learners[learner_id]),
                                                    train_x,
                                                    ind_delta,
                                                    input_monotonic_constraints,
                                                    thread_row_inds_vec[thread_id],
                                                    no_samples_in_nonoverlapping_batch,
                                                    thread_col_inds,
                                                    no_candidate_feature);

                            delete[] thread_col_inds;

                        }
                        else {
                            int *thread_row_inds = new int[n_samples];
                            int *thread_col_inds = new int[n_features];
                            std::copy(row_inds,
                                      row_inds + n_samples,
                                      thread_row_inds);
                            std::copy(col_inds,
                                      col_inds + n_features,
                                      thread_col_inds);

                            #ifdef _OMP
                            #pragma omp critical
                            #endif
                            {
                                shuffle(thread_row_inds,
                                        n_samples,
                                        seeds[learner_id - 1]);
                                shuffle(thread_col_inds,
                                        n_features,
                                        seeds[learner_id - 1]);
                            }

                            kmeans2d_trainer->train(dynamic_cast<Kmeans2d<T> *> (learners[learner_id]),
                                                    train_x,
                                                    ind_delta,
                                                    input_monotonic_constraints,
                                                    thread_row_inds,
                                                    no_train_sample,
                                                    thread_col_inds,
                                                    no_candidate_feature);

                            delete[] thread_row_inds;
                            delete[] thread_col_inds;

                        }

                        #ifdef _OMP
                        #pragma omp barrier
                        #endif
                        {
                            #ifdef _OMP
                            #pragma omp critical
                            #endif
                            learners[learner_id]->predict(train_x,
                                                          *prediction_train_data,
                                                          params.dbm_shrinkage);
                            #ifdef _OMP
                            #pragma omp critical
                            #endif
                            learners[learner_id]->predict(test_x,
                                                          prediction_test_data,
                                                          params.dbm_shrinkage);
                        }
                    }

                    break;

                }
                case 's': {
                    #ifdef _OMP
                    #pragma omp parallel default(shared)
                    {
                        int thread_id = omp_get_thread_num();
                        int learner_id = no_cores * (i - 1) + thread_id + 1;
                    #else
                    {
                        int thread_id = 0, learner_id = i;
                    #endif
                        if (params.dbm_display_training_progress) {
                            #pragma omp critical
                            if(params.dbm_nonoverlapping_training) {
                                std::printf("Learner (%c) No. %d -> "
                                                    "Training Splines at %p "
                                                    "number of samples: %d "
                                                    "number of predictors: %d "
                                                    "number of knots: %d ...\n",
                                            type,
                                            learner_id,
                                            learners[learner_id],
                                            no_samples_in_nonoverlapping_batch,
                                            no_candidate_feature,
                                            params.splines_no_knot);
                            }
                            else {
                                std::printf("Learner (%c) No. %d -> "
                                                    "Training Splines at %p "
                                                    "number of samples: %d "
                                                    "number of predictors: %d "
                                                    "number of knots: %d ...\n",
                                            type,
                                            learner_id,
                                            learners[learner_id],
                                            no_train_sample,
                                            no_candidate_feature,
                                            params.splines_no_knot);
                            }
                        }
                        else {
                            printf(".");
                        }

                        if(params.dbm_nonoverlapping_training) {
                            int *thread_col_inds = new int[n_features];
                            std::copy(col_inds,
                                      col_inds + n_features,
                                      thread_col_inds);

                            #ifdef _OMP
                            #pragma omp critical
                            #endif
                            shuffle(thread_col_inds,
                                    n_features,
                                    seeds[learner_id - 1]);

                            splines_trainer->train(dynamic_cast<Splines<T> *>
                                                   (learners[learner_id]),
                                                   train_x,
                                                   ind_delta,
                                                   input_monotonic_constraints,
                                                   thread_row_inds_vec[thread_id],
                                                   no_samples_in_nonoverlapping_batch,
                                                   thread_col_inds,
                                                   no_candidate_feature);

                            delete[] thread_col_inds;

                        }
                        else {
                            int *thread_row_inds = new int[n_samples];
                            int *thread_col_inds = new int[n_features];
                            std::copy(row_inds,
                                      row_inds + n_samples,
                                      thread_row_inds);
                            std::copy(col_inds,
                                      col_inds + n_features,
                                      thread_col_inds);

                            #ifdef _OMP
                            #pragma omp critical
                            #endif
                            {
                                shuffle(thread_row_inds,
                                        n_samples,
                                        seeds[learner_id - 1]);
                                shuffle(thread_col_inds,
                                        n_features,
                                        seeds[learner_id - 1]);
                            }

                            splines_trainer->train(dynamic_cast<Splines<T> *>
                                                   (learners[learner_id]),
                                                   train_x,
                                                   ind_delta,
                                                   input_monotonic_constraints,
                                                   thread_row_inds,
                                                   no_train_sample,
                                                   thread_col_inds,
                                                   no_candidate_feature);

                            delete[] thread_row_inds;
                            delete[] thread_col_inds;

                        }

                        #ifdef _OMP
                        #pragma omp barrier
                        #endif
                        {
                            #ifdef _OMP
                            #pragma omp critical
                            #endif
                            learners[learner_id]->predict(train_x,
                                                          *prediction_train_data,
                                                          params.dbm_shrinkage);
                            #ifdef _OMP
                            #pragma omp critical
                            #endif
                            learners[learner_id]->predict(test_x,
                                                          prediction_test_data,
                                                          params.dbm_shrinkage);
                        }
                    }

                    break;

                }
                case 'n': {
                    #ifdef _OMP
                    #pragma omp parallel default(shared)
                    {
                        int thread_id = omp_get_thread_num();
                        int learner_id = no_cores * (i - 1) + thread_id + 1;
                    #else
                    {
                        int thread_id = 0, learner_id = i;
                    #endif
                        if (params.dbm_display_training_progress) {
                            #pragma omp critical
                            if(params.dbm_nonoverlapping_training) {
                                std::printf("Learner (%c) No. %d -> "
                                                    "Training Neural Network at %p "
                                                    "number of samples: %d "
                                                    "number of predictors: %d "
                                                    "number of hidden neurons: %d ...\n",
                                            type,
                                            learner_id,
                                            learners[learner_id],
                                            no_samples_in_nonoverlapping_batch,
                                            no_candidate_feature,
                                            params.nn_no_hidden_neurons);
                            }
                            else {
                                std::printf("Learner (%c) No. %d -> "
                                                    "Training Neural Network at %p "
                                                    "number of samples: %d "
                                                    "number of predictors: %d "
                                                    "number of hidden neurons: %d ...\n",
                                            type,
                                            learner_id,
                                            learners[learner_id],
                                            no_train_sample,
                                            no_candidate_feature,
                                            params.nn_no_hidden_neurons);
                            }
                        }
                        else {
                            printf(".");
                        }

                        if(params.dbm_nonoverlapping_training) {
                            int *thread_col_inds = new int[n_features];
                            std::copy(col_inds,
                                      col_inds + n_features,
                                      thread_col_inds);

                            #ifdef _OMP
                            #pragma omp critical
                            #endif
                            shuffle(thread_col_inds,
                                    n_features,
                                    seeds[learner_id - 1]);

                            neural_network_trainer->train(dynamic_cast<Neural_network<T> *>
                                                          (learners[learner_id]),
                                                          train_x,
                                                          ind_delta,
                                                          thread_row_inds_vec[thread_id],
                                                          no_samples_in_nonoverlapping_batch,
                                                          thread_col_inds,
                                                          no_candidate_feature);

                            delete[] thread_col_inds;

                        }
                        else {
                            int *thread_row_inds = new int[n_samples];
                            int *thread_col_inds = new int[n_features];
                            std::copy(row_inds,
                                      row_inds + n_samples,
                                      thread_row_inds);
                            std::copy(col_inds,
                                      col_inds + n_features,
                                      thread_col_inds);

                            #ifdef _OMP
                            #pragma omp critical
                            #endif
                            {
                                shuffle(thread_row_inds,
                                        n_samples,
                                        seeds[learner_id - 1]);
                                shuffle(thread_col_inds,
                                        n_features,
                                        seeds[learner_id - 1]);
                            }

                            neural_network_trainer->train(dynamic_cast<Neural_network<T> *>
                                                          (learners[learner_id]),
                                                          train_x,
                                                          ind_delta,
                                                          thread_row_inds,
                                                          no_train_sample,
                                                          thread_col_inds,
                                                          no_candidate_feature);

                            delete[] thread_row_inds;
                            delete[] thread_col_inds;

                        }

                        #ifdef _OMP
                        #pragma omp barrier
                        #endif
                        {
                            #ifdef _OMP
                            #pragma omp critical
                            #endif
                            learners[learner_id]->predict(train_x,
                                                          *prediction_train_data,
                                                          params.dbm_shrinkage);
                            #ifdef _OMP
                            #pragma omp critical
                            #endif
                            learners[learner_id]->predict(test_x,
                                                          prediction_test_data,
                                                          params.dbm_shrinkage);
                        }
                    }

                    break;

                }
                case 'd': {
                    #ifdef _OMP
                    #pragma omp parallel default(shared)
                    {
                        int thread_id = omp_get_thread_num();
                        int learner_id = no_cores * (i - 1) + thread_id + 1;
                    #else
                    {
                        int thread_id = 0, learner_id = i;
                    #endif
                        if (params.dbm_display_training_progress) {
                            #pragma omp critical
                            if(params.dbm_nonoverlapping_training) {
                                std::printf("Learner (%c) No. %d -> "
                                                    "Training DPC Stairs at %p "
                                                    "number of samples: %d "
                                                    "number of predictors: %d "
                                                    "number of ticks: %d ...\n",
                                            type,
                                            learner_id,
                                            learners[learner_id],
                                            no_samples_in_nonoverlapping_batch,
                                            no_candidate_feature,
                                            params.dpcs_no_ticks);
                            }
                            else {
                                std::printf("Learner (%c) No. %d -> "
                                                    "Training DPC Stairs at %p "
                                                    "number of samples: %d "
                                                    "number of predictors: %d "
                                                    "number of ticks: %d ...\n",
                                            type,
                                            learner_id,
                                            learners[learner_id],
                                            no_train_sample,
                                            no_candidate_feature,
                                            params.dpcs_no_ticks);
                            }
                        }
                        else {
                            printf(".");
                        }

                        if(params.dbm_nonoverlapping_training) {
                            int *thread_col_inds = new int[n_features];
                            std::copy(col_inds,
                                      col_inds + n_features,
                                      thread_col_inds);

                            #ifdef _OMP
                            #pragma omp critical
                            #endif
                            shuffle(thread_col_inds,
                                    n_features,
                                    seeds[learner_id - 1]);

                            dpc_stairs_trainer->train(dynamic_cast<DPC_stairs<T> *>
                                                      (learners[learner_id]),
                                                      train_x,
                                                      ind_delta,
                                                      input_monotonic_constraints,
                                                      thread_row_inds_vec[thread_id],
                                                      no_samples_in_nonoverlapping_batch,
                                                      thread_col_inds,
                                                      no_candidate_feature);

                            delete[] thread_col_inds;

                        }
                        else {
                            int *thread_row_inds = new int[n_samples];
                            int *thread_col_inds = new int[n_features];
                            std::copy(row_inds,
                                      row_inds + n_samples,
                                      thread_row_inds);
                            std::copy(col_inds,
                                      col_inds + n_features,
                                      thread_col_inds);

                            #ifdef _OMP
                            #pragma omp critical
                            #endif
                            {
                                shuffle(thread_row_inds,
                                        n_samples,
                                        seeds[learner_id - 1]);
                                shuffle(thread_col_inds,
                                        n_features,
                                        seeds[learner_id - 1]);
                            }

                            dpc_stairs_trainer->train(dynamic_cast<DPC_stairs<T> *>
                                                      (learners[learner_id]),
                                                      train_x,
                                                      ind_delta,
                                                      input_monotonic_constraints,
                                                      thread_row_inds,
                                                      no_train_sample,
                                                      thread_col_inds,
                                                      no_candidate_feature);

                            delete[] thread_row_inds;
                            delete[] thread_col_inds;

                        }

                        #ifdef _OMP
                        #pragma omp barrier
                        #endif
                        {
                            #ifdef _OMP
                            #pragma omp critical
                            #endif
                            learners[learner_id]->predict(train_x,
                                                          *prediction_train_data,
                                                          params.dbm_shrinkage);
                            #ifdef _OMP
                            #pragma omp critical
                            #endif
                            learners[learner_id]->predict(test_x,
                                                          prediction_test_data,
                                                          params.dbm_shrinkage);
                        }
                    }

                    break;

                }
                default: {
                    std::cout << "Wrong learner type: " << type << std::endl;
                    throw std::invalid_argument("Specified learner does not exist.");
                }
            }

            if (!(i % params.dbm_freq_showing_loss_on_test)) {
                test_loss_record[i / params.dbm_freq_showing_loss_on_test] =
                        loss_function.loss(test_y,
                                           prediction_test_data,
                                           params.dbm_loss_function);
                if (params.dbm_do_perf) {
                    train_loss_record[i / params.dbm_freq_showing_loss_on_test] =
                            loss_function.loss(train_y, *prediction_train_data, params.dbm_loss_function);
                } //dbm_do_perf, record loss on train;

                if(test_loss_record[i / params.dbm_freq_showing_loss_on_test] < lowest_test_loss)
                    lowest_test_loss = test_loss_record[i / params.dbm_freq_showing_loss_on_test];

                if (params.dbm_display_training_progress) {
                    std::cout << std::endl
                              << '(' << i / params.dbm_freq_showing_loss_on_test << ')'
                              << " \tLowest loss on test set: "
                              << lowest_test_loss
                              << std::endl
                              << " \t\tLoss on test set: "
                              << test_loss_record[i / params.dbm_freq_showing_loss_on_test]
                              << std::endl
                              << " \t\tLoss on train set: "
                              << train_loss_record[i / params.dbm_freq_showing_loss_on_test]
                              << std::endl << std::endl;
                }

            }

        }

        loss_function.link_function(*prediction_train_data, params.dbm_loss_function);

        std::cout << std::endl << "Losses on Test Set: " << std::endl;
        for (int i = 0; i < no_bunches_of_learners / params.dbm_freq_showing_loss_on_test + 1; ++i)
            std::cout << "(" << i << ") " << test_loss_record[i] << ' ';
        std::cout << std::endl;

        if (params.dbm_do_perf) {
            std::cout << std::endl << "Losses on Test Set: " << std::endl;
            for (int i = 0; i < no_bunches_of_learners / params.dbm_freq_showing_loss_on_test + 1; ++i)
                std::cout << "(" << i << ") " << train_loss_record[i] << ' ';
            std::cout << std::endl;
        }

        delete[] row_inds;
        delete[] col_inds;
        delete[] seeds;
        delete[] whole_row_inds;
        delete[] thread_row_inds_vec;

    }

    template<typename T>
    void DBM<T>::train(const Data_set<T> &data_set) {

        std::cout << "dbm_no_bunches_of_learners: " << params.dbm_no_bunches_of_learners << std::endl
                  << "dbm_no_cores: " << params.dbm_no_cores << std::endl
                  << "dbm_portion_train_sample: " << params.dbm_portion_train_sample << std::endl
                  << "dbm_no_candidate_feature: " << params.dbm_no_candidate_feature << std::endl
                  << "dbm_shrinkage: " << params.dbm_shrinkage << std::endl
                  << "random_seed in Parameters: " << params.dbm_random_seed << std::endl
                  << "random_seed in Data_set: " << data_set.random_seed << std::endl;

        Time_measurer timer(no_cores);

        Matrix<T> const &train_x = data_set.get_train_x();
        Matrix<T> const &train_y = data_set.get_train_y();
        Matrix<T> const &test_x = data_set.get_test_x();
        Matrix<T> const &test_y = data_set.get_test_y();

        Matrix<T> sorted_train_x = copy(train_x);
        const Matrix<T> sorted_train_x_from = col_sort(sorted_train_x);
//        const Matrix<T> train_x_sorted_to = col_sorted_to(sorted_train_x_from);

        int n_samples = train_x.get_height(), n_features = train_x.get_width();
        int n_test_samples = test_x.get_height();

        no_train_sample = (int)(params.dbm_portion_train_sample * n_samples);
        total_no_feature = n_features;

        int no_samples_in_nonoverlapping_batch = no_train_sample / no_cores - 1;
        int *whole_row_inds = new int[n_samples];
        int **thread_row_inds_vec = new int*[no_cores];
        for(int i = 0; i < no_cores; ++i) {
            thread_row_inds_vec[i] = whole_row_inds + i * no_samples_in_nonoverlapping_batch;
        }

        #ifdef _DEBUG_MODEL
            assert(no_train_sample <= n_samples && no_candidate_feature <= total_no_feature);
        #endif

        Matrix<T> input_monotonic_constraints(n_features, 1, 0);

//#ifdef _DEBUG_MODEL
//        assert(no_train_sample <= n_samples && no_candidate_feature <= total_no_feature);
//
//        for (int i = 0; i < n_features; ++i) {
//            // serves as a check of whether the length of monotonic_constraints
//            // is equal to the length of features in some sense
//
//            assert(input_monotonic_constraints.get(i, 0) == 0 ||
//                   input_monotonic_constraints.get(i, 0) == -1 ||
//                   input_monotonic_constraints.get(i, 0) == 1);
//        }
//#endif

        int *row_inds = new int[n_samples], *col_inds = new int[n_features];

        for (int i = 0; i < n_features; ++i)
            col_inds[i] = i;
        for (int i = 0; i < n_samples; ++i)
            row_inds[i] = i;

        if (prediction_train_data != nullptr)
            delete prediction_train_data;

        if (test_loss_record != nullptr)
            delete[] test_loss_record;
        prediction_train_data = new Matrix<T>(n_samples, 1, 0);
        test_loss_record = new T[no_bunches_of_learners / params.dbm_freq_showing_loss_on_test + 1];
        if (params.dbm_do_perf) {
            if (train_loss_record != nullptr)
                delete[] train_loss_record;
            train_loss_record = new T[no_bunches_of_learners / params.dbm_freq_showing_loss_on_test + 1];
        }

        T lowest_test_loss = std::numeric_limits<T>::max();

        /*
         * ind_delta:
         * col 0: individual delta
         * col 1: denominator of individual delta
         */
        Matrix<T> ind_delta(n_samples, 2, 0);

        Matrix<T> prediction_test_data(n_test_samples, 1, 0);

        #ifdef _OMP
            omp_set_num_threads(no_cores);
        #endif

        unsigned int *seeds = new unsigned int[(no_bunches_of_learners - 1) * no_cores];

        if(params.dbm_random_seed < 0) {
            std::random_device rd;
            std::mt19937 mt(rd());
            std::uniform_real_distribution<T> dist(0, 1000);
            for(int i = 0; i < (no_bunches_of_learners - 1) * no_cores; ++i)
                seeds[i] = (unsigned int) dist(mt);
        }
        else {
            std::mt19937 mt(params.dbm_random_seed);
            std::uniform_real_distribution<T> dist(0, 1000);
            for(int i = 0; i < (no_bunches_of_learners - 1) * no_cores; ++i)
                seeds[i] = (unsigned int) dist(mt);
        }

        std::cout << "Learner " << "(" << learners[0]->get_type() << ") "
                  << " No. " << 0 << " -> ";

        loss_function.calculate_ind_delta(train_y,
                                          *prediction_train_data,
                                          ind_delta,
                                          params.dbm_loss_function);
        mean_trainer->train(dynamic_cast<Global_mean<T> *>(learners[0]),
                            train_x,
                            ind_delta,
                            *prediction_train_data,
                            params.dbm_loss_function);
        learners[0]->predict(train_x,
                             *prediction_train_data);
        learners[0]->predict(test_x,
                             prediction_test_data);

        test_loss_record[0] = loss_function.loss(test_y,
                                                 prediction_test_data,
                                                 params.dbm_loss_function);
        train_loss_record[0] = loss_function.loss(train_y,
                                                  *prediction_train_data,
                                                  params.dbm_loss_function);

        if(test_loss_record[0] < lowest_test_loss)
            lowest_test_loss = test_loss_record[0];

        if (params.dbm_display_training_progress) {
            std::cout << std::endl
                      << '(' << 0 << ')'
                      << " \tLowest loss on test set: "
                      << lowest_test_loss
                      << std::endl
                      << " \t\tLoss on test set: "
                      << test_loss_record[0]
                      << std::endl
                      << " \t\tLoss on train set: "
                      << train_loss_record[0]
                      << std::endl << std::endl;
        }

        if (params.dbm_do_perf) {
            train_loss_record[0] = loss_function.loss(train_y,
                                                      *prediction_train_data,
                                                      params.dbm_loss_function);
        }

        char type;
        for (int i = 1; i < no_bunches_of_learners; ++i) {

            if((!params.dbm_display_training_progress) && i % 10 == 0)
                printf("\n");

            loss_function.calculate_ind_delta(train_y,
                                              *prediction_train_data,
                                              ind_delta,
                                              params.dbm_loss_function);

            if(params.dbm_nonoverlapping_training) {
                std::copy(row_inds, row_inds + n_samples, whole_row_inds);
                shuffle(whole_row_inds,
                        n_samples,
                        seeds[no_cores * (i - 1) + 1]);
            }

            type = learners[(i - 1) * no_cores + 1]->get_type();
            switch (type) {
                case 't': {
                    #ifdef _OMP
                    #pragma omp parallel default(shared)
                    {
                        int thread_id = omp_get_thread_num();
                        int learner_id = no_cores * (i - 1) + thread_id + 1;
                    #else
                    {
                        int thread_id = 0, learner_id = i;
                    #endif
                        if (params.dbm_display_training_progress) {
                            #pragma omp critical
                            if(params.dbm_nonoverlapping_training)
                                std::printf("Learner (%c) No. %d -> "
                                                    "Training Tree at %p "
                                                    "number of samples: %d "
                                                    "max_depth: %d ...\n",
                                            type,
                                            learner_id,
                                            learners[learner_id],
                                            no_samples_in_nonoverlapping_batch,
                                            params.cart_max_depth);
                            else
                                std::printf("Learner (%c) No. %d -> "
                                                    "Training Tree at %p "
                                                    "number of samples: %d "
                                                    "max_depth: %d ...\n",
                                            type,
                                            learner_id,
                                            learners[learner_id],
                                            no_train_sample,
                                            params.cart_max_depth);
                        }
                        else {
                            printf(".");
                        }

                        if(params.dbm_nonoverlapping_training) {
                            int *thread_col_inds = new int[n_features];
                            std::copy(col_inds,
                                      col_inds + n_features,
                                      thread_col_inds);

                            #ifdef _OMP
                            #pragma omp critical
                            #endif
                            shuffle(thread_col_inds,
                                    n_features,
                                    seeds[learner_id - 1]);

                            Matrix<T> first_comp_in_loss = loss_function.first_comp(train_y,
                                                                                    *prediction_train_data,
                                                                                    params.dbm_loss_function);
                            Matrix<T> second_comp_in_loss = loss_function.second_comp(train_y,
                                                                                      *prediction_train_data,
                                                                                      params.dbm_loss_function);

                            tree_trainer->train(dynamic_cast<Tree_node<T> *>(learners[learner_id]),
                                                train_x,
                                                sorted_train_x_from,
                                                train_y,
                                                ind_delta,
                                                *prediction_train_data,
                                                first_comp_in_loss,
                                                second_comp_in_loss,
                                                input_monotonic_constraints,
                                                params.dbm_loss_function,
                                                thread_row_inds_vec[thread_id],
                                                no_samples_in_nonoverlapping_batch,
                                                thread_col_inds,
                                                no_candidate_feature);

                            delete[] thread_col_inds;

                        }
                        else {
                            int *thread_row_inds = new int[n_samples];
                            int *thread_col_inds = new int[n_features];
                            std::copy(row_inds,
                                      row_inds + n_samples,
                                      thread_row_inds);
                            std::copy(col_inds,
                                      col_inds + n_features,
                                      thread_col_inds);

                            #ifdef _OMP
                            #pragma omp critical
                            #endif
                            {
                                shuffle(thread_row_inds,
                                        n_samples,
                                        seeds[learner_id - 1]);
                                shuffle(thread_col_inds,
                                        n_features,
                                        seeds[learner_id - 1]);
                            }

                            Matrix<T> first_comp_in_loss = loss_function.first_comp(train_y,
                                                                                    *prediction_train_data,
                                                                                    params.dbm_loss_function);
                            Matrix<T> second_comp_in_loss = loss_function.second_comp(train_y,
                                                                                      *prediction_train_data,
                                                                                      params.dbm_loss_function);

                            tree_trainer->train(dynamic_cast<Tree_node<T> *>(learners[learner_id]),
                                                train_x,
                                                sorted_train_x_from,
                                                train_y,
                                                ind_delta,
                                                *prediction_train_data,
                                                first_comp_in_loss,
                                                second_comp_in_loss,
                                                input_monotonic_constraints,
                                                params.dbm_loss_function,
                                                thread_row_inds,
                                                no_train_sample,
                                                thread_col_inds,
                                                no_candidate_feature);

                            delete[] thread_row_inds;
                            delete[] thread_col_inds;

                        }

                        tree_trainer->update_loss_reduction(dynamic_cast<Tree_node<T> *>(learners[learner_id]));

                        if(params.cart_prune) {
                            tree_trainer->prune(dynamic_cast<Tree_node<T> *>(learners[learner_id]));
                        }

                        #ifdef _OMP
                        #pragma omp barrier
                        #endif
                        {
                            #ifdef _OMP
                            #pragma omp critical
                            #endif
                            learners[learner_id]->predict(train_x,
                                                          *prediction_train_data,
                                                          params.dbm_shrinkage);
                            #ifdef _OMP
                            #pragma omp critical
                            #endif
                            learners[learner_id]->predict(test_x,
                                                          prediction_test_data,
                                                          params.dbm_shrinkage);

                            if(params.dbm_record_every_tree) {
                                #ifdef _OMP
                                #pragma omp critical
                                #endif
                                {
                                    Tree_info<T> tree_info(dynamic_cast<Tree_node<T> *>
                                                           (learners[learner_id]));
                                    tree_info.print_to_file("trees.txt",
                                                            learner_id);
                                }
                            }
                        }

                    }

                    break;

                }
                case 'l': {
                    #ifdef _OMP
                    #pragma omp parallel default(shared)
                    {
                        int thread_id = omp_get_thread_num();
                        int learner_id = no_cores * (i - 1) + thread_id + 1;
                    #else
                    {
                        int thread_id = 0, learner_id = i;
                    #endif
                        if (params.dbm_display_training_progress) {
                            #pragma omp critical
                            if(params.dbm_nonoverlapping_training) {
                                std::printf("Learner (%c) No. %d -> "
                                                    "Training Linear Regression at %p "
                                                    "number of samples: %d "
                                                    "number of predictors: %d ...\n",
                                            type,
                                            learner_id,
                                            learners[learner_id],
                                            no_samples_in_nonoverlapping_batch,
                                            no_candidate_feature);
                            }
                            else {
                                std::printf("Learner (%c) No. %d -> "
                                                    "Training Linear Regression at %p "
                                                    "number of samples: %d "
                                                    "number of predictors: %d ...\n",
                                            type,
                                            learner_id,
                                            learners[learner_id],
                                            no_train_sample,
                                            no_candidate_feature);
                            }
                        }
                        else {
                            printf(".");
                        }

                        if(params.dbm_nonoverlapping_training) {
                            int *thread_col_inds = new int[n_features];
                            std::copy(col_inds,
                                      col_inds + n_features,
                                      thread_col_inds);

                            #ifdef _OMP
                            #pragma omp critical
                            #endif
                            shuffle(thread_col_inds,
                                    n_features,
                                    seeds[learner_id - 1]);

                            linear_regression_trainer->train(dynamic_cast<Linear_regression<T> *>
                                                             (learners[learner_id]),
                                                             train_x,
                                                             ind_delta,
                                                             input_monotonic_constraints,
                                                             thread_row_inds_vec[thread_id],
                                                             no_samples_in_nonoverlapping_batch,
                                                             thread_col_inds,
                                                             no_candidate_feature);

                            delete[] thread_col_inds;

                        }
                        else {
                            int *thread_row_inds = new int[n_samples];
                            int *thread_col_inds = new int[n_features];
                            std::copy(row_inds,
                                      row_inds + n_samples,
                                      thread_row_inds);
                            std::copy(col_inds,
                                      col_inds + n_features,
                                      thread_col_inds);

                            #ifdef _OMP
                            #pragma omp critical
                            #endif
                            {
                                shuffle(thread_row_inds,
                                        n_samples,
                                        seeds[learner_id - 1]);
                                shuffle(thread_col_inds,
                                        n_features,
                                        seeds[learner_id - 1]);
                            }

                            linear_regression_trainer->train(dynamic_cast<Linear_regression<T> *>
                                                             (learners[learner_id]),
                                                             train_x,
                                                             ind_delta,
                                                             input_monotonic_constraints,
                                                             thread_row_inds,
                                                             no_train_sample,
                                                             thread_col_inds,
                                                             no_candidate_feature);

                            delete[] thread_row_inds;
                            delete[] thread_col_inds;

                        }

                        #ifdef _OMP
                        #pragma omp barrier
                        #endif
                        {
                            #ifdef _OMP
                            #pragma omp critical
                            #endif
                            learners[learner_id]->predict(train_x,
                                                          *prediction_train_data,
                                                          params.dbm_shrinkage);
                            #ifdef _OMP
                            #pragma omp critical
                            #endif
                            learners[learner_id]->predict(test_x,
                                                          prediction_test_data,
                                                          params.dbm_shrinkage);
                        }

                    }

                    break;

                }
                case 'k': {
                    #ifdef _OMP
                    #pragma omp parallel default(shared)
                    {
                        int thread_id = omp_get_thread_num();
                        int learner_id = no_cores * (i - 1) + thread_id + 1;
                    #else
                    {
                        int thread_id = 0, learner_id = i;
                    #endif
                        if (params.dbm_display_training_progress) {
                            #pragma omp critical
                            if(params.dbm_nonoverlapping_training) {
                                std::printf("Learner (%c) No. %d -> "
                                                    "Kmeans2d at %p "
                                                    "number of samples: %d "
                                                    "number of predictors: %d "
                                                    "number of centroids: %d ...\n",
                                            type,
                                            learner_id,
                                            learners[learner_id],
                                            no_samples_in_nonoverlapping_batch,
                                            no_candidate_feature,
                                            params.kmeans_no_centroids);
                            }
                            else {
                                std::printf("Learner (%c) No. %d -> "
                                                    "Kmeans2d at %p "
                                                    "number of samples: %d "
                                                    "number of predictors: %d "
                                                    "number of centroids: %d ...\n",
                                            type,
                                            learner_id,
                                            learners[learner_id],
                                            no_train_sample,
                                            no_candidate_feature,
                                            params.kmeans_no_centroids);
                            }
                        }
                        else {
                            printf(".");
                        }

                        if(params.dbm_nonoverlapping_training) {
                            int *thread_col_inds = new int[n_features];
                            std::copy(col_inds,
                                      col_inds + n_features,
                                      thread_col_inds);

                            #ifdef _OMP
                            #pragma omp critical
                            #endif
                            shuffle(thread_col_inds,
                                    n_features,
                                    seeds[learner_id - 1]);

                            kmeans2d_trainer->train(dynamic_cast<Kmeans2d<T> *> (learners[learner_id]),
                                                    train_x,
                                                    ind_delta,
                                                    input_monotonic_constraints,
                                                    thread_row_inds_vec[thread_id],
                                                    no_samples_in_nonoverlapping_batch,
                                                    thread_col_inds,
                                                    no_candidate_feature);

                            delete[] thread_col_inds;

                        }
                        else {
                            int *thread_row_inds = new int[n_samples];
                            int *thread_col_inds = new int[n_features];
                            std::copy(row_inds,
                                      row_inds + n_samples,
                                      thread_row_inds);
                            std::copy(col_inds,
                                      col_inds + n_features,
                                      thread_col_inds);

                            #ifdef _OMP
                            #pragma omp critical
                            #endif
                            {
                                shuffle(thread_row_inds,
                                        n_samples,
                                        seeds[learner_id - 1]);
                                shuffle(thread_col_inds,
                                        n_features,
                                        seeds[learner_id - 1]);
                            }

                            kmeans2d_trainer->train(dynamic_cast<Kmeans2d<T> *> (learners[learner_id]),
                                                    train_x,
                                                    ind_delta,
                                                    input_monotonic_constraints,
                                                    thread_row_inds,
                                                    no_train_sample,
                                                    thread_col_inds,
                                                    no_candidate_feature);

                            delete[] thread_row_inds;
                            delete[] thread_col_inds;

                        }

                        #ifdef _OMP
                        #pragma omp barrier
                        #endif
                        {
                            #ifdef _OMP
                            #pragma omp critical
                            #endif
                            learners[learner_id]->predict(train_x,
                                                          *prediction_train_data,
                                                          params.dbm_shrinkage);
                            #ifdef _OMP
                            #pragma omp critical
                            #endif
                            learners[learner_id]->predict(test_x,
                                                          prediction_test_data,
                                                          params.dbm_shrinkage);
                        }
                    }

                    break;

                }
                case 's': {
                    #ifdef _OMP
                    #pragma omp parallel default(shared)
                    {
                        int thread_id = omp_get_thread_num();
                        int learner_id = no_cores * (i - 1) + thread_id + 1;
                    #else
                    {
                        int thread_id = 0, learner_id = i;
                    #endif
                        if (params.dbm_display_training_progress) {
                            #pragma omp critical
                            if(params.dbm_nonoverlapping_training) {
                                std::printf("Learner (%c) No. %d -> "
                                                    "Training Splines at %p "
                                                    "number of samples: %d "
                                                    "number of predictors: %d "
                                                    "number of knots: %d ...\n",
                                            type,
                                            learner_id,
                                            learners[learner_id],
                                            no_samples_in_nonoverlapping_batch,
                                            no_candidate_feature,
                                            params.splines_no_knot);
                            }
                            else {
                                std::printf("Learner (%c) No. %d -> "
                                                    "Training Splines at %p "
                                                    "number of samples: %d "
                                                    "number of predictors: %d "
                                                    "number of knots: %d ...\n",
                                            type,
                                            learner_id,
                                            learners[learner_id],
                                            no_train_sample,
                                            no_candidate_feature,
                                            params.splines_no_knot);
                            }
                        }
                        else {
                            printf(".");
                        }

                        if(params.dbm_nonoverlapping_training) {
                            int *thread_col_inds = new int[n_features];
                            std::copy(col_inds,
                                      col_inds + n_features,
                                      thread_col_inds);

                            #ifdef _OMP
                            #pragma omp critical
                            #endif
                            shuffle(thread_col_inds,
                                    n_features,
                                    seeds[learner_id - 1]);

                            splines_trainer->train(dynamic_cast<Splines<T> *>
                                                   (learners[learner_id]),
                                                   train_x,
                                                   ind_delta,
                                                   input_monotonic_constraints,
                                                   thread_row_inds_vec[thread_id],
                                                   no_samples_in_nonoverlapping_batch,
                                                   thread_col_inds,
                                                   no_candidate_feature);

                            delete[] thread_col_inds;

                        }
                        else {
                            int *thread_row_inds = new int[n_samples];
                            int *thread_col_inds = new int[n_features];
                            std::copy(row_inds,
                                      row_inds + n_samples,
                                      thread_row_inds);
                            std::copy(col_inds,
                                      col_inds + n_features,
                                      thread_col_inds);

                            #ifdef _OMP
                            #pragma omp critical
                            #endif
                            {
                                shuffle(thread_row_inds,
                                        n_samples,
                                        seeds[learner_id - 1]);
                                shuffle(thread_col_inds,
                                        n_features,
                                        seeds[learner_id - 1]);
                            }

                            splines_trainer->train(dynamic_cast<Splines<T> *>
                                                   (learners[learner_id]),
                                                   train_x,
                                                   ind_delta,
                                                   input_monotonic_constraints,
                                                   thread_row_inds,
                                                   no_train_sample,
                                                   thread_col_inds,
                                                   no_candidate_feature);

                            delete[] thread_row_inds;
                            delete[] thread_col_inds;

                        }

                        #ifdef _OMP
                        #pragma omp barrier
                        #endif
                        {
                            #ifdef _OMP
                            #pragma omp critical
                            #endif
                            learners[learner_id]->predict(train_x,
                                                          *prediction_train_data,
                                                          params.dbm_shrinkage);
                            #ifdef _OMP
                            #pragma omp critical
                            #endif
                            learners[learner_id]->predict(test_x,
                                                          prediction_test_data,
                                                          params.dbm_shrinkage);
                        }
                    }

                    break;

                }
                case 'n': {
                    #ifdef _OMP
                    #pragma omp parallel default(shared)
                    {
                        int thread_id = omp_get_thread_num();
                        int learner_id = no_cores * (i - 1) + thread_id + 1;
                    #else
                    {
                        int thread_id = 0, learner_id = i;
                    #endif
                        if (params.dbm_display_training_progress) {
                            #pragma omp critical
                            if(params.dbm_nonoverlapping_training) {
                                std::printf("Learner (%c) No. %d -> "
                                                    "Training Neural Network at %p "
                                                    "number of samples: %d "
                                                    "number of predictors: %d "
                                                    "number of hidden neurons: %d ...\n",
                                            type,
                                            learner_id,
                                            learners[learner_id],
                                            no_samples_in_nonoverlapping_batch,
                                            no_candidate_feature,
                                            params.nn_no_hidden_neurons);
                            }
                            else {
                                std::printf("Learner (%c) No. %d -> "
                                                    "Training Neural Network at %p "
                                                    "number of samples: %d "
                                                    "number of predictors: %d "
                                                    "number of hidden neurons: %d ...\n",
                                            type,
                                            learner_id,
                                            learners[learner_id],
                                            no_train_sample,
                                            no_candidate_feature,
                                            params.nn_no_hidden_neurons);
                            }
                        }
                        else {
                            printf(".");
                        }

                        if(params.dbm_nonoverlapping_training) {
                            int *thread_col_inds = new int[n_features];
                            std::copy(col_inds,
                                      col_inds + n_features,
                                      thread_col_inds);
                            #ifdef _OMP
                            #pragma omp critical
                            #endif
                            shuffle(thread_col_inds,
                                    n_features,
                                    seeds[learner_id - 1]);

                            neural_network_trainer->train(dynamic_cast<Neural_network<T> *>
                                                          (learners[learner_id]),
                                                          train_x,
                                                          ind_delta,
                                                          thread_row_inds_vec[thread_id],
                                                          no_samples_in_nonoverlapping_batch,
                                                          thread_col_inds,
                                                          no_candidate_feature);

                            delete[] thread_col_inds;

                        }
                        else {
                            int *thread_row_inds = new int[n_samples];
                            int *thread_col_inds = new int[n_features];
                            std::copy(row_inds,
                                      row_inds + n_samples,
                                      thread_row_inds);
                            std::copy(col_inds,
                                      col_inds + n_features,
                                      thread_col_inds);

                            #ifdef _OMP
                            #pragma omp critical
                            #endif
                            {
                                shuffle(thread_row_inds,
                                        n_samples,
                                        seeds[learner_id - 1]);
                                shuffle(thread_col_inds,
                                        n_features,
                                        seeds[learner_id - 1]);
                            }

                            neural_network_trainer->train(dynamic_cast<Neural_network<T> *>
                                                          (learners[learner_id]),
                                                          train_x,
                                                          ind_delta,
                                                          thread_row_inds,
                                                          no_train_sample,
                                                          thread_col_inds,
                                                          no_candidate_feature);

                            delete[] thread_row_inds;
                            delete[] thread_col_inds;

                        }

                        #ifdef _OMP
                        #pragma omp barrier
                        #endif
                        {
                            #ifdef _OMP
                            #pragma omp critical
                            #endif
                            learners[learner_id]->predict(train_x,
                                                          *prediction_train_data,
                                                          params.dbm_shrinkage);
                            #ifdef _OMP
                            #pragma omp critical
                            #endif
                            learners[learner_id]->predict(test_x,
                                                          prediction_test_data,
                                                          params.dbm_shrinkage);
                        }
                    }

                    break;

                }
                case 'd': {
                    #ifdef _OMP
                    #pragma omp parallel default(shared)
                    {
                        int thread_id = omp_get_thread_num();
                        int learner_id = no_cores * (i - 1) + thread_id + 1;
                    #else
                    {
                        int thread_id = 0, learner_id = i;
                    #endif
                        if (params.dbm_display_training_progress) {
                            #pragma omp critical
                            if(params.dbm_nonoverlapping_training) {
                                std::printf("Learner (%c) No. %d -> "
                                                    "Training DPC Stairs at %p "
                                                    "number of samples: %d "
                                                    "number of predictors: %d "
                                                    "number of ticks: %d ...\n",
                                            type,
                                            learner_id,
                                            learners[learner_id],
                                            no_samples_in_nonoverlapping_batch,
                                            no_candidate_feature,
                                            params.dpcs_no_ticks);
                            }
                            else {
                                std::printf("Learner (%c) No. %d -> "
                                                    "Training DPC Stairs at %p "
                                                    "number of samples: %d "
                                                    "number of predictors: %d "
                                                    "number of ticks: %d ...\n",
                                            type,
                                            learner_id,
                                            learners[learner_id],
                                            no_train_sample,
                                            no_candidate_feature,
                                            params.dpcs_no_ticks);
                            }
                        }
                        else {
                            printf(".");
                        }

                        if(params.dbm_nonoverlapping_training) {
                            int *thread_col_inds = new int[n_features];
                            std::copy(col_inds,
                                      col_inds + n_features,
                                      thread_col_inds);

                            #ifdef _OMP
                            #pragma omp critical
                            #endif
                            shuffle(thread_col_inds,
                                    n_features,
                                    seeds[learner_id - 1]);

                            dpc_stairs_trainer->train(dynamic_cast<DPC_stairs<T> *>
                                                      (learners[learner_id]),
                                                      train_x,
                                                      ind_delta,
                                                      input_monotonic_constraints,
                                                      thread_row_inds_vec[thread_id],
                                                      no_samples_in_nonoverlapping_batch,
                                                      thread_col_inds,
                                                      no_candidate_feature);

                            delete[] thread_col_inds;

                        }
                        else {
                            int *thread_row_inds = new int[n_samples];
                            int *thread_col_inds = new int[n_features];
                            std::copy(row_inds,
                                      row_inds + n_samples,
                                      thread_row_inds);
                            std::copy(col_inds,
                                      col_inds + n_features,
                                      thread_col_inds);

                            #ifdef _OMP
                            #pragma omp critical
                            #endif
                            {
                                shuffle(thread_row_inds,
                                        n_samples,
                                        seeds[learner_id - 1]);
                                shuffle(thread_col_inds,
                                        n_features,
                                        seeds[learner_id - 1]);
                            }

                            dpc_stairs_trainer->train(dynamic_cast<DPC_stairs<T> *>
                                                      (learners[learner_id]),
                                                      train_x,
                                                      ind_delta,
                                                      input_monotonic_constraints,
                                                      thread_row_inds,
                                                      no_train_sample,
                                                      thread_col_inds,
                                                      no_candidate_feature);

                            delete[] thread_row_inds;
                            delete[] thread_col_inds;

                        }

                        #ifdef _OMP
                        #pragma omp barrier
                        #endif
                        {
                            #ifdef _OMP
                            #pragma omp critical
                            #endif
                            learners[learner_id]->predict(train_x,
                                                          *prediction_train_data,
                                                          params.dbm_shrinkage);
                            #ifdef _OMP
                            #pragma omp critical
                            #endif
                            learners[learner_id]->predict(test_x,
                                                          prediction_test_data,
                                                          params.dbm_shrinkage);
                        }
                    }

                    break;

                }
                default: {
                    std::cout << "Wrong learner type: " << type << std::endl;
                    throw std::invalid_argument("Specified learner does not exist.");
                }
            }

            if (!(i % params.dbm_freq_showing_loss_on_test)) {
                test_loss_record[i / params.dbm_freq_showing_loss_on_test] =
                        loss_function.loss(test_y,
                                           prediction_test_data,
                                           params.dbm_loss_function);
                if (params.dbm_do_perf) {
                    train_loss_record[i / params.dbm_freq_showing_loss_on_test] =
                            loss_function.loss(train_y, *prediction_train_data, params.dbm_loss_function);
                } //dbm_do_perf, record loss on train;

                if(test_loss_record[i / params.dbm_freq_showing_loss_on_test] < lowest_test_loss)
                    lowest_test_loss = test_loss_record[i / params.dbm_freq_showing_loss_on_test];

                if (params.dbm_display_training_progress) {
                    std::cout << std::endl
                              << '(' << i / params.dbm_freq_showing_loss_on_test << ')'
                              << " \tLowest loss on test set: "
                              << lowest_test_loss
                              << std::endl
                              << " \t\tLoss on test set: "
                              << test_loss_record[i / params.dbm_freq_showing_loss_on_test]
                              << std::endl
                              << " \t\tLoss on train set: "
                              << train_loss_record[i / params.dbm_freq_showing_loss_on_test]
                              << std::endl << std::endl;
                }

            }

        }

        loss_function.link_function(*prediction_train_data, params.dbm_loss_function);

        std::cout << std::endl << "Losses on Test Set: " << std::endl;
        for (int i = 0; i < no_bunches_of_learners / params.dbm_freq_showing_loss_on_test + 1; ++i)
            std::cout << "(" << i << ") " << test_loss_record[i] << ' ';
        std::cout << std::endl;

        if (params.dbm_do_perf) {
            std::cout << std::endl << "Losses on Test Set: " << std::endl;
            for (int i = 0; i < no_bunches_of_learners / params.dbm_freq_showing_loss_on_test + 1; ++i)
                std::cout << "(" << i << ") " << train_loss_record[i] << ' ';
            std::cout << std::endl;
        }

        delete[] row_inds;
        delete[] col_inds;
        delete[] seeds;
        delete[] whole_row_inds;
        delete[] thread_row_inds_vec;

    }

    template<typename T>
    void DBM<T>::predict(const Matrix<T> &data_x, Matrix<T> &predict_y) {

        int data_height = data_x.get_height();

        #ifdef _DEBUG_MODEL
            assert(total_no_feature == data_x.get_width() &&
                           data_height == predict_y.get_height() &&
                           predict_y.get_width() == 1);
        #endif

        for (int i = 0; i < data_height; ++i)
            predict_y[i][0] = 0;

        if (learners[0]->get_type() == 'm') {
            learners[0]->predict(data_x, predict_y);
        }
        else {
            learners[0]->predict(data_x, predict_y, params.dbm_shrinkage);
        }

        for (int i = 1; i < (no_bunches_of_learners - 1) * no_cores + 1; ++i) {

            learners[i]->predict(data_x, predict_y, params.dbm_shrinkage);

        }

        loss_function.link_function(predict_y, params.dbm_loss_function);

    }

    template <typename T>
    Matrix<T> &DBM<T>::predict(const Matrix<T> &data_x) {
        if(prediction != nullptr) {
            delete prediction;
        }
        prediction = new Matrix<T>(data_x.get_height(), 1, 0);
        predict(data_x, *prediction);
        return *prediction;
    }

}

namespace dbm {

    template <typename T>
    Matrix<T> *DBM<T>::get_prediction_on_train_data() const {

        return prediction_train_data;

    }

    template <typename T>
    T *DBM<T>::get_test_loss() const {

        return test_loss_record;

    }

    template <typename T>
    void DBM<T>::set_loss_function_and_shrinkage(const char &type, const T &shrinkage) {

        params.dbm_loss_function = type;
        params.dbm_shrinkage = shrinkage;

    }

//    template <typename T>
//    Matrix<T> &DBM<T>::partial_dependence_plot(const Matrix<T> &data,
//                                              const int &predictor_ind) {
//        #ifdef _DEBUG_MODEL
//            assert(total_no_feature == data.get_width());
//        #endif
//
//        Matrix<T> modified_data = copy(data);
//
//        int data_height = data.get_height(),
//                *row_inds = new int[data_height],
//                resampling_size = int(data_height * params.pdp_resampling_portion);
//        for(int i = 0; i < data_height; ++i)
//            row_inds[i] = i;
//
//        if(pdp_result != nullptr)
//            delete pdp_result;
//        pdp_result = new Matrix<T>(params.pdp_no_x_ticks, 4, 0);
//        T predictor_min, predictor_max, standard_dev;
//
//        int total_no_resamplings = params.pdp_no_resamplings * no_cores;
//        Matrix<T> bootstraping(params.pdp_no_x_ticks, total_no_resamplings, 0);
//
//        #ifdef _OMP
//
//            omp_set_num_threads(no_cores);
//
//            std::random_device rd;
//            std::mt19937 mt(rd());
//            std::uniform_real_distribution<T> dist(0, 1000);
//            unsigned int *seeds = new unsigned int[total_no_resamplings];
//            for(int i = 0; i < total_no_resamplings; ++i)
//                seeds[i] = (unsigned int)dist(mt);
//
//        #else
//
//            Matrix<T> resampling_prediction(resampling_size, 1, 0);
//
//        #endif
//
//        Time_measurer timer(no_cores);
//        std::cout << std::endl
//                  << "Started bootstraping..."
//                  << std::endl;
//
//        predictor_min = data.get_col_min(predictor_ind),
//                predictor_max = data.get_col_max(predictor_ind);
//
//        for(int i = 0; i < params.pdp_no_x_ticks; ++i) {
//
//            pdp_result->assign(i, 0, predictor_min + i * (predictor_max - predictor_min) / (params.pdp_no_x_ticks - 1));
//
//            for(int j = 0; j < data_height; ++j)
//                modified_data.assign(j, predictor_ind, pdp_result->get(i, 0));
//
//            for(int j = 0; j < params.pdp_no_resamplings; ++j) {
//
//                #ifdef _OMP
//                #pragma omp parallel default(shared)
//                {
//
//                    int resampling_id = no_cores * j + omp_get_thread_num();
//
//                #else
//                {
//                #endif
//
//                    #ifdef _OMP
//
//                        int *thread_row_inds = new int[data_height];
//                        std::copy(row_inds, row_inds + data_height, thread_row_inds);
//                        shuffle(thread_row_inds, data_height, seeds[resampling_id]);
//
//                        Matrix<T> *resampling_prediction = new Matrix<T>(resampling_size, 1, 0);
//                        predict(modified_data.rows(thread_row_inds, resampling_size), *resampling_prediction);
//
//                        bootstraping.assign(i, resampling_id, resampling_prediction->col_average(0));
//
//                        delete[] thread_row_inds;
//                        delete resampling_prediction;
//
//                    #else
//
//                        shuffle(row_inds, data_height);
//
//                        resampling_prediction.clear();
//                        predict(modified_data.rows(row_inds, resampling_size), resampling_prediction);
//
//                        bootstraping.assign(i, j, resampling_prediction.col_average(0));
//
//                    #endif
//
//                }
//
//            }
//
//            pdp_result->assign(i, 1, bootstraping.row_average(i));
//
//            standard_dev = bootstraping.row_std(i);
//
//            pdp_result->assign(i, 2, pdp_result->get(i, 1) - params.pdp_ci_bandwidth / 2.0 * standard_dev);
//            pdp_result->assign(i, 3, pdp_result->get(i, 1) + params.pdp_ci_bandwidth / 2.0 * standard_dev);
//
//        }
//
//        #ifdef _OMP
//            delete[] seeds;
//        #endif
//
//        delete[] row_inds;
//
//        return *pdp_result;
//
//    }
//
//    template <typename T>
//    Matrix<T> &DBM<T>::partial_dependence_plot(const Matrix<T> &data,
//                                              const int &predictor_ind,
//                                              const T &x_tick_min,
//                                              const T &x_tick_max) {
//        #ifdef _DEBUG_MODEL
//            assert(total_no_feature == data.get_width());
//        #endif
//
//        Matrix<T> modified_data = copy(data);
//
//        int data_height = data.get_height(),
//                *row_inds = new int[data_height],
//                resampling_size = int(data_height * params.pdp_resampling_portion);
//        for(int i = 0; i < data_height; ++i)
//            row_inds[i] = i;
//
//        if(pdp_result != nullptr)
//            delete pdp_result;
//        pdp_result = new Matrix<T>(params.pdp_no_x_ticks, 4, 0);
//        T predictor_min, predictor_max, standard_dev;
//
//        int total_no_resamplings = params.pdp_no_resamplings * no_cores;
//        Matrix<T> bootstraping(params.pdp_no_x_ticks, total_no_resamplings, 0);
//
//        #ifdef _OMP
//
//            omp_set_num_threads(no_cores);
//
//            std::random_device rd;
//            std::mt19937 mt(rd());
//            std::uniform_real_distribution<T> dist(0, 1000);
//            unsigned int *seeds = new unsigned int[total_no_resamplings];
//            for(int i = 0; i < total_no_resamplings; ++i)
//                seeds[i] = (unsigned int)dist(mt);
//
//        #else
//
//            Matrix<T> resampling_prediction(resampling_size, 1, 0);
//
//        #endif
//
//        Time_measurer timer(no_cores);
//        std::cout << std::endl
//                  << "Started bootstraping..."
//                  << std::endl;
//
//        predictor_min = x_tick_min,
//                predictor_max = x_tick_max;
//
//        for(int i = 0; i < params.pdp_no_x_ticks; ++i) {
//
//            pdp_result->assign(i, 0, predictor_min + i * (predictor_max - predictor_min) / (params.pdp_no_x_ticks - 1));
//
//            for(int j = 0; j < data_height; ++j)
//                modified_data.assign(j, predictor_ind, pdp_result->get(i, 0));
//
//            for(int j = 0; j < params.pdp_no_resamplings; ++j) {
//
//                #ifdef _OMP
//                #pragma omp parallel default(shared)
//                {
//
//                    int resampling_id = no_cores * j + omp_get_thread_num();
//
//                #else
//                {
//                #endif
//
//                    #ifdef _OMP
//
//                        int *thread_row_inds = new int[data_height];
//                        std::copy(row_inds, row_inds + data_height, thread_row_inds);
//                        shuffle(thread_row_inds, data_height, seeds[resampling_id]);
//
//                        Matrix<T> *resampling_prediction = new Matrix<T>(resampling_size, 1, 0);
//                        predict(modified_data.rows(thread_row_inds, resampling_size), *resampling_prediction);
//
//                        bootstraping.assign(i, resampling_id, resampling_prediction->col_average(0));
//
//                        delete[] thread_row_inds;
//                        delete resampling_prediction;
//
//                    #else
//
//                        shuffle(row_inds, data_height);
//
//                        resampling_prediction.clear();
//                        predict(modified_data.rows(row_inds, resampling_size), resampling_prediction);
//
//                        bootstraping.assign(i, j, resampling_prediction.col_average(0));
//
//                    #endif
//
//                }
//
//            }
//
//            pdp_result->assign(i, 1, bootstraping.row_average(i));
//
//            standard_dev = bootstraping.row_std(i);
//
//            pdp_result->assign(i, 2, pdp_result->get(i, 1) - params.pdp_ci_bandwidth / 2.0 * standard_dev);
//            pdp_result->assign(i, 3, pdp_result->get(i, 1) + params.pdp_ci_bandwidth / 2.0 * standard_dev);
//
//        }
//
//        #ifdef _OMP
//            delete[] seeds;
//        #endif
//
//        delete[] row_inds;
//
//        return *pdp_result;
//
//    }

    template <typename T>
    Matrix<T> &DBM<T>::statistical_significance(const Matrix<T> &data) {

        #ifdef _DEBUG_MODEL
            assert(total_no_feature == data.get_width());
        #endif

        Matrix<T> *modified_data = nullptr;

        int data_height = data.get_height(),
                data_width = data.get_width(),
                *row_inds = new int[data_height],
                resampling_size = int(data_height * params.pdp_resampling_portion);
        for(int i = 0; i < data_height; ++i)
            row_inds[i] = i;

        T predictor_min,
                predictor_max,
                x_tick;

        Matrix<T> x_ticks(total_no_feature, params.pdp_no_x_ticks, 0);
        Matrix<T> means(total_no_feature, params.pdp_no_x_ticks, 0);
        Matrix<T> stds(total_no_feature, params.pdp_no_x_ticks, 0);

        int total_no_resamplings = params.pdp_no_resamplings * no_cores;
        Matrix<T> bootstraping(params.pdp_no_x_ticks, total_no_resamplings, 0);

        ss_result = new Matrix<T>(total_no_feature, 1, 0);

        const int no_probs = 30;
        T z_scores[no_probs] = {0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1,
                          1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2,
                          2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7, 2.8, 2.9, 3};
        T probs[no_probs] = {0.0796, 0.1586, 0.23582, 0.31084, 0.38292, 0.4515, 0.51608, 0.57628, 0.63188, 0.68268,
                       0.72866, 0.76986, 0.8064, 0.83848, 0.86638, 0.8904, 0.91086, 0.92814, 0.94256, 0.9545,
                       0.96428, 0.9722, 0.97856, 0.9836, 0.98758, 0.99068, 0.99306, 0.99488, 0.99626, 0.9973};

        #ifdef _OMP

            omp_set_num_threads(no_cores);

//            std::random_device rd;
//            std::mt19937 mt(rd());
//            std::uniform_real_distribution<T> dist(0, 1000);
//            unsigned int *seeds = new unsigned int[total_no_resamplings];
//            for(int i = 0; i < total_no_resamplings; ++i)
//                seeds[i] = (unsigned int)dist(mt);

            unsigned int *seeds = new unsigned int[(no_bunches_of_learners - 1) * no_cores];

            if(params.dbm_random_seed < 0) {
                std::random_device rd;
                std::mt19937 mt(rd());
                std::uniform_real_distribution<T> dist(0, 1000);
                for(int i = 0; i < (no_bunches_of_learners - 1) * no_cores; ++i)
                    seeds[i] = (unsigned int) dist(mt);
            }
            else {
                std::mt19937 mt(params.dbm_random_seed);
                std::uniform_real_distribution<T> dist(0, 1000);
                for(int i = 0; i < (no_bunches_of_learners - 1) * no_cores; ++i)
                    seeds[i] = (unsigned int) dist(mt);
            }

        #else

            Matrix<T> resampling_prediction(resampling_size, 1, 0);

        #endif

        Time_measurer timer(no_cores);
        std::cout << std::endl
                  << "Started bootstraping..."
                  << std::endl;

        for(int i = 0; i < total_no_feature; ++i) {

            predictor_min = data.get_col_min(i),
                    predictor_max = data.get_col_max(i);

            modified_data = new Matrix<T>(data_height, data_width, 0);
            copy(data, *modified_data);

            for(int j = 0; j < params.pdp_no_x_ticks; ++j) {

                x_tick = predictor_min + j * (predictor_max - predictor_min) / (params.pdp_no_x_ticks - 1);

                for(int k = 0; k < data_height; ++k)
                    modified_data->assign(k, i, x_tick);

                for(int k = 0; k < params.pdp_no_resamplings; ++k) {

                    #ifdef _OMP
                    #pragma omp parallel default(shared)
                    {

                        int resampling_id = no_cores * k + omp_get_thread_num();

                    #else
                    {
                    #endif

                    #ifdef _OMP

                        int *thread_row_inds = new int[data_height];
                        std::copy(row_inds, row_inds + data_height, thread_row_inds);
                        shuffle(thread_row_inds, data_height, seeds[resampling_id]);

                        Matrix<T> *resampling_prediction = new Matrix<T>(resampling_size, 1, 0);
                        predict(modified_data->rows(thread_row_inds, resampling_size), *resampling_prediction);

                        bootstraping.assign(j, resampling_id, resampling_prediction->col_average(0));

                        delete[] thread_row_inds;
                        delete resampling_prediction;

                    #else

                        shuffle(row_inds, data_height);

                        resampling_prediction.clear();
                        predict(modified_data->rows(row_inds, resampling_size), resampling_prediction);

                        bootstraping.assign(j, k, resampling_prediction.col_average(0));

                    #endif

                    }

                }

                x_ticks.assign(i, j, x_tick);
                means.assign(i, j, bootstraping.row_average(j));
                stds.assign(i, j, bootstraping.row_std(j));
            }

            delete modified_data;

            std::cout << "Predictor ( " << i
                      << " ) --> bootstraping completed..."
                      << std::endl;

        }

        if(params.pdp_save_files) {
            x_ticks.print_to_file("x_ticks.txt");
            means.print_to_file("means.txt");
            stds.print_to_file("stds.txt");
        }

        T largest_lower_ci, smallest_higher_ci;

        for(int i = 0; i < total_no_feature; ++i) {

            int j;
            for(j = 0; j < no_probs; ++j) {

                largest_lower_ci = std::numeric_limits<T>::lowest(),
                        smallest_higher_ci = std::numeric_limits<T>::max();

                for(int k = 0; k < params.pdp_no_x_ticks; ++k) {

                    if(means.get(i, k) - z_scores[j] * stds.get(i, k) > largest_lower_ci) {
                        largest_lower_ci = means.get(i, k) - z_scores[j] * stds.get(i, k);
                    }
                    if(means.get(i, k) + z_scores[j] * stds.get(i, k) < smallest_higher_ci) {
                        smallest_higher_ci = means.get(i, k) + z_scores[j] * stds.get(i, k);
                    }

                }

                if(largest_lower_ci < smallest_higher_ci)
                    break;

            }

            ss_result->assign(i, 0, probs[std::max(j - 1, 0)]);

        }

        #ifdef _OMP
            delete[] seeds;
        #endif

        delete[] row_inds;

        return *ss_result;

    }

    template<typename T>
    Matrix<T> &DBM<T>::calibrate_plot(const Matrix<T> &observation,
                              const Matrix<T> &prediction,
                              int resolution,
                              const std::string& file_name) {
        /**
         * Generate data for calibrate plot, containing a pointwise 95 band
         * @param observation the outcome 0-1 variables
         * @param prediction the predictions estimating E(y|x)
         * @param resolution number of significant digits
         * @param file_name (optional) file name to record data of calibrate plot
         */


#ifdef _DEBUG_MODEL
        assert(prediction.get_height() == observation.get_height());
#endif
        std::map<T, dbm::AveragedObservation<T> > plot_data;
        typename std::map<T, dbm::AveragedObservation<T> >::iterator it;
        AveragedObservation<T> avg_obs;
        T x_pos, obs, magnitude = 1.;
        int height = prediction.get_height();

        for (int i = 0; i < resolution; i ++) magnitude *= (T)10.;
        for (int i = 0; i < height; i ++) {
            x_pos = round(prediction.get(i, 0) * magnitude) / magnitude;
            it = plot_data.find(x_pos);
            if (it == plot_data.end()) {
                avg_obs.N = 1;
                avg_obs.sum = observation.get(i, 0);
                avg_obs.sum2 = avg_obs.sum * avg_obs.sum;
                plot_data[x_pos] = avg_obs;
            } else {
                avg_obs = it->second;
                obs = observation.get(i, 0);
                avg_obs.N ++;
                avg_obs.sum += obs;
                avg_obs.sum2 += obs * obs;
                plot_data[x_pos] = avg_obs;
            } // it ?= plot_data.end()
        } // i

        mat_plot_dat = new Matrix<T>((int)plot_data.size(), 3, 0.);
        T sd, avg;
        int i = 0;
        for (it = plot_data.begin(); it != plot_data.end(); ++ it) {
            mat_plot_dat->assign(i, 0, it->first);
            avg_obs = it->second;

            avg = avg_obs.sum / (T)avg_obs.N;
            mat_plot_dat->assign(i, 1, avg);

            sd = avg_obs.sum2 - (T)avg_obs.N * avg * avg;
            if (avg_obs.N == 1) {
                sd = (T)0.;
            } else {
                sd = sqrt(sd / (T)(avg_obs.N - 1));
            }
            mat_plot_dat->assign(i, 2, sd);
            i ++;
        } // it

        std::ofstream fout(file_name);
        if (fout.is_open()) {
            for (i = 0; i < mat_plot_dat->get_height(); i ++) {
                fout << mat_plot_dat->get(i, 0) << " "
                     << mat_plot_dat->get(i, 1) << " "
                     << mat_plot_dat->get(i, 2) << std::endl;
            } // i
            fout.close();
        } // output calibrate.plot to file

        return *mat_plot_dat;
        /* END of Matrix<T>& calibrate_plot()*/
    } // calibrate_plot

    template <typename T>
    void DBM<T>::save_perf_to(const std::string &file_name) {
        if(train_loss_record == nullptr || test_loss_record == nullptr) {
            std::cout << "No training record..." << std::endl;
            return;
        }
        std::ofstream fout(file_name);
        fout << "Iteration" << "\t"
             << "Train Loss" << "\t"
             << "Test Loss" << std::endl;
        for (int i = 0; i < no_bunches_of_learners / params.dbm_freq_showing_loss_on_test + 1; i ++) {
            fout << (i * params.dbm_freq_showing_loss_on_test) << "\t"
                 << train_loss_record[i] << "\t"
                 << test_loss_record[i] << std::endl;
        } // i
        fout.close();
    } // save_perf_to

}

//namespace dbm {
//
//    template <typename T>
//    void DBM<T>::train_two_way_model(const Matrix<T> &data) {
//
//        if(vec_of_two_way_predictions != nullptr) {
//
//            for(int i = 0; i < no_two_way_models; ++i) {
//
//                delete vec_of_two_way_predictions[i];
//
//            }
//            delete[] vec_of_two_way_predictions;
//
//        }
//
//        no_two_way_models = total_no_feature * (total_no_feature - 1) / 2;
//
//        vec_of_two_way_predictions = new Matrix<T> *[no_two_way_models];
//        for(int i = 0; i < no_two_way_models; ++i) {
//
//            vec_of_two_way_predictions[i] = new Matrix<T>(params.twm_no_x_ticks, params.twm_no_x_ticks, 0);
//
//        }
//
//        predictor_x_ticks = new Matrix<T>(total_no_feature, params.twm_no_x_ticks, 0);
//
//        int index_two_way_model = -1;
//
//        Matrix<T> *modified_data = nullptr;
//
//        int data_height = data.get_height(),
//                data_width = data.get_width(),
//                *row_inds = new int[data_height],
//                resampling_size = int(data_height * params.twm_resampling_portion);
//        for(int i = 0; i < data_height; ++i)
//            row_inds[i] = i;
//
//        T predictor_min, predictor_max;
//
//        for(int i = 0; i < total_no_feature; ++i) {
//
//            predictor_max = data.get_col_max(i);
//            predictor_min = data.get_col_min(i);
//
//            predictor_max += 0.2 * std::abs(predictor_max);
//            predictor_min -= 0.2 * std::abs(predictor_min);
//
//            for(int j = 0; j < params.twm_no_x_ticks; ++j) {
//
//                predictor_x_ticks->assign(i, j,
//                                          predictor_min +
//                                                  j * (predictor_max - predictor_min) /
//                                                          (params.twm_no_x_ticks - 1));
//
//            }
//
//        }
//
//        Matrix<T> bootstrapped_predictions(params.twm_no_resamplings * no_cores, 1, 0);
//
//        // end of declarations and initilizations
//
//        #ifdef _OMP
//
//            omp_set_num_threads(no_cores);
//
//            std::random_device rd;
//            std::mt19937 mt(rd());
//            std::uniform_real_distribution<T> dist(0, 1000);
//            unsigned int *seeds = new unsigned int[params.twm_no_resamplings * no_cores];
//            for(int i = 0; i < params.twm_no_resamplings * no_cores; ++i)
//                seeds[i] = (unsigned int)dist(mt);
//
//        #else
//
//            Matrix<T> resampling_prediction(resampling_size, 1, 0);
//
//        #endif
//
//        Time_measurer timer(no_cores);
//        std::cout << std::endl
//                  << "Started bootstraping..."
//                  << std::endl;
//
//        for(int i = 0; i < total_no_feature - 1; ++i) {
//
//            for(int j = i + 1; j < total_no_feature; ++j) {
//
//                ++index_two_way_model;
//
//                for(int k = 0; k < params.twm_no_x_ticks; ++k) {
//
//                    for(int l = 0; l < params.twm_no_x_ticks; ++l) {
//
//                        modified_data = new Matrix<T>(data_height, data_width, 0);
//                        copy(data, *modified_data);
//
//                        for(int m = 0; m < data_height; ++m)
//                            modified_data->assign(m, i, predictor_x_ticks->get(i, k));
//
//                        for(int m = 0; m < data_height; ++m)
//                            modified_data->assign(m, j, predictor_x_ticks->get(j, l));
//
//                        for(int m = 0; m < params.twm_no_resamplings; ++m) {
//
//                            #ifdef _OMP
//                            #pragma omp parallel default(shared)
//                            {
//
//                                int resampling_id = no_cores * m + omp_get_thread_num();
//
//                            #else
//                            {
//                            #endif
//
//                                #ifdef _OMP
//
//                                    int *thread_row_inds = new int[data_height];
//                                    std::copy(row_inds, row_inds + data_height, thread_row_inds);
//                                    shuffle(thread_row_inds, data_height, seeds[resampling_id]);
//
//                                    Matrix<T> *resampling_prediction = new Matrix<T>(resampling_size, 1, 0);
//                                    predict(modified_data->rows(thread_row_inds, resampling_size), *resampling_prediction);
//
//                                    bootstrapped_predictions.assign(resampling_id, 0, resampling_prediction->col_average(0));
//
//                                    delete[] thread_row_inds;
//                                    delete resampling_prediction;
//
//                                #else
//
//                                    shuffle(row_inds, data_height);
//
//                                    resampling_prediction.clear();
//                                    predict(modified_data->rows(row_inds, resampling_size), resampling_prediction);
//
//                                    bootstrapped_predictions.assign(m, 0, resampling_prediction->col_average(0));
//
//                                #endif
//
//                            } // no_cores
//
//                        } // m < params.twm_no_resamplings
//
//                        vec_of_two_way_predictions[index_two_way_model]->assign(k, l, bootstrapped_predictions.col_average(0));
//
//                        delete modified_data;
//
//                    } // l < params.twm_no_x_ticks
//
//                } // k < params.twm_no_x_ticks
//
//                std::cout << "Predictor ( " << i
//                          << " ) and Predictor ( " << j
//                          << " ) --> bootstraping completed..."
//                          << std::endl;
//
//            } // j < total_no_feature
//
//        } // i < total_no_feature
//
//    }
//
//    template <typename T>
//    Matrix<T> &DBM<T>::predict_two_way_model(const Matrix<T> &data_x) {
//
//        if(prediction_two_way != nullptr) {
//
//            delete prediction_two_way;
//
//        }
//
//        int data_height = data_x.get_height();
//
//        #ifdef _DEBUG_MODEL
//            assert(data_x.get_width() == total_no_feature);
//        #endif
//
//        prediction_two_way = new Matrix<T>(data_height, 1, 0);
//
//        int index_two_way_model, m, n;
//
//        for(int i = 0; i < data_height; ++i) {
//
//            index_two_way_model = -1;
//
//            for(int k = 0; k < total_no_feature - 1; ++k) {
//
//                for(int l = k + 1; l < total_no_feature; ++l) {
//
//                    ++index_two_way_model;
//
//                    for(m = 0; m < params.twm_no_x_ticks; ++m) {
//
//                        if(data_x.get(i, k) < predictor_x_ticks->get(k, m)) break;
//
//                    }
//
//                    for(n = 0; n < params.twm_no_x_ticks; ++n) {
//
//                        if(data_x.get(i, l) < predictor_x_ticks->get(l, n)) break;
//
//                    }
//
//                    if(m == 0 && n == 0) {
//
//                        prediction_two_way->assign(i, 0, prediction_two_way->get(i, 0) +
//                                vec_of_two_way_predictions[index_two_way_model]->get(m, n));
//
//                    }
//                    else if(m == params.twm_no_x_ticks && n == params.twm_no_x_ticks) {
//
//                        prediction_two_way->assign(i, 0, prediction_two_way->get(i, 0) +
//                                                         vec_of_two_way_predictions[index_two_way_model]->get(m - 1, n - 1));
//
//                    }
//                    else if(m == 0) {
//
//                        prediction_two_way->assign(i, 0, prediction_two_way->get(i, 0) +
//                                (vec_of_two_way_predictions[index_two_way_model]->get(m, n) +
//                                        vec_of_two_way_predictions[index_two_way_model]->get(m, n - 1)) / 2.0);
//
//                    }
//                    else if(m == params.twm_no_x_ticks) {
//
//                        prediction_two_way->assign(i, 0, prediction_two_way->get(i, 0) +
//                                (vec_of_two_way_predictions[index_two_way_model]->get(m - 1, n) +
//                                        vec_of_two_way_predictions[index_two_way_model]->get(m - 1, n - 1)) / 2.0);
//
//                    }
//                    else if(n == 0) {
//
//                        prediction_two_way->assign(i, 0, prediction_two_way->get(i, 0) +
//                                (vec_of_two_way_predictions[index_two_way_model]->get(m, n) +
//                                        vec_of_two_way_predictions[index_two_way_model]->get(m - 1, n)) / 2.0);
//
//                    }
//                    else if(n == params.twm_no_x_ticks) {
//
//                        prediction_two_way->assign(i, 0, prediction_two_way->get(i, 0) +
//                                (vec_of_two_way_predictions[index_two_way_model]->get(m, n - 1) +
//                                        vec_of_two_way_predictions[index_two_way_model]->get(m - 1, n - 1)) / 2.0);
//
//                    }
//                    else {
//
//                        prediction_two_way->assign(i, 0, prediction_two_way->get(i, 0) +
//                                (vec_of_two_way_predictions[index_two_way_model]->get(m, n - 1) +
//                                 vec_of_two_way_predictions[index_two_way_model]->get(m - 1, n - 1) +
//                                 vec_of_two_way_predictions[index_two_way_model]->get(m, n) +
//                                 vec_of_two_way_predictions[index_two_way_model]->get(m - 1, n)) / 4.0);
//
//                    }
//
//
//                } // l < total_no_feature
//
//            } //  k < total_no_feature - 1
//
//            prediction_two_way->assign(i, 0, prediction_two_way->get(i, 0) / no_two_way_models);
//
//        } // i < data_height
//
//        return *prediction_two_way;
//
//    }
//
//}

namespace dbm {

    template<typename T>
    void save_dbm(const DBM<T> *dbm, std::ofstream &out) {

        out << dbm->no_bunches_of_learners << ' '
            << dbm->no_cores << ' '
            << dbm->no_candidate_feature << ' '
            << dbm->no_train_sample << ' '
            << dbm->params.dbm_loss_function << ' '
            << dbm->params.dbm_shrinkage << ' '
            << dbm->total_no_feature << ' '
            << std::endl;
        char type;
        for (int i = 0; i < (dbm->no_bunches_of_learners - 1) * dbm->no_cores + 1; ++i) {
            type = dbm->learners[i]->get_type();
            switch (type) {

                case 'm': {
                    out << "== Mean " << i << " ==" << std::endl;
                    dbm::save_global_mean(dynamic_cast<Global_mean<T> *>(dbm->learners[i]), out);
                    out << "== End of Mean " << i << " ==" << std::endl;
                    break;
                }

                case 't': {
                    out << "== Tree " << i << " ==" << std::endl;
                    dbm::save_tree_node(dynamic_cast<Tree_node<T> *>(dbm->learners[i]), out);
                    out << "== End of Tree " << i << " ==" << std::endl;
                    break;
                }

                case 'l': {
                    out << "== LinReg " << i << " ==" << std::endl;
                    dbm::save_linear_regression(dynamic_cast<Linear_regression<T> *>(dbm->learners[i]), out);
                    out << "== End of LinReg " << i << " ==" << std::endl;
                    break;
                }

                case 'k': {
                    out << "== Kmeans " << i << " ==" << std::endl;
                    dbm::save_kmeans2d(dynamic_cast<Kmeans2d<T> *>(dbm->learners[i]), out);
                    out << "== End of Kmeans " << i << " ==" << std::endl;
                    break;
                }

                case 's': {
                    out << "== Splines " << i << " ==" << std::endl;
                    dbm::save_splines(dynamic_cast<Splines<T> *>(dbm->learners[i]), out);
                    out << "== End of Splines " << i << " ==" << std::endl;
                    break;
                }


                case 'n': {
                    out << "== NN " << i << " ==" << std::endl;
                    dbm::save_neural_network(dynamic_cast<Neural_network<T> *>(dbm->learners[i]), out);
                    out << "== End of NN " << i << " ==" << std::endl;
                    break;
                }

                case 'd': {
                    out << "== DPCS " << i << " ==" << std::endl;
                    dbm::save_dpc_stairs(dynamic_cast<DPC_stairs<T> *>(dbm->learners[i]), out);
                    out << "== End of DPCS " << i << " ==" << std::endl;
                    break;
                }

                default: {
                    std::cout << "Wrong learner type: " << type <<  std::endl;
                    throw std::invalid_argument("Specified learner does not exist.");
                }

            }

        }

    }

    template<typename T>
    void load_dbm(std::ifstream &in, DBM<T> *&dbm) {

        std::string line;
        std::getline(in, line);

        std::string words[100];
        int count = split_into_words(line, words);

//        assert(count == 7);
        if(count != 7)
            std::cout << "Error in loading DBM: count is " << count << std::endl;

        dbm = new DBM<T>(std::stoi(words[0]), std::stoi(words[1]), std::stoi(words[2]), std::stoi(words[3]), std::stoi(words[6]));
        dbm->set_loss_function_and_shrinkage(words[4].front(), T(std::stod(words[5])));

        Tree_node<T> *temp_tree_ptr;
        Global_mean<T> *temp_mean_ptr;
        Linear_regression<T> *temp_linear_regression_ptr;
        Neural_network<T> *temp_neural_network_ptr;
        Splines<T> *temp_splines_ptr;
        Kmeans2d<T> *temp_kmeans2d_ptr;
        DPC_stairs<T> *temp_dpc_stairs_ptr;

        char type;

        for (int i = 0; i < (dbm->no_bunches_of_learners - 1) * dbm->no_cores + 1; ++i) {

            line.clear();
            std::getline(in, line);

            split_into_words(line, words);

            type = words[1].front();
            switch (type) {

                case 'M': {
                    temp_mean_ptr = nullptr;
                    load_global_mean(in, temp_mean_ptr);
                    dbm->learners[i] = temp_mean_ptr;

                    // skip the end line
                    std::getline(in, line);

                    break;
                }

                case 'T': {
                    temp_tree_ptr = nullptr;
                    load_tree_node(in, temp_tree_ptr);
                    dbm->learners[i] = temp_tree_ptr;

                    // skip the end line
                    std::getline(in, line);

                    break;
                }

                case 'L': {
                    temp_linear_regression_ptr = nullptr;
                    load_linear_regression(in, temp_linear_regression_ptr);
                    dbm->learners[i] = temp_linear_regression_ptr;

                    // skip the end line
                    std::getline(in, line);

                    break;
                }

                case 'K': {
                    temp_kmeans2d_ptr = nullptr;
                    load_kmeans2d(in, temp_kmeans2d_ptr);
                    dbm->learners[i] = temp_kmeans2d_ptr;

                    // skip the end line
                    std::getline(in, line);

                    break;
                }

                case 'S': {
                    temp_splines_ptr = nullptr;
                    load_splines(in, temp_splines_ptr);
                    dbm->learners[i] = temp_splines_ptr;

                    // skip the end line
                    std::getline(in, line);

                    break;
                }

                case 'N': {
                    temp_neural_network_ptr = nullptr;
                    load_neural_network(in, temp_neural_network_ptr);
                    dbm->learners[i] = temp_neural_network_ptr;

                    // skip the end line
                    std::getline(in, line);

                    break;
                }

                case 'D': {
                    temp_dpc_stairs_ptr = nullptr;
                    load_dpc_stairs(in, temp_dpc_stairs_ptr);
                    dbm->learners[i] = temp_dpc_stairs_ptr;

                    // skip the end line
                    std::getline(in, line);

                    break;
                }

                default: {
                    std::cout << "Wrong learner type: " << type <<  std::endl;
                    throw std::invalid_argument("Specified learner does not exist.");
                }

            }

        }

    }

    template<typename T>
    void DBM<T>::save_dbm_to(const std::string &file_name) {

        std::ofstream out(file_name);
        dbm::save_dbm(this, out);

    }

    template<typename T>
    void DBM<T>::load_dbm_from(const std::string &file_name) {

        for (int i = 0; i < (no_bunches_of_learners - 1) * no_cores + 1; ++i) {
            delete learners[i];
        }
        delete[] learners;

        delete tree_trainer;
        delete mean_trainer;
        delete linear_regression_trainer;
        delete neural_network_trainer;
        delete splines_trainer;
        delete kmeans2d_trainer;
        delete dpc_stairs_trainer;

        if(prediction_train_data != nullptr)
            delete prediction_train_data;

        if(test_loss_record != nullptr)
            delete test_loss_record;

        if(pdp_result != nullptr)
            delete pdp_result;

        if(ss_result != nullptr)
            delete ss_result;

        std::ifstream in(file_name);

        std::string line;
        std::getline(in, line);

        std::string words[100];
        int count = split_into_words(line, words);

//        assert(count == 7);
        if(count != 7)
            std::cout << "Error in loading DBM: count is " << count << std::endl;

        no_bunches_of_learners = std::stoi(words[0]);
        no_cores = std::stoi(words[1]);
        total_no_feature = std::stoi(words[6]);
        no_candidate_feature = std::stoi(words[2]);
        no_train_sample = std::stoi(words[3]);

        loss_function = Loss_function<T>(params);
        set_loss_function_and_shrinkage(words[4].front(), T(std::stod(words[5])));

        learners = new Base_learner<T> *[(no_bunches_of_learners - 1) * no_cores + 1];

        for (int i = 0; i < (no_bunches_of_learners - 1) * no_cores + 1; ++i) {
            learners[i] = nullptr;
        }

        tree_trainer = new Fast_tree_trainer<T>(params);
        mean_trainer = new Mean_trainer<T>(params);
        linear_regression_trainer = new Linear_regression_trainer<T>(params);
        neural_network_trainer = new Neural_network_trainer<T>(params);
        splines_trainer = new Splines_trainer<T>(params);
        kmeans2d_trainer = new Kmeans2d_trainer<T>(params);
        dpc_stairs_trainer = new DPC_stairs_trainer<T>(params);

        Tree_node<T> *temp_tree_ptr;
        Global_mean<T> *temp_mean_ptr;
        Linear_regression<T> *temp_linear_regression_ptr;
        Neural_network<T> *temp_neural_network_ptr;
        Splines<T> *temp_splines_ptr;
        Kmeans2d<T> *temp_kmeans2d_ptr;
        DPC_stairs<T> *temp_dpc_stairs_ptr;

        char type;

        for (int i = 0; i < (no_bunches_of_learners - 1) * no_cores + 1; ++i) {

            line.clear();
            std::getline(in, line);

            split_into_words(line, words);

            type = words[1].front();
            switch (type) {

                case 'M': {
                    temp_mean_ptr = nullptr;
                    load_global_mean(in, temp_mean_ptr);
                    learners[i] = temp_mean_ptr;

                    // skip the end line
                    std::getline(in, line);

                    break;
                }

                case 'T': {
                    temp_tree_ptr = nullptr;
                    load_tree_node(in, temp_tree_ptr);
                    learners[i] = temp_tree_ptr;

                    // skip the end line
                    std::getline(in, line);

                    break;
                }

                case 'L': {
                    temp_linear_regression_ptr = nullptr;
                    load_linear_regression(in, temp_linear_regression_ptr);
                    learners[i] = temp_linear_regression_ptr;

                    // skip the end line
                    std::getline(in, line);

                    break;
                }

                case 'K': {
                    temp_kmeans2d_ptr = nullptr;
                    load_kmeans2d(in, temp_kmeans2d_ptr);
                    learners[i] = temp_kmeans2d_ptr;

                    // skip the end line
                    std::getline(in, line);

                    break;
                }

                case 'S': {
                    temp_splines_ptr = nullptr;
                    load_splines(in, temp_splines_ptr);
                    learners[i] = temp_splines_ptr;

                    // skip the end line
                    std::getline(in, line);

                    break;
                }

                case 'N': {
                    temp_neural_network_ptr = nullptr;
                    load_neural_network(in, temp_neural_network_ptr);
                    learners[i] = temp_neural_network_ptr;

                    // skip the end line
                    std::getline(in, line);

                    break;
                }

                case 'D': {
                    temp_dpc_stairs_ptr = nullptr;
                    load_dpc_stairs(in, temp_dpc_stairs_ptr);
                    learners[i] = temp_dpc_stairs_ptr;

                    // skip the end line
                    std::getline(in, line);

                    break;
                }

                default: {
                    std::cout << "Wrong learner type: " << type <<  std::endl;
                    throw std::invalid_argument("Specified learner does not exist.");
                }

            }

        }


    }

}

// AUTO_DBM
namespace dbm {

    template<typename T>
    AUTO_DBM<T>::AUTO_DBM(const Params &params) :
            params(params),
            loss_function(Loss_function<T>(params)){

        no_cores = params.dbm_no_cores;
        no_bunches_of_learners = params.dbm_no_bunches_of_learners;
        no_candidate_feature = params.dbm_no_candidate_feature;

        #ifdef _OMP
            if(no_cores == 0 || no_cores > omp_get_max_threads()) {
                std::cout << std::endl
                          << "================================="
                          << std::endl
                          << "no_cores: " << no_cores
                          << " is ignored, using " << omp_get_max_threads() << " !"
                          << std::endl
                          << "================================="
                          << std::endl;
                no_cores = omp_get_max_threads();
            }
            else if(no_cores < 0)
                throw std::invalid_argument("Specified no_cores is negative.");
        #else
            std::cout << std::endl
                      << "================================="
                      << std::endl
                      << "OpenMP is disabled!"
                      << std::endl
                      << "no_cores: " << no_cores
                      << " is ignored, using " << 1 << " !"
                      << std::endl
                      << "================================="
                      << std::endl;
            no_cores = 1;
        #endif

        learners = new Base_learner<T> *[(no_bunches_of_learners - 1) * no_cores + 1];
        for(int i = 0; i < (no_bunches_of_learners - 1) * no_cores + 1; ++i)
            learners[i] = nullptr;

        mean_trainer = new Mean_trainer<T>(params);
        tree_trainer = new Fast_tree_trainer<T>(params);
        splines_trainer = new Splines_trainer<T>(params);
        linear_regression_trainer = new Linear_regression_trainer<T>(params);
//        neural_network_trainer = new Neural_network_trainer<T>(params);
        kmeans2d_trainer = new Kmeans2d_trainer<T>(params);
        dpc_stairs_trainer = new DPC_stairs_trainer<T>(params);

        names_base_learners = new char[no_base_learners];
        names_base_learners[0] = 't';
        names_base_learners[1] = 's';
        names_base_learners[2] = 'l';
        names_base_learners[3] = 'k';
        names_base_learners[4] = 'd';
//        names_base_learners[5] = 'n';

        portions_base_learners = new T[no_base_learners];
        new_losses_for_base_learners = new T[no_base_learners];

    }

    template<typename T>
    AUTO_DBM<T>::~AUTO_DBM() {

        for (int i = 0; i < (no_bunches_of_learners - 1) * no_cores + 1; ++i) {
            delete learners[i];
        }
        delete[] learners;

        delete tree_trainer;
        delete mean_trainer;
        delete linear_regression_trainer;
        delete neural_network_trainer;
        delete splines_trainer;
        delete kmeans2d_trainer;
        delete dpc_stairs_trainer;

        delete prediction_train_data;
        delete test_loss_record;
        delete pdp_result;
        delete ss_result;

        delete prediction;
        if (train_loss_record != nullptr)
            delete train_loss_record;

        delete[] names_base_learners;
        delete[] portions_base_learners;
        delete[] new_losses_for_base_learners;

        if(vec_of_two_way_predictions != nullptr) {

            for(int i = 0; i < no_two_way_models; ++i) {

                delete vec_of_two_way_predictions[i];

            }
            delete[] vec_of_two_way_predictions;

        }
        delete prediction_two_way;
        delete predictor_x_ticks;

        delete mat_plot_dat;

    }

    template <typename T>
    int AUTO_DBM<T>::base_learner_choose(const Matrix<T> &train_x,
                                         const Matrix<T> &train_y,
                                         const Matrix<T> &test_x,
                                         const Matrix<T> &test_y,
                                         const Matrix<T> &ind_delta,
                                         const Matrix<T> &prediction_test_data,
                                         const Matrix<T> &input_monotonic_constraints,
                                         const int &bunch_no,
                                         const double &type_choose) {

        const T min_portion = 0.2;

        T sum_of_bl_losses = 0, sum_of_portions = 0;

        // end of initialization

        if (params.dbm_display_training_progress) {
            printf("\nSelecting base learner (%f) --> ", type_choose);
        }

        loss_on_train_set = loss_function.loss(train_y,
                                               *prediction_train_data,
                                               params.dbm_loss_function);

        if(bunch_no == 1) {

            for(int i = 0; i < no_base_learners; ++i) {

                new_losses_for_base_learners[i] = loss_on_train_set;
                sum_of_bl_losses += loss_on_train_set;

            } // i

            for(int i = 0; i < no_base_learners; ++i) {

                portions_base_learners[i] =
                        (new_losses_for_base_learners[i] + min_portion * sum_of_bl_losses) /
                        ((1.0 + min_portion * no_base_learners) * sum_of_bl_losses + std::numeric_limits<T>::min());

            } // i

            int i;
            for(i = 0; i < no_base_learners; ++i) {

                sum_of_portions += portions_base_learners[i];

                if(type_choose < sum_of_portions) {

                    break;
                }

            } // i

            return i;

        }
        else {

            for(int i = 0; i < no_base_learners; ++i) {

                sum_of_bl_losses += new_losses_for_base_learners[i];

            } // i

            for(int i = 0; i < no_base_learners; ++i) {

                portions_base_learners[i] =
                        (new_losses_for_base_learners[i] + min_portion * sum_of_bl_losses + std::numeric_limits<T>::min()) /
                        ((1.0 + min_portion * no_base_learners) * sum_of_bl_losses + no_base_learners * std::numeric_limits<T>::min());

            } // i

            int i;
            for(i = 0; i < no_base_learners; ++i) {

                sum_of_portions += portions_base_learners[i];

                if(type_choose < sum_of_portions) {

                    break;
                }

            } // i

            return i;

        }

    }

    template <typename T>
    void AUTO_DBM<T>::update_new_losses_for_bl(const Matrix<T> &train_y, int bl_no) {

        T latest_loss_on_train_set = loss_function.loss(train_y,
                                                        *prediction_train_data,
                                                        params.dbm_loss_function);
        new_losses_for_base_learners[bl_no] = loss_on_train_set - latest_loss_on_train_set;

        new_losses_for_base_learners[bl_no] =
                new_losses_for_base_learners[bl_no] > 0 ? new_losses_for_base_learners[bl_no] : 0;

    }

    template<typename T>
    void AUTO_DBM<T>::train(const Data_set<T> &data_set,
                            const Matrix<T> &input_monotonic_constraints) {

        std::cout << "dbm_no_bunches_of_learners: " << params.dbm_no_bunches_of_learners << std::endl
                  << "dbm_no_cores: " << params.dbm_no_cores << std::endl
                  << "dbm_portion_train_sample: " << params.dbm_portion_train_sample << std::endl
                  << "dbm_no_candidate_feature: " << params.dbm_no_candidate_feature << std::endl
                  << "dbm_shrinkage: " << params.dbm_shrinkage << std::endl
                  << "random_seed in Parameters: " << params.dbm_random_seed << std::endl
                  << "random_seed in Data_set: " << data_set.random_seed << std::endl;


        Time_measurer timer(no_cores);

        Matrix<T> const &train_x = data_set.get_train_x();
        Matrix<T> const &train_y = data_set.get_train_y();
        Matrix<T> const &test_x = data_set.get_test_x();
        Matrix<T> const &test_y = data_set.get_test_y();

        Matrix<T> sorted_train_x = copy(train_x);
        const Matrix<T> sorted_train_x_from = col_sort(sorted_train_x);
        const Matrix<T> train_x_sorted_to = col_sorted_to(sorted_train_x_from);

        int n_samples = train_x.get_height(), n_features = train_x.get_width();
        int n_test_samples = test_x.get_height();

        no_train_sample = (int)(params.dbm_portion_train_sample * n_samples);
        total_no_feature = n_features;

        int no_samples_in_nonoverlapping_batch = no_train_sample / no_cores - 1;
        int *whole_row_inds = new int[n_samples];
        int **thread_row_inds_vec = new int*[no_cores];
        for(int i = 0; i < no_cores; ++i) {
            thread_row_inds_vec[i] = whole_row_inds + i * no_samples_in_nonoverlapping_batch;
        }

        #ifdef _DEBUG_MODEL
            assert(no_train_sample <= n_samples &&
                           no_candidate_feature <= total_no_feature &&
                           input_monotonic_constraints.get_height() == total_no_feature);

            for (int i = 0; i < n_features; ++i) {
                // serves as a check of whether the length of monotonic_constraints
                // is equal to the length of features in some sense

                assert(input_monotonic_constraints.get(i, 0) == 0 ||
                       input_monotonic_constraints.get(i, 0) == -1 ||
                       input_monotonic_constraints.get(i, 0) == 1);
            }
        #endif

        int *row_inds = new int[n_samples], *col_inds = new int[n_features];

        for (int i = 0; i < n_features; ++i)
            col_inds[i] = i;
        for (int i = 0; i < n_samples; ++i)
            row_inds[i] = i;

        if (prediction_train_data != nullptr)
            delete prediction_train_data;
        prediction_train_data = new Matrix<T>(n_samples, 1, 0);

        T lowest_test_loss = std::numeric_limits<T>::max();

        if (test_loss_record != nullptr)
            delete[] test_loss_record;
        test_loss_record = new T[no_bunches_of_learners / params.dbm_freq_showing_loss_on_test + 1];

        if (params.dbm_do_perf) {
            if (train_loss_record != nullptr)
                delete[] train_loss_record;
            train_loss_record = new T[no_bunches_of_learners / params.dbm_freq_showing_loss_on_test + 1];
        }

//        portions_base_learners[0] = params.dbm_portion_for_trees;
//        portions_base_learners[1] = params.dbm_portion_for_lr;
//        portions_base_learners[2] = params.dbm_portion_for_s;
//        portions_base_learners[3] = params.dbm_portion_for_k;
//        portions_base_learners[4] = params.dbm_portion_for_nn;

        /*
         * ind_delta:
         * col 0: individual delta
         * col 1: denominator of individual delta
         */
        Matrix<T> ind_delta(n_samples, 2, 0);

        Matrix<T> prediction_test_data(n_test_samples, 1, 0);

        #ifdef _OMP
            omp_set_num_threads(no_cores);
        #endif

        unsigned int *seeds = new unsigned int[(no_bunches_of_learners - 1) * no_cores];
        double *type_choices = new double[no_bunches_of_learners];

        if(params.dbm_random_seed < 0) {
            std::random_device rd;
            std::mt19937 mt(rd());

            std::uniform_real_distribution<T> dist(0, 1000);
            for(int i = 0; i < (no_bunches_of_learners - 1) * no_cores; ++i)
                seeds[i] = (unsigned int) dist(mt);

            std::uniform_real_distribution<double> dist_0_1(0, 1);
            for(int i = 0; i < no_bunches_of_learners; ++i)
                type_choices[i] = dist_0_1(mt);
        }
        else {
            std::mt19937 mt(params.dbm_random_seed);

            std::uniform_real_distribution<T> dist(0, 1000);
            for(int i = 0; i < (no_bunches_of_learners - 1) * no_cores; ++i)
                seeds[i] = (unsigned int) dist(mt);

            std::uniform_real_distribution<double> dist_0_1(0, 1);
            for(int i = 0; i < no_bunches_of_learners; ++i)
                type_choices[i] = dist_0_1(mt);
        }

        int chosen_bl_index;
        char chosen_bl_type;

        learners[0] = new Global_mean<T>;

        if (params.dbm_display_training_progress) {
            std::cout << "Learner "
                      << "(" << learners[0]->get_type() << ") "
                      << " No. " << 0
                      << " -> ";
        }

        loss_function.calculate_ind_delta(train_y,
                                          *prediction_train_data,
                                          ind_delta,
                                          params.dbm_loss_function);
        mean_trainer->train(dynamic_cast<Global_mean<T> *>(learners[0]),
                            train_x,
                            ind_delta,
                            *prediction_train_data,
                            params.dbm_loss_function);
        learners[0]->predict(train_x,
                             *prediction_train_data);
        learners[0]->predict(test_x,
                             prediction_test_data);

        test_loss_record[0] = loss_function.loss(test_y,
                                                 prediction_test_data,
                                                 params.dbm_loss_function);

        train_loss_record[0] = loss_function.loss(train_y,
                                                  *prediction_train_data,
                                                  params.dbm_loss_function);

        if(test_loss_record[0] < lowest_test_loss)
            lowest_test_loss = test_loss_record[0];

        if (params.dbm_display_training_progress) {
            std::cout << std::endl
                      << '(' << 0 << ')'
                      << " \tLowest loss on test set: "
                      << lowest_test_loss
                      << std::endl
                      << " \t\tLoss on test set: "
                      << test_loss_record[0]
                      << std::endl
                      << " \t\tLoss on train set: "
                      << train_loss_record[0]
                      << std::endl << std::endl;
        }
        else {
            std::cout << "." << std::endl;
        }

        for (int i = 1; i < no_bunches_of_learners; ++i) {

            if ((!params.dbm_display_training_progress) && i % 10 == 0) {
                printf("\n");
            }

            loss_function.calculate_ind_delta(train_y,
                                              *prediction_train_data,
                                              ind_delta,
                                              params.dbm_loss_function);

            if(params.dbm_nonoverlapping_training) {
                std::copy(row_inds, row_inds + n_samples, whole_row_inds);
                shuffle(whole_row_inds,
                        n_samples,
                        seeds[no_cores * (i - 1) + 1]);
            }

            chosen_bl_index = base_learner_choose(train_x,
                                                  train_y,
                                                  test_x,
                                                  test_y,
                                                  ind_delta,
                                                  prediction_test_data,
                                                  input_monotonic_constraints,
                                                  i,
                                                  type_choices[i]);
            chosen_bl_type = names_base_learners[chosen_bl_index];

            if (params.dbm_display_training_progress) {
                printf("%c\n", chosen_bl_type);
                printf("Portions for base learners: ");
                for(int j = 0; j < no_base_learners; ++j) {
                    printf("(%c) %.4f ", names_base_learners[j], portions_base_learners[j]);
                }
                printf("\n");
            }

            if (params.dbm_display_training_progress) {
                printf("Loss Reductions for base learners: ");
                for(int j = 0; j < no_base_learners; ++j) {
                    printf("(%c) %.10f ", names_base_learners[j], new_losses_for_base_learners[j]);
                }
                printf("\n");
            }

            switch (chosen_bl_type) {
                case 't': {

                    for(int j = 0; j < no_cores; ++j)
                        learners[no_cores * (i - 1) + j + 1] = new Tree_node<T>(0);

                    #ifdef _OMP
                    #pragma omp parallel default(shared)
                    {
                        int thread_id = omp_get_thread_num(),
                                learner_id = no_cores * (i - 1) + thread_id + 1;
                    #else
                    {
                        int thread_id = 0, learner_id = i;
                    #endif

                        if (params.dbm_display_training_progress) {
                            #pragma omp critical
                            if(params.dbm_nonoverlapping_training)
                                std::printf("Learner (%c) No. %d -> "
                                                    "Training Tree at %p "
                                                    "number of samples: %d "
                                                    "max_depth: %d ...\n",
                                            chosen_bl_type,
                                            learner_id,
                                            learners[learner_id],
                                            no_samples_in_nonoverlapping_batch,
                                            params.cart_max_depth);
                            else
                                std::printf("Learner (%c) No. %d -> "
                                                    "Training Tree at %p "
                                                    "number of samples: %d "
                                                    "max_depth: %d ...\n",
                                            chosen_bl_type,
                                            learner_id,
                                            learners[learner_id],
                                            no_train_sample,
                                            params.cart_max_depth);
                        }
                        else {
                            printf(".");
                        }

                        if(params.dbm_nonoverlapping_training) {
                            int *thread_col_inds = new int[n_features];
                            std::copy(col_inds,
                                      col_inds + n_features,
                                      thread_col_inds);

                            #ifdef _OMP
                            #pragma omp critical
                            #endif
                            shuffle(thread_col_inds,
                                    n_features,
                                    seeds[learner_id - 1]);

                            Matrix<T> first_comp_in_loss = loss_function.first_comp(train_y,
                                                                                    *prediction_train_data,
                                                                                    params.dbm_loss_function);
                            Matrix<T> second_comp_in_loss = loss_function.second_comp(train_y,
                                                                                      *prediction_train_data,
                                                                                      params.dbm_loss_function);

                            tree_trainer->train(dynamic_cast<Tree_node<T> *>(learners[learner_id]),
                                                train_x,
                                                sorted_train_x_from,
                                                train_y,
                                                ind_delta,
                                                *prediction_train_data,
                                                first_comp_in_loss,
                                                second_comp_in_loss,
                                                input_monotonic_constraints,
                                                params.dbm_loss_function,
                                                thread_row_inds_vec[thread_id],
                                                no_samples_in_nonoverlapping_batch,
                                                thread_col_inds,
                                                no_candidate_feature);

                            delete[] thread_col_inds;

                        }
                        else {
                            int *thread_row_inds = new int[n_samples];
                            int *thread_col_inds = new int[n_features];
                            std::copy(row_inds,
                                      row_inds + n_samples,
                                      thread_row_inds);
                            std::copy(col_inds,
                                      col_inds + n_features,
                                      thread_col_inds);

                            #ifdef _OMP
                            #pragma omp critical
                            #endif
                            {
                                shuffle(thread_row_inds,
                                        n_samples,
                                        seeds[learner_id - 1]);
                                shuffle(thread_col_inds,
                                        n_features,
                                        seeds[learner_id - 1]);
                            }

                            Matrix<T> first_comp_in_loss = loss_function.first_comp(train_y,
                                                                                    *prediction_train_data,
                                                                                    params.dbm_loss_function);
                            Matrix<T> second_comp_in_loss = loss_function.second_comp(train_y,
                                                                                      *prediction_train_data,
                                                                                      params.dbm_loss_function);

                            tree_trainer->train(dynamic_cast<Tree_node<T> *>(learners[learner_id]),
                                                train_x,
                                                sorted_train_x_from,
                                                train_y,
                                                ind_delta,
                                                *prediction_train_data,
                                                first_comp_in_loss,
                                                second_comp_in_loss,
                                                input_monotonic_constraints,
                                                params.dbm_loss_function,
                                                thread_row_inds,
                                                no_train_sample,
                                                thread_col_inds,
                                                no_candidate_feature);

                            delete[] thread_row_inds;
                            delete[] thread_col_inds;

                        }

                        tree_trainer->update_loss_reduction(dynamic_cast<Tree_node<T> *>(learners[learner_id]));

                        if(params.cart_prune) {
                            tree_trainer->prune(dynamic_cast<Tree_node<T> *>(learners[learner_id]));
                        }

                        #ifdef _OMP
                        #pragma omp barrier
                        #endif
                        {
                            #ifdef _OMP
                            #pragma omp critical
                            #endif
                            learners[learner_id]->predict(train_x,
                                                          *prediction_train_data,
                                                          params.dbm_shrinkage);
                            #ifdef _OMP
                            #pragma omp critical
                            #endif
                            learners[learner_id]->predict(test_x,
                                                          prediction_test_data,
                                                          params.dbm_shrinkage);

                            if (params.dbm_record_every_tree) {
                                #ifdef _OMP
                                #pragma omp critical
                                #endif
                                {
                                    Tree_info<T> tree_info(dynamic_cast<Tree_node<T> *>
                                                           (learners[learner_id]));
                                    tree_info.print_to_file("trees.txt",
                                                            learner_id);
                                }
                            }
                        }
                    }

                    break;

                }
                case 'l': {

                    for(int j = 0; j < no_cores; ++j)
                        learners[no_cores * (i - 1) + j + 1] =
                                new Linear_regression<T>(no_candidate_feature,
                                                         params.dbm_loss_function);

                    #ifdef _OMP
                    #pragma omp parallel default(shared)
                    {
                        int thread_id = omp_get_thread_num(),
                                learner_id = no_cores * (i - 1) + thread_id + 1;
                    #else
                    {
                        int thread_id = 0, learner_id = i;
                    #endif

                        if (params.dbm_display_training_progress) {
                            #pragma omp critical
                            if(params.dbm_nonoverlapping_training) {
                                std::printf("Learner (%c) No. %d -> "
                                                    "Training Linear Regression at %p "
                                                    "number of samples: %d "
                                                    "number of predictors: %d ...\n",
                                            chosen_bl_type,
                                            learner_id,
                                            learners[learner_id],
                                            no_samples_in_nonoverlapping_batch,
                                            no_candidate_feature);
                            }
                            else {
                                std::printf("Learner (%c) No. %d -> "
                                                    "Training Linear Regression at %p "
                                                    "number of samples: %d "
                                                    "number of predictors: %d ...\n",
                                            chosen_bl_type,
                                            learner_id,
                                            learners[learner_id],
                                            no_train_sample,
                                            no_candidate_feature);
                            }
                        }
                        else {
                            printf(".");
                        }

                        if(params.dbm_nonoverlapping_training) {
                            int *thread_col_inds = new int[n_features];
                            std::copy(col_inds,
                                      col_inds + n_features,
                                      thread_col_inds);

                            #ifdef _OMP
                            #pragma omp critical
                            #endif
                            shuffle(thread_col_inds,
                                    n_features,
                                    seeds[learner_id - 1]);

                            linear_regression_trainer->train(dynamic_cast<Linear_regression<T> *>
                                                             (learners[learner_id]),
                                                             train_x,
                                                             ind_delta,
                                                             input_monotonic_constraints,
                                                             thread_row_inds_vec[thread_id],
                                                             no_samples_in_nonoverlapping_batch,
                                                             thread_col_inds,
                                                             no_candidate_feature);

                            delete[] thread_col_inds;

                        }
                        else {
                            int *thread_row_inds = new int[n_samples];
                            int *thread_col_inds = new int[n_features];
                            std::copy(row_inds,
                                      row_inds + n_samples,
                                      thread_row_inds);
                            std::copy(col_inds,
                                      col_inds + n_features,
                                      thread_col_inds);

                            #ifdef _OMP
                            #pragma omp critical
                            #endif
                            {
                                shuffle(thread_row_inds,
                                        n_samples,
                                        seeds[learner_id - 1]);
                                shuffle(thread_col_inds,
                                        n_features,
                                        seeds[learner_id - 1]);
                            }

                            linear_regression_trainer->train(dynamic_cast<Linear_regression<T> *>
                                                             (learners[learner_id]),
                                                             train_x,
                                                             ind_delta,
                                                             input_monotonic_constraints,
                                                             thread_row_inds,
                                                             no_train_sample,
                                                             thread_col_inds,
                                                             no_candidate_feature);

                            delete[] thread_row_inds;
                            delete[] thread_col_inds;

                        }

                        #ifdef _OMP
                        #pragma omp barrier
                        #endif
                        {
                            #ifdef _OMP
                            #pragma omp critical
                            #endif
                            learners[learner_id]->predict(train_x, *prediction_train_data,
                                                          params.dbm_shrinkage);
                            #ifdef _OMP
                            #pragma omp critical
                            #endif
                            learners[learner_id]->predict(test_x, prediction_test_data,
                                                          params.dbm_shrinkage);
                        }
                    }

                    break;

                }
                case 'k': {

                    for(int j = 0; j < no_cores; ++j)
                        learners[no_cores * (i - 1) + j + 1] =
                                new Kmeans2d<T>(params.kmeans_no_centroids,
                                                params.dbm_loss_function);

                    #ifdef _OMP
                    #pragma omp parallel default(shared)
                    {
                        int thread_id = omp_get_thread_num(),
                                learner_id = no_cores * (i - 1) + thread_id + 1;
                    #else
                    {
                        int thread_id = 0, learner_id = i;
                    #endif

                        if (params.dbm_display_training_progress) {
                            #pragma omp critical
                            if(params.dbm_nonoverlapping_training) {
                                std::printf("Learner (%c) No. %d -> "
                                                    "Kmeans2d at %p "
                                                    "number of samples: %d "
                                                    "number of predictors: %d "
                                                    "number of centroids: %d ...\n",
                                            chosen_bl_type,
                                            learner_id,
                                            learners[learner_id],
                                            no_samples_in_nonoverlapping_batch,
                                            no_candidate_feature,
                                            params.kmeans_no_centroids);
                            }
                            else {
                                std::printf("Learner (%c) No. %d -> "
                                                    "Kmeans2d at %p "
                                                    "number of samples: %d "
                                                    "number of predictors: %d "
                                                    "number of centroids: %d ...\n",
                                            chosen_bl_type,
                                            learner_id,
                                            learners[learner_id],
                                            no_train_sample,
                                            no_candidate_feature,
                                            params.kmeans_no_centroids);
                            }
                        }
                        else {
                            printf(".");
                        }

                        if(params.dbm_nonoverlapping_training) {
                            int *thread_col_inds = new int[n_features];
                            std::copy(col_inds,
                                      col_inds + n_features,
                                      thread_col_inds);

                            #ifdef _OMP
                            #pragma omp critical
                            #endif
                            shuffle(thread_col_inds,
                                    n_features,
                                    seeds[learner_id - 1]);

                            kmeans2d_trainer->train(dynamic_cast<Kmeans2d<T> *> (learners[learner_id]),
                                                    train_x,
                                                    ind_delta,
                                                    input_monotonic_constraints,
                                                    thread_row_inds_vec[thread_id],
                                                    no_samples_in_nonoverlapping_batch,
                                                    thread_col_inds,
                                                    no_candidate_feature);

                            delete[] thread_col_inds;

                        }
                        else {
                            int *thread_row_inds = new int[n_samples];
                            int *thread_col_inds = new int[n_features];
                            std::copy(row_inds,
                                      row_inds + n_samples,
                                      thread_row_inds);
                            std::copy(col_inds,
                                      col_inds + n_features,
                                      thread_col_inds);

                            #ifdef _OMP
                            #pragma omp critical
                            #endif
                            {
                                shuffle(thread_row_inds,
                                        n_samples,
                                        seeds[learner_id - 1]);
                                shuffle(thread_col_inds,
                                        n_features,
                                        seeds[learner_id - 1]);
                            }

                            kmeans2d_trainer->train(dynamic_cast<Kmeans2d<T> *> (learners[learner_id]),
                                                    train_x,
                                                    ind_delta,
                                                    input_monotonic_constraints,
                                                    thread_row_inds,
                                                    no_train_sample,
                                                    thread_col_inds,
                                                    no_candidate_feature);

                            delete[] thread_row_inds;
                            delete[] thread_col_inds;

                        }

                        #ifdef _OMP
                        #pragma omp barrier
                        #endif
                        {
                            #ifdef _OMP
                            #pragma omp critical
                            #endif
                            learners[learner_id]->predict(train_x,
                                                          *prediction_train_data,
                                                          params.dbm_shrinkage);
                            #ifdef _OMP
                            #pragma omp critical
                            #endif
                            learners[learner_id]->predict(test_x,
                                                          prediction_test_data,
                                                          params.dbm_shrinkage);
                        }
                    }

                    break;

                }
                case 's': {

                    for(int j = 0; j < no_cores; ++j)
                        learners[no_cores * (i - 1) + j + 1] =
                                new Splines<T>(params.splines_no_knot,
                                               params.dbm_loss_function,
                                               params.splines_hinge_coefficient);

                    #ifdef _OMP
                    #pragma omp parallel default(shared)
                    {
                        int thread_id = omp_get_thread_num(),
                                learner_id = no_cores * (i - 1) + thread_id + 1;
                    #else
                    {
                        int thread_id = 0, learner_id = i;
                    #endif

                        if (params.dbm_display_training_progress) {
                            #pragma omp critical
                            if(params.dbm_nonoverlapping_training) {
                                std::printf("Learner (%c) No. %d -> "
                                                    "Training Splines at %p "
                                                    "number of samples: %d "
                                                    "number of predictors: %d "
                                                    "number of knots: %d ...\n",
                                            chosen_bl_type,
                                            learner_id,
                                            learners[learner_id],
                                            no_samples_in_nonoverlapping_batch,
                                            no_candidate_feature,
                                            params.splines_no_knot);
                            }
                            else {
                                std::printf("Learner (%c) No. %d -> "
                                                    "Training Splines at %p "
                                                    "number of samples: %d "
                                                    "number of predictors: %d "
                                                    "number of knots: %d ...\n",
                                            chosen_bl_type,
                                            learner_id,
                                            learners[learner_id],
                                            no_train_sample,
                                            no_candidate_feature,
                                            params.splines_no_knot);
                            }
                        }
                        else {
                            printf(".");
                        }

                        if(params.dbm_nonoverlapping_training) {
                            int *thread_col_inds = new int[n_features];
                            std::copy(col_inds,
                                      col_inds + n_features,
                                      thread_col_inds);

                            #ifdef _OMP
                            #pragma omp critical
                            #endif
                            shuffle(thread_col_inds,
                                    n_features,
                                    seeds[learner_id - 1]);

                            splines_trainer->train(dynamic_cast<Splines<T> *>
                                                   (learners[learner_id]),
                                                   train_x,
                                                   ind_delta,
                                                   input_monotonic_constraints,
                                                   thread_row_inds_vec[thread_id],
                                                   no_samples_in_nonoverlapping_batch,
                                                   thread_col_inds,
                                                   no_candidate_feature);

                            delete[] thread_col_inds;

                        }
                        else {
                            int *thread_row_inds = new int[n_samples];
                            int *thread_col_inds = new int[n_features];
                            std::copy(row_inds,
                                      row_inds + n_samples,
                                      thread_row_inds);
                            std::copy(col_inds,
                                      col_inds + n_features,
                                      thread_col_inds);

                            #ifdef _OMP
                            #pragma omp critical
                            #endif
                            {
                                shuffle(thread_row_inds,
                                        n_samples,
                                        seeds[learner_id - 1]);
                                shuffle(thread_col_inds,
                                        n_features,
                                        seeds[learner_id - 1]);
                            }

                            splines_trainer->train(dynamic_cast<Splines<T> *>
                                                   (learners[learner_id]),
                                                   train_x,
                                                   ind_delta,
                                                   input_monotonic_constraints,
                                                   thread_row_inds,
                                                   no_train_sample,
                                                   thread_col_inds,
                                                   no_candidate_feature);

                            delete[] thread_row_inds;
                            delete[] thread_col_inds;

                        }

                        #ifdef _OMP
                        #pragma omp barrier
                        #endif
                        {
                            #ifdef _OMP
                            #pragma omp critical
                            #endif
                            learners[learner_id]->predict(train_x,
                                                          *prediction_train_data,
                                                          params.dbm_shrinkage);
                            #ifdef _OMP
                            #pragma omp critical
                            #endif
                            learners[learner_id]->predict(test_x,
                                                          prediction_test_data,
                                                          params.dbm_shrinkage);
                        }
                    }

                    break;

                }
                case 'n': {

                    for(int j = 0; j < no_cores; ++j)
                        learners[no_cores * (i - 1) + j + 1] =
                                new Neural_network<T>(no_candidate_feature,
                                                      params.nn_no_hidden_neurons,
                                                      params.dbm_loss_function);

                    #ifdef _OMP
                    #pragma omp parallel default(shared)
                    {
                        int thread_id = omp_get_thread_num(),
                                learner_id = no_cores * (i - 1) + thread_id + 1;
                    #else
                    {
                        int thread_id = 0, learner_id = i;
                    #endif

                        if (params.dbm_display_training_progress) {
                        #pragma omp critical
                            if(params.dbm_nonoverlapping_training) {
                                std::printf("Learner (%c) No. %d -> "
                                                    "Training Neural Network at %p "
                                                    "number of samples: %d "
                                                    "number of predictors: %d "
                                                    "number of hidden neurons: %d ...\n",
                                            chosen_bl_type,
                                            learner_id,
                                            learners[learner_id],
                                            no_samples_in_nonoverlapping_batch,
                                            no_candidate_feature,
                                            params.nn_no_hidden_neurons);
                            }
                            else {
                                std::printf("Learner (%c) No. %d -> "
                                                    "Training Neural Network at %p "
                                                    "number of samples: %d "
                                                    "number of predictors: %d "
                                                    "number of hidden neurons: %d ...\n",
                                            chosen_bl_type,
                                            learner_id,
                                            learners[learner_id],
                                            no_train_sample,
                                            no_candidate_feature,
                                            params.nn_no_hidden_neurons);
                            }
                        }
                        else {
                            printf(".");
                        }

                        if(params.dbm_nonoverlapping_training) {
                            int *thread_col_inds = new int[n_features];
                            std::copy(col_inds,
                                      col_inds + n_features,
                                      thread_col_inds);

                            #ifdef _OMP
                            #pragma omp critical
                            #endif
                            shuffle(thread_col_inds,
                                    n_features,
                                    seeds[learner_id - 1]);

                            neural_network_trainer->train(dynamic_cast<Neural_network<T> *>
                                                          (learners[learner_id]),
                                                          train_x,
                                                          ind_delta,
                                                          thread_row_inds_vec[thread_id],
                                                          no_samples_in_nonoverlapping_batch,
                                                          thread_col_inds,
                                                          no_candidate_feature);

                            delete[] thread_col_inds;

                        }
                        else {
                            int *thread_row_inds = new int[n_samples];
                            int *thread_col_inds = new int[n_features];
                            std::copy(row_inds,
                                      row_inds + n_samples,
                                      thread_row_inds);
                            std::copy(col_inds,
                                      col_inds + n_features,
                                      thread_col_inds);

                            #ifdef _OMP
                            #pragma omp critical
                            #endif
                            {
                                shuffle(thread_row_inds,
                                        n_samples,
                                        seeds[learner_id - 1]);
                                shuffle(thread_col_inds,
                                        n_features,
                                        seeds[learner_id - 1]);
                            }

                            neural_network_trainer->train(dynamic_cast<Neural_network<T> *>
                                                          (learners[learner_id]),
                                                          train_x,
                                                          ind_delta,
                                                          thread_row_inds,
                                                          no_train_sample,
                                                          thread_col_inds,
                                                          no_candidate_feature);

                            delete[] thread_row_inds;
                            delete[] thread_col_inds;

                        }

                        #ifdef _OMP
                        #pragma omp barrier
                        #endif
                        {
                            #ifdef _OMP
                            #pragma omp critical
                            #endif
                            learners[learner_id]->predict(train_x,
                                                          *prediction_train_data,
                                                          params.dbm_shrinkage);
                            #ifdef _OMP
                            #pragma omp critical
                            #endif
                            learners[learner_id]->predict(test_x,
                                                          prediction_test_data,
                                                          params.dbm_shrinkage);
                        }
                    }

                    break;

                }
                case 'd': {

                    for(int j = 0; j < no_cores; ++j)
                        learners[no_cores * (i - 1) + j + 1] =
                                new DPC_stairs<T>(no_candidate_feature,
                                                  params.dbm_loss_function,
                                                  params.dpcs_no_ticks);

                    #ifdef _OMP
                    #pragma omp parallel default(shared)
                    {
                        int thread_id = omp_get_thread_num();
                        int learner_id = no_cores * (i - 1) + thread_id + 1;
                    #else
                    {
                        int thread_id = 0, learner_id = i;
                    #endif

                        if (params.dbm_display_training_progress) {
                            #pragma omp critical
                            if(params.dbm_nonoverlapping_training) {
                                std::printf("Learner (%c) No. %d -> "
                                                    "Training DPC Stairs at %p "
                                                    "number of samples: %d "
                                                    "number of predictors: %d "
                                                    "number of ticks: %d ...\n",
                                            chosen_bl_type,
                                            learner_id,
                                            learners[learner_id],
                                            no_samples_in_nonoverlapping_batch,
                                            no_candidate_feature,
                                            params.dpcs_no_ticks);
                            }
                            else {
                                std::printf("Learner (%c) No. %d -> "
                                                    "Training DPC Stairs at %p "
                                                    "number of samples: %d "
                                                    "number of predictors: %d "
                                                    "number of ticks: %d ...\n",
                                            chosen_bl_type,
                                            learner_id,
                                            learners[learner_id],
                                            no_train_sample,
                                            no_candidate_feature,
                                            params.dpcs_no_ticks);
                            }
                        }
                        else {
                            printf(".");
                        }

                        if(params.dbm_nonoverlapping_training) {
                            int *thread_col_inds = new int[n_features];
                            std::copy(col_inds,
                                      col_inds + n_features,
                                      thread_col_inds);

                            #ifdef _OMP
                            #pragma omp critical
                            #endif
                            shuffle(thread_col_inds,
                                    n_features,
                                    seeds[learner_id - 1]);

                            dpc_stairs_trainer->train(dynamic_cast<DPC_stairs<T> *>
                                                      (learners[learner_id]),
                                                      train_x,
                                                      ind_delta,
                                                      input_monotonic_constraints,
                                                      thread_row_inds_vec[thread_id],
                                                      no_samples_in_nonoverlapping_batch,
                                                      thread_col_inds,
                                                      no_candidate_feature);

                            delete[] thread_col_inds;

                        }
                        else {
                            int *thread_row_inds = new int[n_samples];
                            int *thread_col_inds = new int[n_features];
                            std::copy(row_inds,
                                      row_inds + n_samples,
                                      thread_row_inds);
                            std::copy(col_inds,
                                      col_inds + n_features,
                                      thread_col_inds);

                            #ifdef _OMP
                            #pragma omp critical
                            #endif
                            {
                                shuffle(thread_row_inds,
                                        n_samples,
                                        seeds[learner_id - 1]);
                                shuffle(thread_col_inds,
                                        n_features,
                                        seeds[learner_id - 1]);
                            }
                            dpc_stairs_trainer->train(dynamic_cast<DPC_stairs<T> *>
                                                      (learners[learner_id]),
                                                      train_x,
                                                      ind_delta,
                                                      input_monotonic_constraints,
                                                      thread_row_inds,
                                                      no_train_sample,
                                                      thread_col_inds,
                                                      no_candidate_feature);

                            delete[] thread_row_inds;
                            delete[] thread_col_inds;

                        }

                        #ifdef _OMP
                        #pragma omp barrier
                        #endif
                        {
                            #ifdef _OMP
                            #pragma omp critical
                            #endif
                            learners[learner_id]->predict(train_x,
                                                          *prediction_train_data,
                                                          params.dbm_shrinkage);
                            #ifdef _OMP
                            #pragma omp critical
                            #endif
                            learners[learner_id]->predict(test_x,
                                                          prediction_test_data,
                                                          params.dbm_shrinkage);
                        }
                    }

                    break;

                }
                default: {
                    std::cout << "Wrong learner type: " << chosen_bl_type << std::endl;
                    throw std::invalid_argument("Specified learner does not exist.");
                }
            } // switch

            if (!(i % params.dbm_freq_showing_loss_on_test)) {
                test_loss_record[i / params.dbm_freq_showing_loss_on_test] =
                        loss_function.loss(test_y,
                                           prediction_test_data,
                                           params.dbm_loss_function);

                if (params.dbm_do_perf) {
                    train_loss_record[i / params.dbm_freq_showing_loss_on_test] =
                            loss_function.loss(train_y, *prediction_train_data, params.dbm_loss_function);
                } //dbm_do_perf, record loss on train;

                if(test_loss_record[i / params.dbm_freq_showing_loss_on_test] < lowest_test_loss)
                    lowest_test_loss = test_loss_record[i / params.dbm_freq_showing_loss_on_test];

                if (params.dbm_display_training_progress) {
                    std::cout << std::endl
                              << '(' << i / params.dbm_freq_showing_loss_on_test << ')'
                              << " \tLowest loss on test set: "
                              << lowest_test_loss
                              << std::endl
                              << " \t\tLoss on test set: "
                              << test_loss_record[i / params.dbm_freq_showing_loss_on_test]
                              << std::endl
                              << " \t\tLoss on train set: "
                              << train_loss_record[i / params.dbm_freq_showing_loss_on_test]
                              << std::endl << std::endl;
                }

            }

            update_new_losses_for_bl(train_y, chosen_bl_index);

        } // i < no_bunches_of_learners

        loss_function.link_function(*prediction_train_data,
                                    params.dbm_loss_function);

        std::cout << std::endl << "Losses on Test Set: " << std::endl;
        for (int i = 0; i < no_bunches_of_learners / params.dbm_freq_showing_loss_on_test + 1; ++i)
            std::cout << "(" << i << ") " << test_loss_record[i] << ' ';
        std::cout << std::endl;

        if (params.dbm_do_perf) {
            std::cout << std::endl << "Losses on Train Set: " << std::endl;
            for (int i = 0; i < no_bunches_of_learners / params.dbm_freq_showing_loss_on_test + 1; ++i)
                std::cout << "(" << i << ") " << train_loss_record[i] << ' ';
            std::cout << std::endl;
        }

        delete[] row_inds;
        delete[] col_inds;
        delete[] seeds;
        delete[] whole_row_inds;
        delete[] thread_row_inds_vec;

    }

    template<typename T>
    void AUTO_DBM<T>::train(const Data_set<T> &data_set) {

        std::cout << "dbm_no_bunches_of_learners: " << params.dbm_no_bunches_of_learners << std::endl
                  << "dbm_no_cores: " << params.dbm_no_cores << std::endl
                  << "dbm_portion_train_sample: " << params.dbm_portion_train_sample << std::endl
                  << "dbm_no_candidate_feature: " << params.dbm_no_candidate_feature << std::endl
                  << "dbm_shrinkage: " << params.dbm_shrinkage << std::endl
                  << "random_seed in Parameters: " << params.dbm_random_seed << std::endl
                  << "random_seed in Data_set: " << data_set.random_seed << std::endl;


        Time_measurer timer(no_cores);

        Matrix<T> const &train_x = data_set.get_train_x();
        Matrix<T> const &train_y = data_set.get_train_y();
        Matrix<T> const &test_x = data_set.get_test_x();
        Matrix<T> const &test_y = data_set.get_test_y();

        Matrix<T> sorted_train_x = copy(train_x);
        const Matrix<T> sorted_train_x_from = col_sort(sorted_train_x);
        const Matrix<T> train_x_sorted_to = col_sorted_to(sorted_train_x_from);

        int n_samples = train_x.get_height(), n_features = train_x.get_width();
        int n_test_samples = test_x.get_height();

        no_train_sample = (int)(params.dbm_portion_train_sample * n_samples);
        total_no_feature = n_features;

        int no_samples_in_nonoverlapping_batch = no_train_sample / no_cores - 1;
        int *whole_row_inds = new int[n_samples];
        int **thread_row_inds_vec = new int*[no_cores];
        for(int i = 0; i < no_cores; ++i) {
            thread_row_inds_vec[i] = whole_row_inds + i * no_samples_in_nonoverlapping_batch;
        }

        #ifdef _DEBUG_MODEL
            assert(no_train_sample <= n_samples && no_candidate_feature <= total_no_feature);
        #endif

        Matrix<T> input_monotonic_constraints(n_features, 1, 0);

//#ifdef _DEBUG_MODEL
//        assert(no_train_sample <= n_samples && no_candidate_feature <= total_no_feature);
//
//        for (int i = 0; i < n_features; ++i) {
//            // serves as a check of whether the length of monotonic_constraints
//            // is equal to the length of features in some sense
//
//            assert(input_monotonic_constraints.get(i, 0) == 0 ||
//                   input_monotonic_constraints.get(i, 0) == -1 ||
//                   input_monotonic_constraints.get(i, 0) == 1);
//        }
//#endif

        int *row_inds = new int[n_samples], *col_inds = new int[n_features];

        for (int i = 0; i < n_features; ++i)
            col_inds[i] = i;
        for (int i = 0; i < n_samples; ++i)
            row_inds[i] = i;

        if (prediction_train_data != nullptr)
            delete prediction_train_data;
        prediction_train_data = new Matrix<T>(n_samples, 1, 0);

        T lowest_test_loss = std::numeric_limits<T>::max();

        if (test_loss_record != nullptr)
            delete[] test_loss_record;
        test_loss_record = new T[no_bunches_of_learners / params.dbm_freq_showing_loss_on_test + 1];

        if (params.dbm_do_perf) {
            if (train_loss_record != nullptr)
                delete[] train_loss_record;
            train_loss_record = new T[no_bunches_of_learners / params.dbm_freq_showing_loss_on_test + 1];
        }

//        portions_base_learners[0] = params.dbm_portion_for_trees;
//        portions_base_learners[1] = params.dbm_portion_for_lr;
//        portions_base_learners[2] = params.dbm_portion_for_s;
//        portions_base_learners[3] = params.dbm_portion_for_k;
//        portions_base_learners[4] = params.dbm_portion_for_nn;

        /*
         * ind_delta:
         * col 0: individual delta
         * col 1: denominator of individual delta
         */
        Matrix<T> ind_delta(n_samples, 2, 0);

        Matrix<T> prediction_test_data(n_test_samples, 1, 0);

        #ifdef _OMP
            omp_set_num_threads(no_cores);
        #endif

        unsigned int *seeds = new unsigned int[(no_bunches_of_learners - 1) * no_cores];
        double *type_choices = new double[no_bunches_of_learners];

        if(params.dbm_random_seed < 0) {
            std::random_device rd;
            std::mt19937 mt(rd());

            std::uniform_real_distribution<T> dist(0, 1000);
            for(int i = 0; i < (no_bunches_of_learners - 1) * no_cores; ++i)
                seeds[i] = (unsigned int) dist(mt);

            std::uniform_real_distribution<double> dist_0_1(0, 1);
            for(int i = 0; i < no_bunches_of_learners; ++i)
                type_choices[i] = dist_0_1(mt);
        }
        else {
            std::mt19937 mt(params.dbm_random_seed);

            std::uniform_real_distribution<T> dist(0, 1000);
            for(int i = 0; i < (no_bunches_of_learners - 1) * no_cores; ++i)
                seeds[i] = (unsigned int) dist(mt);

            std::uniform_real_distribution<double> dist_0_1(0, 1);
            for(int i = 0; i < no_bunches_of_learners; ++i)
                type_choices[i] = dist_0_1(mt);
        }

        int chosen_bl_index;
        char chosen_bl_type;

        learners[0] = new Global_mean<T>;

        if (params.dbm_display_training_progress) {
            std::cout << "Learner "
                      << "(" << learners[0]->get_type() << ") "
                      << " No. " << 0
                      << " -> ";
        }

        loss_function.calculate_ind_delta(train_y,
                                          *prediction_train_data,
                                          ind_delta,
                                          params.dbm_loss_function);
        mean_trainer->train(dynamic_cast<Global_mean<T> *>(learners[0]),
                            train_x,
                            ind_delta,
                            *prediction_train_data,
                            params.dbm_loss_function);
        learners[0]->predict(train_x,
                             *prediction_train_data);
        learners[0]->predict(test_x,
                             prediction_test_data);

        test_loss_record[0] = loss_function.loss(test_y,
                                                 prediction_test_data,
                                                 params.dbm_loss_function);

        train_loss_record[0] = loss_function.loss(train_y,
                                                  *prediction_train_data,
                                                  params.dbm_loss_function);

        if(test_loss_record[0] < lowest_test_loss)
            lowest_test_loss = test_loss_record[0];

        if (params.dbm_display_training_progress) {
            std::cout << std::endl
                      << '(' << 0 << ')'
                      << " \tLowest loss on test set: "
                      << lowest_test_loss
                      << std::endl
                      << " \t\tLoss on test set: "
                      << test_loss_record[0]
                      << std::endl
                      << " \t\tLoss on train set: "
                      << train_loss_record[0]
                      << std::endl << std::endl;
        }
        else {
            std::cout << "." << std::endl;
        }

        for (int i = 1; i < no_bunches_of_learners; ++i) {

            if ((!params.dbm_display_training_progress) && i % 10 == 0) {
                printf("\n");
            }

            loss_function.calculate_ind_delta(train_y,
                                              *prediction_train_data,
                                              ind_delta,
                                              params.dbm_loss_function);

            if(params.dbm_nonoverlapping_training) {
                std::copy(row_inds, row_inds + n_samples, whole_row_inds);
                shuffle(whole_row_inds,
                        n_samples,
                        seeds[no_cores * (i - 1) + 1]);
            }

            chosen_bl_index = base_learner_choose(train_x,
                                                  train_y,
                                                  test_x,
                                                  test_y,
                                                  ind_delta,
                                                  prediction_test_data,
                                                  input_monotonic_constraints,
                                                  i,
                                                  type_choices[i]);
            chosen_bl_type = names_base_learners[chosen_bl_index];

            if (params.dbm_display_training_progress) {
                printf("%c\n", chosen_bl_type);
                printf("Portions for base learners: ");
                for(int j = 0; j < no_base_learners; ++j) {
                    printf("(%c) %.4f ", names_base_learners[j], portions_base_learners[j]);
                }
                printf("\n");
            }

            if (params.dbm_display_training_progress) {
                printf("Loss Reductions for base learners: ");
                for(int j = 0; j < no_base_learners; ++j) {
                    printf("(%c) %.10f ", names_base_learners[j], new_losses_for_base_learners[j]);
                }
                printf("\n");
            }

            switch (chosen_bl_type) {
                case 't': {

                    for(int j = 0; j < no_cores; ++j)
                        learners[no_cores * (i - 1) + j + 1] = new Tree_node<T>(0);

                    #ifdef _OMP
                    #pragma omp parallel default(shared)
                    {
                        int thread_id = omp_get_thread_num(),
                                learner_id = no_cores * (i - 1) + thread_id + 1;
                    #else
                    {
                        int thread_id = 0, learner_id = i;
                    #endif

                        if (params.dbm_display_training_progress) {
                            #pragma omp critical
                            if(params.dbm_nonoverlapping_training)
                                std::printf("Learner (%c) No. %d -> "
                                                    "Training Tree at %p "
                                                    "number of samples: %d "
                                                    "max_depth: %d ...\n",
                                            chosen_bl_type,
                                            learner_id,
                                            learners[learner_id],
                                            no_samples_in_nonoverlapping_batch,
                                            params.cart_max_depth);
                            else
                                std::printf("Learner (%c) No. %d -> "
                                                    "Training Tree at %p "
                                                    "number of samples: %d "
                                                    "max_depth: %d ...\n",
                                            chosen_bl_type,
                                            learner_id,
                                            learners[learner_id],
                                            no_train_sample,
                                            params.cart_max_depth);
                        }
                        else {
                            printf(".");
                        }

                        if(params.dbm_nonoverlapping_training) {
                            int *thread_col_inds = new int[n_features];
                            std::copy(col_inds,
                                      col_inds + n_features,
                                      thread_col_inds);

                            #ifdef _OMP
                            #pragma omp critical
                            #endif
                            shuffle(thread_col_inds,
                                    n_features,
                                    seeds[learner_id - 1]);

                            Matrix<T> first_comp_in_loss = loss_function.first_comp(train_y,
                                                                                    *prediction_train_data,
                                                                                    params.dbm_loss_function);
                            Matrix<T> second_comp_in_loss = loss_function.second_comp(train_y,
                                                                                      *prediction_train_data,
                                                                                      params.dbm_loss_function);

                            tree_trainer->train(dynamic_cast<Tree_node<T> *>(learners[learner_id]),
                                                train_x,
                                                sorted_train_x_from,
                                                train_y,
                                                ind_delta,
                                                *prediction_train_data,
                                                first_comp_in_loss,
                                                second_comp_in_loss,
                                                input_monotonic_constraints,
                                                params.dbm_loss_function,
                                                thread_row_inds_vec[thread_id],
                                                no_samples_in_nonoverlapping_batch,
                                                thread_col_inds,
                                                no_candidate_feature);

                            delete[] thread_col_inds;

                        }
                        else {
                            int *thread_row_inds = new int[n_samples];
                            int *thread_col_inds = new int[n_features];
                            std::copy(row_inds,
                                      row_inds + n_samples,
                                      thread_row_inds);
                            std::copy(col_inds,
                                      col_inds + n_features,
                                      thread_col_inds);

                            #ifdef _OMP
                            #pragma omp critical
                            #endif
                            {
                                shuffle(thread_row_inds,
                                        n_samples,
                                        seeds[learner_id - 1]);
                                shuffle(thread_col_inds,
                                        n_features,
                                        seeds[learner_id - 1]);
                            }
                            Matrix<T> first_comp_in_loss = loss_function.first_comp(train_y,
                                                                                    *prediction_train_data,
                                                                                    params.dbm_loss_function);
                            Matrix<T> second_comp_in_loss = loss_function.second_comp(train_y,
                                                                                      *prediction_train_data,
                                                                                      params.dbm_loss_function);

                            tree_trainer->train(dynamic_cast<Tree_node<T> *>(learners[learner_id]),
                                                train_x,
                                                sorted_train_x_from,
                                                train_y,
                                                ind_delta,
                                                *prediction_train_data,
                                                first_comp_in_loss,
                                                second_comp_in_loss,
                                                input_monotonic_constraints,
                                                params.dbm_loss_function,
                                                thread_row_inds,
                                                no_train_sample,
                                                thread_col_inds,
                                                no_candidate_feature);

                            delete[] thread_row_inds;
                            delete[] thread_col_inds;

                        }

                        tree_trainer->update_loss_reduction(dynamic_cast<Tree_node<T> *>(learners[learner_id]));

                        if(params.cart_prune) {
                            tree_trainer->prune(dynamic_cast<Tree_node<T> *>(learners[learner_id]));
                        }

                        #ifdef _OMP
                        #pragma omp barrier
                        #endif
                        {
                            #ifdef _OMP
                            #pragma omp critical
                            #endif
                            learners[learner_id]->predict(train_x,
                                                          *prediction_train_data,
                                                          params.dbm_shrinkage);
                            #ifdef _OMP
                            #pragma omp critical
                            #endif
                            learners[learner_id]->predict(test_x,
                                                          prediction_test_data,
                                                          params.dbm_shrinkage);

                            if (params.dbm_record_every_tree) {
                                #ifdef _OMP
                                #pragma omp critical
                                #endif
                                {
                                    Tree_info<T> tree_info(dynamic_cast<Tree_node<T> *>
                                                           (learners[learner_id]));
                                    tree_info.print_to_file("trees.txt",
                                                            learner_id);
                                }
                            }
                        }
                    }

                    break;

                }
                case 'l': {

                    for(int j = 0; j < no_cores; ++j)
                        learners[no_cores * (i - 1) + j + 1] =
                                new Linear_regression<T>(no_candidate_feature,
                                                         params.dbm_loss_function);

                    #ifdef _OMP
                    #pragma omp parallel default(shared)
                    {
                        int thread_id = omp_get_thread_num(),
                                learner_id = no_cores * (i - 1) + thread_id + 1;
                    #else
                    {
                        int thread_id = 0, learner_id = i;
                    #endif

                        if (params.dbm_display_training_progress) {
                            #pragma omp critical
                            if(params.dbm_nonoverlapping_training) {
                                std::printf("Learner (%c) No. %d -> "
                                                    "Training Linear Regression at %p "
                                                    "number of samples: %d "
                                                    "number of predictors: %d ...\n",
                                            chosen_bl_type,
                                            learner_id,
                                            learners[learner_id],
                                            no_samples_in_nonoverlapping_batch,
                                            no_candidate_feature);
                            }
                            else {
                                std::printf("Learner (%c) No. %d -> "
                                                    "Training Linear Regression at %p "
                                                    "number of samples: %d "
                                                    "number of predictors: %d ...\n",
                                            chosen_bl_type,
                                            learner_id,
                                            learners[learner_id],
                                            no_train_sample,
                                            no_candidate_feature);
                            }
                        }
                        else {
                            printf(".");
                        }

                        if(params.dbm_nonoverlapping_training) {
                            int *thread_col_inds = new int[n_features];
                            std::copy(col_inds,
                                      col_inds + n_features,
                                      thread_col_inds);

                            #ifdef _OMP
                            #pragma omp critical
                            #endif
                            shuffle(thread_col_inds,
                                    n_features,
                                    seeds[learner_id - 1]);

                            linear_regression_trainer->train(dynamic_cast<Linear_regression<T> *>
                                                             (learners[learner_id]),
                                                             train_x,
                                                             ind_delta,
                                                             input_monotonic_constraints,
                                                             thread_row_inds_vec[thread_id],
                                                             no_samples_in_nonoverlapping_batch,
                                                             thread_col_inds,
                                                             no_candidate_feature);

                            delete[] thread_col_inds;

                        }
                        else {
                            int *thread_row_inds = new int[n_samples];
                            int *thread_col_inds = new int[n_features];
                            std::copy(row_inds,
                                      row_inds + n_samples,
                                      thread_row_inds);
                            std::copy(col_inds,
                                      col_inds + n_features,
                                      thread_col_inds);

                            #ifdef _OMP
                            #pragma omp critical
                            #endif
                            {
                                shuffle(thread_row_inds,
                                        n_samples,
                                        seeds[learner_id - 1]);
                                shuffle(thread_col_inds,
                                        n_features,
                                        seeds[learner_id - 1]);
                            }

                            linear_regression_trainer->train(dynamic_cast<Linear_regression<T> *>
                                                             (learners[learner_id]),
                                                             train_x,
                                                             ind_delta,
                                                             input_monotonic_constraints,
                                                             thread_row_inds,
                                                             no_train_sample,
                                                             thread_col_inds,
                                                             no_candidate_feature);

                            delete[] thread_row_inds;
                            delete[] thread_col_inds;

                        }

                        #ifdef _OMP
                        #pragma omp barrier
                        #endif
                        {
                            #ifdef _OMP
                            #pragma omp critical
                            #endif
                            learners[learner_id]->predict(train_x, *prediction_train_data,
                                                          params.dbm_shrinkage);
                            #ifdef _OMP
                            #pragma omp critical
                            #endif
                            learners[learner_id]->predict(test_x, prediction_test_data,
                                                          params.dbm_shrinkage);
                        }
                    }

                    break;

                }
                case 'k': {

                    for(int j = 0; j < no_cores; ++j)
                        learners[no_cores * (i - 1) + j + 1] =
                                new Kmeans2d<T>(params.kmeans_no_centroids,
                                                params.dbm_loss_function);

                    #ifdef _OMP
                    #pragma omp parallel default(shared)
                    {
                        int thread_id = omp_get_thread_num(),
                                learner_id = no_cores * (i - 1) + thread_id + 1;
                    #else
                    {
                        int thread_id = 0, learner_id = i;
                    #endif

                        if (params.dbm_display_training_progress) {
                            #pragma omp critical
                            if(params.dbm_nonoverlapping_training) {
                                std::printf("Learner (%c) No. %d -> "
                                                    "Kmeans2d at %p "
                                                    "number of samples: %d "
                                                    "number of predictors: %d "
                                                    "number of centroids: %d ...\n",
                                            chosen_bl_type,
                                            learner_id,
                                            learners[learner_id],
                                            no_samples_in_nonoverlapping_batch,
                                            no_candidate_feature,
                                            params.kmeans_no_centroids);
                            }
                            else {
                                std::printf("Learner (%c) No. %d -> "
                                                    "Kmeans2d at %p "
                                                    "number of samples: %d "
                                                    "number of predictors: %d "
                                                    "number of centroids: %d ...\n",
                                            chosen_bl_type,
                                            learner_id,
                                            learners[learner_id],
                                            no_train_sample,
                                            no_candidate_feature,
                                            params.kmeans_no_centroids);
                            }
                        }
                        else {
                            printf(".");
                        }

                        if(params.dbm_nonoverlapping_training) {
                            int *thread_col_inds = new int[n_features];
                            std::copy(col_inds,
                                      col_inds + n_features,
                                      thread_col_inds);

                            #ifdef _OMP
                            #pragma omp critical
                            #endif
                            shuffle(thread_col_inds,
                                    n_features,
                                    seeds[learner_id - 1]);

                            kmeans2d_trainer->train(dynamic_cast<Kmeans2d<T> *> (learners[learner_id]),
                                                    train_x,
                                                    ind_delta,
                                                    input_monotonic_constraints,
                                                    thread_row_inds_vec[thread_id],
                                                    no_samples_in_nonoverlapping_batch,
                                                    thread_col_inds,
                                                    no_candidate_feature);

                            delete[] thread_col_inds;

                        }
                        else {
                            int *thread_row_inds = new int[n_samples];
                            int *thread_col_inds = new int[n_features];
                            std::copy(row_inds,
                                      row_inds + n_samples,
                                      thread_row_inds);
                            std::copy(col_inds,
                                      col_inds + n_features,
                                      thread_col_inds);

                            #ifdef _OMP
                            #pragma omp critical
                            #endif
                            {
                                shuffle(thread_row_inds,
                                        n_samples,
                                        seeds[learner_id - 1]);
                                shuffle(thread_col_inds,
                                        n_features,
                                        seeds[learner_id - 1]);
                            }

                            kmeans2d_trainer->train(dynamic_cast<Kmeans2d<T> *> (learners[learner_id]),
                                                    train_x,
                                                    ind_delta,
                                                    input_monotonic_constraints,
                                                    thread_row_inds,
                                                    no_train_sample,
                                                    thread_col_inds,
                                                    no_candidate_feature);

                            delete[] thread_row_inds;
                            delete[] thread_col_inds;

                        }

                        #ifdef _OMP
                        #pragma omp barrier
                        #endif
                        {
                            #ifdef _OMP
                            #pragma omp critical
                            #endif
                            learners[learner_id]->predict(train_x,
                                                          *prediction_train_data,
                                                          params.dbm_shrinkage);
                            #ifdef _OMP
                            #pragma omp critical
                            #endif
                            learners[learner_id]->predict(test_x,
                                                          prediction_test_data,
                                                          params.dbm_shrinkage);
                        }
                    }

                    break;

                }
                case 's': {

                    for(int j = 0; j < no_cores; ++j)
                        learners[no_cores * (i - 1) + j + 1] =
                                new Splines<T>(params.splines_no_knot,
                                               params.dbm_loss_function,
                                               params.splines_hinge_coefficient);

                    #ifdef _OMP
                    #pragma omp parallel default(shared)
                    {
                        int thread_id = omp_get_thread_num(),
                                learner_id = no_cores * (i - 1) + thread_id + 1;
                    #else
                    {
                        int thread_id = 0, learner_id = i;
                    #endif

                        if (params.dbm_display_training_progress) {
                            #pragma omp critical
                            if(params.dbm_nonoverlapping_training) {
                                std::printf("Learner (%c) No. %d -> "
                                                    "Training Splines at %p "
                                                    "number of samples: %d "
                                                    "number of predictors: %d "
                                                    "number of knots: %d ...\n",
                                            chosen_bl_type,
                                            learner_id,
                                            learners[learner_id],
                                            no_samples_in_nonoverlapping_batch,
                                            no_candidate_feature,
                                            params.splines_no_knot);
                            }
                            else {
                                std::printf("Learner (%c) No. %d -> "
                                                    "Training Splines at %p "
                                                    "number of samples: %d "
                                                    "number of predictors: %d "
                                                    "number of knots: %d ...\n",
                                            chosen_bl_type,
                                            learner_id,
                                            learners[learner_id],
                                            no_train_sample,
                                            no_candidate_feature,
                                            params.splines_no_knot);
                            }
                        }
                        else {
                            printf(".");
                        }

                        if(params.dbm_nonoverlapping_training) {
                            int *thread_col_inds = new int[n_features];
                            std::copy(col_inds,
                                      col_inds + n_features,
                                      thread_col_inds);

                            #ifdef _OMP
                            #pragma omp critical
                            #endif
                            shuffle(thread_col_inds,
                                    n_features,
                                    seeds[learner_id - 1]);

                            splines_trainer->train(dynamic_cast<Splines<T> *>
                                                   (learners[learner_id]),
                                                   train_x,
                                                   ind_delta,
                                                   input_monotonic_constraints,
                                                   thread_row_inds_vec[thread_id],
                                                   no_samples_in_nonoverlapping_batch,
                                                   thread_col_inds,
                                                   no_candidate_feature);

                            delete[] thread_col_inds;

                        }
                        else {
                            int *thread_row_inds = new int[n_samples];
                            int *thread_col_inds = new int[n_features];
                            std::copy(row_inds,
                                      row_inds + n_samples,
                                      thread_row_inds);
                            std::copy(col_inds,
                                      col_inds + n_features,
                                      thread_col_inds);

                            #ifdef _OMP
                            #pragma omp critical
                            #endif
                            {
                                shuffle(thread_row_inds,
                                        n_samples,
                                        seeds[learner_id - 1]);
                                shuffle(thread_col_inds,
                                        n_features,
                                        seeds[learner_id - 1]);
                            }

                            splines_trainer->train(dynamic_cast<Splines<T> *>
                                                   (learners[learner_id]),
                                                   train_x,
                                                   ind_delta,
                                                   input_monotonic_constraints,
                                                   thread_row_inds,
                                                   no_train_sample,
                                                   thread_col_inds,
                                                   no_candidate_feature);

                            delete[] thread_row_inds;
                            delete[] thread_col_inds;

                        }

                        #ifdef _OMP
                        #pragma omp barrier
                        #endif
                        {
                            #ifdef _OMP
                            #pragma omp critical
                            #endif
                            learners[learner_id]->predict(train_x,
                                                          *prediction_train_data,
                                                          params.dbm_shrinkage);
                            #ifdef _OMP
                            #pragma omp critical
                            #endif
                            learners[learner_id]->predict(test_x,
                                                          prediction_test_data,
                                                          params.dbm_shrinkage);
                        }
                    }

                    break;

                }
                case 'n': {

                    for(int j = 0; j < no_cores; ++j)
                        learners[no_cores * (i - 1) + j + 1] =
                                new Neural_network<T>(no_candidate_feature,
                                                      params.nn_no_hidden_neurons,
                                                      params.dbm_loss_function);

                    #ifdef _OMP
                    #pragma omp parallel default(shared)
                    {
                        int thread_id = omp_get_thread_num(),
                                learner_id = no_cores * (i - 1) + thread_id + 1;
                    #else
                    {
                        int thread_id = 0, learner_id = i;
                    #endif

                        if (params.dbm_display_training_progress) {
                            #pragma omp critical
                            if(params.dbm_nonoverlapping_training) {
                                std::printf("Learner (%c) No. %d -> "
                                                    "Training Neural Network at %p "
                                                    "number of samples: %d "
                                                    "number of predictors: %d "
                                                    "number of hidden neurons: %d ...\n",
                                            chosen_bl_type,
                                            learner_id,
                                            learners[learner_id],
                                            no_samples_in_nonoverlapping_batch,
                                            no_candidate_feature,
                                            params.nn_no_hidden_neurons);
                            }
                            else {
                                std::printf("Learner (%c) No. %d -> "
                                                    "Training Neural Network at %p "
                                                    "number of samples: %d "
                                                    "number of predictors: %d "
                                                    "number of hidden neurons: %d ...\n",
                                            chosen_bl_type,
                                            learner_id,
                                            learners[learner_id],
                                            no_train_sample,
                                            no_candidate_feature,
                                            params.nn_no_hidden_neurons);
                            }
                        }
                        else {
                            printf(".");
                        }

                        if(params.dbm_nonoverlapping_training) {
                            int *thread_col_inds = new int[n_features];
                            std::copy(col_inds,
                                      col_inds + n_features,
                                      thread_col_inds);

                            #ifdef _OMP
                            #pragma omp critical
                            #endif
                            shuffle(thread_col_inds,
                                    n_features,
                                    seeds[learner_id - 1]);

                            neural_network_trainer->train(dynamic_cast<Neural_network<T> *>
                                                          (learners[learner_id]),
                                                          train_x,
                                                          ind_delta,
                                                          thread_row_inds_vec[thread_id],
                                                          no_samples_in_nonoverlapping_batch,
                                                          thread_col_inds,
                                                          no_candidate_feature);

                            delete[] thread_col_inds;

                        }
                        else {
                            int *thread_row_inds = new int[n_samples];
                            int *thread_col_inds = new int[n_features];
                            std::copy(row_inds,
                                      row_inds + n_samples,
                                      thread_row_inds);
                            std::copy(col_inds,
                                      col_inds + n_features,
                                      thread_col_inds);

                            #ifdef _OMP
                            #pragma omp critical
                            #endif
                            {
                                shuffle(thread_row_inds,
                                        n_samples,
                                        seeds[learner_id - 1]);
                                shuffle(thread_col_inds,
                                        n_features,
                                        seeds[learner_id - 1]);
                            }

                            neural_network_trainer->train(dynamic_cast<Neural_network<T> *>
                                                          (learners[learner_id]),
                                                          train_x,
                                                          ind_delta,
                                                          thread_row_inds,
                                                          no_train_sample,
                                                          thread_col_inds,
                                                          no_candidate_feature);

                            delete[] thread_row_inds;
                            delete[] thread_col_inds;

                        }

                        #ifdef _OMP
                        #pragma omp barrier
                        #endif
                        {
                            #ifdef _OMP
                            #pragma omp critical
                            #endif
                            learners[learner_id]->predict(train_x,
                                                          *prediction_train_data,
                                                          params.dbm_shrinkage);
                            #ifdef _OMP
                            #pragma omp critical
                            #endif
                            learners[learner_id]->predict(test_x,
                                                          prediction_test_data,
                                                          params.dbm_shrinkage);
                        }
                    }

                    break;

                }
                case 'd': {

                    for(int j = 0; j < no_cores; ++j)
                        learners[no_cores * (i - 1) + j + 1] =
                                new DPC_stairs<T>(no_candidate_feature,
                                                  params.dbm_loss_function,
                                                  params.dpcs_no_ticks);

                    #ifdef _OMP
                    #pragma omp parallel default(shared)
                    {
                        int thread_id = omp_get_thread_num();
                        int learner_id = no_cores * (i - 1) + thread_id + 1;
                    #else
                    {
                        int thread_id = 0, learner_id = i;
                    #endif

                        if (params.dbm_display_training_progress) {
                            #pragma omp critical
                            if(params.dbm_nonoverlapping_training) {
                                std::printf("Learner (%c) No. %d -> "
                                                    "Training DPC Stairs at %p "
                                                    "number of samples: %d "
                                                    "number of predictors: %d "
                                                    "number of ticks: %d ...\n",
                                            chosen_bl_type,
                                            learner_id,
                                            learners[learner_id],
                                            no_samples_in_nonoverlapping_batch,
                                            no_candidate_feature,
                                            params.dpcs_no_ticks);
                            }
                            else {
                                std::printf("Learner (%c) No. %d -> "
                                                    "Training DPC Stairs at %p "
                                                    "number of samples: %d "
                                                    "number of predictors: %d "
                                                    "number of ticks: %d ...\n",
                                            chosen_bl_type,
                                            learner_id,
                                            learners[learner_id],
                                            no_train_sample,
                                            no_candidate_feature,
                                            params.dpcs_no_ticks);
                            }
                        }
                        else {
                            printf(".");
                        }

                        if(params.dbm_nonoverlapping_training) {
                            int *thread_col_inds = new int[n_features];
                            std::copy(col_inds,
                                      col_inds + n_features,
                                      thread_col_inds);

                            #ifdef _OMP
                            #pragma omp critical
                            #endif
                            shuffle(thread_col_inds,
                                    n_features,
                                    seeds[learner_id - 1]);

                            dpc_stairs_trainer->train(dynamic_cast<DPC_stairs<T> *>
                                                      (learners[learner_id]),
                                                      train_x,
                                                      ind_delta,
                                                      input_monotonic_constraints,
                                                      thread_row_inds_vec[thread_id],
                                                      no_samples_in_nonoverlapping_batch,
                                                      thread_col_inds,
                                                      no_candidate_feature);

                            delete[] thread_col_inds;

                        }
                        else {
                            int *thread_row_inds = new int[n_samples];
                            int *thread_col_inds = new int[n_features];
                            std::copy(row_inds,
                                      row_inds + n_samples,
                                      thread_row_inds);
                            std::copy(col_inds,
                                      col_inds + n_features,
                                      thread_col_inds);

                            #ifdef _OMP
                            #pragma omp critical
                            #endif
                            {
                                shuffle(thread_row_inds,
                                        n_samples,
                                        seeds[learner_id - 1]);
                                shuffle(thread_col_inds,
                                        n_features,
                                        seeds[learner_id - 1]);
                            }

                            dpc_stairs_trainer->train(dynamic_cast<DPC_stairs<T> *>
                                                      (learners[learner_id]),
                                                      train_x,
                                                      ind_delta,
                                                      input_monotonic_constraints,
                                                      thread_row_inds,
                                                      no_train_sample,
                                                      thread_col_inds,
                                                      no_candidate_feature);

                            delete[] thread_row_inds;
                            delete[] thread_col_inds;

                        }

                        #ifdef _OMP
                        #pragma omp barrier
                        #endif
                        {
                            #ifdef _OMP
                            #pragma omp critical
                            #endif
                            learners[learner_id]->predict(train_x,
                                                          *prediction_train_data,
                                                          params.dbm_shrinkage);
                            #ifdef _OMP
                            #pragma omp critical
                            #endif
                            learners[learner_id]->predict(test_x,
                                                          prediction_test_data,
                                                          params.dbm_shrinkage);
                        }
                    }

                    break;

                }
                default: {
                    std::cout << "Wrong learner type: " << chosen_bl_type << std::endl;
                    throw std::invalid_argument("Specified learner does not exist.");
                }
            } // switch

            if (!(i % params.dbm_freq_showing_loss_on_test)) {
                test_loss_record[i / params.dbm_freq_showing_loss_on_test] =
                        loss_function.loss(test_y,
                                           prediction_test_data,
                                           params.dbm_loss_function);

                if (params.dbm_do_perf) {
                    train_loss_record[i / params.dbm_freq_showing_loss_on_test] =
                            loss_function.loss(train_y, *prediction_train_data, params.dbm_loss_function);
                } //dbm_do_perf, record loss on train;

                if(test_loss_record[i / params.dbm_freq_showing_loss_on_test] < lowest_test_loss)
                    lowest_test_loss = test_loss_record[i / params.dbm_freq_showing_loss_on_test];

                if (params.dbm_display_training_progress) {
                    std::cout << std::endl
                              << '(' << i / params.dbm_freq_showing_loss_on_test << ')'
                              << " \tLowest loss on test set: "
                              << lowest_test_loss
                              << std::endl
                              << " \t\tLoss on test set: "
                              << test_loss_record[i / params.dbm_freq_showing_loss_on_test]
                              << std::endl
                              << " \t\tLoss on train set: "
                              << train_loss_record[i / params.dbm_freq_showing_loss_on_test]
                              << std::endl << std::endl;
                }

            }

            update_new_losses_for_bl(train_y, chosen_bl_index);

        } // i < no_bunches_of_learners

        loss_function.link_function(*prediction_train_data,
                                    params.dbm_loss_function);

        std::cout << std::endl << "Losses on Test Set: " << std::endl;
        for (int i = 0; i < no_bunches_of_learners / params.dbm_freq_showing_loss_on_test + 1; ++i)
            std::cout << "(" << i << ") " << test_loss_record[i] << ' ';
        std::cout << std::endl;

        if (params.dbm_do_perf) {
            std::cout << std::endl << "Losses on Train Set: " << std::endl;
            for (int i = 0; i < no_bunches_of_learners / params.dbm_freq_showing_loss_on_test + 1; ++i)
                std::cout << "(" << i << ") " << train_loss_record[i] << ' ';
            std::cout << std::endl;
        }

        delete[] row_inds;
        delete[] col_inds;
        delete[] seeds;
        delete[] type_choices;
        delete[] whole_row_inds;
        delete[] thread_row_inds_vec;

    }

    template<typename T>
    void AUTO_DBM<T>::predict(const Matrix<T> &data_x,
                              Matrix<T> &predict_y) {

        int data_height = data_x.get_height();

        #ifdef _DEBUG_MODEL
            assert(total_no_feature == data_x.get_width() &&
                   data_height == predict_y.get_height() &&
                   predict_y.get_width() == 1);
        #endif

        for (int i = 0; i < data_height; ++i)
            predict_y[i][0] = 0;

        if (learners[0]->get_type() == 'm') {
            learners[0]->predict(data_x, predict_y);
        }
        else {
            learners[0]->predict(data_x, predict_y, params.dbm_shrinkage);
        }

        for (int i = 1; i < (no_bunches_of_learners - 1) * no_cores + 1; ++i) {

            learners[i]->predict(data_x, predict_y, params.dbm_shrinkage);

        }

        loss_function.link_function(predict_y, params.dbm_loss_function);

    }

    template <typename T>
    Matrix<T> &AUTO_DBM<T>::predict(const Matrix<T> &data_x) {
        if(prediction != nullptr) {
            delete prediction;
        }
        prediction = new Matrix<T>(data_x.get_height(), 1, 0);
        predict(data_x, *prediction);
        return *prediction;
    }

}

namespace dbm {

    template <typename T>
    Matrix<T> *AUTO_DBM<T>::get_prediction_on_train_data() const {

        return prediction_train_data;

    }

    template <typename T>
    T *AUTO_DBM<T>::get_test_loss() const {

        return test_loss_record;

    }

    template <typename T>
    void AUTO_DBM<T>::set_loss_function_and_shrinkage(const char &type, const T &shrinkage) {

        params.dbm_loss_function = type;
        params.dbm_shrinkage = shrinkage;

    }

    template <typename T>
    Matrix<T> &AUTO_DBM<T>::partial_dependence_plot(const Matrix<T> &data,
                                               const int &predictor_ind) {
        #ifdef _DEBUG_MODEL
            assert(total_no_feature == data.get_width());
        #endif

        Matrix<T> modified_data = copy(data);

        int data_height = data.get_height(),
                *row_inds = new int[data_height],
                resampling_size = int(data_height * params.pdp_resampling_portion);
        for(int i = 0; i < data_height; ++i)
            row_inds[i] = i;

        if(pdp_result != nullptr)
            delete pdp_result;
        pdp_result = new Matrix<T>(params.pdp_no_x_ticks, 4, 0);
        T predictor_min, predictor_max, standard_dev;

        int total_no_resamplings = params.pdp_no_resamplings * no_cores;
        Matrix<T> bootstraping(params.pdp_no_x_ticks, total_no_resamplings, 0);

        #ifdef _OMP

            omp_set_num_threads(no_cores);

            unsigned int *seeds = new unsigned int[(no_bunches_of_learners - 1) * no_cores];

            if(params.dbm_random_seed < 0) {
                std::random_device rd;
                std::mt19937 mt(rd());
                std::uniform_real_distribution<T> dist(0, 1000);
                for(int i = 0; i < (no_bunches_of_learners - 1) * no_cores; ++i)
                    seeds[i] = (unsigned int) dist(mt);
            }
            else {
                std::mt19937 mt(params.dbm_random_seed);
                std::uniform_real_distribution<T> dist(0, 1000);
                for(int i = 0; i < (no_bunches_of_learners - 1) * no_cores; ++i)
                    seeds[i] = (unsigned int) dist(mt);
            }

        #else

            Matrix<T> resampling_prediction(resampling_size, 1, 0);

        #endif

        Time_measurer timer(no_cores);
        std::cout << std::endl
                  << "Started bootstraping..."
                  << std::endl;

        predictor_min = data.get_col_min(predictor_ind),
                predictor_max = data.get_col_max(predictor_ind);

        for(int i = 0; i < params.pdp_no_x_ticks; ++i) {

            pdp_result->assign(i, 0, predictor_min + i * (predictor_max - predictor_min) / (params.pdp_no_x_ticks - 1));

            for(int j = 0; j < data_height; ++j)
                modified_data.assign(j, predictor_ind, pdp_result->get(i, 0));

            for(int j = 0; j < params.pdp_no_resamplings; ++j) {

                #ifdef _OMP
                #pragma omp parallel default(shared)
                {

                    int resampling_id = no_cores * j + omp_get_thread_num();

                #else
                {
                #endif

                    #ifdef _OMP

                        int *thread_row_inds = new int[data_height];
                        std::copy(row_inds, row_inds + data_height, thread_row_inds);
                        shuffle(thread_row_inds, data_height, seeds[resampling_id]);

                        Matrix<T> *resampling_prediction = new Matrix<T>(resampling_size, 1, 0);
                        predict(modified_data.rows(thread_row_inds, resampling_size), *resampling_prediction);

                        bootstraping.assign(i, resampling_id, resampling_prediction->col_average(0));

                        delete[] thread_row_inds;
                        delete resampling_prediction;

                    #else

                        shuffle(row_inds, data_height);

                        resampling_prediction.clear();
                        predict(modified_data.rows(row_inds, resampling_size), resampling_prediction);

                        bootstraping.assign(i, j, resampling_prediction.col_average(0));

                    #endif

                }

            }

            pdp_result->assign(i, 1, bootstraping.row_average(i));

            standard_dev = bootstraping.row_std(i);

            pdp_result->assign(i, 2, pdp_result->get(i, 1) - params.pdp_ci_bandwidth / 2.0 * standard_dev);
            pdp_result->assign(i, 3, pdp_result->get(i, 1) + params.pdp_ci_bandwidth / 2.0 * standard_dev);

        }

        #ifdef _OMP
            delete[] seeds;
        #endif

        delete[] row_inds;

        return *pdp_result;

    }

    template <typename T>
    Matrix<T> &AUTO_DBM<T>::partial_dependence_plot(const Matrix<T> &data,
                                               const int &predictor_ind,
                                               const T &x_tick_min,
                                               const T &x_tick_max) {
        #ifdef _DEBUG_MODEL
            assert(total_no_feature == data.get_width());
        #endif

        Matrix<T> modified_data = copy(data);

        int data_height = data.get_height(),
                *row_inds = new int[data_height],
                resampling_size = int(data_height * params.pdp_resampling_portion);
        for(int i = 0; i < data_height; ++i)
            row_inds[i] = i;

        if(pdp_result != nullptr)
            delete pdp_result;
        pdp_result = new Matrix<T>(params.pdp_no_x_ticks, 4, 0);
        T predictor_min, predictor_max, standard_dev;

        int total_no_resamplings = params.pdp_no_resamplings * no_cores;
        Matrix<T> bootstraping(params.pdp_no_x_ticks, total_no_resamplings, 0);

        #ifdef _OMP

            omp_set_num_threads(no_cores);

            unsigned int *seeds = new unsigned int[(no_bunches_of_learners - 1) * no_cores];

            if(params.dbm_random_seed < 0) {
                std::random_device rd;
                std::mt19937 mt(rd());
                std::uniform_real_distribution<T> dist(0, 1000);
                for(int i = 0; i < (no_bunches_of_learners - 1) * no_cores; ++i)
                    seeds[i] = (unsigned int) dist(mt);
            }
            else {
                std::mt19937 mt(params.dbm_random_seed);
                std::uniform_real_distribution<T> dist(0, 1000);
                for(int i = 0; i < (no_bunches_of_learners - 1) * no_cores; ++i)
                    seeds[i] = (unsigned int) dist(mt);
            }

        #else

            Matrix<T> resampling_prediction(resampling_size, 1, 0);

        #endif

        Time_measurer timer(no_cores);
        std::cout << std::endl
                  << "Started bootstraping..."
                  << std::endl;

        predictor_min = x_tick_min,
                predictor_max = x_tick_max;

        for(int i = 0; i < params.pdp_no_x_ticks; ++i) {

            pdp_result->assign(i, 0, predictor_min + i * (predictor_max - predictor_min) / (params.pdp_no_x_ticks - 1));

            for(int j = 0; j < data_height; ++j)
                modified_data.assign(j, predictor_ind, pdp_result->get(i, 0));

            for(int j = 0; j < params.pdp_no_resamplings; ++j) {

                #ifdef _OMP
                #pragma omp parallel default(shared)
                {

                    int resampling_id = no_cores * j + omp_get_thread_num();

                #else
                {
                #endif

                    #ifdef _OMP

                        int *thread_row_inds = new int[data_height];
                        std::copy(row_inds, row_inds + data_height, thread_row_inds);
                        shuffle(thread_row_inds, data_height, seeds[resampling_id]);

                        Matrix<T> *resampling_prediction = new Matrix<T>(resampling_size, 1, 0);
                        predict(modified_data.rows(thread_row_inds, resampling_size), *resampling_prediction);

                        bootstraping.assign(i, resampling_id, resampling_prediction->col_average(0));

                        delete[] thread_row_inds;
                        delete resampling_prediction;

                    #else

                        shuffle(row_inds, data_height);

                        resampling_prediction.clear();
                        predict(modified_data.rows(row_inds, resampling_size), resampling_prediction);

                        bootstraping.assign(i, j, resampling_prediction.col_average(0));

                    #endif

                }

            }

            pdp_result->assign(i, 1, bootstraping.row_average(i));

            standard_dev = bootstraping.row_std(i);

            pdp_result->assign(i, 2, pdp_result->get(i, 1) - params.pdp_ci_bandwidth / 2.0 * standard_dev);
            pdp_result->assign(i, 3, pdp_result->get(i, 1) + params.pdp_ci_bandwidth / 2.0 * standard_dev);

        }

        #ifdef _OMP
            delete[] seeds;
        #endif

        delete[] row_inds;

        return *pdp_result;

    }

    template <typename T>
    Matrix<T> &AUTO_DBM<T>::statistical_significance(const Matrix<T> &data) {

        #ifdef _DEBUG_MODEL
            assert(total_no_feature == data.get_width());
        #endif

        Matrix<T> *modified_data = nullptr;

        int data_height = data.get_height(),
                data_width = data.get_width(),
                *row_inds = new int[data_height],
                resampling_size = int(data_height * params.pdp_resampling_portion);
        for(int i = 0; i < data_height; ++i)
            row_inds[i] = i;

        T predictor_min,
                predictor_max,
                x_tick;

        Matrix<T> x_ticks(total_no_feature, params.pdp_no_x_ticks, 0);
        Matrix<T> means(total_no_feature, params.pdp_no_x_ticks, 0);
        Matrix<T> stds(total_no_feature, params.pdp_no_x_ticks, 0);

        int total_no_resamplings = params.pdp_no_resamplings * no_cores;
        Matrix<T> bootstraping(params.pdp_no_x_ticks, total_no_resamplings, 0);

        ss_result = new Matrix<T>(total_no_feature, 1, 0);

        const int no_probs = 30;
        T z_scores[no_probs] = {0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1,
                                1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2,
                                2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7, 2.8, 2.9, 3};
        T probs[no_probs] = {0.0796, 0.1586, 0.23582, 0.31084, 0.38292, 0.4515, 0.51608, 0.57628, 0.63188, 0.68268,
                             0.72866, 0.76986, 0.8064, 0.83848, 0.86638, 0.8904, 0.91086, 0.92814, 0.94256, 0.9545,
                             0.96428, 0.9722, 0.97856, 0.9836, 0.98758, 0.99068, 0.99306, 0.99488, 0.99626, 0.9973};

        #ifdef _OMP

            omp_set_num_threads(no_cores);

            unsigned int *seeds = new unsigned int[(no_bunches_of_learners - 1) * no_cores];

            if(params.dbm_random_seed < 0) {
                std::random_device rd;
                std::mt19937 mt(rd());
                std::uniform_real_distribution<T> dist(0, 1000);
                for(int i = 0; i < (no_bunches_of_learners - 1) * no_cores; ++i)
                    seeds[i] = (unsigned int) dist(mt);
            }
            else {
                std::mt19937 mt(params.dbm_random_seed);
                std::uniform_real_distribution<T> dist(0, 1000);
                for(int i = 0; i < (no_bunches_of_learners - 1) * no_cores; ++i)
                    seeds[i] = (unsigned int) dist(mt);
            }

        #else

            Matrix<T> resampling_prediction(resampling_size, 1, 0);

        #endif

        Time_measurer timer(no_cores);
        std::cout << std::endl
                  << "Started bootstraping..."
                  << std::endl;

        for(int i = 0; i < total_no_feature; ++i) {

            predictor_min = data.get_col_min(i),
                    predictor_max = data.get_col_max(i);

            modified_data = new Matrix<T>(data_height, data_width, 0);
            copy(data, *modified_data);

            for(int j = 0; j < params.pdp_no_x_ticks; ++j) {

                x_tick = predictor_min + j * (predictor_max - predictor_min) / (params.pdp_no_x_ticks - 1);

                for(int k = 0; k < data_height; ++k)
                    modified_data->assign(k, i, x_tick);

                for(int k = 0; k < params.pdp_no_resamplings; ++k) {

                    #ifdef _OMP
                    #pragma omp parallel default(shared)
                    {

                        int resampling_id = no_cores * k + omp_get_thread_num();

                    #else
                    {
                    #endif

                        #ifdef _OMP

                            int *thread_row_inds = new int[data_height];
                            std::copy(row_inds, row_inds + data_height, thread_row_inds);
                            shuffle(thread_row_inds, data_height, seeds[resampling_id]);

                            Matrix<T> *resampling_prediction = new Matrix<T>(resampling_size, 1, 0);
                            predict(modified_data->rows(thread_row_inds, resampling_size), *resampling_prediction);

                            bootstraping.assign(j, resampling_id, resampling_prediction->col_average(0));

                            delete[] thread_row_inds;
                            delete resampling_prediction;

                        #else

                            shuffle(row_inds, data_height);

                            resampling_prediction.clear();
                            predict(modified_data->rows(row_inds, resampling_size), resampling_prediction);

                            bootstraping.assign(j, k, resampling_prediction.col_average(0));

                        #endif

                    }

                }

                x_ticks.assign(i, j, x_tick);
                means.assign(i, j, bootstraping.row_average(j));
                stds.assign(i, j, bootstraping.row_std(j));
            }

            delete modified_data;

            std::cout << "Predictor ( " << i
                      << " ) --> bootstraping completed..."
                      << std::endl;

        }

        if(params.pdp_save_files) {
            x_ticks.print_to_file("x_ticks.txt");
            means.print_to_file("means.txt");
            stds.print_to_file("stds.txt");
        }

        T largest_lower_ci, smallest_higher_ci;

        for(int i = 0; i < total_no_feature; ++i) {

            int j;
            for(j = 0; j < no_probs; ++j) {

                largest_lower_ci = std::numeric_limits<T>::lowest(),
                        smallest_higher_ci = std::numeric_limits<T>::max();

                for(int k = 0; k < params.pdp_no_x_ticks; ++k) {

                    if(means.get(i, k) - z_scores[j] * stds.get(i, k) > largest_lower_ci) {
                        largest_lower_ci = means.get(i, k) - z_scores[j] * stds.get(i, k);
                    }
                    if(means.get(i, k) + z_scores[j] * stds.get(i, k) < smallest_higher_ci) {
                        smallest_higher_ci = means.get(i, k) + z_scores[j] * stds.get(i, k);
                    }

                }

                if(largest_lower_ci < smallest_higher_ci)
                    break;

            }

            ss_result->assign(i, 0, probs[std::max(j - 1, 0)]);

        }

        #ifdef _OMP
            delete[] seeds;
        #endif

        delete[] row_inds;

        return *ss_result;

    }

    template<typename T>
    Matrix<T> &AUTO_DBM<T>::calibrate_plot(const Matrix<T> &observation,
                                      const Matrix<T> &prediction,
                                      int resolution,
                                      const std::string& file_name) {
        /**
         * Generate data for calibrate plot, containing a pointwise 95 band
         * @param observation the outcome 0-1 variables
         * @param prediction the predictions estimating E(y|x)
         * @param resolution number of significant digits
         * @param file_name (optional) file name to record data of calibrate plot
         */


#ifdef _DEBUG_MODEL
        assert(prediction.get_height() == observation.get_height());
#endif
        std::map<T, dbm::AveragedObservation<T> > plot_data;
        typename std::map<T, dbm::AveragedObservation<T> >::iterator it;
        AveragedObservation<T> avg_obs;
        T x_pos, obs, magnitude = 1.;
        int height = prediction.get_height();

        for (int i = 0; i < resolution; i ++) magnitude *= (T)10.;
        for (int i = 0; i < height; i ++) {
            x_pos = round(prediction.get(i, 0) * magnitude) / magnitude;
            it = plot_data.find(x_pos);
            if (it == plot_data.end()) {
                avg_obs.N = 1;
                avg_obs.sum = observation.get(i, 0);
                avg_obs.sum2 = avg_obs.sum * avg_obs.sum;
                plot_data[x_pos] = avg_obs;
            } else {
                avg_obs = it->second;
                obs = observation.get(i, 0);
                avg_obs.N ++;
                avg_obs.sum += obs;
                avg_obs.sum2 += obs * obs;
                plot_data[x_pos] = avg_obs;
            } // it ?= plot_data.end()
        } // i

        mat_plot_dat = new Matrix<T>((int)plot_data.size(), 3, 0.);
        T sd, avg;
        int i = 0;
        for (it = plot_data.begin(); it != plot_data.end(); ++ it) {
            mat_plot_dat->assign(i, 0, it->first);
            avg_obs = it->second;

            avg = avg_obs.sum / (T)avg_obs.N;
            mat_plot_dat->assign(i, 1, avg);

            sd = avg_obs.sum2 - (T)avg_obs.N * avg * avg;
            if (avg_obs.N == 1) {
                sd = (T)0.;
            } else {
                sd = sqrt(sd / (T)(avg_obs.N - 1));
            }
            mat_plot_dat->assign(i, 2, sd);
            i ++;
        } // it

        std::ofstream fout(file_name);
        if (fout.is_open()) {
            for (i = 0; i < mat_plot_dat->get_height(); i ++) {
                fout << mat_plot_dat->get(i, 0) << " "
                     << mat_plot_dat->get(i, 1) << " "
                     << mat_plot_dat->get(i, 2) << std::endl;
            } // i
            fout.close();
        } // output calibrate.plot to file

        return *mat_plot_dat;
        /* END of Matrix<T>& calibrate_plot()*/
    } // calibrate_plot

    template <typename T>
    void AUTO_DBM<T>::save_perf_to(const std::string &file_name) {
        if(train_loss_record == nullptr || test_loss_record == nullptr) {
            std::cout << "No training record..." << std::endl;
            return;
        }
        std::ofstream fout(file_name);
        fout << "Iteration" << "\t"
             << "Train Loss" << "\t"
             << "Test Loss" << std::endl;
        for (int i = 0; i < no_bunches_of_learners / params.dbm_freq_showing_loss_on_test + 1; i ++) {
            fout << (i * params.dbm_freq_showing_loss_on_test) << "\t"
                 << train_loss_record[i] << "\t"
                 << test_loss_record[i] << std::endl;
        } // i
        fout.close();
    } // save_perf_to

}

//namespace dbm {
//
//    template <typename T>
//    void AUTO_DBM<T>::train_two_way_model(const Matrix<T> &data) {
//
//        if(vec_of_two_way_predictions != nullptr) {
//
//            for(int i = 0; i < no_two_way_models; ++i) {
//
//                delete vec_of_two_way_predictions[i];
//
//            }
//            delete[] vec_of_two_way_predictions;
//
//        }
//
//        no_two_way_models = total_no_feature * (total_no_feature - 1) / 2;
//
//        vec_of_two_way_predictions = new Matrix<T> *[no_two_way_models];
//        for(int i = 0; i < no_two_way_models; ++i) {
//
//            vec_of_two_way_predictions[i] = new Matrix<T>(params.twm_no_x_ticks, params.twm_no_x_ticks, 0);
//
//        }
//
//        predictor_x_ticks = new Matrix<T>(total_no_feature, params.twm_no_x_ticks, 0);
//
//        int index_two_way_model = -1;
//
//        Matrix<T> *modified_data = nullptr;
//
//        int data_height = data.get_height(),
//                data_width = data.get_width(),
//                *row_inds = new int[data_height],
//                resampling_size = int(data_height * params.twm_resampling_portion);
//        for(int i = 0; i < data_height; ++i)
//            row_inds[i] = i;
//
//        T predictor_min, predictor_max;
//
//        for(int i = 0; i < total_no_feature; ++i) {
//
//            predictor_max = data.get_col_max(i);
//            predictor_min = data.get_col_min(i);
//
//            predictor_max += 0.2 * std::abs(predictor_max);
//            predictor_min -= 0.2 * std::abs(predictor_min);
//
//            for(int j = 0; j < params.twm_no_x_ticks; ++j) {
//
//                predictor_x_ticks->assign(i, j,
//                                          predictor_min +
//                                          j * (predictor_max - predictor_min) /
//                                          (params.twm_no_x_ticks - 1));
//
//            }
//
//        }
//
//        Matrix<T> bootstrapped_predictions(params.twm_no_resamplings * no_cores, 1, 0);
//
//        // end of declarations and initilizations
//
//#ifdef _OMP
//
//        omp_set_num_threads(no_cores);
//
//        std::random_device rd;
//        std::mt19937 mt(rd());
//        std::uniform_real_distribution<T> dist(0, 1000);
//        unsigned int *seeds = new unsigned int[params.twm_no_resamplings * no_cores];
//        for(int i = 0; i < params.twm_no_resamplings * no_cores; ++i)
//            seeds[i] = (unsigned int)dist(mt);
//
//#else
//
//        Matrix<T> resampling_prediction(resampling_size, 1, 0);
//
//#endif
//
//        Time_measurer timer(no_cores);
//        std::cout << std::endl
//                  << "Started bootstraping..."
//                  << std::endl;
//
//        for(int i = 0; i < total_no_feature - 1; ++i) {
//
//            for(int j = i + 1; j < total_no_feature; ++j) {
//
//                ++index_two_way_model;
//
//                for(int k = 0; k < params.twm_no_x_ticks; ++k) {
//
//                    for(int l = 0; l < params.twm_no_x_ticks; ++l) {
//
//                        modified_data = new Matrix<T>(data_height, data_width, 0);
//                        copy(data, *modified_data);
//
//                        for(int m = 0; m < data_height; ++m)
//                            modified_data->assign(m, i, predictor_x_ticks->get(i, k));
//
//                        for(int m = 0; m < data_height; ++m)
//                            modified_data->assign(m, j, predictor_x_ticks->get(j, l));
//
//                        for(int m = 0; m < params.twm_no_resamplings; ++m) {
//
//#ifdef _OMP
//#pragma omp parallel default(shared)
//                            {
//
//                                int resampling_id = no_cores * m + omp_get_thread_num();
//
//#else
//                                {
//#endif
//
//#ifdef _OMP
//
//                                int *thread_row_inds = new int[data_height];
//                                std::copy(row_inds, row_inds + data_height, thread_row_inds);
//                                shuffle(thread_row_inds, data_height, seeds[resampling_id]);
//
//                                Matrix<T> *resampling_prediction = new Matrix<T>(resampling_size, 1, 0);
//                                predict(modified_data->rows(thread_row_inds, resampling_size), *resampling_prediction);
//
//                                bootstrapped_predictions.assign(resampling_id, 0, resampling_prediction->col_average(0));
//
//                                delete[] thread_row_inds;
//                                delete resampling_prediction;
//
//#else
//
//                                shuffle(row_inds, data_height);
//
//                                    resampling_prediction.clear();
//                                    predict(modified_data->rows(row_inds, resampling_size), resampling_prediction);
//
//                                    bootstrapped_predictions.assign(m, 0, resampling_prediction->col_average(0));
//
//#endif
//
//                            } // no_cores
//
//                        } // m < params.twm_no_resamplings
//
//                        vec_of_two_way_predictions[index_two_way_model]->assign(k, l, bootstrapped_predictions.col_average(0));
//
//                        delete modified_data;
//
//                    } // l < params.twm_no_x_ticks
//
//                } // k < params.twm_no_x_ticks
//
//                std::cout << "Predictor ( " << i
//                          << " ) and Predictor ( " << j
//                          << " ) --> bootstraping completed..."
//                          << std::endl;
//
//            } // j < total_no_feature
//
//        } // i < total_no_feature
//
//    }
//
//    template <typename T>
//    Matrix<T> &AUTO_DBM<T>::predict_two_way_model(const Matrix<T> &data_x) {
//
//        if(prediction_two_way != nullptr) {
//
//            delete prediction_two_way;
//
//        }
//
//        int data_height = data_x.get_height();
//
//        #ifdef _DEBUG_MODEL
//            assert(data_x.get_width() == total_no_feature);
//        #endif
//
//        prediction_two_way = new Matrix<T>(data_height, 1, 0);
//
//        int index_two_way_model, m, n;
//
//        for(int i = 0; i < data_height; ++i) {
//
//            index_two_way_model = -1;
//
//            for(int k = 0; k < total_no_feature - 1; ++k) {
//
//                for(int l = k + 1; l < total_no_feature; ++l) {
//
//                    ++index_two_way_model;
//
//                    for(m = 0; m < params.twm_no_x_ticks; ++m) {
//
//                        if(data_x.get(i, k) < predictor_x_ticks->get(k, m)) break;
//
//                    }
//
//                    for(n = 0; n < params.twm_no_x_ticks; ++n) {
//
//                        if(data_x.get(i, l) < predictor_x_ticks->get(l, n)) break;
//
//                    }
//
//                    if(m == 0 && n == 0) {
//
//                        prediction_two_way->assign(i, 0, prediction_two_way->get(i, 0) +
//                                                         vec_of_two_way_predictions[index_two_way_model]->get(m, n));
//
//                    }
//                    else if(m == params.twm_no_x_ticks && n == params.twm_no_x_ticks) {
//
//                        prediction_two_way->assign(i, 0, prediction_two_way->get(i, 0) +
//                                                         vec_of_two_way_predictions[index_two_way_model]->get(m - 1, n - 1));
//
//                    }
//                    else if(m == 0) {
//
//                        prediction_two_way->assign(i, 0, prediction_two_way->get(i, 0) +
//                                                         (vec_of_two_way_predictions[index_two_way_model]->get(m, n) +
//                                                          vec_of_two_way_predictions[index_two_way_model]->get(m, n - 1)) / 2.0);
//
//                    }
//                    else if(m == params.twm_no_x_ticks) {
//
//                        prediction_two_way->assign(i, 0, prediction_two_way->get(i, 0) +
//                                                         (vec_of_two_way_predictions[index_two_way_model]->get(m - 1, n) +
//                                                          vec_of_two_way_predictions[index_two_way_model]->get(m - 1, n - 1)) / 2.0);
//
//                    }
//                    else if(n == 0) {
//
//                        prediction_two_way->assign(i, 0, prediction_two_way->get(i, 0) +
//                                                         (vec_of_two_way_predictions[index_two_way_model]->get(m, n) +
//                                                          vec_of_two_way_predictions[index_two_way_model]->get(m - 1, n)) / 2.0);
//
//                    }
//                    else if(n == params.twm_no_x_ticks) {
//
//                        prediction_two_way->assign(i, 0, prediction_two_way->get(i, 0) +
//                                                         (vec_of_two_way_predictions[index_two_way_model]->get(m, n - 1) +
//                                                          vec_of_two_way_predictions[index_two_way_model]->get(m - 1, n - 1)) / 2.0);
//
//                    }
//                    else {
//
//                        prediction_two_way->assign(i, 0, prediction_two_way->get(i, 0) +
//                                                         (vec_of_two_way_predictions[index_two_way_model]->get(m, n - 1) +
//                                                          vec_of_two_way_predictions[index_two_way_model]->get(m - 1, n - 1) +
//                                                          vec_of_two_way_predictions[index_two_way_model]->get(m, n) +
//                                                          vec_of_two_way_predictions[index_two_way_model]->get(m - 1, n)) / 4.0);
//
//                    }
//
//
//                } // l < total_no_feature
//
//            } //  k < total_no_feature - 1
//
//            prediction_two_way->assign(i, 0, prediction_two_way->get(i, 0) / no_two_way_models);
//
//        } // i < data_height
//
//        return *prediction_two_way;
//
//    }
//
//}

namespace dbm {

    template<typename T>
    void save_auto_dbm(const AUTO_DBM<T> *auto_dbm, std::ofstream &out) {

        out << auto_dbm->no_bunches_of_learners << ' '
            << auto_dbm->no_cores << ' '
            << auto_dbm->no_candidate_feature << ' '
            << auto_dbm->no_train_sample << ' '
            << auto_dbm->params.dbm_loss_function << ' '
            << auto_dbm->params.dbm_shrinkage << ' '
            << auto_dbm->total_no_feature << ' '
            << std::endl;

        for(int i = 0; i < auto_dbm->no_base_learners; ++i) {

            out << auto_dbm->names_base_learners[i] << ' ';

        }
        out << std::endl;

        for(int i = 0; i < auto_dbm->no_base_learners; ++i) {

            out << auto_dbm->portions_base_learners[i] << ' ';

        }
        out << std::endl;

        char type;
        for (int i = 0; i < (auto_dbm->no_bunches_of_learners - 1) * auto_dbm->no_cores + 1; ++i) {
            type = auto_dbm->learners[i]->get_type();
            switch (type) {

                case 'm': {
                    out << "== Mean " << i << " ==" << std::endl;
                    dbm::save_global_mean(dynamic_cast<Global_mean<T> *>(auto_dbm->learners[i]), out);
                    out << "== End of Mean " << i << " ==" << std::endl;
                    break;
                }

                case 't': {
                    out << "== Tree " << i << " ==" << std::endl;
                    dbm::save_tree_node(dynamic_cast<Tree_node<T> *>(auto_dbm->learners[i]), out);
                    out << "== End of Tree " << i << " ==" << std::endl;
                    break;
                }

                case 'l': {
                    out << "== LinReg " << i << " ==" << std::endl;
                    dbm::save_linear_regression(dynamic_cast<Linear_regression<T> *>(auto_dbm->learners[i]), out);
                    out << "== End of LinReg " << i << " ==" << std::endl;
                    break;
                }

                case 'k': {
                    out << "== Kmeans " << i << " ==" << std::endl;
                    dbm::save_kmeans2d(dynamic_cast<Kmeans2d<T> *>(auto_dbm->learners[i]), out);
                    out << "== End of Kmeans " << i << " ==" << std::endl;
                    break;
                }

                case 's': {
                    out << "== Splines " << i << " ==" << std::endl;
                    dbm::save_splines(dynamic_cast<Splines<T> *>(auto_dbm->learners[i]), out);
                    out << "== End of Splines " << i << " ==" << std::endl;
                    break;
                }


                case 'n': {
                    out << "== NN " << i << " ==" << std::endl;
                    dbm::save_neural_network(dynamic_cast<Neural_network<T> *>(auto_dbm->learners[i]), out);
                    out << "== End of NN " << i << " ==" << std::endl;
                    break;
                }

                case 'd': {
                    out << "== DPCS " << i << " ==" << std::endl;
                    dbm::save_dpc_stairs(dynamic_cast<DPC_stairs<T> *>(auto_dbm->learners[i]), out);
                    out << "== End of DPCS " << i << " ==" << std::endl;
                    break;
                }

                default: {
                    std::cout << "Wrong learner type: " << type <<  std::endl;
                    throw std::invalid_argument("Specified learner does not exist.");
                }

            }

        }

    }

    template<typename T>
    void AUTO_DBM<T>::save_auto_dbm_to(const std::string &file_name) {

        std::ofstream out(file_name);
        dbm::save_auto_dbm(this, out);

    }

    template<typename T>
    void AUTO_DBM<T>::load_auto_dbm_from(const std::string &file_name) {

        for (int i = 0; i < (no_bunches_of_learners - 1) * no_cores + 1; ++i) {
            delete learners[i];
        }

        delete prediction_train_data;
        prediction_train_data = nullptr;
        delete test_loss_record;
        test_loss_record = nullptr;
        delete pdp_result;
        pdp_result = nullptr;
        delete ss_result;
        ss_result = nullptr;

        std::ifstream in(file_name);

        std::string line;
        std::getline(in, line);

        std::string words[100];
        int count = split_into_words(line, words);

//        assert(count == 7);
        if(count != 7)
            std::cout << "Error in loading AUTO DBM: count is " << count << std::endl;

        no_bunches_of_learners = std::stoi(words[0]);
        no_cores = std::stoi(words[1]);

        total_no_feature = std::stoi(words[6]);
        no_candidate_feature = std::stoi(words[2]);
        no_train_sample = std::stoi(words[3]);

        set_loss_function_and_shrinkage(words[4].front(), T(std::stod(words[5])));

        line.clear();
        std::getline(in, line);
        count = split_into_words(line, words);
        #ifdef _DEBUG_MODEL
            assert(count == no_base_learners);
        #endif
        for(int i = 0; i < no_base_learners; ++i) {
            names_base_learners[i] = words[i].front();
        }

        line.clear();
        std::getline(in, line);
        count = split_into_words(line, words);
        #ifdef _DEBUG_MODEL
            assert(count == no_base_learners);
        #endif
        for(int i = 0; i < no_base_learners; ++i) {
            portions_base_learners[i] = (T)std::stod(words[i]);
        }

        Tree_node<T> *temp_tree_ptr;
        Global_mean<T> *temp_mean_ptr;
        Linear_regression<T> *temp_linear_regression_ptr;
        Neural_network<T> *temp_neural_network_ptr;
        Splines<T> *temp_splines_ptr;
        Kmeans2d<T> *temp_kmeans2d_ptr;
        DPC_stairs<T> *temp_dpc_stairs_ptr;

        char type;

        for (int i = 0; i < (no_bunches_of_learners - 1) * no_cores + 1; ++i) {

            line.clear();
            std::getline(in, line);

            split_into_words(line, words);

            type = words[1].front();
            switch (type) {

                case 'M': {
                    temp_mean_ptr = nullptr;
                    load_global_mean(in, temp_mean_ptr);
                    learners[i] = temp_mean_ptr;

                    // skip the end line
                    std::getline(in, line);

                    break;
                }

                case 'T': {
                    temp_tree_ptr = nullptr;
                    load_tree_node(in, temp_tree_ptr);
                    learners[i] = temp_tree_ptr;

                    // skip the end line
                    std::getline(in, line);

                    break;
                }

                case 'L': {
                    temp_linear_regression_ptr = nullptr;
                    load_linear_regression(in, temp_linear_regression_ptr);
                    learners[i] = temp_linear_regression_ptr;

                    // skip the end line
                    std::getline(in, line);

                    break;
                }

                case 'K': {
                    temp_kmeans2d_ptr = nullptr;
                    load_kmeans2d(in, temp_kmeans2d_ptr);
                    learners[i] = temp_kmeans2d_ptr;

                    // skip the end line
                    std::getline(in, line);

                    break;
                }

                case 'S': {
                    temp_splines_ptr = nullptr;
                    load_splines(in, temp_splines_ptr);
                    learners[i] = temp_splines_ptr;

                    // skip the end line
                    std::getline(in, line);

                    break;
                }

                case 'N': {
                    temp_neural_network_ptr = nullptr;
                    load_neural_network(in, temp_neural_network_ptr);
                    learners[i] = temp_neural_network_ptr;

                    // skip the end line
                    std::getline(in, line);

                    break;
                }

                case 'D': {
                    temp_dpc_stairs_ptr = nullptr;
                    load_dpc_stairs(in, temp_dpc_stairs_ptr);
                    learners[i] = temp_dpc_stairs_ptr;

                    // skip the end line
                    std::getline(in, line);

                    break;
                }

                default: {
                    std::cout << "Wrong learner type: " << type <<  std::endl;
                    throw std::invalid_argument("Specified learner does not exist.");
                }

            }

        }


    }

}

// explicit instantiation
namespace dbm {

    template void save_dbm<double>(const DBM<double> *dbm, std::ofstream &out);

    template void save_dbm<float>(const DBM<float> *dbm, std::ofstream &out);

    template void load_dbm<double>(std::ifstream &in, DBM<double> *&dbm);

    template void load_dbm<float>(std::ifstream &in, DBM<float> *&dbm);

    template void save_auto_dbm<double>(const AUTO_DBM<double> *auto_dbm, std::ofstream &out);

    template void save_auto_dbm<float>(const AUTO_DBM<float> *auto_dbm, std::ofstream &out);

}


#include "partial_dependence_plot.inc"

#include "interact.inc"






