//
// Created by xixuan on 10/10/16.
//

#ifndef DBM_CODE_MODEL_H
#define DBM_CODE_MODEL_H

//#ifndef _DEBUG_MODEL
//#define _DEBUG_MODEL
//#endif

#include "matrix.h"
#include "data_set.h"
#include "base_learner.h"
#include "base_learner_trainer.h"
#include "tools.h"

#include <string>
#include <fstream>
#include <cmath>
#include <map>
#include <algorithm>

namespace dbm {

    template<typename T>
    class DBM;

    template<typename T>
    void save_dbm(const DBM<T> *dbm,
                  std::ofstream &out);

    template<typename T>
    void load_dbm(std::ifstream &in,
                  DBM<T> *&dbm);

    template<typename T>
    class AUTO_DBM;

    template<typename T>
    void save_auto_dbm(const AUTO_DBM<T> *auto_dbm,
                       std::ofstream &out);

    template<typename T>
    class Regressor {
    public:

        virtual void train(const Data_set<T> &data_set,
                           const Matrix<T> &input_monotonic_constraints) = 0;

        virtual void train(const Data_set<T> &data_set) = 0;

        virtual void predict(const Matrix<T> &data_x,
                             Matrix<T> &predict_y) = 0;

        virtual Matrix<T> &predict(const Matrix<T> &data_x) = 0;

    };

    template<typename T>
    class DBM : public Regressor<T> {
    private:
        int no_bunches_of_learners;
        int no_cores;

        int total_no_feature;
        int no_candidate_feature;
        int no_train_sample;

        Base_learner<T> **learners = nullptr;

//        Tree_trainer<T> *tree_trainer = nullptr;
        Fast_tree_trainer<T> *tree_trainer = nullptr;
        Mean_trainer<T> *mean_trainer = nullptr;
        Linear_regression_trainer<T> *linear_regression_trainer = nullptr;
        Neural_network_trainer<T> *neural_network_trainer = nullptr;
        Splines_trainer<T> *splines_trainer = nullptr;
        Kmeans2d_trainer<T> *kmeans2d_trainer = nullptr;
        DPC_stairs_trainer<T> *dpc_stairs_trainer = nullptr;

        Params params;
        Loss_function<T> loss_function;

        Matrix<T> *prediction_train_data = nullptr;

        T *test_loss_record = nullptr;
        T *train_loss_record = nullptr;

        Matrix<T> *pdp_result = nullptr;
        Matrix<T> *ss_result = nullptr;

        Matrix<T> *prediction = nullptr;

        int no_two_way_models;
        Matrix<T> **vec_of_two_way_predictions = nullptr;
        Matrix<T> *prediction_two_way = nullptr;
        Matrix<T> *predictor_x_ticks = nullptr;

        Matrix<T> *mat_plot_dat = nullptr;

    public:

        DBM(int no_bunches_of_learners,
            int no_cores,
            int no_candidate_feature,
            int no_train_sample,
            int total_no_feature);

        DBM(const Params &params);

        ~DBM();

        // comments on monotonic_constraints
        // 1: positive relationship; 0: anything; -1: negative relationship
        // by only allowing 1, 0, -1, we could be able to check if the length is correct in some sense

        void train(const Data_set<T> &data_set,
                   const Matrix<T> &input_monotonic_constraints);

        void train(const Data_set<T> &data_set);

        void predict(const Matrix<T> &data_x,
                     Matrix<T> &predict_y);

        Matrix<T> &predict(const Matrix<T> &data_x);

        /*
         *  TOOLS
         */

        Matrix<T> *get_prediction_on_train_data() const;
        T *get_test_loss() const;

        void set_loss_function_and_shrinkage(const char &type, const T &shrinkage);

        Matrix<T> &partial_dependence_plot(const Matrix<T> &data,
                                          const int &predictor_ind);

        Matrix<T> &partial_dependence_plot(const Matrix<T> &data,
                                          const int &predictor_ind,
                                          const T &x_tick_min,
                                          const T &x_tick_max);

        Matrix<T> &partial_dependence_plot(const Matrix<T> &data,
                                           const int *predictor_ind,
                                           int no_predictor);

        Matrix<T> &statistical_significance(const Matrix<T> &data);

        Matrix<T> &calibrate_plot(const Matrix<T> &observation,
                                  const Matrix<T> &prediction,
                                  int resolution,
                                  const std::string& file_name = "");

        void save_perf_to(const std::string &file_name);

        T interact(const Matrix<T>& data,
                   const int* predictor_ind,
                   int no_predictor);
        /*
         * two-way models
         */

//        void train_two_way_model(const Matrix<T> &data);
//
//        Matrix<T> &predict_two_way_model(const Matrix<T> &data_x);

        /*
         *  IO
         */
        friend void save_dbm<>(const DBM *dbm,
                               std::ofstream &out);
        friend void load_dbm<>(std::ifstream &in,
                               DBM *&dbm);

        void save_dbm_to(const std::string &file_name);
        void load_dbm_from(const std::string &file_name);

    };

    template<typename T>
    class AUTO_DBM : public Regressor<T> {
    private:
        int no_bunches_of_learners;
        int no_cores;

        int total_no_feature;
        int no_candidate_feature;
        int no_train_sample;

        Base_learner<T> **learners = nullptr;

        Fast_tree_trainer<T> *tree_trainer = nullptr;
        Mean_trainer<T> *mean_trainer = nullptr;
        Linear_regression_trainer<T> *linear_regression_trainer = nullptr;
        Neural_network_trainer<T> *neural_network_trainer = nullptr;
        Splines_trainer<T> *splines_trainer = nullptr;
        Kmeans2d_trainer<T> *kmeans2d_trainer = nullptr;
        DPC_stairs_trainer<T> *dpc_stairs_trainer = nullptr;

        Params params;
        Loss_function<T> loss_function;

        Matrix<T> *prediction_train_data = nullptr;

        T *test_loss_record = nullptr;
        T *train_loss_record = nullptr;

        Matrix<T> *pdp_result = nullptr;
        Matrix<T> *ss_result = nullptr;

        Matrix<T> *prediction = nullptr;

        static const int no_base_learners = 5;
        char *names_base_learners = nullptr;
        T loss_on_train_set;

        T *portions_base_learners = nullptr;
        T *new_losses_for_base_learners = nullptr;

        int no_two_way_models;
        Matrix<T> **vec_of_two_way_predictions = nullptr;
        Matrix<T> *prediction_two_way = nullptr;
        Matrix<T> *predictor_x_ticks = nullptr;

        Matrix<T> *mat_plot_dat = nullptr;

        int base_learner_choose(const Matrix<T> &train_x,
                                const Matrix<T> &train_y,
                                const Matrix<T> &test_x,
                                const Matrix<T> &test_y,
                                const Matrix<T> &ind_delta,
                                const Matrix<T> &prediction_test_data,
                                const Matrix<T> &input_monotonic_constraints,
                                const int &bunch_no,
                                const double &type_choose);

        void update_new_losses_for_bl(const Matrix<T> &train_y, int bl_no);

    public:

        AUTO_DBM(const Params &params);

        ~AUTO_DBM();

        // comments on monotonic_constraints
        // 1: positive relationship; 0: anything; -1: negative relationship
        // by only allowing 1, 0, -1, we could be able to check if the length is correct in some sense

        void train(const Data_set<T> &data_set,
                   const Matrix<T> &input_monotonic_constraints);

        void train(const Data_set<T> &data_set);

        void predict(const Matrix<T> &data_x,
                     Matrix<T> &predict_y);

        Matrix<T> &predict(const Matrix<T> &data_x);

        /*
         *  TOOLS
         */

        Matrix<T> *get_prediction_on_train_data() const;
        T *get_test_loss() const;

        void set_loss_function_and_shrinkage(const char &type, const T &shrinkage);

        Matrix<T> &partial_dependence_plot(const Matrix<T> &data,
                                           const int &predictor_ind);

        Matrix<T> &partial_dependence_plot(const Matrix<T> &data,
                                           const int &predictor_ind,
                                           const T &x_tick_min,
                                           const T &x_tick_max);

        Matrix<T> &statistical_significance(const Matrix<T> &data);

        Matrix<T> &calibrate_plot(const Matrix<T> &observation,
                                  const Matrix<T> &prediction,
                                  int resolution,
                                  const std::string& file_name = "");

        void save_perf_to(const std::string &file_name);

        T interact(const Matrix<T>& data,
                   const int* predictor_ind,
                   int no_predictor);

        /*
         * two-way models
         */

//        void train_two_way_model(const Matrix<T> &data);
//
//        Matrix<T> &predict_two_way_model(const Matrix<T> &data_x);

        /*
         *  IO
         */
        friend void save_auto_dbm<>(const AUTO_DBM *auto_dbm,
                                    std::ofstream &out);

        void save_auto_dbm_to(const std::string &file_name);
        void load_auto_dbm_from(const std::string &file_name);

    };

}


#endif //DBM_CODE_MODEL_H




