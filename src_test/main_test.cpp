#include <iostream>

#include "data_set.h"
#include "base_learner.h"
#include "base_learner_trainer.h"
#include "model.h"

using namespace std;

void train_test_save_load_auto_dbm();

void train_test_save_load_dbm();

void train_test_save_load_nn();

void train_test_save_load_dpcs();

void prepare_data();

int main() {

//    train_test_save_load_dbm();
    train_test_save_load_auto_dbm();

    return 0;

}

void prepare_data() {
    string file_name = "train_data.txt";
    dbm::make_data<float>(file_name, 10000, 30, 'b');
}

void train_test_save_load_auto_dbm() {
    int n_samples = 136573, n_features = 50, n_width = 51;
    dbm::Matrix<float> train_data(n_samples, n_width, "numerai_training_data.csv", ',');

    dbm::add_nans_to_mat(train_data, 5, 1);

    int *col_inds = new int[n_features];

    for (int i = 0; i < n_features; ++i)
        col_inds[i] = i;

//    dbm::Matrix<float> train_x = train_data.cols(col_inds, n_features);
//    dbm::Matrix<float> train_y = train_data.col(n_features);

    int no_rows = 136573;
    int *row_inds = new int[no_rows];
    for(int i = 0; i < no_rows; ++i)
        row_inds[i] = i;

    dbm::Matrix<float> train_x = train_data.submatrix(row_inds, no_rows, col_inds, n_features);
    dbm::Matrix<float> train_y = train_data.col(n_features).rows(row_inds, no_rows);

    // ================

    dbm::Matrix<float> monotonic_constraints(n_features, 1, 0);
//    for(int i = 0; i < n_features; ++i)
//        monotonic_constraints.assign(i, 0, 1);

    // ================

    dbm::Data_set<float> data_set(train_x, train_y, 0.1, 1);
    dbm::Matrix<float> train_prediction(data_set.get_train_x().get_height(), 1, 0);
    dbm::Matrix<float> test_prediction(data_set.get_test_x().get_height(), 1, 0);
    dbm::Matrix<float> re_test_prediction(data_set.get_test_x().get_height(), 1, 0);

    // ================
//    string param_string = "dbm_no_bunches_of_learners 20000 dbm_no_cores 3 dbm_loss_function b "
//            "dbm_portion_train_sample 0.3 dbm_no_candidate_feature 30 dbm_shrinkage 0.001";
    string param_string = "dbm_no_bunches_of_learners 30 dbm_no_cores 3 dbm_loss_function b "
            "dbm_portion_train_sample 0.3 dbm_no_candidate_feature 30 dbm_shrinkage 0.1 "
            "dbm_random_seed -1 ";

    dbm::Params params = dbm::set_params(param_string);
    dbm::AUTO_DBM<float> auto_dbm(params);

//    auto_dbm.train(data_set);
    auto_dbm.train(data_set, monotonic_constraints);

    auto_dbm.predict(data_set.get_train_x(), train_prediction);
    auto_dbm.predict(data_set.get_test_x(), test_prediction);

    dbm::Matrix<float> pred = auto_dbm.predict(data_set.get_test_x());
    pred.print_to_file("pred.txt");

    dbm::Matrix<float> pdp = auto_dbm.partial_dependence_plot(data_set.get_train_x(), 6);
//    pdp.print_to_file("pdp.txt");

//    dbm::Matrix<float> ss = auto_dbm.statistical_significance(data_set.get_train_x());
//    ss.print_to_file("ss.txt");

    auto_dbm.save_auto_dbm_to("dbm.txt");

    auto_dbm.save_perf_to("performance.txt");

    auto_dbm.calibrate_plot(data_set.get_test_y(), pred, 20, "cal_plot.txt");

//    const int predictor_ind = 6;
//    float h_value = auto_dbm.interact(data_set.get_train_x(), &predictor_ind, n_features);
//    cout << "H Value: " << h_value << endl;

    // ===================

    dbm::AUTO_DBM<float> re_auto_dbm(params);

    re_auto_dbm.load_auto_dbm_from("dbm.txt");

    re_auto_dbm.save_auto_dbm_to("re_dbm.txt");

    re_auto_dbm.predict(data_set.get_test_x(), re_test_prediction);

//    dbm::Matrix<float> temp = dbm::hori_merge(*auto_dbm.get_prediction_on_train_data(), dbm::hori_merge(train_prediction, twm_pred));
    dbm::Matrix<float> temp = dbm::hori_merge(*auto_dbm.get_prediction_on_train_data(), train_prediction);
//    dbm::Matrix<float> check = dbm::hori_merge(dbm::hori_merge(data_set.get_train_x(), data_set.get_train_y()), temp);
    dbm::Matrix<float> check = dbm::hori_merge(data_set.get_train_y(), temp);
    check.print_to_file("check_train_and_pred.txt");

    dbm::Matrix<float> combined = dbm::hori_merge(test_prediction, re_test_prediction);
//    dbm::Matrix<float> result = dbm::hori_merge(dbm::hori_merge(data_set.get_test_x(), data_set.get_test_y()), combined);
    dbm::Matrix<float> result = dbm::hori_merge(data_set.get_test_y(), combined);
    result.print_to_file("whole_result.txt");

    dbm::Matrix<float> oos_data(13512, 50, "numerai_tournament_data.csv", ',');
    dbm::Matrix<float> oos_prediction = oos_data.col(0);
    oos_prediction.clear();
    auto_dbm.predict(oos_data, oos_prediction);


    oos_prediction.print_to_file("oos_prediction.txt");

    delete[] col_inds;
}

void train_test_save_load_dbm() {
    int n_samples = 136573, n_features = 50, n_width = 51;

    dbm::Matrix<float> train_data(n_samples, n_width, "numerai_training_data.csv", ',');

    dbm::add_nans_to_mat(train_data, 2, 1);

    int *col_inds = new int[n_features];

    for (int i = 0; i < n_features; ++i)
        col_inds[i] = i;

//    dbm::Matrix<float> train_x = train_data.cols(col_inds, n_features);
//    dbm::Matrix<float> train_y = train_data.col(n_features);

    int no_rows = 136573;
    int *row_inds = new int[no_rows];
    for(int i = 0; i < no_rows; ++i)
        row_inds[i] = i;

    dbm::Matrix<float> train_x = train_data.submatrix(row_inds, no_rows, col_inds, n_features);
    dbm::Matrix<float> train_y = train_data.col(n_features).rows(row_inds, no_rows);

    // ================

    dbm::Matrix<float> monotonic_constraints(n_features, 1, 0);
//    for(int i = 0; i < n_features; ++i)
//        monotonic_constraints.assign(i, 0, 1);

    // ================

    dbm::Data_set<float> data_set(train_x, train_y, 0.1, 1);
    dbm::Matrix<float> train_prediction(data_set.get_train_x().get_height(), 1, 0);
    dbm::Matrix<float> test_prediction(data_set.get_test_x().get_height(), 1, 0);
    dbm::Matrix<float> re_test_prediction(data_set.get_test_x().get_height(), 1, 0);

    // ================

//    string param_string = "dbm_no_bunches_of_learners 25000 dbm_no_cores 3 dbm_loss_function n "
//            "dbm_portion_train_sample 0.3 dbm_no_candidate_feature 30 dbm_shrinkage 0.0005 "
//            "dbm_portion_for_trees 0 dbm_portion_for_lr 0 dbm_portion_for_s 1 "
//            "dbm_portion_for_k 0 dbm_portion_for_d 0 ";
    string param_string = "dbm_no_bunches_of_learners 31 dbm_no_cores 3 dbm_loss_function b "
            "dbm_portion_train_sample 0.3 dbm_no_candidate_feature 30 dbm_shrinkage 0.1 "
            "dbm_portion_for_trees 0.2 dbm_portion_for_lr 0.2 dbm_portion_for_s 0.2 "
            "dbm_portion_for_k 0.2 dbm_portion_for_d 0.2 dbm_random_seed 1 ";

    dbm::Params params = dbm::set_params(param_string);
    dbm::DBM<float> dbm(params);

//    dbm.train(data_set);
    dbm.train(data_set, monotonic_constraints);

//    dbm.train_two_way_model(data_set.get_train_x());
//    dbm::Matrix<float> twm_pred = dbm.predict_two_way_model(data_set.get_train_x());

    dbm.predict(data_set.get_train_x(), train_prediction);
    dbm.predict(data_set.get_test_x(), test_prediction);

    dbm::Matrix<float> pred = dbm.predict(data_set.get_test_x());
    pred.print_to_file("pred.txt");

//    dbm::Matrix<float> pdp = dbm.partial_dependence_plot(data_set.get_train_x(), 6);
//    pdp.print_to_file("pdp.txt");
//
//    dbm::Matrix<float> ss = dbm.statistical_significance(data_set.get_train_x());
//    ss.print_to_file("ss.txt");

    dbm.save_dbm_to("dbm.txt");

//    {
//        ofstream out("dbm.txt");
//        dbm::save_dbm(&dbm, out);
//    }

    dbm.save_perf_to("performance.txt");

    dbm.calibrate_plot(data_set.get_test_y(), pred, 20, "cal_plot.txt");

//    const int predictor_ind = 6;
//    float h_value = dbm.interact(data_set.get_train_x(), &predictor_ind, n_features);
//    cout << "H Value: " << h_value << endl;

    // ===================

    dbm::DBM<float> re_dbm(params);

    re_dbm.load_dbm_from("dbm.txt");

//    {
//        ifstream in("dbm.txt");
//        dbm::load_dbm(in, re_dbm);
//    }

    re_dbm.save_dbm_to("re_dbm.txt");

//    {
//        ofstream out("re_dbm.txt");
//        dbm::save_dbm(re_dbm, out);
//    }

    re_dbm.predict(data_set.get_test_x(), re_test_prediction);

//    dbm::Matrix<float> temp = dbm::hori_merge(*dbm.get_prediction_on_train_data(), dbm::hori_merge(train_prediction, twm_pred));
    dbm::Matrix<float> temp = dbm::hori_merge(*dbm.get_prediction_on_train_data(), train_prediction);
//    dbm::Matrix<float> check = dbm::hori_merge(dbm::hori_merge(data_set.get_train_x(), data_set.get_train_y()), temp);
    dbm::Matrix<float> check = dbm::hori_merge(data_set.get_train_y(), temp);
    check.print_to_file("check_train_and_pred.txt");

    dbm::Matrix<float> combined = dbm::hori_merge(test_prediction, re_test_prediction);
//    dbm::Matrix<float> result = dbm::hori_merge(dbm::hori_merge(data_set.get_test_x(), data_set.get_test_y()), combined);
    dbm::Matrix<float> result = dbm::hori_merge(data_set.get_test_y(), combined);
    result.print_to_file("whole_result.txt");

    dbm::Matrix<float> oos_data(13512, 50, "numerai_tournament_data.csv", ',');
    dbm::Matrix<float> oos_prediction = oos_data.col(0);
    oos_prediction.clear();
    dbm.predict(oos_data, oos_prediction);

    oos_prediction.print_to_file("oos_prediction.txt");

    delete[] col_inds;
    delete[] row_inds;
}

