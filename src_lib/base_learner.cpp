//
// Created by xixuan on 10/10/16.
//

#include "base_learner.h"
#include "tools.h"

#include <cassert>
#include <cmath>
#include <iostream>
#include <fstream>
#include <sstream>

namespace dbm {

    template
    class Tree_node<float>;

    template
    class Tree_node<double>;

    template
    class Linear_regression<double>;

    template
    class Linear_regression<float>;

    template
    class DPC_stairs<double>;

    template
    class DPC_stairs<float>;

    template
    class Kmeans2d<double>;

    template
    class Kmeans2d<float>;

    template
    class Global_mean<float>;

    template
    class Global_mean<double>;

    template
    class Neural_network<float>;

    template
    class Neural_network<double>;

    template
    class Splines<double>;

    template
    class Splines<float>;

}

namespace dbm {

    template
    class Tree_info<double>;

    template
    class Tree_info<float>;

}


// global mean
namespace dbm {

    template <typename T>
    Global_mean<T>::Global_mean() : Base_learner<T>('m') {}

    template <typename T>
    Global_mean<T>::~Global_mean() {};

    template <typename T>
    T Global_mean<T>::predict_for_row(const Matrix<T> &data,
                                      int row_ind) {
        return mean;
    }

    template <typename T>
    void Global_mean<T>::predict(const Matrix<T> &data_x,
                                 Matrix<T> &prediction,
                                 const T shrinkage,
                                 const int *row_inds,
                                 int no_rows) {

        if (row_inds == NULL) {
            int data_height = data_x.get_height();
            #ifdef _DEBUG_BASE_LEARNER
            assert(data_height == prediction.get_height() && prediction.get_width() == 1);
            #endif
            T predicted_value;
            for (int i = 0; i < data_height; ++i) {
                predicted_value = prediction.get(i, 0) +
                        shrinkage * predict_for_row(data_x, i);
                prediction.assign(i, 0, predicted_value);
            }
        } else {
            #ifdef _DEBUG_BASE_LEARNER
            assert(no_rows > 0 && prediction.get_height() == 1);
            #endif
            T predicted_value;
            for (int i = 0; i < no_rows; ++i) {
                predicted_value = prediction.get(row_inds[i], 0) +
                        shrinkage * predict_for_row(data_x, row_inds[i]);
                prediction.assign(row_inds[i], 0, predicted_value);
            }
        }

    }

}

// neural networks
namespace dbm {

    template <typename T>
    Neural_network<T>::Neural_network(int no_predictors,
                                      int no_hidden_neurons,
                                      char loss_type) :
            Base_learner<T>('n'),
            no_predictors(no_predictors),
            no_hidden_neurons(no_hidden_neurons),
            loss_type(loss_type){

        col_inds = new int[no_predictors];

        input_weight = new Matrix<T>(no_hidden_neurons, no_predictors + 1);
        hidden_weight = new Matrix<T>(1, no_hidden_neurons + 1);

    }

    template <typename T>
    Neural_network<T>::~Neural_network<T>() {
        delete input_weight;
        delete hidden_weight;
        delete[] col_inds;
        input_weight = nullptr, hidden_weight = nullptr, col_inds = nullptr;
    }

    template <typename T>
    inline T Neural_network<T>::activation(const T &input) {
        return 1 / (1 + std::exp( - input));
    }

    template <typename T>
    void Neural_network<T>::forward(const Matrix<T> &input_output, Matrix<T> &hidden_output, T &output_output) {
        double ip = 0;
        for(int i = 0; i < no_hidden_neurons; ++i) {
            for(int j = 0; j < no_predictors + 1; ++j)
                ip += input_weight->get(i, j) * input_output.get(j, 0);
            hidden_output.assign(i, 0, activation(ip));
        }
        hidden_output.assign(no_hidden_neurons, 0, 1);

        output_output = 0;
        for(int j = 0; j < no_hidden_neurons + 1; ++j)
            output_output += hidden_weight->get(0, j) * hidden_output.get(j, 0);
    }

    template <typename T>
    T Neural_network<T>::predict_for_row(const Matrix<T> &data,
                                         int row_ind) {

        Matrix<T> input_output(no_predictors + 1, 1, 0);
        Matrix<T> hidden_output(no_hidden_neurons + 1, 1, 0);
        T output_output;

        for(int i = 0; i < no_predictors; ++i)
            input_output.assign(i, 0, data.get(row_ind, col_inds[i]));
        input_output.assign(no_predictors, 0, 1);
        forward(input_output, hidden_output, output_output);

        if(std::isnan(output_output))
            return 0.0;

        /*
         * @TODO check with Simon what to do if base learner predicts negative values
         */
        switch (loss_type) {
            case 'n':
                return output_output;
            case 'p':
                return std::log(output_output > TOLERANCE ?  output_output : TOLERANCE);
            case 'b':
                return output_output;
            case 't':
                return std::log(output_output > TOLERANCE ?  output_output : TOLERANCE);
            default:
                throw std::invalid_argument("Specified distribution does not exist.");
        }
    }

    template <typename T>
    void Neural_network<T>::predict(const Matrix<T> &data_x,
                                    Matrix<T> &prediction,
                                    const T shrinkage,
                                    const int *row_inds,
                                    int no_rows) {

        if (row_inds == NULL) {
            int data_height = data_x.get_height();
            #ifdef _DEBUG_BASE_LEARNER
            assert(data_height == prediction.get_height() && prediction.get_width() == 1);
            #endif
            T predicted_value;
            for (int i = 0; i < data_height; ++i) {
                predicted_value = prediction.get(i, 0) +
                        shrinkage * predict_for_row(data_x, i);
                prediction.assign(i, 0, predicted_value);
            }
        } else {
            #ifdef _DEBUG_BASE_LEARNER
            assert(no_rows > 0 && prediction.get_height() == 1);
            #endif
            T predicted_value;
            for (int i = 0; i < no_rows; ++i) {
                predicted_value = prediction.get(row_inds[i], 0) +
                        shrinkage * predict_for_row(data_x, row_inds[i]);
                prediction.assign(row_inds[i], 0, predicted_value);
            }
        }

    }

}

// kmeans2d
namespace dbm {

    template <typename T>
    Kmeans2d<T>::Kmeans2d(int no_centroids,
                          char loss_type) :
            Base_learner<T>('k'),
            no_centroids(no_centroids),
            loss_type(loss_type){

        col_inds = new int[no_predictors];
        col_inds[0] = -1;
        col_inds[1] = -1;

        centroids = new T*[no_centroids];
        for(int i = 0; i < no_centroids; ++i)
            centroids[i] = new T[no_predictors];

        predictions = new T[no_centroids];
    }

    template <typename T>
    Kmeans2d<T>::~Kmeans2d() {
        delete[] col_inds;
        for(int i = 0; i < no_centroids; ++i)
            delete[] centroids[i];
        delete[] centroids;
        delete[] predictions;
        centroids = nullptr, predictions = nullptr;
    };

    template <typename T>
    T Kmeans2d<T>::distance(const Matrix<T> &data,
                          const int &row_ind,
                          const int &centroid_ind) {

        double result = 0;
        for(int i = 0; i < no_predictors; ++i)
            result += (centroids[centroid_ind][i] - data.get(row_ind, col_inds[i])) *
                    (centroids[centroid_ind][i] - data.get(row_ind, col_inds[i]));
        return std::sqrt(result);

    }

    template <typename T>
    T Kmeans2d<T>::predict_for_row(const Matrix<T> &data,
                                 int row_ind) {
        T lowest_dist = std::numeric_limits<T>::max(),
                result = NAN, dist;
        for(int i = 0; i < no_centroids; ++i) {
            dist = distance(data, row_ind, i);
            if(dist < lowest_dist) {
                lowest_dist = dist;
                result = predictions[i];
            }
        }

        if(std::isnan(result))
            return 0.0;

        switch (loss_type) {
            case 'n':
                return result;
            case 'p':
                return std::log(result > TOLERANCE ?  result : TOLERANCE);
            case 'b':
                return result;
            case 't':
                return std::log(result > TOLERANCE ?  result : TOLERANCE);
            default:
                throw std::invalid_argument("Specified distribution does not exist.");
        }
    }

    template <typename T>
    void Kmeans2d<T>::predict(const Matrix<T> &data_x,
                            Matrix<T> &prediction,
                            const T shrinkage,
                            const int *row_inds,
                            int no_rows) {

        if (row_inds == NULL) {
            int data_height = data_x.get_height();
            #ifdef _DEBUG_BASE_LEARNER
             assert(data_height == prediction.get_height() && prediction.get_width() == 1);
            #endif
            T predicted_value;
            for (int i = 0; i < data_height; ++i) {
                predicted_value = prediction.get(i, 0) +
                                  shrinkage * predict_for_row(data_x, i);
                prediction.assign(i, 0, predicted_value);
            }
        } else {
            #ifdef _DEBUG_BASE_LEARNER
                assert(no_rows > 0 && prediction.get_height() == 1);
            #endif
            T predicted_value;
            for (int i = 0; i < no_rows; ++i) {
                predicted_value = prediction.get(row_inds[i], 0) +
                                  shrinkage * predict_for_row(data_x, row_inds[i]);
                prediction.assign(row_inds[i], 0, predicted_value);
            }
        }

    }

}

// splines
namespace dbm {

    template <typename T>
    inline T Splines<T>::x_left_hinge(T &x, T &y, T &knot) {
        return std::max(T(0), hinge_coefficient * (knot - x));
    }

    template <typename T>
    inline T Splines<T>::x_right_hinge(T &x, T &y, T &knot) {
        return std::max(T(0), hinge_coefficient * (x - knot));
    }

    template <typename T>
    inline T Splines<T>::y_left_hinge(T &x, T &y, T &knot) {
        return std::max(T(0), hinge_coefficient * (knot - y));
    }

    template <typename T>
    inline T Splines<T>::y_right_hinge(T &x, T &y, T &knot) {
        return std::max(T(0), hinge_coefficient * (y - knot));
    }

    template <typename T>
    Splines<T>::Splines(int no_knots,
                        char loss_type,
                        T hinge_coefficient) :
            Base_learner<T>('s'),
            no_knots(no_knots),
            loss_type(loss_type),
            hinge_coefficient(hinge_coefficient){

        col_inds = new int[no_predictors];
        col_inds[0] = -1;
        col_inds[0] = -1;

        x_knots = new T[no_knots];

        x_left_coefs = new T[no_knots];
        x_right_coefs = new T[no_knots];

        y_knots = new T[no_knots];

        y_left_coefs = new T[no_knots];
        y_right_coefs = new T[no_knots];

    }

    template <typename T>
    Splines<T>::~Splines() {
        delete[] col_inds;
        delete[] x_knots;
        delete[] x_left_coefs;
        delete[] x_right_coefs;
        delete[] y_knots;
        delete[] y_left_coefs;
        delete[] y_right_coefs;
        x_knots = nullptr, x_left_coefs = nullptr, x_right_coefs = nullptr,
                y_knots = nullptr, y_left_coefs = nullptr, y_right_coefs = nullptr;
    };

    template <typename T>
    T Splines<T>::predict_for_row(const Matrix<T> &data,
                                  int row_ind) {
        double result = 0;
        T x = data.get(row_ind, col_inds[0]), y = data.get(row_ind, col_inds[1]);
        for(int i = 0; i < no_knots; ++i) {
            result += x_left_hinge(x, y, x_knots[i]) * x_left_coefs[i];
            result += x_right_hinge(x, y, x_knots[i]) * x_right_coefs[i];
            result += y_left_hinge(x, y, y_knots[i]) * y_left_coefs[i];
            result += y_right_hinge(x, y, y_knots[i]) * y_right_coefs[i];
        }

        if(std::isnan(result))
            return 0.0;

        switch (loss_type) {
            case 'n':
                return result;
            case 'p':
                return std::log(result > TOLERANCE ?  result : TOLERANCE);
            case 'b':
                return result;
            case 't':
                return std::log(result > TOLERANCE ?  result : TOLERANCE);
            default:
                throw std::invalid_argument("Specified distribution does not exist.");
        }
    }

    template <typename T>
    void Splines<T>::predict(const Matrix<T> &data_x,
                             Matrix<T> &prediction,
                             const T shrinkage,
                             const int *row_inds,
                             int no_rows) {

        if (row_inds == NULL) {
            int data_height = data_x.get_height();
            #ifdef _DEBUG_BASE_LEARNER
                assert(data_height == prediction.get_height() && prediction.get_width() == 1);
            #endif
            T predicted_value;
            for (int i = 0; i < data_height; ++i) {
                predicted_value = prediction.get(i, 0) +
                        shrinkage * predict_for_row(data_x, i);
                prediction.assign(i, 0, predicted_value);
            }
        } else {
            #ifdef _DEBUG_BASE_LEARNER
                assert(no_rows > 0 && prediction.get_height() == 1);
            #endif
            T predicted_value;
            for (int i = 0; i < no_rows; ++i) {
                predicted_value = prediction.get(row_inds[i], 0) +
                        shrinkage * predict_for_row(data_x, row_inds[i]);
                prediction.assign(row_inds[i], 0, predicted_value);
            }
        }

    }

}

// linear regression
namespace dbm {

    template <typename T>
    Linear_regression<T>::Linear_regression(int no_predictors,
                                            char loss_type) :
            Base_learner<T>('l'),
            no_predictors(no_predictors),
            loss_type(loss_type) {
        col_inds = new int[no_predictors];
        coefs_no_intercept = new T[no_predictors];
    }

    template <typename T>
    Linear_regression<T>::~Linear_regression() {
        delete[] col_inds;
        delete[] coefs_no_intercept;
        col_inds = nullptr, coefs_no_intercept = nullptr;
    };

    template <typename T>
    T Linear_regression<T>::predict_for_row(const Matrix<T> &data,
                                            int row_ind) {
        double result = 0;
        for(int i = 0; i < no_predictors; ++i) {
            result += data.get(row_ind, col_inds[i]) * coefs_no_intercept[i];
        }
        result += intercept;

        if(std::isnan(result))
            return 0.0;

        switch (loss_type) {
            case 'n':
                return result;
            case 'p':
                return std::log(result > TOLERANCE ?  result : TOLERANCE);
            case 'b':
                return result;
            case 't':
                return std::log(result > TOLERANCE ?  result : TOLERANCE);
            default:
                throw std::invalid_argument("Specified distribution does not exist.");
        }
    }

    template <typename T>
    void Linear_regression<T>::predict(const Matrix<T> &data_x,
                                       Matrix<T> &prediction,
                                       const T shrinkage,
                                       const int *row_inds,
                                       int no_rows) {

        if (row_inds == NULL) {
            int data_height = data_x.get_height();
            #ifdef _DEBUG_BASE_LEARNER
            assert(data_height == prediction.get_height() && prediction.get_width() == 1);
            #endif
            T predicted_value;
            for (int i = 0; i < data_height; ++i) {
                predicted_value = prediction.get(i, 0) +
                        shrinkage * predict_for_row(data_x, i);
                prediction.assign(i, 0, predicted_value);
            }
        } else {
            #ifdef _DEBUG_BASE_LEARNER
            assert(no_rows > 0 && prediction.get_height() == 1);
            #endif
            T predicted_value;
            for (int i = 0; i < no_rows; ++i) {
                predicted_value = prediction.get(row_inds[i], 0) +
                        shrinkage * predict_for_row(data_x, row_inds[i]);
                prediction.assign(row_inds[i], 0, predicted_value);
            }
        }

    }

}

// dpc stairs
namespace dbm {

    template <typename T>
    DPC_stairs<T>::DPC_stairs(int no_predictors,
                              char loss_type,
                              int no_ticks) :
            Base_learner<T>('d'),
            no_predictors(no_predictors),
            loss_type(loss_type),
            no_ticks(no_ticks){

        col_inds = new int[no_predictors];

        coefs = new T[no_predictors];
        ticks = new T[no_ticks];
        predictions = new T[no_ticks + 1];

    }

    template <typename T>
    DPC_stairs<T>::~DPC_stairs() {
        delete[] col_inds;
        delete[] coefs;
        delete[] ticks;
        delete[] predictions;
    };

    template <typename T>
    T DPC_stairs<T>::predict_for_row(const Matrix<T> &data,
                                            int row_ind) {
        double dpc_val = 0;
        for(int i = 0; i < no_predictors; ++i) {
            dpc_val += coefs[i] * data.get(row_ind, col_inds[i]);
        }

        if(std::isnan(dpc_val)) {
            return 0.0;
        }

        int j = 0;
        while (j < no_ticks && dpc_val > ticks[j]) {
            ++j;
        }

        if(std::isnan(predictions[j]))
            return 0.0;

        switch (loss_type) {
            case 'n':
                return predictions[j];
            case 'p':
                return std::log(predictions[j] > TOLERANCE ? predictions[j] : TOLERANCE);
            case 'b':
                return predictions[j];
            case 't':
                return std::log(predictions[j] > TOLERANCE ? predictions[j] : TOLERANCE);
            default:
                throw std::invalid_argument("Specified distribution does not exist.");
        }
    }

    template <typename T>
    void DPC_stairs<T>::predict(const Matrix<T> &data_x,
                                       Matrix<T> &prediction,
                                       const T shrinkage,
                                       const int *row_inds,
                                       int no_rows) {

        if (row_inds == NULL) {
            int data_height = data_x.get_height();
            #ifdef _DEBUG_BASE_LEARNER
                assert(data_height == prediction.get_height() && prediction.get_width() == 1);
            #endif
            T predicted_value;
            for (int i = 0; i < data_height; ++i) {
                predicted_value = prediction.get(i, 0) +
                                  shrinkage * predict_for_row(data_x, i);
                prediction.assign(i, 0, predicted_value);
            }
        } else {
            #ifdef _DEBUG_BASE_LEARNER
                assert(no_rows > 0 && prediction.get_height() == 1);
            #endif
            T predicted_value;
            for (int i = 0; i < no_rows; ++i) {
                predicted_value = prediction.get(row_inds[i], 0) +
                                  shrinkage * predict_for_row(data_x, row_inds[i]);
                prediction.assign(row_inds[i], 0, predicted_value);
            }
        }

    }

}

// trees
namespace dbm {

    template<typename T>
    Tree_node<T>::Tree_node(int depth) :
            Base_learner<T>('t'),
            depth(depth),
            column(-1),
            split_value(0),
            loss_reduction(std::numeric_limits<T>::max()),
            last_node(false),
            prediction(0),
            no_training_samples(0){}

    template<typename T>
    Tree_node<T>::Tree_node(int depth,
                            int column,
                            bool last_node,
                            T split_value,
                            T loss,
                            T prediction,
                            int no_tr_samples) :
            Base_learner<T>('t'),
            depth(depth),
            column(column),
            split_value(split_value),
            loss_reduction(loss),
            last_node(last_node),
            prediction(prediction),
            no_training_samples(no_tr_samples) {}

    template<typename T>
    Tree_node<T>::~Tree_node() {

        if (this == nullptr) return;

        delete right;
        right = nullptr;

        delete left;
        left = nullptr;

    }

    template<typename T>
    T Tree_node<T>::predict_for_row(const Matrix<T> &data_x,
                                    int row_ind) {
        if (last_node)
            return prediction;
        if(std::isnan(data_x.get(row_ind, column)))
            return 0.0;
        if (data_x.get(row_ind, column) > split_value) {
            #ifdef _DEBUG_BASE_LEARNER
            assert(right != NULL);
            #endif
            return right->predict_for_row(data_x, row_ind);
        } else {
            #ifdef _DEBUG_BASE_LEARNER
            assert(left != NULL);
            #endif
            return left->predict_for_row(data_x, row_ind);
        }
    }

    template<typename T>
    void Tree_node<T>::predict(const Matrix<T> &data_x,
                               Matrix<T> &prediction,
                               const T shrinkage,
                               const int *row_inds,
                               int no_rows) {

        if (row_inds == NULL) {
            int data_height = data_x.get_height();
            #ifdef _DEBUG_BASE_LEARNER
            assert(data_height == prediction.get_height() && prediction.get_width() == 1);
            #endif
            T predicted_value;
            for (int i = 0; i < data_height; ++i) {
                predicted_value = prediction.get(i, 0) +
                        shrinkage * predict_for_row(data_x, i);
                prediction.assign(i, 0, predicted_value);
            }
        } else {
            #ifdef _DEBUG_BASE_LEARNER
            assert(no_rows > 0 && prediction.get_height() == 1);
            #endif
            T predicted_value;
            for (int i = 0; i < no_rows; ++i) {
                predicted_value = prediction.get(row_inds[i], 0) +
                        shrinkage * predict_for_row(data_x, row_inds[i]);
                prediction.assign(row_inds[i], 0, predicted_value);
            }
        }

    }
}






/* ========================
 *
 * Tools for base learners
 *
 * ========================
 */

// for global means
namespace dbm {

    template <typename T>
    void save_global_mean(const Global_mean<T> *mean, std::ofstream &out) {
        out << mean->mean << std::endl;
    }

    template <typename T>
    void load_global_mean(std::ifstream &in, Global_mean<T> *&mean) {
        std::string line;
        std::getline(in, line);

        std::string words[100];
        int count = split_into_words(line, words);

//        assert(count == 1);
        if(count != 1)
            std::cout << "Error in loading Global Mean: count is " << count << std::endl;

        mean = new Global_mean<T>;
        mean->mean = T(std::stod(words[0]));
    }

}

//for neural networks
namespace dbm {
    
    template <typename T>
    void save_neural_network(const Neural_network<T> *neural_network, std::ofstream &out) {

        out << neural_network->no_predictors << ' '
            << neural_network->no_hidden_neurons << ' '
            << neural_network->loss_type << std::endl;

        for(int i = 0; i < neural_network->no_predictors; ++i)
            out << neural_network->col_inds[i] << ' ';
        out << std::endl;

        for(int i = 0; i < neural_network->no_hidden_neurons; ++i) {
            for(int j = 0; j < neural_network->no_predictors + 1; ++j)
                out << neural_network->input_weight->get(i, j) << ' ';
            out << std::endl;
        }

        for(int i = 0; i < neural_network->no_hidden_neurons + 1; ++i)
            out << neural_network->hidden_weight->get(0, i) << ' ';

        out << std::endl;

    }
    
    template <typename T>
    void load_neural_network(std::ifstream &in, Neural_network<T> *&neural_network) {
        std::string line;
        std::string words[500];

        std::getline(in, line);
        int count = split_into_words(line, words);
        #ifdef _DEBUG_BASE_LEARNER
        assert(count == 3);
        #endif
        neural_network = new Neural_network<T>(std::stoi(words[0]),
                                               std::stoi(words[1]),
                                               words[2].front());

        line.clear();
        std::getline(in, line);
        count = split_into_words(line, words);
        #ifdef _DEBUG_BASE_LEARNER
        assert(count == neural_network->no_predictors);
        #endif
        for(int i = 0; i < count; ++i)
            neural_network->col_inds[i] = std::stoi(words[i]);

        for(int i = 0; i < neural_network->no_hidden_neurons; ++i) {

            line.clear();
            std::getline(in, line);
            count = split_into_words(line, words);
            #ifdef _DEBUG_BASE_LEARNER
            assert(count == neural_network->no_predictors + 1);
            #endif
            for(int j = 0; j < count; ++j)
                neural_network->input_weight->assign(i,
                                                     j,
                                                     T(std::stod(words[j])));

        }

        line.clear();
        std::getline(in, line);
        count = split_into_words(line, words);
        #ifdef _DEBUG_BASE_LEARNER
        assert(count == neural_network->no_hidden_neurons + 1);
        #endif
        for(int i = 0; i < count; ++i)
            neural_network->hidden_weight->assign(0,
                                                  i,
                                                  T(std::stod(words[i])));

    }
    
}

// for kmeans2d
namespace dbm {

    template <typename T>
    void save_kmeans2d(const Kmeans2d<T> *kmeans2d,
                     std::ofstream &out) {

        out << kmeans2d->no_centroids << ' '
            << kmeans2d->loss_type
            << std::endl;

        for(int i = 0; i < kmeans2d->no_predictors; ++i)
            out << kmeans2d->col_inds[i] << ' ';
        out << std::endl;

        for(int i = 0; i < kmeans2d->no_centroids; ++i) {

            for(int j = 0; j < kmeans2d->no_predictors; ++j)
                out << kmeans2d->centroids[i][j] << ' ';
            out << std::endl;

        }

        for(int i = 0; i < kmeans2d->no_centroids; ++i)
            out << kmeans2d->predictions[i] << ' ';
        out << std::endl;

    }

    template <typename T>
    void load_kmeans2d(std::ifstream &in,
                     Kmeans2d<T> *&kmeans2d) {


        std::string line;
        std::string words[500];

        std::getline(in, line);
        int count = split_into_words(line, words);
        #ifdef _DEBUG_BASE_LEARNER
            assert(count == 2);
        #endif
        kmeans2d = new Kmeans2d<T>(std::stoi(words[0]),
                                   words[1].front());

        line.clear();
        std::getline(in, line);
        count = split_into_words(line, words);
        #ifdef _DEBUG_BASE_LEARNER
            assert(count == kmeans2d->no_predictors);
        #endif
        for(int i = 0; i < count; ++i)
            kmeans2d->col_inds[i] = std::stoi(words[i]);

        for(int i = 0; i < kmeans2d->no_centroids; ++i) {

            line.clear();
            std::getline(in, line);
            count = split_into_words(line, words);
            #ifdef _DEBUG_BASE_LEARNER
                assert(count == kmeans2d->no_predictors);
            #endif
            for(int j = 0; j < kmeans2d->no_predictors; ++j)
                kmeans2d->centroids[i][j] = T(std::stod(words[j]));

        }

        line.clear();
        std::getline(in, line);
        count = split_into_words(line, words);
        #ifdef _DEBUG_BASE_LEARNER
            assert(count == kmeans2d->no_centroids);
        #endif
        for(int i = 0; i < count; ++i)
            kmeans2d->predictions[i] = T(std::stod(words[i]));

    }

}

// for splines
namespace dbm {

    template <typename T>
    void save_splines(const Splines<T> *splines,
                      std::ofstream &out) {

        out << splines->no_knots << ' '
            << splines->loss_type << ' '
            << splines->hinge_coefficient << ' '
            << std::endl;

        for(int i = 0; i < splines->no_predictors; ++i)
            out << splines->col_inds[i] << ' ';
        out << std::endl;

        for(int i = 0; i < splines->no_knots; ++i)
            out << splines->x_knots[i] << ' ';
        out << std::endl;
        for(int i = 0; i < splines->no_knots; ++i)
            out << splines->x_left_coefs[i] << ' ';
        out << std::endl;
        for(int i = 0; i < splines->no_knots; ++i)
            out << splines->x_right_coefs[i] << ' ';
        out << std::endl;

        for(int i = 0; i < splines->no_knots; ++i)
            out << splines->y_knots[i] << ' ';
        out << std::endl;
        for(int i = 0; i < splines->no_knots; ++i)
            out << splines->y_left_coefs[i] << ' ';
        out << std::endl;
        for(int i = 0; i < splines->no_knots; ++i)
            out << splines->y_right_coefs[i] << ' ';
        out << std::endl;

    }

    template <typename T>
    void load_splines(std::ifstream &in,
                      Splines<T> *&splines) {

        std::string line;
        std::string words[500];

        std::getline(in, line);
        int count = split_into_words(line, words);
        #ifdef _DEBUG_BASE_LEARNER
            assert(count == 3);
        #endif
        splines = new Splines<T>(std::stoi(words[0]),
                                 words[1].front(),
                                 std::stod(words[2]));

        line.clear();
        std::getline(in, line);
        count = split_into_words(line, words);
        #ifdef _DEBUG_BASE_LEARNER
            assert(count == splines->no_predictors);
        #endif
        for(int i = 0; i < count; ++i)
            splines->col_inds[i] = std::stoi(words[i]);

        line.clear();
        std::getline(in, line);
        count = split_into_words(line, words);
        #ifdef _DEBUG_BASE_LEARNER
            assert(count == splines->no_knots);
        #endif
        for(int i = 0; i < count; ++i)
            splines->x_knots[i] = T(std::stod(words[i]));

        line.clear();
        std::getline(in, line);
        count = split_into_words(line, words);
        #ifdef _DEBUG_BASE_LEARNER
            assert(count == splines->no_knots);
        #endif
        for(int i = 0; i < count; ++i)
            splines->x_left_coefs[i] = T(std::stod(words[i]));

        line.clear();
        std::getline(in, line);
        count = split_into_words(line, words);
        #ifdef _DEBUG_BASE_LEARNER
            assert(count == splines->no_knots);
        #endif
        for(int i = 0; i < count; ++i)
            splines->x_right_coefs[i] = T(std::stod(words[i]));

        line.clear();
        std::getline(in, line);
        count = split_into_words(line, words);
        #ifdef _DEBUG_BASE_LEARNER
            assert(count == splines->no_knots);
        #endif
        for(int i = 0; i < count; ++i)
            splines->y_knots[i] = T(std::stod(words[i]));

        line.clear();
        std::getline(in, line);
        count = split_into_words(line, words);
        #ifdef _DEBUG_BASE_LEARNER
            assert(count == splines->no_knots);
        #endif
        for(int i = 0; i < count; ++i)
            splines->y_left_coefs[i] = T(std::stod(words[i]));

        line.clear();
        std::getline(in, line);
        count = split_into_words(line, words);
        #ifdef _DEBUG_BASE_LEARNER
            assert(count == splines->no_knots);
        #endif
        for(int i = 0; i < count; ++i)
            splines->y_right_coefs[i] = T(std::stod(words[i]));

    }

}

// for linear regression
namespace dbm {

    template <typename T>
    void save_linear_regression(const Linear_regression<T> *linear_regression,
                                std::ofstream &out) {

        out << linear_regression->no_predictors << ' '
            << linear_regression->loss_type << std::endl;

        for(int i = 0; i < linear_regression->no_predictors; ++i)
            out << linear_regression->col_inds[i] << ' ';
        out << std::endl;

        for(int i = 0; i < linear_regression->no_predictors; ++i)
            out << linear_regression->coefs_no_intercept[i] << ' ';
        out << std::endl;

        out << linear_regression->intercept << std::endl;
    }

    template <typename T>
    void load_linear_regression(std::ifstream &in, Linear_regression<T> *&linear_regression) {
        std::string line;
        std::string words[500];

        std::getline(in, line);
        int count = split_into_words(line, words);
        #ifdef _DEBUG_BASE_LEARNER
        assert(count == 2);
        #endif
        linear_regression = new Linear_regression<T>(std::stoi(words[0]),
                                                     words[1].front());

        line.clear();
        std::getline(in, line);
        count = split_into_words(line, words);
        #ifdef _DEBUG_BASE_LEARNER
        assert(count == linear_regression->no_predictors);
        #endif
        for(int i = 0; i < count; ++i)
            linear_regression->col_inds[i] = std::stoi(words[i]);

        line.clear();
        std::getline(in, line);
        count = split_into_words(line, words);
        #ifdef _DEBUG_BASE_LEARNER
        assert(count == linear_regression->no_predictors);
        #endif
        for(int i = 0; i < count; ++i)
            linear_regression->coefs_no_intercept[i] = T(std::stod(words[i]));

        line.clear();
        std::getline(in, line);
        count = split_into_words(line, words);
            #ifdef _DEBUG_BASE_LEARNER
        assert(count == 1);
            #endif
        linear_regression->intercept = T(std::stod(words[0]));
    }

}

// for dpc stairs
namespace dbm {

    template <typename T>
    void save_dpc_stairs(const DPC_stairs<T> *dpc_stairs,
                         std::ofstream &out) {

        out << dpc_stairs->no_predictors << ' '
            << dpc_stairs->loss_type << ' '
            << dpc_stairs->no_ticks
            << std::endl;

        for(int i = 0; i < dpc_stairs->no_predictors; ++i)
            out << dpc_stairs->col_inds[i] << ' ';
        out << std::endl;

        for(int i = 0; i < dpc_stairs->no_predictors; ++i)
            out << dpc_stairs->coefs[i] << ' ';
        out << std::endl;

        for(int i = 0; i < dpc_stairs->no_ticks; ++i)
            out << dpc_stairs->ticks[i] << ' ';
        out << std::endl;

        for(int i = 0; i < dpc_stairs->no_ticks; ++i)
            out << dpc_stairs->predictions[i] << ' ';
        out << std::endl;
    }

    template <typename T>
    void load_dpc_stairs(std::ifstream &in,
                         DPC_stairs<T> *&dpc_stairs) {
        std::string line;
        std::string words[500];

        std::getline(in, line);
        int count = split_into_words(line, words);
        #ifdef _DEBUG_BASE_LEARNER
            assert(count == 3);
        #endif
        dpc_stairs = new DPC_stairs<T>(std::stoi(words[0]),
                                       words[1].front(),
                                       std::stoi(words[2]));

        line.clear();
        std::getline(in, line);
        count = split_into_words(line, words);
        #ifdef _DEBUG_BASE_LEARNER
            assert(count == dpc_stairs->no_predictors);
        #endif
        for(int i = 0; i < count; ++i)
            dpc_stairs->col_inds[i] = std::stoi(words[i]);

        line.clear();
        std::getline(in, line);
        count = split_into_words(line, words);
        #ifdef _DEBUG_BASE_LEARNER
            assert(count == dpc_stairs->no_predictors);
        #endif
        for(int i = 0; i < count; ++i)
            dpc_stairs->coefs[i] = std::stod(words[i]);

        line.clear();
        std::getline(in, line);
        count = split_into_words(line, words);
        #ifdef _DEBUG_BASE_LEARNER
            assert(count == dpc_stairs->no_ticks);
        #endif
        for(int i = 0; i < count; ++i)
            dpc_stairs->ticks[i] = std::stod(words[i]);

        line.clear();
        std::getline(in, line);
        count = split_into_words(line, words);
        #ifdef _DEBUG_BASE_LEARNER
            assert(count == dpc_stairs->no_ticks);
        #endif
        for(int i = 0; i < count; ++i)
            dpc_stairs->predictions[i] = std::stod(words[i]);
    }

}

// for trees
namespace dbm {

    template<typename T>
    void save_tree_node(const Tree_node<T> *node, std::ofstream &out) {
        if (node == nullptr) {
            out << "#" << '\n';
        } else {
            out << node->depth << ' '
                << node->column << ' '
                << node->last_node << ' '
                << node->split_value << ' '
                << node->loss_reduction << ' '
                << node->prediction << ' '
                << node->no_training_samples << ' '
                << std::endl;
            save_tree_node(node->right, out);
            save_tree_node(node->left, out);
        }
    }

    template<typename T>
    bool readNextToken(int &depth,
                       int &column,
                       bool &last_node,
                       T &split_value,
                       T &loss,
                       T &prediction,
                       int &no_tr_samples,
                       std::istream &in,
                       bool &isNumber) {

        isNumber = false;

        if (in.eof()) return false;

        std::string line;
        std::getline(in, line);

        std::string words[100];
        int count = split_into_words(line, words);

        if (!count || words[0] == "==")
            return false;

        if (words[0] != "#") {
            isNumber = true;
            depth = std::stoi(words[0]);
            column = std::stoi(words[1]);
            last_node = bool(std::stoi(words[2]));
            split_value = T(std::stod(words[3]));
            loss = T(std::stod(words[4]));
            prediction = T(std::stod(words[5]));
            no_tr_samples = std::stoi(words[6]);
        }
        return true;
    }

    template<typename T>
    void load_tree_node(std::ifstream &in, Tree_node<T> *&node) {
        int depth, column, no_tr_samples;
        T split_value, loss, prediction;
        bool last_node = false;
        bool isNumber;
        if (!readNextToken(depth,
                           column,
                           last_node,
                           split_value,
                           loss,
                           prediction,
                           no_tr_samples,
                           in,
                           isNumber))
            return;
        if (isNumber) {
            node = new Tree_node<T>(depth,
                                    column,
                                    last_node,
                                    split_value,
                                    loss,
                                    prediction,
                                    no_tr_samples);
            load_tree_node(in, node->right);
            load_tree_node(in, node->left);
        }
    }

    template<typename T>
    void delete_tree(Tree_node<T> *tree) {
        delete tree;
        tree = nullptr;
    }

    template<typename T>
    void print_tree_info(const dbm::Tree_node<T> *tree) {
        if (tree->last_node) {
            std::cout << "depth: " << tree->depth << ' '
                      << "column: " << tree->column << ' '
                      << "split_value: " << tree->split_value << ' '
                      << "loss: " << tree->loss_reduction << ' '
                      << "last_node: " << tree->last_node << ' '
                      << "prediction: " << tree->prediction << ' '
                      << "no_training_sample: " << tree->no_training_samples
                      << std::endl;
            std::cout << "==========" << std::endl;
            return;
        }
        std::cout << "depth: " << tree->depth << ' '
                  << "column: " << tree->column << ' '
                  << "split_value: " << tree->split_value << ' '
                  << "loss: " << tree->loss_reduction << ' '
                  << "last_node: " << tree->last_node << ' '
                  << "prediction: " << tree->prediction << ' '
                  << "no_training_sample: " << tree->no_training_samples
                  << std::endl;
        std::cout << "==========" << std::endl;
        print_tree_info(tree->right);
        print_tree_info(tree->left);
    }

    template<typename T>
    void Tree_info<T>::get_depth(const dbm::Tree_node<T> *tree) {
        if (tree->last_node) {
            depth = std::max(depth, tree->depth);
            return;
        }
        get_depth(tree->right);
        get_depth(tree->left);
    }

    template<typename T>
    void Tree_info<T>::fill(const dbm::Tree_node<T> *tree, int h) {

        std::ostringstream temporary;
        temporary << "(" << tree->depth << ")";


        if (tree->last_node) {
            temporary << " " << tree->prediction;
            tree_nodes[h][tree->depth] = temporary.str();
            return;
        }

        temporary << " n:" << tree->no_training_samples
                  << " l:" << tree->loss_reduction
                  << " c:" << tree->column
                  << " v:" << tree->split_value;
        tree_nodes[h][tree->depth] = temporary.str();
        int next_higher = h - std::max(1, int(height / std::pow(2.0, tree->depth + 2))),
                next_lower = h + int(height / std::pow(2.0, tree->depth + 2));
        fill(tree->right, next_higher);
        fill(tree->left, next_lower);
    }

    template<typename T>
    Tree_info<T>::Tree_info(const dbm::Tree_node<T> *tree) {

        get_depth(tree);

        height = int(std::pow(2.0, depth));

        tree_nodes = new std::string *[height];
        for (int i = 0; i < height; ++i) {
            tree_nodes[i] = new std::string[depth + 1];
        }

        for (int i = 0; i < height; ++i) {
            for (int j = 0; j < depth + 1; ++j)
                tree_nodes[i][j] = "";
        }

        fill(tree, height / 2);

    }

    template<typename T>
    Tree_info<T>::~Tree_info() {

        for (int i = 0; i < height; ++i) {
            delete[] tree_nodes[i];
        }
        delete[] tree_nodes;

    }

    template<typename T>
    void Tree_info<T>::print() const {
        for (int i = 0; i < height; ++i) {
            for (int j = 0; j < depth + 1; ++j) {
                std::cout << tree_nodes[i][j] << "\t\t";
            }
            std::cout << std::endl;
        }
    }

    template<typename T>
    void Tree_info<T>::print_to_file(const std::string &file_name,
                                     const int & number) const {
        std::ofstream file(file_name.c_str(), std::ios_base::app);
        file << std::endl;
        file << "=======================  Tree "
             << number
             << "  ======================="
             << std::endl;
        for (int i = 0; i < height; ++i) {
            for (int j = 0; j < depth + 1; ++j) {
                file << tree_nodes[i][j] << "\t\t";
            }
            file << std::endl;
        }
        file.close();
    }

}

// explicit instantiation

namespace dbm {

    template void save_global_mean<double>(const Global_mean<double> *mean, std::ofstream &out);

    template void save_global_mean<float>(const Global_mean<float> *mean, std::ofstream &out);

    template void load_global_mean<float>(std::ifstream &in, Global_mean<float> *&mean);

    template void load_global_mean<double>(std::ifstream &in, Global_mean<double> *&mean);


    template void save_neural_network<double>(const Neural_network<double> *neural_network, std::ofstream &out);

    template void save_neural_network<float>(const Neural_network<float> *neural_network, std::ofstream &out);

    template void load_neural_network<double>(std::ifstream &in, Neural_network<double> *&neural_network);

    template void load_neural_network<float>(std::ifstream &in, Neural_network<float> *&neural_network);

    template void save_kmeans2d<double>(const Kmeans2d<double> *kmeans2d, std::ofstream &out);

    template void save_kmeans2d<float>(const Kmeans2d<float> *kmeans2d, std::ofstream &out);

    template void load_kmeans2d<double>(std::ifstream &in, Kmeans2d<double> *&kmeans2d);

    template void load_kmeans2d<float>(std::ifstream &in, Kmeans2d<float> *&kmeans2d);


    template void save_splines<double>(const Splines<double> *splines, std::ofstream &out);

    template void save_splines<float>(const Splines<float> *splines, std::ofstream &out);

    template void load_splines<double>(std::ifstream &in, Splines<double> *&splines);

    template void load_splines<float>(std::ifstream &in, Splines<float> *&splines);


    template void save_linear_regression<double>(const Linear_regression<double> *linear_regression, std::ofstream &out);

    template void save_linear_regression<float>(const Linear_regression<float> *linear_regression, std::ofstream &out);

    template void load_linear_regression<double>(std::ifstream &in, Linear_regression<double> *&linear_regression);

    template void load_linear_regression<float>(std::ifstream &in, Linear_regression<float> *&linear_regression);


    template void save_dpc_stairs<double>(const DPC_stairs<double> *dpc_stairs, std::ofstream &out);

    template void save_dpc_stairs<float>(const DPC_stairs<float> *dpc_stairs, std::ofstream &out);

    template void load_dpc_stairs<double>(std::ifstream &in, DPC_stairs<double> *&dpc_stairs);

    template void load_dpc_stairs<float>(std::ifstream &in, DPC_stairs<float> *&dpc_stairs);


    template void save_tree_node<double>(const Tree_node<double> *node, std::ofstream &out);

    template void save_tree_node<float>(const Tree_node<float> *node, std::ofstream &out);

    template void load_tree_node<double>(std::ifstream &in, Tree_node<double> *&node);

    template void load_tree_node<float>(std::ifstream &in, Tree_node<float> *&node);

    template void delete_tree<double>(Tree_node<double> *tree);

    template void delete_tree<float>(Tree_node<float> *tree);

    template void print_tree_info<double>(const dbm::Tree_node<double> *tree);

    template void print_tree_info<float>(const dbm::Tree_node<float> *tree);

}