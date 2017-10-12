//
// Created by xixuan on 10/10/16.
//

#ifndef DBM_CODE_BASE_LEARNER_H
#define DBM_CODE_BASE_LEARNER_H

#define TOLERANCE 1e-30

//#ifndef _DEBUG_BASE_LEARNER
//#define _DEBUG_BASE_LEARNER
//#endif

#include "matrix.h"

#include <string>
#include <functional>



// base learner
namespace dbm {

    template<typename T>
    class Base_learner {
    protected:
        char learner_type;

        virtual T predict_for_row(const Matrix<T> &data_x,
                                  int row_ind) = 0;

    public:
        Base_learner(const char &type) : learner_type(type) {};
        virtual ~Base_learner() {};

        char get_type() const { return learner_type; };

        virtual void
        predict(const Matrix<T> &data_x,
                Matrix<T> &prediction,
                const T shrinkage = 1,
                const int *row_inds = NULL,
                int no_rows = 0) = 0;
    };

}

// global mean
namespace dbm {

    template <typename T>
    class Global_mean;

    template <typename T>
    class Mean_trainer;

    template <typename T>
    void save_global_mean(const Global_mean<T> *mean,
                          std::ofstream &out);

    template <typename T>
    void load_global_mean(std::ifstream &in,
                          Global_mean<T> *&mean);

    template <typename T>
    class Global_mean : public Base_learner<T> {
    private:
        T mean = 0;
        T predict_for_row(const Matrix<T> &data_x,
                          int row_ind);
    public:
        Global_mean();
        ~Global_mean();

        void predict(const Matrix<T> &data_x,
                     Matrix<T> &prediction,
                     const T shrinkage = 1,
                     const int *row_inds = nullptr,
                     int no_rows = 0);

        friend void save_global_mean<>(const Global_mean<T> *mean,
                                       std::ofstream &out);

        friend void load_global_mean<>(std::ifstream &in,
                                       Global_mean<T> *&mean);

        friend class Mean_trainer<T>;

    };

}

// kmeans2d
namespace dbm {

    template <typename T>
    class Kmeans2d;

    template <typename T>
    class Kmeans2d_trainer;

    template <typename T>
    void save_kmeans2d(const Kmeans2d<T> *kmeans2d,
                     std::ofstream &out);

    template <typename T>
    void load_kmeans2d(std::ifstream &in,
                     Kmeans2d<T> *&kmeans2d);

    template <typename T>
    class Kmeans2d : public Base_learner<T> {
    private:

        static const int no_predictors = 2;
        int no_centroids;
        char loss_type;

        int *col_inds;

        T **centroids;
        T *predictions;

        T distance(const Matrix<T> &train_x,
                   const int &row_ind,
                   const int &centroid_ind);

        T predict_for_row(const Matrix<T> &data_x,
                          int row_ind);
    public:

        Kmeans2d(int no_centroids, char loss_type);
        ~Kmeans2d();

        void predict(const Matrix<T> &data_x,
                     Matrix<T> &prediction,
                     const T shrinkage = 1,
                     const int *row_inds = nullptr,
                     int no_rows = 0);

        friend void save_kmeans2d<>(const Kmeans2d<T> *kmeans2d,
                                    std::ofstream &out);

        friend void load_kmeans2d<>(std::ifstream &in,
                                    Kmeans2d<T> *&kmeans2d);

        friend class Kmeans2d_trainer<T>;

    };

}

// splines
namespace dbm {

    template <typename T>
    class Splines;

    template <typename T>
    class Splines_trainer;

    template <typename T>
    void save_splines(const Splines<T> *splines,
                      std::ofstream &out);

    template <typename T>
    void load_splines(std::ifstream &in,
                      Splines<T> *&splines);

    template <typename T>
    class Splines : public Base_learner<T> {
    private:
        static const int no_predictors = 2;

        int no_knots;
        char loss_type;
        T hinge_coefficient;

        int *col_inds;

        T x_left_hinge(T &x, T &y, T &knot);
        T x_right_hinge(T &x, T &y, T &knot);
        T *x_knots;
        T *x_left_coefs;
        T *x_right_coefs;

        T y_left_hinge(T &x, T &y, T &knot);
        T y_right_hinge(T &x, T &y, T &knot);
        T *y_knots;
        T *y_left_coefs;
        T *y_right_coefs;

        T predict_for_row(const Matrix<T> &data_x,
                          int row_ind);
    public:
        Splines(int no_knots, char loss_type, T hinge_coefficient);
        ~Splines();

        void predict(const Matrix<T> &data_x,
                     Matrix<T> &prediction,
                     const T shrinkage = 1,
                     const int *row_inds = nullptr,
                     int no_rows = 0);

        friend void save_splines<>(const Splines<T> *splines,
                                   std::ofstream &out);

        friend void load_splines<>(std::ifstream &in,
                                   Splines<T> *&splines);

        friend class Splines_trainer<T>;

    };

}

// neural networks
namespace dbm {

    template <typename T>
    class Neural_network;

    template <typename T>
    class Neural_network_trainer;

    template <typename T>
    void save_neural_network(const Neural_network<T> *neural_network,
                             std::ofstream &out);

    template <typename T>
    void load_neural_network(std::ifstream &in,
                             Neural_network<T> *&neural_network);

    template <typename T>
    class Neural_network : public Base_learner<T> {
    private:
        int no_predictors;
        int no_hidden_neurons;

        char loss_type;

        int *col_inds = nullptr;

        // no_hidden_neurons * (no_predictors + 1)
        Matrix<T> *input_weight;
        // 1 * (no_hidden_neurons + 1)
        Matrix<T> *hidden_weight;

        T activation(const T &input);
        void forward(const Matrix<T> &input_output, Matrix<T> &hidden_output, T &output_output);

        T predict_for_row(const Matrix<T> &data_x,
                          int row_ind);

    public:
        Neural_network(int no_predictors,
                       int no_hidden_neurons,
                       char loss_type);
        ~Neural_network();

        void predict(const Matrix<T> &data_x,
                     Matrix<T> &prediction,
                     const T shrinkage = 1,
                     const int *row_inds = nullptr,
                     int no_rows = 0);

        friend void save_neural_network<>(const Neural_network<T> *neural_network,
                                          std::ofstream &out);

        friend void load_neural_network<>(std::ifstream &in,
                                          Neural_network<T> *&neural_network);

        friend class Neural_network_trainer<T>;

    };

}

// linear regression
namespace dbm {

    template <typename T>
    class Linear_regression;

    template <typename T>
    class Linear_regression_trainer;

    template <typename T>
    void save_linear_regression(const Linear_regression<T> *linear_regression,
                                std::ofstream &out);

    template <typename T>
    void load_linear_regression(std::ifstream &in,
                                Linear_regression<T> *&linear_regression);

    template <typename T>
    class Linear_regression : public Base_learner<T> {
    private:
        int no_predictors;
        char loss_type;

        int *col_inds = nullptr;

        T intercept;
        T *coefs_no_intercept = nullptr;

        T predict_for_row(const Matrix<T> &data_x,
                          int row_ind);
    public:
        Linear_regression(int no_predictors,
                          char loss_type);
        ~Linear_regression();

        void predict(const Matrix<T> &data_x,
                     Matrix<T> &prediction,
                     const T shrinkage = 1,
                     const int *row_inds = nullptr,
                     int no_rows = 0);

        friend void save_linear_regression<>(const Linear_regression<T> *linear_regression,
                                             std::ofstream &out);

        friend void load_linear_regression<>(std::ifstream &in,
                                             Linear_regression<T> *&linear_regression);

        friend class Linear_regression_trainer<T>;

    };

}

// dpc stairs
namespace dbm {

    template <typename T>
    class DPC_stairs;

    template <typename T>
    class DPC_stairs_trainer;

    template <typename T>
    void save_dpc_stairs(const DPC_stairs<T> *dpc_stairs,
                         std::ofstream &out);

    template <typename T>
    void load_dpc_stairs(std::ifstream &in,
                         DPC_stairs<T> *&dpc_stairs);

    template <typename T>
    class DPC_stairs : public Base_learner<T> {
    private:
        int no_predictors;
        char loss_type;
        int no_ticks;

        int *col_inds = nullptr;

        T *coefs = nullptr;
        T *ticks = nullptr;
        T *predictions = nullptr;

        T predict_for_row(const Matrix<T> &data_x,
                          int row_ind);
    public:
        DPC_stairs(int no_predictors,
                   char loss_type,
                   int no_ticks);
        ~DPC_stairs();

        void predict(const Matrix<T> &data_x,
                     Matrix<T> &prediction,
                     const T shrinkage = 1,
                     const int *row_inds = nullptr,
                     int no_rows = 0);

        friend void save_dpc_stairs<>(const DPC_stairs<T> *dpc_stairs,
                                      std::ofstream &out);

        friend void load_dpc_stairs<>(std::ifstream &in,
                                      DPC_stairs<T> *&dpc_stairs);

        friend class DPC_stairs_trainer<T>;

    };

}

// trees
namespace dbm {

    template<typename T>
    class Tree_node;

    template<typename T>
    class Tree_trainer;

    template <typename T>
    class Fast_tree_trainer;

    template<typename T>
    class Tree_info;

    template<typename T>
    void save_tree_node(const Tree_node<T> *node,
                        std::ofstream &out);
    template<typename T>
    void load_tree_node(std::ifstream &in,
                        Tree_node<T> *&tree);

    template<typename T>
    void delete_tree(Tree_node<T> *tree);
    template<typename T>
    void print_tree_info(const dbm::Tree_node<T> *tree);

    template<typename T>
    class Tree_node : public Base_learner<T> {
    private:

        Tree_node *right = nullptr;
        Tree_node *left = nullptr;

        int depth;

        int column;
        T split_value;
        T loss_reduction;

        bool last_node;
        T prediction;

        int no_training_samples;

        T predict_for_row(const Matrix<T> &data_x,
                          int row_ind);

    public:

        Tree_node(int depth);

        Tree_node(int depth,
                  int column,
                  bool last_node,
                  T split_value,
                  T loss,
                  T prediction,
                  int no_tr_samples);

        ~Tree_node();

        void predict(const Matrix<T> &data_x,
                     Matrix<T> &prediction,
                     const T shrinkage = 1,
                     const int *row_inds = nullptr,
                     int no_rows = 0);

        friend void save_tree_node<>(const Tree_node<T> *node,
                                     std::ofstream &out);

        friend void load_tree_node<>(std::ifstream &in,
                                     Tree_node<T> *&tree);

        friend void delete_tree<>(Tree_node<T> *tree);

        friend void print_tree_info<>(const Tree_node<T> *tree);

        friend class Tree_trainer<T>;
        friend class Fast_tree_trainer<T>;

        friend class Tree_info<T>;

    };

}




/*
 * tools for base learners
 */

// for global means
namespace dbm {

    template <typename T>
    void save_global_mean(const Global_mean<T> *mean, std::ofstream &out);

    template <typename T>
    void load_global_mean(std::ifstream &in, Global_mean<T> *&mean);

}

// for linear regression
namespace dbm {

    template <typename T>
    void save_linear_regression(const Linear_regression<T> *linear_regression, std::ofstream &out);

    template <typename T>
    void load_linear_regression(std::ifstream &in, Linear_regression<T> *&linear_regression);

}

// for trees
namespace dbm {

    template <typename T>
    void save_tree_node(const Tree_node<T> *node, std::ofstream &out);

    template <typename T>
    void load_tree_node(std::ifstream &in, Tree_node<T> *&node);

    template <typename T>
    void delete_tree(Tree_node<T> *tree);

    // display tree information
    template<typename T>
    void print_tree_info(const dbm::Tree_node<T> *tree);

    template<typename T>
    class Tree_info {
    private:
        std::string **tree_nodes;
        int depth = 0;
        int height = 0;

        void get_depth(const dbm::Tree_node<T> *tree);

        void fill(const dbm::Tree_node<T> *tree, int h);

    public:
        Tree_info(const dbm::Tree_node<T> *tree);

        ~Tree_info();

        void print() const;

        void print_to_file(const std::string &file_name, const int & number) const;
    };

}

#endif //DBM_CODE_BASE_LEARNER_H



