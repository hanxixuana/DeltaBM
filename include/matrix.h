//
// Created by xixuan on 10/10/16.
//

#ifndef DBM_CODE_MATRIX_H
#define DBM_CODE_MATRIX_H

//#ifndef _DEBUG_MATRIX
//#define _DEBUG_MATRIX
//#endif

//#ifndef _CD_INDICATOR
//#define _CD_INDICATOR
//#endif

#if defined(_BLAS) || defined(_MKL)
#undef _PLAIN_MATRIX_OP
#else
#define _PLAIN_MATRIX_OP
#endif

#include <string>
#include <cstring>
#ifdef _MKL
#include "mkl.h"
#endif

namespace dbm {

    // prototypes for friend functions
    template<typename T>
    class Matrix;

    template <typename T>
    Matrix<T> transpose(const Matrix<T> &matrix);

    template <typename T>
    Matrix<T> plus(const Matrix<T> &left, const Matrix<T> &right);

#ifdef _MKL
    template<> Matrix<float> plus(const Matrix<float> &left, const Matrix<float> &right);
    template<> Matrix<double> plus(const Matrix<double> &left, const Matrix<double> &right);
#endif

    template <typename T>
    Matrix<T> substract(const Matrix<T> &left, const Matrix<T> &right);

    template <typename T>
    Matrix<T> inner_product(const Matrix<T> &left, const Matrix<T> &right);

#ifdef _MKL
    template<> Matrix<float> inner_product(const Matrix<float> &lhs, const Matrix<float> &rhs);
    template<> Matrix<double> inner_product(const Matrix<double> &lhs, const Matrix<double> &rhs);

    float* inner_product(const Matrix<float> &lhs, const float *rhs);
    double* inner_product(const Matrix<double> &lhs, const double *rhs);
#endif

    template <typename T>
    T determinant(const Matrix<T> &matrix);

    template <typename T>
    Matrix<T> inverse(const Matrix<T> &matrix);

#ifdef _MKL
    template<> Matrix<float> inverse(const Matrix<float> &rhs);
    template<> Matrix<double> inverse(const Matrix<double> &rhs);
#endif

    template<typename T>
    Matrix<T> vert_merge(const Matrix<T> &upper, const Matrix<T> &lower);

    template<typename T>
    Matrix<T> hori_merge(const Matrix<T> &left, const Matrix<T> &right);

    template<typename T>
    Matrix<T> copy(const Matrix<T> &target);

    template<typename T>
    void copy(const Matrix<T> &target, Matrix<T> &to);

    template <typename T>
    Matrix<T> col_sort(Matrix<T> &data);

    template <typename T>
    Matrix<T> col_sorted_to(const Matrix<T> &sorted_from);

    // matrix class
    template<typename T>
    class Matrix {

    private:

        int height;
        int width;

        T **data = nullptr;
        int *col_labels = nullptr;
        int *row_labels = nullptr;

    public:

        //=======================================
        // constructors, destructor and IO tools
        //=======================================
        Matrix();

        Matrix(int height, int width);

        Matrix(int height, int width, const T &value);

        Matrix(int height, int width, std::string file_name, const char &delimiter = '\t');
	
		Matrix(const Matrix<T>& rhs);
	
		Matrix<T>& operator=(const Matrix<T>& rhs);

        ~Matrix();

        void print() const;

//        std::string print_to_string() const;

        void print_to_file(const std::string &file_name, const char &delimiter = '\t') const;

        //=======================
        // dimensions and ranges
        //=======================
        int get_width() const { return width; }

        int get_height() const { return height; }

        T get_col_max(int col_index,
                      const int *row_inds = nullptr, 
                      int no_rows = 0) const;

        T get_col_min(int col_index,
                      const int *row_inds = nullptr, 
                      int no_rows = 0) const;

        //===============
        // unique values
        //===============
        // returns the number of unique values
        // sort and put unique values in the beginning of values
        int unique_vals_col(int j, T *values,
                            const int *row_inds = nullptr, int no_rows = 0) const;

        //======
        // clear
        //======
        void clear();

        //==============
        // shuffle rows
        //==============
        void row_shuffle(); // it does not shuffle row_labels and row_labels will not match rows
        Matrix row_shuffled_to() const;  // it shuffles both rows and row_labels to a new Matrix<T>

        //============
        // assignment
        //============
        #if _DEBUG_MATRIX

            void assign_row_label(int i, const int &label);

            void assign_col_label(int j, const int &label);

        #endif

        void assign(int i, int j, const T &value);

        void assign_col(int j, T *column);

        void assign_row(int i, T *row);

        // matrix[i][j] returns a reference to [(i+1), (j+1)]'th element
        // matrix[i] returns a pointer to (i + 1)'th row
        T *operator[](int k);

        //=========================================
        // get element, rows, columns, submatrices
        //=========================================
        T get(int i, int j) const;

        Matrix col(int col_index) const;

        Matrix row(int row_index) const;

        Matrix cols(const int *col_indices, int no_cols) const;

        Matrix rows(const int *row_indices, int no_rows) const;

        Matrix submatrix(const int *row_indices, int no_rows, const int *col_indices, int no_cols) const;

        //=============================================================
        // split into two Matrix<T> according to a col and a threshold
        //=============================================================
        int n_larger_in_col(int col_index, const T &threshold,
                            const int *row_inds = nullptr, int no_rows = 0) const;

        int n_smaller_or_eq_in_col(int col_index, const T &threshold,
                                   const int *row_inds = nullptr, int no_rows = 0) const;

        int inds_larger_in_col(int col_index, const T &threshold, int *indices,
                               const int *row_inds = nullptr, int no_rows = 0) const;

        int inds_smaller_or_eq_in_col(int col_index, const T &threshold, int *indices,
                                      const int *row_inds = nullptr, int no_rows = 0) const;

        void inds_split(int col_inds, const T &threshold, int *larger_inds,
                        int *smaller_inds, int *n_two_inds,
                        const int *row_inds = nullptr, int no_rows = 0) const;

        Matrix vert_split_l(int col_index, const T &threshold) const;

        Matrix vert_split_s(int col_index, const T &threshold) const;

        //===================================
        // average in a col for certain rows
        //===================================
        T average_col_for_rows(int col_index, const int *row_inds = nullptr, int no_rows = 0) const;

        void ul_average_col_for_rows(int col_index, const T &threshold, T *two_average,
                                     const int *row_inds = nullptr, int no_rows = 0) const;

        //================
        // math operations
        //================

        T row_sum(const int &row_ind) const;
        T col_sum(const int &col_ind) const;

        T row_average(const int &row_ind) const;
        T col_average(const int &col_ind) const;

        T row_std(const int &row_ind) const;
        T col_std(const int &col_ind) const;

        friend Matrix transpose<>(const Matrix &matrix);

        friend Matrix plus<>(const Matrix &left, const Matrix &right);

#ifdef _MKL
        friend Matrix<float> plus<>(const Matrix<float> &lhs, const Matrix<float> &rhs);
        friend Matrix<double> plus<>(const Matrix<double> &lhs, const Matrix<double> &rhs);
#endif

        friend Matrix substract<>(const Matrix &left, const Matrix &right);

        friend Matrix inner_product<>(const Matrix &left, const Matrix &right);

#ifdef _MKL
        friend Matrix<float> inner_product<>(const Matrix<float> &lhs, const Matrix<float> &rhs);
        friend Matrix<double> inner_product<>(const Matrix<double> &lhs, const Matrix<double> &rhs);

        friend float* inner_product(const Matrix<float> &lhs, const float *rhs);
        friend double* inner_product(const Matrix<double> &lhs, const double *rhs);
#endif

        friend T determinant<>(const Matrix &matrix);

        friend Matrix inverse<>(const Matrix &matrix);

#ifdef _MKL
        friend Matrix<float> inverse<>(const Matrix<float> &rhs);
        friend Matrix<double> inverse<>(const Matrix<double> &rhs);
#endif

        T frobenius_norm() const;

        bool is_symmetric() const;

        T dominant_eigen_decomp(dbm::Matrix<T> &eigen_vector, int no_iterations = 1000);

        // in-place
        void columnwise_centering();

        void scaling(const T &scalar);

        void inplace_elewise_prod_mat_with_row_vec(const Matrix &row);

        //===================
        // columnwise sorting
        //===================
        friend Matrix col_sort<>(Matrix &data);

        friend Matrix col_sorted_to<>(const Matrix &sorted_from);

        //===========================================================================================
        // vertical merge of two Matrix<T>, row labels are combined and column labels are from upper
        //===========================================================================================
        friend Matrix vert_merge<>(const Matrix &upper, const Matrix &lower);

        friend Matrix hori_merge<>(const Matrix<T> &left, const Matrix<T> &right);

        //================================
        // deep copy target to a new Matrix<T>
        //================================
        friend Matrix copy<>(const Matrix &target);

        friend void copy<>(const Matrix &target, Matrix &to);

    };

}

#endif //DBM_CODE_MATRIX_H




