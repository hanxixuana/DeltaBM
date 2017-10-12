//
// Created by xixuan on 10/10/16.
//

#include "loss_function.h"

#include <cassert>
#include <cmath>
#include <stdexcept>
#include <algorithm>

namespace dbm {

    template
    class Loss_function<double>;

    template
    class Loss_function<float>;

}

namespace dbm {

    template <typename T>
    Loss_function<T>::Loss_function(const Params &params) :
            params(params) {};

    template<typename T>
    inline T Loss_function<T>::loss(const Matrix<T> &train_y,
                                    const Matrix<T> &prediction,
                                    const char &dist,
                                    const T beta,
                                    const int *row_inds,
                                    int no_rows) const {
        int train_y_height = train_y.get_height();

        #ifdef _DEBUG_LOSS_FUNCTION
            assert(train_y.get_width() == 1);
        #endif

        /*
         *  1. Remember that a link function may be needed when calculating losses
         *  2. Also remember to put beta in it
         */

        if (row_inds == NULL) {
            switch (dist) {
                case 'n': {
                    // mean suared error
                    double result = 0;
                    for (int i = 0; i < train_y_height; ++i) {
                        result += (train_y.get(i, 0) - prediction.get(i, 0) - beta) *
                                (train_y.get(i, 0) - prediction.get(i, 0) - beta);
                    }
                    return result / (double) train_y_height;
                }
                case 'p': {
                    // negative log likelihood of poission distribution
                    double result = 0;
                    for (int i = 0; i < train_y_height; ++i) {
                        result += std::exp(prediction.get(i, 0) + beta) -
                                  train_y.get(i, 0) * (prediction.get(i, 0) + beta);
                    }
                    return result / (double) train_y_height;
                }
                case 'b': {
                    // negative log likelihood of bernoulli distribution
                    auto prob = [](auto &&f, auto &&b) {
                        auto temp = 1.0 / (1.0 + std::exp( - f - b));
                        return temp < MAX_PROB_BERNOULLI ?
                               (temp > MIN_PROB_BERNOULLI ? temp : MIN_PROB_BERNOULLI) : MAX_PROB_BERNOULLI;
                    };
                    auto nll = [&prob](auto &&y, auto &&f, auto &&b) {
                        auto p = prob(f, b);
                        return - y * std::log(p) - (1 - y) * std::log(1 - p);
                    };
                    double result = 0;
                    for (int i = 0; i < train_y_height; ++i) {
                        result += nll(train_y.get(i, 0), prediction.get(i, 0), beta);
                    }
                    return result / (double) train_y_height;
                }
                case 't': {
                    double result = 0;
                    for (int i = 0; i < train_y_height; ++i) {
                        result += std::pow(std::exp(prediction.get(i, 0) + beta), 2.0 - params.tweedie_p) /
                                          (2.0 - params.tweedie_p) -
                                train_y.get(i, 0) * std::pow(std::exp(prediction.get(i, 0)),
                                                             1.0 - params.tweedie_p) / (1.0 - params.tweedie_p);
                    }
                    return result / (double) train_y_height;
                }
                default: {
                    throw std::invalid_argument("Specified distribution does not exist.");
                }
            }
        } else {
            switch (dist) {
                case 'n': {
                    double result = 0;
                    std::for_each(row_inds, row_inds + no_rows, [&result, &train_y, &prediction, &beta](int index)
                    {
                        result += (train_y.get(index, 0) - prediction.get(index, 0) - beta) *
                                  (train_y.get(index, 0) - prediction.get(index, 0) - beta);
                    });
//                    for (int i = 0; i < no_rows; ++i) {
//                        result += (train_y.get(row_inds[i], 0) - prediction.get(row_inds[i], 0) - beta) *
//                                (train_y.get(row_inds[i], 0) - prediction.get(row_inds[i], 0) - beta);
//                    }
                    return result / (double) no_rows;
                }
                case 'p': {
                    double result = 0;
                    std::for_each(row_inds, row_inds + no_rows, [&result, &train_y, &prediction, &beta](int index)
                    {
                        result += std::exp(prediction.get(index, 0) + beta) -
                                  train_y.get(index, 0) * (prediction.get(index, 0) + beta);
                    });
//                    for (int i = 0; i < no_rows; ++i) {
//                        result += std::exp(prediction.get(row_inds[i], 0) + beta) -
//                                  train_y.get(row_inds[i], 0) * (prediction.get(row_inds[i], 0) + beta);
//                    }
                    return result / (double) no_rows;
                }
                case 'b': {
                    auto prob = [](auto &&f, auto &&b) {
                        auto temp = 1.0 / (1.0 + std::exp( - f - b));
                        return temp < MAX_PROB_BERNOULLI ?
                               (temp > MIN_PROB_BERNOULLI ? temp : MIN_PROB_BERNOULLI) : MAX_PROB_BERNOULLI;
                    };
                    auto nll = [&prob](auto &&y, auto &&f, auto &&b) {
                        auto p = prob(f, b);
                        return - y * std::log(p) - (1 - y) * std::log(1 - p);
                    };
                    double result = 0;
                    std::for_each(row_inds, row_inds + no_rows, [&result, &train_y, &prediction, &beta, &prob, &nll](int index)
                    {
                        result += nll(train_y.get(index, 0), prediction.get(index, 0), beta);
                    });
//                    for (int i = 0; i < no_rows; ++i) {
//                        result += nll(train_y.get(row_inds[i], 0), prediction.get(row_inds[i], 0), beta);
//                    }
                    return result / (double) no_rows;
                }
                case 't': {
                    double result = 0;
                    std::for_each(row_inds, row_inds + no_rows, [&result, &train_y, &prediction, &beta, this](int index)
                    {
                        result += std::pow(std::exp(prediction.get(index, 0) + beta),
                                           2.0 - params.tweedie_p) / (2.0 - params.tweedie_p) -
                                  train_y.get(index, 0) * std::pow(std::exp(prediction.get(index, 0)),
                                                                         1.0 - params.tweedie_p) / (1.0 - params.tweedie_p);
                    });
//                    for (int i = 0; i < no_rows; ++i) {
//                        result += std::pow(std::exp(prediction.get(row_inds[i], 0) + beta),
//                                           2.0 - params.tweedie_p) / (2.0 - params.tweedie_p) -
//                                train_y.get(row_inds[i], 0) * std::pow(std::exp(prediction.get(row_inds[i], 0)),
//                                                   1.0 - params.tweedie_p) / (1.0 - params.tweedie_p);
//                    }
                    return result / (double) no_rows;
                }
                default: {
                    throw std::invalid_argument("Specified distribution does not exist.");
                }
            }
        }
    }


    template<typename T>
    inline T Loss_function<T>::estimate_mean(const Matrix<T> &ind_delta,
                                             const char &dist,
                                             const int *row_inds,
                                             int no_rows) const {

        int ind_delta_height = ind_delta.get_height();

        #ifdef _DEBUG_LOSS_FUNCTION
            assert(ind_delta.get_width() == 2);
        #endif

        if (row_inds == nullptr) {
            switch (dist) {
                case 'n': {
                    double result = 0;
                    for (int i = 0; i < ind_delta_height; ++i) {
                        result += ind_delta.get(i, 0);
                    }
                    return result / (double) ind_delta_height;
                }
                case 'p': {
                    double y_sum = 0, exp_pred_sum = 0;
                    for (int i = 0; i < ind_delta_height; ++i) {
                        y_sum += ind_delta.get(i, 0);
                        exp_pred_sum += ind_delta.get(i, 1);
                    }
                    return std::log(y_sum / exp_pred_sum);
                }
                case 'b': {
                    double numerator = 0, denominator = 0;
                    for (int i = 0; i < ind_delta_height; ++i) {
                        numerator += ind_delta.get(i, 0);
                        denominator += ind_delta.get(i, 1);
                    }
                    return numerator / denominator;
                }
                case 't': {
                    double numerator = 0, denominator = 0;
                    for (int i = 0; i < ind_delta_height; ++i) {
                        numerator += ind_delta.get(i, 0);
                        denominator += ind_delta.get(i, 1);
                    }
                    numerator += MIN_NUMERATOR_TWEEDIE;
                    return std::log(numerator / denominator);
                }
                default: {
                    throw std::invalid_argument("Specified distribution does not exist.");
                }
            }
        } else {
            switch (dist) {
                case 'n': {
                    double result = 0;
                    std::for_each(row_inds, row_inds + no_rows, [&result, &ind_delta](int index)
                    {
                        result += ind_delta.get(index, 0);
                    });
//                    for (int i = 0; i < no_rows; ++i) {
//                        result += ind_delta.get(row_inds[i], 0);
//                    }
                    return result / (double) no_rows;
                }
                case 'p': {
                    double y_sum = 0, exp_pred_sum = 0;
                    std::for_each(row_inds, row_inds + no_rows, [&y_sum, &exp_pred_sum, &ind_delta](int index)
                    {
                        y_sum += ind_delta.get(index, 0);
                        exp_pred_sum += ind_delta.get(index, 1);
                    });
//                    for (int i = 0; i < no_rows; ++i) {
//                        y_sum += ind_delta.get(row_inds[i], 0);
//                        exp_pred_sum += ind_delta.get(row_inds[i], 1);
//                    }
                    return std::log(y_sum / exp_pred_sum);
                }
                case 'b': {
                    double numerator = 0, denominator = 0;
                    std::for_each(row_inds, row_inds + no_rows, [&numerator, &denominator, &ind_delta](int index)
                    {
                        numerator += ind_delta.get(index, 0);
                        denominator += ind_delta.get(index, 1);
                    });
//                    int row_index;
//                    for (int i = 0; i < no_rows; ++i) {
//                        row_index = row_inds[i];
//                        numerator += ind_delta.get(row_index, 0);
//                        denominator += ind_delta.get(row_index, 1);
//                    }
                    return numerator / denominator;
                }
                case 't': {
                    double numerator = 0, denominator = 0;
                    std::for_each(row_inds, row_inds + no_rows, [&numerator, &denominator, &ind_delta](int index)
                    {
                        numerator += ind_delta.get(index, 0);
                        denominator += ind_delta.get(index, 1);
                    });
//                    for (int i = 0; i < no_rows; ++i) {
//                        numerator += ind_delta.get(row_inds[i], 0);
//                        denominator += ind_delta.get(row_inds[i], 1);
//                    }
                    numerator += MIN_NUMERATOR_TWEEDIE;
                    return std::log(numerator / denominator);
                }
                default: {
                    throw std::invalid_argument("Specified distribution does not exist.");
                }
            }
        }

    }

    template <typename T>
    inline void Loss_function<T>::link_function(Matrix<T> &in_and_out,
                                         char &dist) {

        int lpp_height = in_and_out.get_height();

        #ifdef _DEBUG_LOSS_FUNCTION
            assert(in_and_out.get_width() == 1);
        #endif

        double temp = 0;
        switch (dist) {
            case 'n': {
                //do nothing
                break;
            }
            case 'p': {
                for(int i = 0; i < lpp_height; ++i) {
                    temp = std::exp(in_and_out.get(i, 0));
                    in_and_out.assign(i, 0, temp);
                }
                break;
            }
            case 'b': {
                for(int i = 0; i < lpp_height; ++i) {
                    temp = 1.0 / ( 1.0 + std::exp(-in_and_out.get(i, 0)) );
                    in_and_out.assign(i, 0, temp);
                }
                break;
            }
            case 't': {
                for(int i = 0; i < lpp_height; ++i) {
                    temp = std::exp(in_and_out.get(i, 0));
                    in_and_out.assign(i, 0, temp);
                }
                break;
            }
            default: {
                throw std::invalid_argument("Specified distribution does not exist.");
            }
        }

    }

    template <typename T>
    inline T Loss_function<T>::inversed_link_function(T value, const char &dist) {

        switch (dist) {
            case 'n': {
                return value;
            }
            case 'p': {
                return std::log(value > TOLERANCE_INVERSE_LINK_FUNC ? value : TOLERANCE_INVERSE_LINK_FUNC);
            }
            case 'b': {
                return value;
            }
            case 't': {
                return std::log(value > TOLERANCE_INVERSE_LINK_FUNC ? value : TOLERANCE_INVERSE_LINK_FUNC);
            }
            default: {
                throw std::invalid_argument("Specified distribution does not exist.");
            }
        }

    }

    template <typename T>
    inline void Loss_function<T>::calculate_ind_delta(const Matrix<T> &train_y,
                                               const Matrix<T> &prediction,
                                               Matrix<T> &ind_delta,
                                               const char &dist,
                                               const int *row_inds,
                                               int no_rows) {

        if( row_inds == nullptr) {

            int y_height = train_y.get_height();

            #ifdef _DEBUG_LOSS_FUNCTION
                assert(y_height == prediction.get_height() &&
                               prediction.get_height() == ind_delta.get_height() &&
                               ind_delta.get_width() == 2);
            #endif

            switch (dist) {
                case 'n': {
                    for(int i = 0; i < y_height; ++i) {
                        ind_delta.assign(i, 0, train_y.get(i, 0) - prediction.get(i, 0));
                        ind_delta.assign(i, 1, 1);
                    }
                    break;
                }
                case 'p': {
                    for(int i = 0; i < y_height; ++i) {
                        ind_delta.assign(i, 0, train_y.get(i, 0));
                        ind_delta.assign(i, 1, std::exp(prediction.get(i, 0)));
                    }
                    break;
                }
                case 'b': {
                    auto numerator = [](auto &&y, auto &&f) {
                        auto p = 1.0 / (1.0 + std::exp( - f));
                        return y - p;
                    };
                    auto denominator = [](auto &&f) {
                        auto p = 1.0 / (1.0 + std::exp( - f));
                        return p * (1.0 - p);
                    };
                    for(int i = 0; i < y_height; ++i) {
                        ind_delta.assign(i, 0, numerator(train_y.get(i, 0), prediction.get(i, 0)));
                        ind_delta.assign(i, 1, denominator(prediction.get(i, 0)));
                    }
                    break;
                }
                case 't': {
                    for(int i = 0; i < y_height; ++i) {
                        ind_delta.assign(i, 0, train_y.get(i, 0) * std::exp(prediction.get(i, 0) * (1 - params.tweedie_p)));
                        ind_delta.assign(i, 1, std::exp(prediction.get(i, 0) * (2 - params.tweedie_p)));
                    }
                    break;
                }
                default: {
                    throw std::invalid_argument("Specified distribution does not exist.");
                }
            }
        }
        else {
            #ifdef _DEBUG_LOSS_FUNCTION
                assert(no_rows > 0);
            #endif
            switch (dist) {
                case 'n': {
                    for(int i = 0; i < no_rows; ++i) {
                        ind_delta.assign(row_inds[i], 0, train_y.get(row_inds[i], 0) - prediction.get(row_inds[i], 0));
                        ind_delta.assign(row_inds[i], 1, 1);
                    }
                    break;
                }
                case 'p': {
                    for(int i = 0; i < no_rows; ++i) {
                        ind_delta.assign(row_inds[i], 0, train_y.get(row_inds[i], 0));
                        ind_delta.assign(row_inds[i], 1, std::exp(prediction.get(row_inds[i], 0)));
                    }
                    break;
                }
                case 'b': {
                    auto numerator = [](auto &&y, auto &&f) {
                        auto p = 1.0 / (1.0 + std::exp( - f));
                        return y - p;
                    };
                    auto denominator = [](auto &&f) {
                        auto p = 1.0 / (1.0 + std::exp( - f));
                        return p * (1.0 - p);
                    };
                    for(int i = 0; i < no_rows; ++i) {
                        ind_delta.assign(row_inds[i], 0, numerator(train_y.get(row_inds[i], 0), prediction.get(row_inds[i], 0)));
                        ind_delta.assign(row_inds[i], 1, denominator(prediction.get(row_inds[i], 0)));
                    }
                    break;
                }
                case 't': {
                    for(int i = 0; i < no_rows; ++i) {
                        ind_delta.assign(row_inds[i], 0, train_y.get(row_inds[i], 0) * std::exp(prediction.get(row_inds[i], 0) * (1 - params.tweedie_p)));
                        ind_delta.assign(row_inds[i], 1, std::exp(prediction.get(row_inds[i], 0) * (2 - params.tweedie_p)));
                    }
                    break;
                }
                default: {
                    throw std::invalid_argument("Specified distribution does not exist.");
                }
            }
        }

    }

    template <typename T>
    inline Matrix<T> Loss_function<T>::first_comp(const Matrix<T> &train_y,
                                           const Matrix<T> &prediction,
                                           const char loss_function_type,
                                           const int *row_inds,
                                           int no_rows) {

        if (row_inds == nullptr) {

            int height = train_y.get_height();
            Matrix<T> result(height, 1, 0);

            #ifdef _DEBUG_LOSS_FUNCTION
                assert(train_y.get_height() == prediction.get_height());
            #endif

            switch (loss_function_type) {
                case 'n': {
                    for(int i = 0; i < height; ++i) {
                        result.assign(i, 0, 2 * (prediction.get(i, 0) - train_y.get(i, 0)));
                    }
                    return result;
                }
                case 'p': {
                    for(int i = 0; i < height; ++i) {
                        result.assign(i, 0, - train_y.get(i, 0));
                    }
                    return result;
                }
                case 'b': {
                    double prob;
                    for(int i = 0; i < height; ++i) {
                        prob = 1.0 / (1.0 + std::exp(-prediction.get(i, 0)));
                        result.assign(i, 0, - 0.5 * prob * (1.0 - prob) );
                    }
                    return result;
                }
                case 't': {
                    for(int i = 0; i < height; ++i) {
                        result.assign(i, 0,
                                      2 * train_y.get(i, 0) / (1 - params.tweedie_p) *
                                      std::exp(prediction.get(i, 0) * (1 - params.tweedie_p)) );
                    }
                    return result;
                }
                default: {
                    throw std::invalid_argument("Specified distribution does not exist.");
                }
            }

        }
        else {

            #ifdef _DEBUG_LOSS_FUNCTION
                assert(no_rows > 0);
            #endif

            Matrix<T> result(no_rows, 1, 0);

            switch (loss_function_type) {
                case 'n': {
                    for(int i = 0; i < no_rows; ++i) {
                        result.assign(i, 0, 2 * (prediction.get(row_inds[i], 0) - train_y.get(row_inds[i], 0)));
                    }
                    return result;
                }
                case 'p': {
                    for(int i = 0; i < no_rows; ++i) {
                        result.assign(i, 0, - train_y.get(row_inds[i], 0));
                    }
                    return result;
                }
                case 'b': {
                    double prob;
                    for(int i = 0; i < no_rows; ++i) {
                        prob = 1.0 / (1.0 + std::exp(-prediction.get(row_inds[i], 0)));
                        result.assign(i, 0, - 0.5 * prob * (1.0 - prob) );
                    }
                    return result;
                }
                case 't': {
                    for(int i = 0; i < no_rows; ++i) {
                        result.assign(i, 0,
                                      2 * train_y.get(row_inds[i], 0) / (1.0 - params.tweedie_p) *
                                              std::exp(prediction.get(row_inds[i], 0) * (1.0 - params.tweedie_p)) );
                    }
                    return result;
                }
                default: {
                    throw std::invalid_argument("Specified distribution does not exist.");
                }
            }

        }

    }

    template <typename T>
    inline Matrix<T> Loss_function<T>::second_comp(const Matrix<T> &train_y,
                                            const Matrix<T> &prediction,
                                            const char loss_function_type,
                                            const int *row_inds,
                                            int no_rows) {

        if (row_inds == nullptr) {

            int height = train_y.get_height();

            #ifdef _DEBUG_LOSS_FUNCTION
                assert(train_y.get_height() == prediction.get_height());
            #endif

            switch (loss_function_type) {
                case 'n': {
                    return Matrix<T> (height, 1, 1.0);
                }
                case 'p': {
                    Matrix<T> result(height, 1, 0);
                    for(int i = 0; i < height; ++i) {
                        result.assign(i, 0, std::exp(prediction.get(i, 0)));
                    }
                    return result;
                }
                case 'b': {
                    return Matrix<T> (height, 1, 0);
                }
                case 't': {
                    Matrix<T> result(height, 1, 0);
                    for(int i = 0; i < height; ++i) {
                        result.assign(i, 0, 2.0 / (2.0 - params.tweedie_p) *
                                      std::exp(prediction.get(i, 0) * (2.0 - params.tweedie_p)) );
                    }
                    return result;
                }
                default: {
                    throw std::invalid_argument("Specified distribution does not exist.");
                }
            }

        }
        else {

            #ifdef _DEBUG_LOSS_FUNCTION
                assert(no_rows > 0);
            #endif

            switch (loss_function_type) {
                case 'n': {
                    return Matrix<T> (no_rows, 1, 1);
                }
                case 'p': {
                    Matrix<T> result(no_rows, 1, 0);
                    for(int i = 0; i < no_rows; ++i) {
                        result.assign(i, 0, std::exp(prediction.get(row_inds[i], 0)));
                    }
                    return result;
                }
                case 'b': {
                    return Matrix<T> (no_rows, 1, 0);
                }
                case 't': {
                    Matrix<T> result(no_rows, 1, 0);
                    for(int i = 0; i < no_rows; ++i) {
                        result.assign(i, 0,
                                      2.0 / (2.0 - params.tweedie_p) *
                                      std::exp(prediction.get(row_inds[i], 0) * (2.0 - params.tweedie_p)) );
                    }
                    return result;
                }
                default: {
                    throw std::invalid_argument("Specified distribution does not exist.");
                }
            }

        }

    }

    template <typename T>
    inline T Loss_function<T>::loss_reduction(const T first_comp_in_loss,
                                       const T second_comp_in_loss,
                                       const T beta,
                                       const char loss_function_type) {

        switch (loss_function_type) {

            case 'n': {
                return beta *(first_comp_in_loss + beta * second_comp_in_loss);
            }
            case 'p': {
                return first_comp_in_loss * beta + second_comp_in_loss * (std::exp(beta) - 1);
            }
            case 'b': {
                return first_comp_in_loss * beta * beta;
            }
            case 't': {
                return first_comp_in_loss * (1.0 - std::exp(beta * (1.0 - params.tweedie_p))) +
                        second_comp_in_loss * (std::exp(beta * (2.0 - params.tweedie_p)) - 1.0);
            }
            default: {
                throw std::invalid_argument("Specified distribution does not exist.");
            }

        }

    }

}



