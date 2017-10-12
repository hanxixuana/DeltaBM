//
// Created by yizhou on 28/12/16.
//

namespace dbm{

    void gen_predictor_list(int choose_index, const int* predictor_ind,
                            int* partial_predictor_ind, int& no_predictor) {
        /**
         * generate predictor list according to choose_index
         * to determine which predictors are used to predict
         */

        int x = choose_index, k = 0;
        no_predictor = 0;
        while(x > 0) {
            if (! (x % 2))
                partial_predictor_ind[no_predictor ++] = predictor_ind[k];
            x /= 2;
            k ++;
        } // while

    } // gen_predictor_list

    template<typename T>
    T DBM<T>::interact(const Matrix<T>& data,
                       const int* predictor_ind,
                       int no_predictor) {
        /**
         * computes Friedman’s H-statistic to assess the relative
         * strength of interaction effects in non-linear models.
         * H is on the scale of [0-1]
         * with higher values indicating larger interaction effects
         * @param data dataset used to train
         * @param predictor_ind index of predictors involved in calculation
         * @param no_predictor number of predictors
         * @return value of H
         * @TODO add bootstrapping
         */

        Matrix<T> modified_data = data;

        int data_height = data.get_height();

        Matrix<T> prediction(data_height, 1, T(0.));
        Matrix<T> full_prediction(data_height, 1, T(0.));
        Matrix<T> total_prediction(data_height, 1, T(0.));
        Matrix<T> tmp_prediction(data_height, 1, T(0.));

        int *partial_predictor_ind = new int[no_predictor];
        int no_partial_predictor;
        int sgn;
        T average;

        for (int combination = 1; combination < (1 << no_predictor); combination ++) {
            gen_predictor_list(combination, predictor_ind, partial_predictor_ind, no_partial_predictor);
            sgn = (no_partial_predictor % 2 == no_predictor % 2) ? 1 : -1;

            for (int i = 0; i < data_height; i++) {
                prediction.clear();

                for (int j = 0; j < data_height; j++) {
                    for (int k = 0; k < no_partial_predictor; k++) {
                        modified_data.assign(j, partial_predictor_ind[k], data.get(i, partial_predictor_ind[k]));
                    } // k
                } // j

                predict(modified_data, prediction);
                tmp_prediction.assign(i, 0, prediction.col_average(0));
            } // i

            average = tmp_prediction.col_average(0);
            for (int i = 0; i < data_height; i ++) {
                total_prediction.assign(i, 0, sgn * (tmp_prediction.get(i, 0) - average));
            } // i
            if (combination == (1 << no_predictor) - 1) {
                for (int i = 0; i < data_height; i ++) {
                    full_prediction.assign(i, 0, tmp_prediction.get(i, 0) - average);
                } // i
            } // if
        } // combination

        T top = 0, btm = 0;
        for (int i = 0; i < data_height; i ++) {
            top += total_prediction.get(i, 0) * total_prediction.get(i, 0);
            btm += full_prediction.get(i, 0) * full_prediction.get(i, 0);
        } // i

        T H = sqrt(top / btm);

        delete[] partial_predictor_ind;

        return H;
    } // dbm::interact

    template<typename T>
    T AUTO_DBM<T>::interact(const Matrix<T>& data,
                       const int* predictor_ind,
                       int no_predictor) {
        /**
         * computes Friedman’s H-statistic to assess the relative
         * strength of interaction effects in non-linear models.
         * H is on the scale of [0-1]
         * with higher values indicating larger interaction effects
         * @param data dataset used to train
         * @param predictor_ind index of predictors involved in calculation
         * @param no_predictor number of predictors
         * @return value of H
         * @TODO add bootstrapping
         */

        Matrix<T> modified_data = data;

        int data_height = data.get_height();

        Matrix<T> prediction(data_height, 1, T(0.));
        Matrix<T> full_prediction(data_height, 1, T(0.));
        Matrix<T> total_prediction(data_height, 1, T(0.));
        Matrix<T> tmp_prediction(data_height, 1, T(0.));

        int *partial_predictor_ind = new int[no_predictor];
        int no_partial_predictor;
        int sgn;
        T average;

        for (int combination = 1; combination < (1 << no_predictor); combination ++) {
            gen_predictor_list(combination, predictor_ind, partial_predictor_ind, no_partial_predictor);
            sgn = (no_partial_predictor % 2 == no_predictor % 2) ? 1 : -1;

            for (int i = 0; i < data_height; i++) {
                prediction.clear();

                for (int j = 0; j < data_height; j++) {
                    for (int k = 0; k < no_partial_predictor; k++) {
                        modified_data.assign(j, partial_predictor_ind[k], data.get(i, partial_predictor_ind[k]));
                    } // k
                } // j

                predict(modified_data, prediction);
                tmp_prediction.assign(i, 0, prediction.col_average(0));
            } // i

            average = tmp_prediction.col_average(0);
            for (int i = 0; i < data_height; i ++) {
                total_prediction.assign(i, 0, sgn * (tmp_prediction.get(i, 0) - average));
            } // i
            if (combination == (1 << no_predictor) - 1) {
                for (int i = 0; i < data_height; i ++) {
                    full_prediction.assign(i, 0, tmp_prediction.get(i, 0) - average);
                } // i
            } // if
        } // combination

        T top = 0, btm = 0;
        for (int i = 0; i < data_height; i ++) {
            top += total_prediction.get(i, 0) * total_prediction.get(i, 0);
            btm += full_prediction.get(i, 0) * full_prediction.get(i, 0);
        } // i

        T H = sqrt(top / btm);

        delete[] partial_predictor_ind;

        return H;
    } // auto_dbm::interact

} // namespace dbm
