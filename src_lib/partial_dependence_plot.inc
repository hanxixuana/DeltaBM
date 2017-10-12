//
// Created by yizhou on 28/12/16.
//

namespace dbm {

    void gen_tick_inc_list(int total_index, int no_predictor, int no_tick,
                          int *row_ind) {
        /**
         * Generate row indices for multi-way partial dependence plot to
         * choose values in subset of sample.
         * @param total_index index indicating the choose. A no_tick based integer
         * @param no_predictor number of predictor
         * @param no_tick number of values for each predictor
         * @return row_ind[] array containing row indices for each predictor
         */

        int x = total_index;
        for (int i = 0; i < no_predictor; i ++) {
            row_ind[i] = x % no_tick;
            x /= no_tick;
        } // i
    } // gen_tick_inc_list

    template<typename T>
    Matrix<T> &DBM<T>::partial_dependence_plot(const Matrix<T> &data,
                                               const int &predictor_ind) {
#ifdef _DEBUG_MODEL
        assert(total_no_feature == data.get_width());
#endif

        Matrix<T> modified_data = copy(data);

        int data_height = data.get_height(),
                *row_inds = new int[data_height],
                resampling_size = int(data_height * params.pdp_resampling_portion);
        for (int i = 0; i < data_height; i++)
            row_inds[i] = i;

        if (pdp_result != nullptr)
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

        for (int i = 0; i < params.pdp_no_x_ticks; i++) {

            pdp_result->assign(i, 0, predictor_min + i * (predictor_max - predictor_min) / (params.pdp_no_x_ticks - 1));

            for (int j = 0; j < data_height; j ++)
                modified_data.assign(j, predictor_ind, pdp_result->get(i, 0));

            for (int j = 0; j < params.pdp_no_resamplings; j ++) {

#ifdef _OMP
#pragma omp parallel default(shared)
                {
                    int resampling_id = no_cores * j + omp_get_thread_num();

                    int *thread_row_inds = new int[data_height];
                    std::copy(row_inds, row_inds + data_height, thread_row_inds);
                    shuffle(thread_row_inds, data_height, seeds[resampling_id]);

                    Matrix<T> *resampling_prediction = new Matrix<T>(resampling_size, 1, 0);
                    predict(modified_data.rows(thread_row_inds, resampling_size), *resampling_prediction);

                    bootstraping.assign(i, resampling_id, resampling_prediction->col_average(0));

                    delete[] thread_row_inds;
                    delete resampling_prediction;
                } // omp parallel default

#else

                shuffle(row_inds, data_height);

                resampling_prediction.clear();
                predict(modified_data.rows(row_inds, resampling_size), resampling_prediction);

                bootstraping.assign(i, j, resampling_prediction.col_average(0));

#endif
            } // j

            pdp_result->assign(i, 1, bootstraping.row_average(i));

            standard_dev = bootstraping.row_std(i);

            pdp_result->assign(i, 2, pdp_result->get(i, 1) - params.pdp_ci_bandwidth / 2.0 * standard_dev);
            pdp_result->assign(i, 3, pdp_result->get(i, 1) + params.pdp_ci_bandwidth / 2.0 * standard_dev);

        } // i

#ifdef _OMP
        delete[] seeds;
#endif

        delete[] row_inds;

        return *pdp_result;

    } // partial_dependence_plot

    template<typename T>
    Matrix<T> &DBM<T>::partial_dependence_plot(const Matrix<T> &data,
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
        for (int i = 0; i < data_height; i++)
            row_inds[i] = i;

        if (pdp_result != nullptr)
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

        for (int i = 0; i < params.pdp_no_x_ticks; i++) {

            pdp_result->assign(i, 0, predictor_min + i * (predictor_max - predictor_min) / (params.pdp_no_x_ticks - 1));

            for (int j = 0; j < data_height; j ++)
                modified_data.assign(j, predictor_ind, pdp_result->get(i, 0));

            for (int j = 0; j < params.pdp_no_resamplings; j ++) {

#ifdef _OMP
#pragma omp parallel default(shared)
                {
                    int resampling_id = no_cores * j + omp_get_thread_num();

                    int *thread_row_inds = new int[data_height];
                    std::copy(row_inds, row_inds + data_height, thread_row_inds);
                    shuffle(thread_row_inds, data_height, seeds[resampling_id]);

                    Matrix<T> *resampling_prediction = new Matrix<T>(resampling_size, 1, 0);
                    predict(modified_data.rows(thread_row_inds, resampling_size), *resampling_prediction);

                    bootstraping.assign(i, resampling_id, resampling_prediction->col_average(0));

                    delete[] thread_row_inds;
                    delete resampling_prediction;
                } // omp parallel default

#else

                shuffle(row_inds, data_height);

                resampling_prediction.clear();
                predict(modified_data.rows(row_inds, resampling_size), resampling_prediction);

                bootstraping.assign(i, j, resampling_prediction.col_average(0));

#endif

            } // j

            pdp_result->assign(i, 1, bootstraping.row_average(i));

            standard_dev = bootstraping.row_std(i);

            pdp_result->assign(i, 2, pdp_result->get(i, 1) - params.pdp_ci_bandwidth / 2.0 * standard_dev);
            pdp_result->assign(i, 3, pdp_result->get(i, 1) + params.pdp_ci_bandwidth / 2.0 * standard_dev);

        } // i

#ifdef _OMP
        delete[] seeds;
#endif

        delete[] row_inds;

        return *pdp_result;

    }

    template<typename T>
    Matrix<T> &DBM<T>::partial_dependence_plot(const Matrix<T> &data,
                                               const int *predictor_ind,
                                               int no_predictor) {
        /**
         * generate multi-way partial dependence plot data
         * @param data dataset used to predict
         * @param predictor_ind index of predictors
         * @param no_predictor number of predictors
         */

#ifdef _DEBUG_MODEL
        assert(total_no_feature == data.get_width());
#endif

        Matrix<T> modified_data = data;

        int data_height = data.get_height();
        int *row_inds = new int[data_height];
        int resampling_size = int(data_height * params.pdp_resampling_portion);

        for (int i = 0; i < data_height; i++)
            row_inds[i] = i;

        if (pdp_result != nullptr)
            delete pdp_result;
        int pdp_result_height = 1;
        for (int i = 0; i < no_predictor; i++)
            pdp_result_height *= params.pdp_no_x_ticks;
        pdp_result = new Matrix<T>(pdp_result_height, no_predictor + 2, 0);

        T *predictor_min = new T[no_predictor];
        T *predictor_max = new T[no_predictor];
        T *predictor_tick_inc = new T[no_predictor];
        T standard_deviation;

        int total_no_resamplings = params.pdp_no_resamplings * no_cores;
        Matrix<T> bootstraping(1, total_no_resamplings, 0);

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


        for (int i = 0; i < no_predictor; i ++) {
            predictor_min[i] = data.get_col_min(predictor_ind[i]);
            predictor_max[i] = data.get_col_max(predictor_ind[i]);
            predictor_tick_inc[i] = (predictor_max[i] - predictor_min[i]) / (T)(params.pdp_no_x_ticks - 1);
        } // i

        int pdp_predictor_inc[no_predictor];
        for (int i = 0; i < pdp_result_height; i++) {

            gen_tick_inc_list(i, no_predictor, params.pdp_no_x_ticks, pdp_predictor_inc);
            for (int j = 0; j < no_predictor; j ++) {
                pdp_result->assign(i, j, predictor_min[j] + predictor_tick_inc[j] * pdp_predictor_inc[j]);
            } // j

            for (int j = 0; j < data_height; j ++){
                for (int k = 0; k < no_predictor; k ++) {
                    modified_data.assign(j, predictor_ind[k], pdp_result->get(i, predictor_ind[k]));
                } // k
            } // j

            for (int j = 0; j < params.pdp_no_resamplings; j ++) {

#ifdef _OMP
#pragma omp parallel default(shared)

                {
                    int resampling_id = no_cores * j + omp_get_thread_num();

                    int *thread_row_inds = new int[data_height];
                    std::copy(row_inds, row_inds + data_height, thread_row_inds);
                    shuffle(thread_row_inds, data_height, seeds[resampling_id]);

                    Matrix<T> *resampling_prediction = new Matrix<T>(resampling_size, 1, 0);
                    predict(modified_data.rows(thread_row_inds, resampling_size), *resampling_prediction);

                    bootstraping.assign(0, resampling_id, resampling_prediction->col_average(0));

                    delete[] thread_row_inds;
                    delete resampling_prediction;
                } // omp parallel default

#else

                shuffle(row_inds, data_height);

                resampling_prediction.clear();
                predict(modified_data.rows(row_inds, resampling_size), resampling_prediction);

                bootstraping.assign(0, j, resampling_prediction.col_average(0));

#endif
            } // j

            pdp_result->assign(i, no_predictor, bootstraping.row_average(0));

            standard_deviation = bootstraping.row_std(0);
            pdp_result->assign(i, no_predictor + 1, standard_deviation);

        } // i

#ifdef _OMP
        delete[] seeds;
#endif

        delete[] row_inds;

        delete[] predictor_min;
        delete[] predictor_max;
        delete[] predictor_tick_inc;

        return *pdp_result;
    } // partial_dependence_plot

} // namespace dbm
