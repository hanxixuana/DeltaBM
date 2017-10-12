library(rPython)

python.exec('import sys; import os')

python.exec('sys.path.append(os.getcwd())')
python.exec('sys.path.append(os.getcwd() + \'/dbm_r\')')

python.exec('import dbm_py as dbm')

python.exec('import numpy as np')
python.exec('import pandas as pd')

create_params <- function(params_name, string) {

    if(missing(string)) {
        python.exec(paste(params_name, ' = dbm.Params()', sep = ""))
    } else {
        python.exec(paste(params_name, ' = dbm.Params()', sep = ""))
        python.exec(paste(params_name, '.set_params(\'', string, '\')', sep = ""))
    }

}

print_params <- function(params_name) {

    python.exec(paste(params_name, '.print_all()'))

}


create_dbm <- function(model_name, params_name) {

    if(python.get(paste('\'', params_name, '\' in dir()', sep = ""))) {
        python.exec(paste(model_name, ' = dbm.DBM(', params_name, ')', sep = ""))
    } else {
        stop(paste('Params:', params_name, 'has not been defined.'))
    }

}

train_dbm <- function(model_name, data_x, data_y, portion_for_validating) {

    if(python.get(paste('\'', model_name, '\' not in dir()', sep = ""))) {
        stop(paste('DBM:', model_name, 'has not been defined.', sep = ""))
    }

    if(!(class(data_x) == 'matrix' || class(data_x) == 'data.frame')) {
        stop('data_x should be either matrix or data.frame.')
    }

    if(!(class(data_y) == 'matrix' || class(data_y) == 'data.frame')) {
        stop('data_x should be either matrix or data.frame.')
    }

    if(class(data_x) == 'matrix' && class(data_y) == 'matrix') {
        python.assign('data_x_list', data_x)
        python.assign('data_y_list', data_y)

        python.exec('data_x = dbm.np2darray_to_float_matrix(np.array(data_x_list))')
        python.exec('data_y = dbm.np2darray_to_float_matrix(np.array(data_y_list))')

        python.exec(paste('ds = dbm.Data_set(data_x, data_y,', toString(portion_for_validating), ')', sep = ""))

        python.exec(paste(model_name, '.train(ds)'))
    } else if (class(data_x) == 'data.frame' && class(data_y) == 'data.frame') {
        python.assign('data_x_dict', data_x)
        python.assign('data_y_dict', data_y)

        python.exec('data_x = dbm.np2darray_to_float_matrix(pd.DataFrame(data_x_dict).as_matrix())')
        python.exec('data_y = dbm.np2darray_to_float_matrix(pd.DataFrame(data_y_dict).as_matrix())')

        python.exec(paste('ds = dbm.Data_set(data_x, data_y,', toString(portion_for_validating), ')', sep = ""))

        python.exec(paste(model_name, '.train(ds)'))
    } else {
        stop('data_x and data_y should be of the same type.')
    }

}

dbm_predict <- function(model_name, data_x) {

    if(python.get(paste('\'', model_name, '\' not in dir()', sep = ""))) {
        stop(paste('DBM:', model_name, 'has not been defined.', sep = ""))
    }

    if(class(data_x) == 'matrix') {
        python.assign('data_x_list', data_x)

        python.exec('data_x = dbm.np2darray_to_float_matrix(np.array(data_x_list))')
        python.exec(paste('prediction = ', model_name, '.predict(data_x)', sep = ""))
        python.exec('prediction = dbm.float_matrix_to_np2darray(prediction).tolist()')

        prediction = matrix(python.get('prediction'))

        return(prediction)

    } else if (class(data_x) == 'data.frame') {
        python.assign('data_x_dict', data_x)

        python.exec('data_x = dbm.np2darray_to_float_matrix(pd.DataFrame(data_x_dict).as_matrix())')

        python.exec(paste('prediction = ', model_name, '.predict(data_x)', sep = ""))
        python.exec('prediction = dbm.float_matrix_to_np2darray(prediction).tolist()')

        prediction = data.frame(python.get('prediction'))
        colnames(prediction) = c('predict')

        return(prediction)

    } else {
        stop('data_x should be either matrix or data.frame.')
    }

}

save_dbm <- function(model_name, file_name) {

    if(python.get(paste('\'', model_name, '\' not in dir()', sep = ""))) {
        stop(paste('DBM:', model_name, 'has not been defined.', sep = ""))
    }

    python.exec(paste(model_name, '.save(\'', file_name, '\')', sep = ""))

}

load_dbm <- function(model_name, file_name) {

    python.exec(paste(model_name, '= dbm.DBM(dbm.Params())', sep = ""))
    python.exec(paste(model_name, '.load(\'', file_name, '\')', sep = ""))

}

pdp <- function(model_name, data_x, feature_index) {

    if(python.get(paste('\'', model_name, '\' not in dir()', sep = ""))) {
        stop(paste('DBM:', model_name, 'has not been defined.', sep = ""))
    }

    if(class(data_x) == 'matrix') {
        python.assign('data_x_list', data_x)

        python.exec('data_x = dbm.np2darray_to_float_matrix(np.array(data_x_list))')

        python.exec(paste('pdp = ', model_name, '.pdp(data_x, ', feature_index, ')', sep = ""))
        python.exec('pdp = dbm.float_matrix_to_np2darray(pdp)')

        x_ticks = matrix(python.get("pdp[:, 0].tolist()"))
        mean = matrix(python.get("pdp[:, 1].tolist()"))
        upper = matrix(python.get("pdp[:, 2].tolist()"))
        lower = matrix(python.get("pdp[:, 3].tolist()"))

        return(matrix(c(x_ticks, mean, upper, lower), ncol = 4))

    } else if (class(data_x) == 'data.frame') {
        python.assign('data_x_dict', data_x)

        python.exec('data_x = dbm.np2darray_to_float_matrix(pd.DataFrame(data_x_dict).as_matrix())')

        python.exec(paste('pdp = ', model_name, '.pdp(data_x, ', feature_index, ')', sep = ""))
        python.exec('pdp = dbm.float_matrix_to_np2darray(pdp)')

        x_ticks = matrix(python.get("pdp[:, 0].tolist()"))
        mean = matrix(python.get("pdp[:, 1].tolist()"))
        upper = matrix(python.get("pdp[:, 2].tolist()"))
        lower = matrix(python.get("pdp[:, 3].tolist()"))

        return(data.frame(matrix(c(x_ticks, mean, upper, lower), ncol = 4)))

    } else {
        stop('data_x should be either matrix or data.frame.')
    }

}

ss <- function(model_name, data_x) {

    if(python.get(paste('\'', model_name, '\' not in dir()', sep = ""))) {
        stop(paste('DBM:', model_name, 'has not been defined.', sep = ""))
    }

    if(class(data_x) == 'matrix') {
        python.assign('data_x_list', data_x)

        python.exec('data_x = dbm.np2darray_to_float_matrix(np.array(data_x_list))')

        python.exec(paste('ss = ', model_name, '.ss(data_x)', sep = ""))
        python.exec('ss = dbm.float_matrix_to_np2darray(ss).tolist()')

        ss = matrix(python.get("ss"))

        return(ss)

    } else if (class(data_x) == 'data.frame') {
        python.assign('data_x_dict', data_x)

        python.exec('data_x = dbm.np2darray_to_float_matrix(pd.DataFrame(data_x_dict).as_matrix())')

        python.exec(paste('ss = ', model_name, '.ss(data_x)', sep = ""))
        python.exec('ss = dbm.float_matrix_to_np2darray(ss).tolist()')

        ss = data.frame(python.get("ss"))
        colnames(ss) = c('significance')

        return(ss)

    } else {
        stop('data_x should be either matrix or data.frame.')
    }

}