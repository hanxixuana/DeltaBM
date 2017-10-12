#!/usr/bin/env python

import numpy as np
import os
import pandas as pd
import sys
import random

random.seed(1)

sys.path.append(os.getcwd())

import dbm_py.interface as dbm

x = dbm.Matrix(500000, 10)
x_nd_array = dbm.float_matrix_to_np2darray(x)
y_nd_array = x_nd_array[:, 0] ** 2 / 2  - \
             x_nd_array[:, 3] * 2 * np.exp(x_nd_array[:, 6] / 10) + \
             4 * np.cos(x_nd_array[:, 8] / 2) * np.exp(x_nd_array[:, 5] / 2) + \
             1.5 * np.random.randn(500000)
y_nd_array = np.round(np.abs(y_nd_array))

y = dbm.np2darray_to_float_matrix(y_nd_array[:, np.newaxis])

c = dbm.Data_set(x, y, 0.2, 1)
train_x = c.get_train_x()

s = 'dbm_no_bunches_of_learners 30 dbm_no_cores 3 dbm_loss_function p ' \
    'dbm_portion_train_sample 0.3 dbm_no_candidate_feature 5 dbm_shrinkage 0.1 ' \
    'dbm_random_seed 1 '
params = dbm.Params()
params.set_params(s)
# =====================================

auto_model = dbm.AUTO_DBM(params)
auto_model.train(c)

auto_predict = auto_model.predict(c.get_validate_x())

auto_result = pd.DataFrame(np.concatenate([auto_predict.to_np2darray(), c.get_validate_y().to_np2darray()], 1))

auto_model.save('auto_dbm.txt')
# =====================================

model = dbm.DBM(params)

model.train(c)

predict = model.predict(c.get_validate_x())

result = pd.DataFrame(np.concatenate([predict.to_np2darray(), c.get_validate_y().to_np2darray()], 1))

model.save('dbm.txt')

pdp = model.pdp(c.get_train_x(), 3)

ss = model.ss(c.get_train_x())

re_model = dbm.DBM(dbm.Params())
re_model.load('dbm.txt')

re_predict = re_model.predict(c.get_validate_x())

re_result = pd.DataFrame(np.concatenate([predict.to_np2darray(),
                                         re_predict.to_np2darray(),
                                         c.get_validate_y().to_np2darray()], 1))

re_pdp = re_model.pdp(c.get_train_x(), 3)

re_ss = re_model.ss(c.get_train_x())

re_model.train(c)
