source('dbm_r.R')

no_samples = 10000
no_features = 30
data_x = matrix(rnorm(no_samples * no_features), no_samples, no_features)
data_y = matrix(cos(data_x[, 1]) * exp(data_x[, 3] + sin(data_x[, 5]) / 2) / (abs(data_x[, 8]) + 0.3)) +
                matrix(rnorm(no_samples), no_samples, 1)

create_params('pms', paste('dbm_no_bunches_of_learners 11 ',
                            'dbm_no_candidate_feature 10 ',
                            'dbm_shrinkage 0.25 ',
                            'cart_portion_candidate_split_point 0.01', setp = ""))

create_dbm('model', 'pms')
train_dbm('model', data_x, data_y, 0.2)

prediction = dbm_predict('model', data_x)

save_dbm('model', 'dbm.txt')
load_dbm('re_model', 'dbm.txt')

re_prediction = dbm_predict('re_model', data_x)
result = cbind(prediction, re_prediction)

pdp_result = pdp('model', data_x, 6)

ss_result = ss('model', data_x)