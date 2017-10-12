.. DBM documentation master file, created by
   sphinx-quickstart on Thu Mar  2 15:40:49 2017.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to DBM's documentation!
===============================

.. toctree::
   :maxdepth: 2
   :caption: Contents:

This is the document of the Python APIs of Delta Boosting Machine. Classes and functions are listed and described.

Classes
=======

Matrix
------

.. autoclass:: dbm_py.interface.Matrix
   :members: __init__, shape, get, show, save, clear, assign, from_np2darray, to_np2darray

Data Set
--------

.. autoclass:: dbm_py.interface.Data_set
   :members: __init__, get_train_x, get_train_y, get_validate_x, get_validate_y

Parameters
----------

    +-----------------------------------------------+-------+---------------------------------------------+
    |Parameter Name                                 |Type   |Meaning                                      |
    +-----------------------------------------------+-------+---------------------------------------------+
    |dbm_no_bunches_of_learners                     |int    |number of boostraped BLs                     |
    +-----------------------------------------------+-------+---------------------------------------------+
    |dbm_no_candidate_feature                       |int    |number of features for each BL               |
    +-----------------------------------------------+-------+(< total number of features)                 +
    |                                               |       |                                             |
    +-----------------------------------------------+-------+---------------------------------------------+
    |dbm_portion_train_sample                       |double |percentage for training each BL              |
    +-----------------------------------------------+-------+---------------------------------------------+
    |dbm_no_cores                                   |int    |number of BL in each bunch                   |
    +-----------------------------------------------+-------+(number of cores used)                       +
    |                                               |       |                                             |
    +-----------------------------------------------+-------+---------------------------------------------+
    |dbm_loss_function                              |char   |(n)ormal, (b)ernoulli, (p)oisson             |
    +-----------------------------------------------+-------+or (t)weedie                                 +
    |                                               |       |                                             |
    +-----------------------------------------------+-------+---------------------------------------------+
    |dbm_display_training_progress                  |bool   |whether to display training progress or not  |
    +-----------------------------------------------+-------+---------------------------------------------+
    |dbm_record_every_tree                          |bool   |whether to record trees in a file or not     |
    +-----------------------------------------------+-------+---------------------------------------------+
    |dbm_freq_showing_loss_on_test                  |int    |show loss on test after how many             |
    +-----------------------------------------------+-------+bunches of BLs                               +
    |                                               |       |                                             |
    +-----------------------------------------------+-------+---------------------------------------------+
    |dbm_shrinkage                                  |double |shrinkage for each BL                        |
    +-----------------------------------------------+-------+---------------------------------------------+
    |dbm_nonoverlapping_training                    |int    |whether to BLs in a bunch use                |
    +-----------------------------------------------+-------+nonoverlapping samples or not                +
    |                                               |       |                                             |
    +-----------------------------------------------+-------+---------------------------------------------+
    |dbm_remove_rows_containing_nans                |int    |whether to remove rows containing NaNs in    |
    +-----------------------------------------------+-------+training every BL                            +
    |                                               |       |                                             |
    +-----------------------------------------------+-------+---------------------------------------------+
    |dbm_min_no_samples_per_bl                      |int    |minimal number of samples for trainin        |
    +-----------------------------------------------+-------+every BL                                     +
    |                                               |       |                                             |
    +-----------------------------------------------+-------+---------------------------------------------+
    |dbm_portion_for_trees                          |double |percentage of BLs using trees                |
    +-----------------------------------------------+-------+---------------------------------------------+
    |dbm_random_seed                                |int    |random seed (random < 0 and fixed >= 0)      |
    +-----------------------------------------------+-------+---------------------------------------------+
    |dbm_portion_for_lr                             |double |percentage of BLs using linear regression    |
    +-----------------------------------------------+-------+---------------------------------------------+
    |dbm_portion_for_s                              |double |percentage of BLs using splines              |
    +-----------------------------------------------+-------+---------------------------------------------+
    |dbm_portion_for_k                              |double |percentage of BLs using k-means              |
    +-----------------------------------------------+-------+---------------------------------------------+
    |dbm_portion_for_nn                             |double |should be 0                                  |
    +-----------------------------------------------+-------+---------------------------------------------+
    |dbm_portion_for_d                              |double |percentage of BLs using dominating           |
    +-----------------------------------------------+-------+principal component stairs                   +
    |                                               |       |                                             |
    +-----------------------------------------------+-------+---------------------------------------------+
    |dbm_accumulated_portion                        |double |unused                                       |
    +-----------------------------------------------+-------+                                             +
    |    _shrinkage_for_selected_b                  |       |                                             |
    +-----------------------------------------------+-------+---------------------------------------------+
    |dbm_portion_shrinkage_for_unselected_bl        |double |unused                                       |
    +-----------------------------------------------+-------+---------------------------------------------+
    |tweedie_p                                      |double |p of tweedie should in (1, 2)                |
    +-----------------------------------------------+-------+---------------------------------------------+
    |splines_no_knot                                |int    |number of knots of splines                   |
    +-----------------------------------------------+-------+---------------------------------------------+
    |splines_portion_of_pairs                       |double |percentage of pairs of perdictors            |
    +-----------------------------------------------+-------+considered                                   +
    |                                               |       |                                             |
    +-----------------------------------------------+-------+---------------------------------------------+
    |splines_regularization                         |double |ridge regression penalty                     |
    +-----------------------------------------------+-------+---------------------------------------------+
    |splines_hinge_coefficient                      |double |coefficient in splines                       |
    +-----------------------------------------------+-------+---------------------------------------------+
    |kmeans_no_centroids                            |int    |number of centroids                          |
    +-----------------------------------------------+-------+---------------------------------------------+
    |kmeans_max_iteration                           |int    |max number of iterations of training         |
    +-----------------------------------------------+-------+---------------------------------------------+
    |kmeans_tolerance                               |double |max tolerated error                          |
    +-----------------------------------------------+-------+---------------------------------------------+
    |kmeans_fraction_of_pairs                       |double |percentage of pairs of predictors            |
    +-----------------------------------------------+-------+considered                                   +
    |                                               |       |                                             |
    +-----------------------------------------------+-------+---------------------------------------------+
    |nn_no_hidden_neurons                           |int    |number of hidden neurons                     |
    +-----------------------------------------------+-------+---------------------------------------------+
    |nn_step_size                                   |double |stochastic gradient descent step size        |
    +-----------------------------------------------+-------+---------------------------------------------+
    |nn_validate_portion                            |double |percentage of samples used for validating    |
    +-----------------------------------------------+-------+---------------------------------------------+
    |nn_batch_size                                  |int    |number of samples in a batch                 |
    +-----------------------------------------------+-------+---------------------------------------------+
    |nn_max_iteration                               |int    |maximal number of iterations of training     |
    +-----------------------------------------------+-------+---------------------------------------------+
    |nn_no_rise_of_loss_on_validate                 |int    |maximal number rises of loss on              |
    +-----------------------------------------------+-------+validation set                               +
    |                                               |       |                                             |
    +-----------------------------------------------+-------+---------------------------------------------+
    |cart_min_samples_in_a_node                     |int    |minimal numbers in a node of a tree          |
    +-----------------------------------------------+-------+---------------------------------------------+
    |cart_max_depth                                 |int    |maximal numbers of levels of a tree          |
    +-----------------------------------------------+-------+---------------------------------------------+
    |cart_prune                                     |int    |whether to prune after training              |
    +-----------------------------------------------+-------+---------------------------------------------+
    |lr_regularization                              |double |ridge regression penalty                     |
    +-----------------------------------------------+-------+---------------------------------------------+
    |dpcs_no_ticks                                  |int    |number of stairs in the direction of         |
    +-----------------------------------------------+-------+dominating principal component               +
    |                                               |       |                                             |
    +-----------------------------------------------+-------+---------------------------------------------+
    |dpcs_range_shrinkage_of_ticks                  |double |shrinkage of the range in the direction of   |
    +-----------------------------------------------+-------+dominating principal component               +
    |                                               |       |                                             |
    +-----------------------------------------------+-------+---------------------------------------------+
    |dbm_do_perf                                    |bool   |whether to record performance on both        |
    +-----------------------------------------------+-------+training sets                                +
    |                                               |       |                                             |
    +-----------------------------------------------+-------+---------------------------------------------+
    |pdp_no_x_ticks                                 |int    |number of ticks in x-axis                    |
    +-----------------------------------------------+-------+---------------------------------------------+
    |pdp_no_resamplings                             |int    |number of resamplings for bootstrapping      |
    +-----------------------------------------------+-------+---------------------------------------------+
    |pdp_resampling_portion                         |double |percentage of samples in each bootstrap      |
    +-----------------------------------------------+-------+---------------------------------------------+
    |pdp_ci_bandwidth                               |double |width of the confidence interval             |
    +-----------------------------------------------+-------+---------------------------------------------+
    |pdp_save_files                                 |int    |whether to save the result                   |
    +-----------------------------------------------+-------+---------------------------------------------+

.. autoclass:: dbm_py.interface.Params
    :members: __init__, set_params, print_all

Delta Boosting Machines
-----------------------

.. autoclass:: dbm_py.interface.DBM
   :members: __init__, train, train_with_monotonic_constraints, predict, pdp, ss, calibrate_plot, interact, save_performance, save, load

Delta Boosting Machines with Automatic BL Selection
---------------------------------------------------

.. autoclass:: dbm_py.interface.AUTO_DBM
   :members: __init__, train, train_with_monotonic_constraints, predict, pdp, ss, calibrate_plot, interact, save_performance, save, load

Functions
=========

.. autofunction:: dbm_py.interface.np2darray_to_float_matrix

.. autofunction:: dbm_py.interface.float_matrix_to_np2darray

.. autofunction:: dbm_py.interface.string_to_params


