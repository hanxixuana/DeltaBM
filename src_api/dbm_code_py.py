#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
sys.path.append(os.path.abspath('../api'))

import dbm_py.lib_dbm_cpp_to_python_interface as dbm_cpp_interface
import numpy as np


class Matrix(object):

    def __init__(self,
                 height = None,
                 width = None,
                 val = None,
                 file_name = None,
                 sep = None,
                 mat = None):
        """
        This is the class of Matrix used in DBM. To feed the training
        and prediction data to DBM, they should be converted to
        Matrix first of all. The Matrix interface provides four ways
        of initialization, i.e. initialization with random values in
        [-1, 1], initialization with a user-provided value,
        initialization from a file and initialization with a
        Float_Matrix object. One may also initializing a matrix with
        any values and then use the method from_np2darray to transfer
        the values from a numpy array of the same shape to it.

        Initialization with random values in [-1, 1]
        :param height: height of the matrix
        :param width: width of the matrix

        Initialization with a user-provided value
        :param val: a particular value for initialization

        Initialization from a file
        :param file_name: file name of the file where data comes from
        :param sep: seperator used in the file

        Initialization with a Float_Matrix object
        :param mat: a Float_Matrix object

        Note:
            1. When initializing from a file, the format should be
            correct. One may first of all save a matrix to a file
            and look at the file and see how it looks like.
            2. Avoid directly using Float_Matrix.
            3. Converting tools np2darray_to_float_matrix and
            float_matrix_to_np2darray are provided.

        """

        if(height is not None
           and width is not None
           and mat is None):

            if(val is None
               and file_name is None
               and sep is None):
                self.mat = dbm_cpp_interface.Float_Matrix(height,
                                                          width)

            elif(val is not None
                 and file_name is None
                 and sep is None):
                self.mat = dbm_cpp_interface.Float_Matrix(height,
                                                          width,
                                                          val)

            elif(val is None
                 and file_name is not None
                 and sep is None):
                self.mat = dbm_cpp_interface.Float_Matrix(height,
                                                          width,
                                                          file_name)

            elif(val is None
                 and file_name is not None
                 and sep is not None):
                self.mat = dbm_cpp_interface.Float_Matrix(height,
                                                          width,
                                                          file_name,
                                                          sep)

            else:
                raise ValueError('Error!')

        elif(height is None
             and width is None
             and mat is not None):
            self.mat = mat

    # shape
    def shape(self):
        """
        Return a list containing the shape of the matrix.

        :return: [matrix height, matrix width]
        """
        return [self.mat.get_height(), self.mat.get_width()]

    # get
    def get(self, i, j):
        """
        Access to a particular element in the matrix.

        :param i: height of the element
        :param j: width of the element
        :return: the element

        Note: i and j should be in the correct ranges
        """
        return self.mat.get(i, j)

    # print to screen
    def show(self):
        """
        Print to screen the data stored in the matrix.
        """
        self.mat.show()

    # save to file
    def save(self, file_name, sep = '\t'):
        """
        Save the data stored in it to a file.

        :param file_name: a string
        :param sep: a character
        """
        self.mat.print_to_file(file_name, sep)

    # clear all values to 0
    def clear(self):
        """
        Set all elements to 0.
        """
        self.mat.clear()

    # elementwise assignment
    def assign(self, i, j, val):
        """
        Assign a value to a particular element.

        :param i: height of the element
        :param j: width of the element
        :param val: value to be assigned
        """
        self.mat.assign(i, j, val)

    # conversion of Numpy 2d array to Matrix
    def from_np2darray(self, source):
        """
        Assign the data stored in a two-dimensional numpy array to
        this matrix.

        :param source: a two-dimensional numpy array of the same shape as this matrix
        """
        try:
            assert source.shape.__len__() == 2
            assert self.mat.get_height() == source.shape[0]
            assert self.mat.get_width() == source.shape[1]
        except AssertionError as e:
            print(source.shape)
            print((self.mat.get_height(), self.mat.get_width()))
            raise ValueError('The Numpy array may not have the same '
                             'shape as the target.')
        for i in range(self.mat.get_height()):
            for j in range(self.mat.get_width()):
                self.mat.assign(i, j, source[i][j])

    # conversion of Matrix to Numpy 2d array
    def to_np2darray(self):
        """
        Assign the data stored in this matrix to a two-dimensional numpy array and return it.

        :return: a two-dimensional numpy array of the same shape as this matrix
        """
        result = np.zeros([self.mat.get_height(),
                           self.mat.get_width()])
        for i in range(self.mat.get_height()):
            for j in range(self.mat.get_width()):
                result[i][j] = self.mat.get(i, j)
        return result



class Data_set(object):

    def __init__(self, data_x, data_y, portion_for_validating, random_seed = -1):
        """
        This is the class of Data_set that provides an easy to tool for splitting all data into training and validating
        parts.

        :param data_x: a Matrix object
        :param data_y: a Matrix object
        :param portion_for_validating: percentage of the whole data used for validating
        :param random_seed: optional random seed (random if negative or fixed if non-negative)
        """
        self.data_set = \
            dbm_cpp_interface.Data_Set(data_x.mat,
                                       data_y.mat,
                                       portion_for_validating,
                                       random_seed)

    def get_train_x(self):
        """
        Return the part of predictors for training.

        :return: a Matrix object
        """
        return Matrix(mat = self.data_set.get_train_x())

    def get_train_y(self):
        """
        Return the part of responses for training.

        :return: a Matrix object
        """
        return Matrix(mat = self.data_set.get_train_y())

    def get_validate_x(self):
        """
        Return the part of predictors for validating.

        :return: a Matrix object
        """
        return Matrix(mat = self.data_set.get_test_x())

    def get_validate_y(self):
        """
        Return the part of responses for validating.

        :return: a Matrix object
        """
        return Matrix(mat = self.data_set.get_test_y())

class Params(object):

    def __init__(self, params = None):
        """
        This is class of Params storing parameters used in DBM.

        :param params: a Params object
        """
        if params is None:
            self.params = dbm_cpp_interface.Params()
        else:
            self.params = params

    def set_params(self, string, sep = ' '):
        """
        Set values of parameters.

        Usage: [sep] represents the character used as the separator

                'parameter_name[sep]parameter_value'
                'parameter_name[sep]parameter_value[sep]parameter_name[sep]parameter_value'

        :param string: a string storing the parameters to be set
        :param sep: separator used in the string
        """
        self.params = dbm_cpp_interface.set_params(string, sep)

    def print_all(self):
        """
        Print all parameters and their values to the screen.
        """
        attrs = [attr for attr in dir(self.params)
                 if not callable(attr) and not attr.startswith("__")]
        for attr in attrs:
            print("%s = %s" % (attr, getattr(self.params, attr)))

class DBM(object):

    def __init__(self, params):
        """
        This is the class of DBM.

        :param params: a Params object
        """
        self.dbm = dbm_cpp_interface.DBM(params.params)

    def train(self, data_set):
        """
        Train the DBM.

        :param data_set: a Data_set object
        """
        self.dbm.train_val_no_const(data_set.data_set)

    def train_with_monotonic_constraints(self,
                                         data_set,
                                         monotonic_constraints):
        """
        Train the DBM with monotonic constraints. Negative values
        indicate a decreasing relationship and positive values indicates
        an increasing relationship.

        :param data_set: a Data_set object
        :param monotonic_constraints: a Matrix object of the dimension p*1
        """
        self.dbm.train_val_const(data_set.data_set,
                                    monotonic_constraints.mat)

    def predict(self, data_x):
        """
        Predict if it has been trained or it has been loaded from
        a trained model.

        :param data_x: a Matrix object
        :return:
        """
        data_y = Matrix(data_x.shape()[0], 1, 0)
        self.dbm.predict_in_place(data_x.mat, data_y.mat)
        return data_y

    def pdp(self, data_x, feature_index):
        """
        Calculate the data used in partial dependence plots.

        :param data_x: a Matrix object used for calculating
        :param feature_index: the index of the predictor of interest (the No. of the column)
        :return: a Matrix object storing the data used in partial dependence plots
        """
        return Matrix(mat = self.dbm.pdp_auto(data_x.mat,
                                              feature_index))

    def ss(self, data_x):
        """
        Calculate statistical signifiance of every predictor.

        :param data_x: a Matrix object used for calculating
        :return: a Matrix object storing P-values for every predictor
        """
        return Matrix(mat =
                      self.dbm.statistical_significance(data_x.mat))

    def calibrate_plot(self,
                       observation,
                       prediction,
                       resolution,
                       file_name = ''):
        """
        This is exactly the same as the one in GBM in R.

        :param observation: a Matrix object
        :param prediction: a Matrix object
        :param resolution: a scalar
        :param file_name: save the result if provided
        :return: a Matrix object
        """
        return Matrix(mat = self.dbm.calibrate_plot(observation.mat,
                                                    prediction.mat,
                                                    resolution,
                                                    file_name))

    def interact(self, data,
                 predictor_ind,
                 total_no_predictor):
        """
        This is exactly the same as the one in GBM in R.

        :param data: a Matrix object
        :param predictor_ind: a Matrix object
        :param total_no_predictor: a scalar
        :return: a scalar
        """
        return self.dbm.interact(data.mat,
                                 predictor_ind,
                                 total_no_predictor)

    def save_performance(self, file_name):
        """
        Save the training and validating losses.

        :param file_name: a string
        """
        self.dbm.save_performance(file_name)

    def save(self, file_name):
        """
        Save the DBM after trained.

        :param file_name: a string
        """
        self.dbm.save_dbm(file_name)

    def load(self, file_name):
        """
        Load from a file.

        :param file_name: a string
        """
        self.dbm.load_dbm(file_name)

class AUTO_DBM(object):

    def __init__(self, params):
        """
        This is the class of DBM.

        :param params: a Params object
        """
        self.auto_dbm = dbm_cpp_interface.AUTO_DBM(params.params)

    def train(self, data_set):
        """
        Train the DBM.

        :param data_set: a Data_set object
        """
        self.auto_dbm.train_val_no_const(data_set.data_set)

    def train_with_monotonic_constraints(self,
                                         data_set,
                                         monotonic_constraints):
        """
        Train the DBM with monotonic constraints. Negative values
        indicate a decreasing relationship, positive values indicates
        an increasing relationship and zero indicates no constraints.

        :param data_set: a Data_set object
        :param monotonic_constraints: a Matrix object of the dimension p*1
        """
        self.auto_dbm.train_val_const(data_set.data_set,
                                 monotonic_constraints.mat)

    def predict(self, data_x):
        """
        Predict if it has been trained or it has been loaded from a trained model.

        :param data_x: a Matrix object
        :return:
        """
        data_y = Matrix(data_x.shape()[0], 1, 0)
        self.auto_dbm.predict_in_place(data_x.mat, data_y.mat)
        return data_y

    def pdp(self, data_x, feature_index):
        """
        Calculate the data used in partial dependence plots.

        :param data_x: a Matrix object used for calculating
        :param feature_index: the index of the predictor of interest (the No. of the column)
        :return: a Matrix object storing the data used in partial dependence plots
        """
        return Matrix(mat = self.auto_dbm.pdp_auto(data_x.mat,
                                              feature_index))

    def ss(self, data_x):
        """
        Calculate statistical signifiance of every predictor.

        :param data_x: a Matrix object used for calculating
        :return: a Matrix object storing P-values for every predictor
        """
        return Matrix(mat =
                      self.auto_dbm.statistical_significance(data_x.mat))

    def calibrate_plot(self,
                       observation,
                       prediction,
                       resolution,
                       file_name):
        """
        This is exactly the same as the one in GBM in R.

        :param observation: a Matrix object
        :param prediction: a Matrix object
        :param resolution: a scalar
        :param file_name: save the result if provided
        :return: a Matrix object
        """
        return Matrix(mat = self.auto_dbm.calibrate_plot(observation.mat,
                                                    prediction.mat,
                                                    resolution,
                                                    file_name))

    def interact(self, data, predictor_ind, total_no_predictor):
        """
        This is exactly the same as the one in GBM in R.

        :param data: a Matrix object
        :param predictor_ind: a Matrix object
        :param total_no_predictor: a scalar
        :return: a scalar
        """
        return self.auto_dbm.interact(data.mat,
                                 predictor_ind,
                                 total_no_predictor)

    def save_performance(self, file_name):
        """
        Save the training and validating losses.

        :param file_name: a string
        """
        self.auto_dbm.save_performance(file_name)

    def save(self, file_name):
        """
        Save the DBM after trained.

        :param file_name: a string
        """
        self.auto_dbm.save_dbm(file_name)

    def load(self, file_name):
        """
        Load from a file.

        :param file_name: a string
        """
        self.auto_dbm.load_dbm(file_name)

def np2darray_to_float_matrix(source):
    """
    Convert a two-dimensional numpy array to a Matrix.

    :param source: a two-dimensional numpy array
    :return: a Matrix object of the same shape as the numpy array
    """
    try:
        assert type(source) is np.ndarray
    except AssertionError as e:
        raise ValueError('The argument may not be a Numpy array.')
    try:
        assert source.shape.__len__() == 2
    except AssertionError as e:
        raise ValueError('The argument may not be a 2d array.')
    target = Matrix(source.shape[0], source.shape[1], 0)
    target.from_np2darray(source)
    return target

def float_matrix_to_np2darray(source):
    """
    Convert a Matrix to a two-dimensional numpy array.

    :param source: a Matrix object
    :return: a two-dimensional numpy array of the same shape as the Matrix
    """
    try:
        assert type(source) is Matrix
    except AssertionError as e:
        raise ValueError('The argument may not be a Matrix.')
    return source.to_np2darray()

def string_to_params(string, sep = ' '):
    """
    Directly transfer a string to a Params object.

    :param string: a string
    :param sep: a character
    :return: a Params object
    """
    return Params(params = dbm_cpp_interface.set_params(string, sep))