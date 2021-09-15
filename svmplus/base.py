#!/usr/bin/env python
# -*- coding: utf-8 -*-

import six
from abc import ABCMeta
import numpy as np
from sklearn.base import BaseEstimator
from numpy import linalg as LA


class BaseSVMPlus(six.with_metaclass(ABCMeta, BaseEstimator)):
    """Base class for SVM plus classification
    """

    def __init__(self, C, gamma,
                 kernel_x, degree_x, gamma_x,
                 kernel_xstar, degree_xstar, gamma_xstar,
                 tol):

        if gamma == 0:
            msg = ("The gamma value of 0.0 is invalid. Use 'auto' to set"
                   " gamma to a value of 1 / n_features.")
            raise ValueError(msg)

        self.C = C
        self.gamma = gamma
        self.kernel_x = kernel_x
        self.degree_x = degree_x
        self.gamma_x = gamma_x
        self.kernel_xstar = kernel_xstar
        self.degree_xstar = degree_xstar
        self.gamma_xstar = gamma_xstar
        self.tol = tol

    def fit(self, X, XStar, y):
        print("fit function")


    def project(self, X):
        print("project function")


    def predict(self, X):
        print("predict function")


    def decision_function(self, X):
        print("decision function")



    @staticmethod
    # Linear kernel
    def _linear_kernel(x1, x2, param = None):
        return np.dot(x1, x2)


    @staticmethod
    # Polynomial kernel
    def _poly_kernel(x1, x2, param = 2):
        return (1 + np.dot(x1, x2)) ** param


    @staticmethod
    # Radial basis kernel
    def _rbf_kernel(x1, x2, param = 1.0):
        return np.exp(-(LA.norm(x1 - x2) ** 2) * param)
