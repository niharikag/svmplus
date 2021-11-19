#!/usr/bin/env python
# -*- coding: utf-8 -*-
import six
from abc import ABCMeta
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn import svm
from sklearn.utils import check_X_y
from svmplus.base import BaseSVMPlus


class LibSVMPlus(six.with_metaclass(ABCMeta, BaseSVMPlus, BaseEstimator)):
    """Base class for SVM plus classification
    """

    #_kernels = ["linear", "poly", "rbf"]

    def __init__(self, C=1, gamma=1,
                 kernel_x = 'rbf', degree_x = 3, gamma_x ='auto',
                 kernel_xstar = 'rbf', degree_xstar = 3, gamma_xstar ='auto',
                 tol = 1e-5):

        super(LibSVMPlus, self).__init__(C, gamma,
                                      kernel_x, degree_x, gamma_x,
                                      kernel_xstar, degree_xstar, gamma_xstar,
                                      tol)


    def fit(self, X, XStar, y):
        """Fit the SVM model according to the given training data.
        """
        y = np.array(y).reshape(len(X), 1)
        X, y = check_X_y(X, y.flat, 'csr')
        XStar, y = check_X_y(XStar, y.flat, 'csr')
        n_samples, n_features = X.shape
        y = y.reshape(n_samples, 1)

        if self.kernel_x == "linear":
            kernel_method = self._linear_kernel
            kernel_param = None
        elif self.kernel_x == "poly":
            kernel_method = self._poly_kernel
            kernel_param = self.degree_x
        else:
            kernel_method = self._rbf_kernel
            if self.gamma_x == 'auto':
                self.gamma_x = 1 / n_features
            kernel_param = self.gamma_x

        if self.kernel_xstar == "linear":
            kernel_method_star = self._linear_kernel
            kernel_param_star = None
        elif self.kernel_xstar == "poly":
            kernel_method_star = self._poly_kernel
            kernel_param_star = self.degree_xstar
        else:
            kernel_method_star = self._rbf_kernel
            if self.gamma_xstar == 'auto':
                self.gamma_xstar = 1 / XStar.shape[1]
            kernel_param_star = self.gamma_xstar

        # compute the matrix K and KStar (n_samples X n_samples) using kernel function
        K = np.zeros((n_samples, n_samples))
        KStar = np.zeros((n_samples, n_samples))
        for i in range(n_samples):
            for j in range(n_samples):
                K[i, j] = kernel_method(X[i, :], X[j, :], kernel_param)
                KStar[i, j] = kernel_method_star(XStar[i, :], XStar[j, :], kernel_param_star)

        cls_labels = np.unique(y)
        n_class = len(cls_labels)

        if n_class == 2:
            n_class = 1

        self.n_class = n_class
        self.model = []

        for i in range(n_class):
            y_temp = np.array(y)
            y[y != (i + 1)] = -1
            y[y == (i + 1)] = 1

            # append bias
            K = K + 1
            KStar = KStar + 1

            G = np.eye(n_samples) - np.linalg.inv(np.eye(n_samples) + (self.C / self.gamma) * KStar)
            G = (1 / self.C) * G
            H = np.multiply(K, y * np.transpose(y))
            Q = H + G
            #D = np.transpose(range(n_samples))
            #prob = np.concatenate((D, Q), axis = 1)
            model = svm._libsvm.fit(Q, np.ones(n_samples), svm_type=2, nu = 1 / n_samples,
                                   kernel = 'precomputed', tol = self.tol)

            sv_x = X[model[0]]
            sv_y = y[model[0]]

            coeff = model[3]
            dc = abs(coeff[0]).reshape(len(coeff[0]),1)
            dual_coef = np.multiply(sv_y, dc)
            support_vectors = sv_x  # support vector's features
            m = [support_vectors, dual_coef]

            if n_class == 1:
                self.model = m
            else:
                if i == 0:
                    self.model = list([m])
                else:
                    self.model.append(m)  # list(self.model, [m])

            y = y_temp[:]



    def project(self, X):
        if self.kernel_x == "linear":
            kernel_method = self._linear_kernel
            kernel_param = None
        elif self.kernel_x == "poly":
            kernel_method = self._poly_kernel
            kernel_param = self.degree_x
        else:
            kernel_method = self._rbf_kernel
            kernel_param = self.gamma_x


        if self.n_class == 1:
            y_project = np.zeros(len(X))
            m = self.model
            for i in range(len(X)):
                s = 0
                for a, sv in zip(m[1], m[0]):
                    s += a * kernel_method(X[i], sv, kernel_param)
                y_project[i] = s
        else:
            y_project = np.zeros((len(X), self.n_class))
            for cls in range(self.n_class) :
                m = self.model[cls]
                for i in range(len(X)):
                    s = 0
                    for a, sv in zip(m[1], m[0]):
                        s += np.multiply(a, kernel_method(X[i], sv, kernel_param))
                    y_project[i, cls] = s
        return y_project


    def predict(self, X):
        if self.n_class == 1:
            y_predict = np.sign(self.project(X))
        else:
            y_predict = -1 * np.ones((len(X), 1))
            y_project = self.project(X)
            for cls in range(self.n_class):
                y_temp = y_project[:, cls]
                y_predict[y_temp >= 0] = cls+1

        return y_predict


    def decision_function(self, X):
        return self.project(X)


