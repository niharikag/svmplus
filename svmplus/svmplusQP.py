#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Multiclass SVM Plus Classification(SVPC)
https://github.com/TBD
niharika gauraha<niharika.gauraha@farmbio.uu.se>
Ola Spjuth<ola.spjuth@farmbio.uu.se>
Learns a binary classifier based on SVM Plus:
LUPI paradigm. Uses scikit learn.
Licensed under a Creative Commons Attribution-NonCommercial 4.0
International License.
Based on SVM+ by Vapnik et al.
"""


import six
from abc import ABCMeta
import numpy as np
from sklearn.base import BaseEstimator
from cvxopt import matrix, solvers
from base  import BaseSVMPlus
from sklearn.utils import check_X_y


class QPSVMPlus(six.with_metaclass(ABCMeta, BaseSVMPlus, BaseEstimator)):
    def __init__(self, C=1, gamma=1,
                 kernel_x = 'rbf', degree_x = 3, gamma_x ='auto',
                 kernel_xstar = 'rbf', degree_xstar = 3, gamma_xstar ='auto',
                 tol = 1e-5):

        super(QPSVMPlus, self).__init__(C, gamma,
                 kernel_x, degree_x, gamma_x,
                 kernel_xstar, degree_xstar, gamma_xstar,
                 tol)


    def fit(self, X, XStar, y):
        """Fit the SVM model according to the given training data.
                """
        n_samples, n_features = X.shape
        X, y = check_X_y(X, y.flat, 'csr')
        XStar, y = check_X_y(XStar, y.flat, 'csr')
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

            P1 = np.concatenate((np.multiply(K, y * np.transpose(y)) + KStar / float(self.gamma),
                                 KStar / float(self.gamma)), axis=1)

            P2 = np.concatenate((KStar / float(self.gamma), KStar / float(self.gamma)), axis=1)
            P = np.concatenate((P1, P2), axis=0)
            A = np.concatenate((np.ones((2 * n_samples, 1)),
                                np.concatenate((y, np.zeros((n_samples, 1))))), axis=1)
            A = A.T
            b = np.array([[n_samples * self.C], [0]])
            G = -np.eye(2 * n_samples)
            h = np.zeros((2 * n_samples, 1))

            Q = - (self.C / (self.gamma)) * np.sum(KStar, axis=1)  # [sum(KStar[:,i]) for i in range(n_samples)]
            q = np.concatenate((-1 + Q, Q))

            sol = solvers.qp(matrix(P, tc='d'), matrix(q, tc='d'), matrix(G, tc='d'), matrix(h, tc='d'),
                             matrix(A, tc='d'), matrix(b, tc='d'))
            # Lagrange multipliers
            alpha = np.array(sol['x'][0:n_samples])
            beta = sol['x'][n_samples:2 * n_samples]

            # compute b_star first
            wxstar = (1 / self.gamma) * np.matmul(KStar, (alpha + beta - self.C))
            wxstar_idx = np.ravel(beta) > self.tol

            if (~wxstar_idx).all():  # no beta > tol
                b_star = max(-wxstar)
            else:
                b_star = np.mean(-wxstar[wxstar_idx])

            # Support vectors have non zero lagrange multipliers
            sv = np.ravel(alpha) > self.tol  # tolerance
            sv_x = X[sv]
            sv_y = y[sv]

            d = np.multiply(y, alpha)
            wx = np.matmul(K, d)
            temp = np.multiply(y, (1 - wxstar - b_star))
            wx = -wx + temp
            if (~sv).all():  # no alpha > tol
                lb = np.max(wx[y == 1])
                ub = np.min(wx[y == -1])
                b = (lb + ub) / 2
            else:
                b = np.mean(wx[sv])

            dual_coef = np.multiply(sv_y, alpha[sv])
            intercept = b
            support_vectors = sv_x  # support vector's features
            m = [support_vectors, dual_coef, intercept]

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
                    s += np.multiply(a, kernel_method(X[i], sv, kernel_param))
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
            y_predict = np.sign(self.project(X) +self.model[2])
        else:
            y_predict = -1 * np.ones((len(X), 1))
            y_project = self.project(X)
            for cls in range(self.n_class):
                y_temp = y_project[:, cls] + self.model[cls][2]
                y_predict[y_temp >= 0] = cls+1

        return y_predict


    def decision_function(self, X):
        return self.project(X)


