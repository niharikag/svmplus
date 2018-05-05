# -*- coding: utf-8 -*-

"""Main module."""

from svmplusQP import QPSVMPlus
from libsvmplus import LibSVMPlus

def SVMPlus(svm_type = "QP", C=1, gamma=1,
                 kernel_x = 'rbf', degree_x = 3, gamma_x ='auto',
                 kernel_xstar = 'rbf', degree_xstar = 3, gamma_xstar ='auto',
                 tol = 1e-10):

    if svm_type == "QP":
        svmp = QPSVMPlus(C, gamma,
                         kernel_x, degree_x, gamma_x,
                         kernel_xstar, degree_xstar, gamma_xstar,
                         tol)
    else:
        svmp = LibSVMPlus(C, gamma,
                         kernel_x, degree_x, gamma_x,
                         kernel_xstar, degree_xstar, gamma_xstar,
                         tol)
    return svmp
