# -*- coding: utf-8 -*-

"""Main module."""

from svmplus.svmplusQP import QPSVMPlus
from svmplus.libsvmplus import LibSVMPlus

def SVMPlus(svm_type = "QP", C=1, gamma=1,
                 kernel_x = 'rbf', degree_x = 3, gamma_x ='auto',
                 kernel_xstar = 'rbf', degree_xstar = 3, gamma_xstar ='auto',
                 tol = 1e-10):
    """
        SVMPlus for classification problems.

        Creates and returns an instance of the class specified
        in the svm_type.

        We refer to [1] for theoretical details of LUPI and SVM+,
        and we refer to [2] for implementation details of SVM+ in MATLAB.

        Parameters
        ----------
        svm_type : str
            optimization techiniques based on : QP, LibSVM, LibLinear etc.
            Currently it supports only "QP" and "LIBSVM".
        C : int
            cost of constraints violation
        gamma : int
            parameter needed for priviledged information
        kernel_x : str
            the kernel used for standard training data
        degree_x :
            parameter needed for polynomial kernel for training data
        gamma_x :
            parameter needed for rbf kernel for training data
        kernel_xstar :
            the kernel used for priviledged information (PI)
        degree_xstar :
            parameter needed for polynomial kernel for PI
        gamma_xstar :
            parameter needed for rbf kernel for PI
        tol :
            tolerance of dual variables

        Returns
        -------
        svmp
            an instance of the class specified in the svm_type.

        References
        ----------
        .. [1] Vladimir et. al, Neural Networks, 2009, 22, pp 544–557.
                \url{https://doi.org/10.1016/j.neunet.2009.06.042}

        .. [2] Li et. al, 2016.
            \url{https://github.com/okbalefthanded/svmplus_matlab}

        Examples
        --------
        >>> import numpy as np
        >>> import svmplus
        >>> X1 = np.array([[3,1], [3,-1], [5,1], [5,-1]])
        >>> X2 = np.array([[1,0], [0,1], [0,-1], [-1,0]])
        >>> X3 = np.array([[1,10], [3,11]])
        >>> X_train = np.concatenate((X1,X2))
        >>> X_train = np.concatenate((X_train, X3))
        >>> y_train = np.array([1,1,1,1,2,2,2,2,3,3]).reshape(10, 1)
        >>> XStar = np.eye(10)
        >>> X_test = np.array([[4,1], [5,0], [4,0], [4,1],
        >>>              [0, 0], [-1,-2], [1,2], [1,-2]])
        >>> y_test = np.array([1,1,1,1, 2, 2, 2, 2]).reshape(8, 1)

        >>> svmp = svmplus.SVMPlus(svm_type="QP", C=1000, gamma=.1, kernel_x="rbf", degree_x=2, gamma_x=.00001,
        >>>                kernel_xstar="rbf", degree_xstar=2, gamma_xstar=0.00001)

        >>> svmp.fit(X_train, XStar, np.array(y_train).reshape(len(X_train),1))
        >>> y_predict = svmp.predict(X_test)

        """
    if svm_type == "QP":
        svmp = QPSVMPlus(C, gamma,
                         kernel_x, degree_x, gamma_x,
                         kernel_xstar, degree_xstar, gamma_xstar,
                         tol)
    elif svm_type == "LIBSVM":
        svmp = LibSVMPlus(C, gamma,
                         kernel_x, degree_x, gamma_x,
                         kernel_xstar, degree_xstar, gamma_xstar,
                         tol)
    else:
        print("SVM type not supported")
    return svmp
