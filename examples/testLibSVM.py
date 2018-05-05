import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
import scipy.ndimage
import sklearn.preprocessing as preproc
import svmplus

def prepareDigitData():
    #digits = load_digits(n_class=4)
    digits = load_digits()
    X = digits.data
    y = digits.target
    #y[y == 0] = -1
    y = y+1

    resizedImage = np.zeros((len(X), 16))
    for i in range(len(X)):
        originalImage = X[i].reshape(8, 8)
        originalImage = scipy.ndimage.zoom(originalImage, 4 / (8), order=0)
        resizedImage[i] = originalImage.reshape(16, )

    XStar = preproc.scale(X, axis = 1)
    X = preproc.scale(resizedImage, axis = 1)

    X_train, X_test, y_train, y_test, indices_train, indices_valid = \
        train_test_split(X, y, range(len(X)), test_size=0.5, stratify = y)

    XStar = XStar[indices_train]
    return X_train, X_test, y_train, y_test, XStar


def testLinearSVMPlus():
    X_train, X_test, y_train, y_test, XStar = prepareDigitData()

    y_test = np.array(y_test)

    svmp = svmplus.SVMPlus(svm_type="LIBSVM", C=10, kernel_x="linear", kernel_xstar="linear", tol=1e-10)
    svmp.fit(X_train, XStar, y_train)
    y_predict = svmp.predict(X_test)
    correct = np.sum(y_predict.flat == y_test)
    print("Prediction accuracy of SVM")
    print("%d out of %d predictions correct" % (correct, len(y_predict)))


def testPolynomial():
    X_train, X_test, y_train, y_test, XStar = prepareDigitData()

    svmp = svmplus.SVMPlus(svm_type="LIBSVM", C=10, gamma=.01, kernel_x="poly",
                   kernel_xstar="poly", tol=1e-10)
    svmp.fit(X_train, XStar, y_train)
    y_predict = svmp.predict(X_test)
    correct = np.sum(y_predict.flat == y_test)
    print("Prediction accuracy of SVM")
    print("%d out of %d predictions correct" % (correct, len(y_predict)))



def testRbfSVMPlus():
    X_train, X_test, y_train, y_test, XStar = prepareDigitData()

    # train and predict using SVM plus
    svmp = svmplus.SVMPlus(svm_type="LIBSVM", C=10, gamma=.00001, kernel_x="rbf", gamma_x=.000001,
                   kernel_xstar="rbf", gamma_xstar=.000001, tol=1e-10)
    svmp.fit(X_train, XStar, y_train)
    y_predict = svmp.predict(X_test)
    correct = np.sum(y_predict.flat == y_test)
    print("Prediction accuracy of SVM")
    print("%d out of %d predictions correct" % (correct, len(y_predict)))



def testSVMPlus():
    # X_train, X_test, y_train, y_test, XStar = 2*np.eye(3), 3*np.eye(3), [1,1,-1], [1,1,-1], 2*np.eye(3)
    # X1 = np.array([[3, 1], [3, -1], [6, 1], [6, -1]])
    # X2 = np.array([[1, 0], [0, 1], [0, -1], [-1, 0]])
    # X_train = XStar = np.concatenate((X1, X2))
    # y_train = [1, 1, 1, 1, -1, -1, -1, -1]

    X_train = np.array([[17, 24, 1, 8, 15],
                        [23, 5, 7, 14, 16],
                        [4, 6, 13, 20, 22],
                        [10, 12, 19, 21, 3],
                        [11, 18, 25, 2, 9]])

    y_train = [1, 1, -1, 1, -1]

    XStar = np.eye(5)
    XStar[0, 0] = 1
    XStar[1, 1] = 2
    XStar[2, 2] = 3
    XStar[3, 3] = 4
    XStar[4, 4] = 5
    # train and predict using SVM plus
    # svmp = SVMPlus(C=1, gamma=1, kernel_x= "rbf", gamma_x = .0019,
    #               kernel_xstar="rbf", gamma_xstar = 0.0568)
    # svmp.fit(X_train, XStar, y_train)

    svmp = svmplus.SVMPlus(svm_type="LIBSVM", C=1, gamma=1, kernel_x="linear", degree_x=2, gamma_x=.0019,
                   kernel_xstar="linear", degree_xstar=2, gamma_xstar=0.0568)
    svmp.fit(X_train, XStar, y_train)
    # y_predict = svmp.predict(X_test)
    # print(y_predict)


def test3Class():
    X1 = np.array([[3, 1], [3, -1], [5, 1], [5, -1]])
    X2 = np.array([[1, 0], [0, 1], [0, -1], [-1, 0]])
    X3 = np.array([[1, 10], [3, 11]])

    X_train = np.concatenate((X1, X2))
    X_train = np.concatenate((X_train, X3))

    y_train = np.array([1, 1, 1, 1, 2, 2, 2, 2, 3, 3]).reshape(10, 1)
    XStar = np.eye(10)
    X_test = np.array([[4, 1], [5, 0], [4, 0], [4, 1],
                       [0, 0], [-1, -2], [1, 2], [1, -2]])
    y_test = np.array([1, 1, 1, 1, 2, 2, 2, 2]).reshape(8, 1)

    svmp = svmplus.SVMPlus(svm_type = "LIBSVM", C=10, gamma=.01, kernel_x="rbf", degree_x=3, gamma_x=.001,
                   kernel_xstar="rbf", degree_xstar=3, gamma_xstar=0.001, tol = 1e-10)
    svmp.fit(X_train, XStar, y_train)
    y_predict = svmp.predict(X_test)
    print(y_predict)
    correct = np.sum(y_predict == y_test)
    print("Prediction accuracy of SVM")
    print("%d out of %d predictions correct" % (correct, len(y_predict)))


if __name__ == "__main__":
    #testSVMPlus()
    #testLinearSVMPlus()
    #testPolynomial()
    #testRbfSVMPlus()
    test3Class()