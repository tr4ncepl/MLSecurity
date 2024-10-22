import time

import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils import check_random_state
from sklearn.preprocessing import LabelEncoder
from featureSelection import *
from sklearn.metrics import *


def projection_simplex(v, z=1):
    n_features = v.shape[0]
    u = np.sort(v)[::-1]
    cssv = np.cumsum(u)-z
    ind = np.arange(n_features) + 1
    cond = u - cssv / ind > 0

    rho = ind[cond][-1]
    theta = cssv[cond][-1] / float(rho)
    w = np.maximum(v - theta, 0)
    return w


def normalize(X, axis=-1, order=2):
    l2 = np.atleast_1d(np.linalg.norm(X, order, axis))
    l2[l2 == 0] = 1
    return X / np.expand_dims(l2, axis)


class MulticlassSVM(BaseEstimator, ClassifierMixin):

    def __init__(self, C=1, max_iter=50, tol=0.05,
                 random_state=None, verbose=0):
        self.C = C
        self.max_iter = max_iter
        self.tol = tol,
        self.random_state = random_state
        self.verbose = verbose

    def _partial_gradient(self, X, y, i):

        g = np.dot(X[i], self.coef_.T) + 1

        g[y[i]] -= 1

        return g

    def _violation(self, g, y, i):

        smallest = np.inf
        for k in range(g.shape[0]):
            if k == y[i] and self.dual_coef_[k, i] >= self.C:
                continue
            elif k != y[i] and self.dual_coef_[k, i] >= 0:
                continue

            smallest = min(smallest, g[k])

        return g.max() - smallest

    def _solve_subproblem(self, g, y, norms, i):

        Ci = np.zeros(g.shape[0])
        Ci[y[i]] = self.C

        beta_hat = norms[i] * (Ci - self.dual_coef_[:, i]) + g / norms[i]
        z = self.C * norms[i]


        beta = projection_simplex(beta_hat, z)

        return Ci - self.dual_coef_[:, i] - beta / norms[i]

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self._label_encoder = LabelEncoder()
        y = self._label_encoder.fit_transform(y)
        n_classes = len(self._label_encoder.classes_)

        self.dual_coef_ = np.zeros((n_classes, n_samples), dtype=np.float64)
        self.coef_ = np.zeros((n_classes, n_features))

        norms = np.sqrt(np.sum(X ** 2, axis=1))

        rs = check_random_state(self.random_state)
        ind = np.arange(n_samples)
        rs.shuffle(ind)

        violation_init = None
        for it in range(self.max_iter):
            violation_sum = 0

            for ii in range(n_samples):
                i = ind[ii]

                if norms[i] == 0:
                    continue

                g = self._partial_gradient(X, y, i)
                v = self._violation(g, y, i)
                violation_sum += v

                if v < 1e-12:
                    continue

                delta = self._solve_subproblem(g, y, norms, i)

                self.coef_ += (delta * X[i][:, np.newaxis]).T
                self.dual_coef_[:, i] += delta

            if it == 0:
                violation_init = violation_sum

            vratio = violation_sum / violation_init

            if self.verbose >= 1:
                print("iter", it + 1, "violation", vratio)

            if vratio < self.tol:
                if self.verbose >= 1:
                    print("Converged")
                break

        return self


    def aha(self,dane):
        return dane

    def predict(self, X):
        decision = np.dot(X, self.coef_.T)
        pred = decision.argmax(axis=1)
        self.predictions = self._label_encoder.inverse_transform(pred)
        return self._label_encoder.inverse_transform(pred)




def start(t, n,c1, iter, tol, ver):
    np.set_printoptions(suppress=True)
    data = pd.read_csv('train.csv')
    test = pd.read_csv('test.csv')

    if t == 1:
        dane, indexes = univariateSelection(data, n)
    elif t == 2:
        dane, indexes = rge(data, n)
    elif t == 3:
        dane, indexes = boruta(data, n)
    elif t == 4:
        dane, indexes = featureImportance(data, n)
    elif t == 0:
        dane = data
        indexes = list(data.columns)

    test = test[indexes]
    x1 = dane.drop(data.columns[-1], axis=1)


    x1 = normalize(x1)
    y1 = dane.iloc[:, -1]
    x_test = test.drop(test.columns[-1], axis=1)
    y_test = test.iloc[:, -1]
    x_test = np.array(x_test)
    y_test = np.array(y_test)
    x_test = normalize(x_test)
    x2 = np.array(x1)
    y2 = np.array(y1)
    t0 = time.time()
    about ="Starting training algorithm for C = "+str(c1)+ " iterations = "+ str(iter) + "and tolerance = " +str(tol)
    clf = MulticlassSVM(C=c1, tol=tol, max_iter=iter, random_state=0, verbose=ver)
    clf.fit(x2, y2)
    t2 =time.time()
    print("Accuracy of train classification", clf.score(x2, y2))
    print("Training ended ")
    print("Starting fitting model and predicting classes ")

    training = t2-t0
    acc = clf.score(x_test, y_test)
    predictions = clf.predictions
    t1 = time.time()


    print("Test data accuracy : ", acc)
    total = t1-t0
    accu = accuracy_score(y_test, predictions)
    prec = precision_score(y_test, predictions, average='weighted')
    rec = recall_score(y_test, predictions, average='weighted')
    f1 = f1_score(y_test, predictions, average='weighted')
    print("Accuracy: ", accu)
    print("Precision: ", prec)
    print("Recall", rec)
    print("F1 Score:", f1)
    print("Total time : ", total)
    print(predictions)

    final =about + "\nTraining ended in: " + str(training)+ "seconds \nStarting predictinon of test data"+ "\n####End of prediction####\nAccuracy:   " + str(accu) + "\nPrecision:   " + str(prec) + "\nRecall:   " + str(
        rec) + "\nF1 Score:  " + str(f1) + "\n" + "Total time:   " + str(total)
    return final










