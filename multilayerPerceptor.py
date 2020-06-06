from __future__ import print_function, division
import numpy as np
import math

import pandas as pd
from featureSelection import *
from sklearn import datasets
import time


def to_categorical(x, n_col=None):
    if not n_col:
        n_col = np.amax(x) + 1
    one_hot = np.zeros((x.shape[0], n_col))
    one_hot[np.arange(x.shape[0]), x] = 1
    return one_hot


def normalize(X, axis=-1, order=2):
    l2 = np.atleast_1d(np.linalg.norm(X, order, axis))
    l2[l2 == 0] = 1
    return X / np.expand_dims(l2, axis)


def accuracy_score(y_true, y_pred):
    accuracy = np.sum(y_true == y_pred, axis=0) / len(y_true)
    return accuracy


def shuffle_data(X, y, seed=None):
    if seed:
        np.random.seed(seed)
    idx = np.arange(X.shape[0])
    np.random.shuffle(idx)
    return X[idx], y[idx]





class Loss(object):
    def loss(self, y_true, y_pred):
        return NotImplementedError()

    def gradient(self, y, y_pred):
        raise NotImplementedError()

    def acc(self, y, y_pred):
        return 0


class CrossEntropy(Loss):
    def __init__(self): pass

    def loss(self, y, p):
        # Avoid division by zero
        #p = np.clip(p, 1e-15, 1 - 1e-15)
        return - y * np.log(p) - (1 - y) * np.log(1 - p)

    def gradient(self, y, p):
        # Avoid division by zero
        p = np.clip(p, 1e-15, 1 - 1e-15)
        return - (y / p) + (1 - y) / (1 - p)


class Sigmoid():
    def __call__(self, x):
        return 1 / (1 + np.exp(-x))

    def gradient(self, x):
        return self.__call__(x) * (1 - self.__call__(x))


class Softmax():
    def __call__(self, x):
        e_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
        return e_x / np.sum(e_x, axis=-1, keepdims=True)

    def gradient(self, x):
        p = self.__call__(x)
        return p * (1 - p)


class MultilayerPerceptron():

    def __init__(self, n_hidden, n_iterations=3000, learning_rate=0.01):
        self.n_hidden = n_hidden
        self.n_iterations = n_iterations
        self.learning_rate = learning_rate
        self.hidden_activation = Sigmoid()
        self.output_activation = Softmax()
        self.loss = CrossEntropy()

    def _initialize_weights(self, X, y):
        n_samples, n_features = X.shape
        _, n_outputs = y.shape
        print(n_outputs)
        # Hidden layer
        limit = 1 / math.sqrt(n_features)
        self.W = np.random.uniform(-limit, limit, (n_features, self.n_hidden))
        self.w0 = np.zeros((1, self.n_hidden))

        # Output layer
        limit = 1 / math.sqrt(self.n_hidden)
        self.V = np.random.uniform(-limit, limit, (self.n_hidden, n_outputs))
        self.v0 = np.zeros((1, n_outputs))

    def fit(self, X, y):
        self._initialize_weights(X, y)

        for i in range(self.n_iterations):
            hidden_input = X.dot(self.W) + self.w0
            hidden_output = self.hidden_activation(hidden_input)

            output_layer_input = hidden_output.dot(self.V) + self.v0
            y_pred = self.output_activation(output_layer_input)

            grad_wrt_out_l_input = self.loss.gradient(y, y_pred) * self.output_activation.gradient(output_layer_input)
            grad_v = hidden_output.T.dot(grad_wrt_out_l_input)
            grad_v0 = np.sum(grad_wrt_out_l_input, axis=0, keepdims=True)

            grad_wrt_hidden_l_input = grad_wrt_out_l_input.dot(self.V.T) * self.hidden_activation.gradient(hidden_input)
            grad_w = X.T.dot(grad_wrt_hidden_l_input)
            grad_w0 = np.sum(grad_wrt_hidden_l_input, axis=0, keepdims=True)

            self.V -= self.learning_rate * grad_v
            self.v0 -= self.learning_rate * grad_v0
            self.W -= self.learning_rate * grad_w
            self.w0 -= self.learning_rate * grad_w0

    def predict(self, X):
        hidden_input = X.dot(self.W) + self.w0
        hidden_output = self.hidden_activation(hidden_input)
        output_layer_input = hidden_output.dot(self.V) + self.v0
        y_pred = self.output_activation(output_layer_input)
        return y_pred


def start(t, n, num, it, rate ):

    train = pd.read_csv('train.csv')
    if t==1:
        dane, indexes = univariateSelection(train, n)
    elif t==2:
        dane, indexes = rge(train, n)
    elif t==3:
        dane, indexes = boruta(train, n)
    elif t==4:
        dane, indexes = featureImportance(train, n)
    elif t== 0:
        dane = train
        indexes = list(train.columns)


    test = pd.read_csv('test.csv')
    test = test[indexes]

    test_labels = test.iloc[:, -1]
    test_data = test.drop(test.columns[-1], axis=1)

    test_labels = to_categorical(test_labels)
    test_data = normalize(test_data)

    test_data = test_data.to_numpy()

    train_labels = dane.iloc[:, -1]

    train_data = dane.drop(dane.columns[-1], axis=1)

    train_labels = to_categorical(train_labels)


    train_data = normalize(train_data)
    train_data = train_data.to_numpy()
    print("Starting of training and building model with ", num," neurons in HL, ",it," iterations and ", rate," learning rate")
    t0 = time.time()
    clf = MultilayerPerceptron(n_hidden=num,
                               n_iterations=it,
                               learning_rate=rate)

    print("Fitting data into a model")
    clf.fit(train_data, train_labels)
    y_pred = np.argmax(clf.predict(test_data), axis=1)
    y_test = np.argmax(test_labels, axis=1)
    print(y_pred)
    accuracy = accuracy_score(y_test, y_pred)
    t1 = time.time()
    total = t1-t0
    print("Accuracy:", accuracy)
    print("Total time : ", total)

    final = "Accuracy: " + str(accuracy) + "\n" + "Total time: " + str(total)
    return final



