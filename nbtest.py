from __future__ import division, print_function

import time

import numpy as np
import math

import pandas as pd

from featureSelection import univariateSelection, featureImportance
from multilayerPerceptor import shuffle_data, to_categorical, rge, boruta


def accuracy_score(y_true, y_pred):
    accuracy = np.sum(y_true == y_pred, axis=0) / len(y_true)
    return accuracy

def normalize(X, axis=-1, order=2):
    """ Normalize the dataset X """
    l2 = np.atleast_1d(np.linalg.norm(X, order, axis))
    l2[l2 == 0] = 1
    return X / np.expand_dims(l2, axis)


class NaiveBayes():
    """The Gaussian Naive Bayes classifier. """
    def fit(self, X, y):
        self.X, self.y = X, y
        self.classes = np.unique(y)
        self.parameters = []
        # Calculate the mean and variance of each feature for each class
        for i, c in enumerate(self.classes):
            # Only select the rows where the label equals the given class
            X_where_c = X[np.where(y == c)]
            self.parameters.append([])
            # Add the mean and variance for each feature (column)
            for col in X_where_c.T:
                parameters = {"mean": col.mean(), "var": col.var()}
                self.parameters[i].append(parameters)

    def _calculate_likelihood(self, mean, var, x):
        """ Gaussian likelihood of the data x given mean and var """
        eps = 1e-4 # Added in denominator to prevent division by zero
        coeff = 1.0 / math.sqrt(2.0 * math.pi * var + eps)
        exponent = math.exp(-(math.pow(x - mean, 2) / (2 * var + eps)))
        return coeff * exponent

    def _calculate_prior(self, c):
        """ Calculate the prior of class c
        (samples where class == c / total number of samples)"""
        frequency = np.mean(self.y == c)
        return frequency

    def _classify(self, sample):
        """ Classification using Bayes Rule P(Y|X) = P(X|Y)*P(Y)/P(X),
            or Posterior = Likelihood * Prior / Scaling Factor
        P(Y|X) - The posterior is the probability that sample x is of class y given the
                 feature values of x being distributed according to distribution of y and the prior.
        P(X|Y) - Likelihood of data X given class distribution Y.
                 Gaussian distribution (given by _calculate_likelihood)
        P(Y)   - Prior (given by _calculate_prior)
        P(X)   - Scales the posterior to make it a proper probability distribution.
                 This term is ignored in this implementation since it doesn't affect
                 which class distribution the sample is most likely to belong to.
        Classifies the sample as the class that results in the largest P(Y|X) (posterior)
        """
        posteriors = []
        # Go through list of classes
        for i, c in enumerate(self.classes):
            # Initialize posterior as prior
            posterior = self._calculate_prior(c)
            print(posterior)
            # Naive assumption (independence):
            # P(x1,x2,x3|Y) = P(x1|Y)*P(x2|Y)*P(x3|Y)
            # Posterior is product of prior and likelihoods (ignoring scaling factor)
            for feature_value, params in zip(sample, self.parameters[i]):
                # Likelihood of feature value given distribution of feature values given y
                likelihood = self._calculate_likelihood(params["mean"], params["var"], feature_value)
                posterior *= likelihood
            posteriors.append(posterior)

        # Return the class with the largest posterior probability
        return self.classes[np.argmax(posteriors)]

    def predict(self, X):
        """ Predict the class labels of the samples in X """
        y_pred = [self._classify(sample) for sample in X]
        return y_pred

def start(t, n):

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

    #test_labels = to_categorical(test_labels)
    test_data = normalize(test_data)

    test_data = test_data.to_numpy()

    train_labels = dane.iloc[:, -1]
    train_data = dane.drop(dane.columns[-1], axis=1)

    #train_labels = to_categorical(train_labels)

    train_data = normalize(train_data)
    train_data = train_data.to_numpy()

    t0 = time.time()
    clf = NaiveBayes()

    print("Fitting data into a model")
    clf.fit(train_data, train_labels)
    y_pred = clf.predict(test_data)
    #y_test = np.argmax(test_labels, axis=1)

    accuracy = accuracy_score(test_labels, y_pred)
    t1 = time.time()
    total = t1-t0
    print("Accuracy:", accuracy)
    print("Total time : ", total)


    final = "Accuracy: " + str(accuracy) + "\n" + "Total time: " + str(total)
    return final

