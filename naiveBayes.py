import numpy as np
import math
import pandas as pd
from sklearn import datasets
from sklearn.model_selection import train_test_split
from featureSelection import *
from mpmath import mp
from sklearn import preprocessing


class gaussClf:

    def separate_by_classes(self, X, y):
        ''' This function separates our dataset in subdatasets by classes '''
        self.classes = np.unique(y)
        classes_index = {}
        subdatasets = {}
        cls, counts = np.unique(y, return_counts=True)
        self.class_freq = dict(zip(cls, counts))
        print(self.class_freq)
        for class_type in self.classes:
            classes_index[class_type] = np.argwhere(y==class_type)
            subdatasets[class_type] = X[classes_index[class_type], :]
            self.class_freq[class_type] = self.class_freq[class_type]/sum(list(self.class_freq.values()))
        return subdatasets

    def fit(self, X, y):
        ''' The fitting function '''
        separated_X = self.separate_by_classes(X, y)
        self.means = {}
        self.std = {}
        for class_type in self.classes:
            # Here we calculate the mean and the standart deviation from datasets
            self.means[class_type] = np.mean(separated_X[class_type], axis=0)[0]
            self.std[class_type] = np.std(separated_X[class_type], axis=0)[0]

    def calculate_probability(self, x, mean, stdev):
        ''' This function calculates the class probability using gaussian distribution '''

        if mean ==0:
            mean = 0.01
        if stdev == 0:
            stdev = 0.01

        exponent = mp.exp(-((x - mean) ** 2 / (2 * stdev ** 2)))
        return (1 / (math.sqrt(2 * math.pi) * stdev)) * exponent

    def predict_proba(self, X):
        ''' This function predicts the probability for every class '''
        self.class_prob = {cls: math.log1p(self.class_freq[cls]) for cls in self.classes}
        for cls in self.classes:
            for i in range(len(self.means)):

                self.class_prob[cls] += math.log1p(self.calculate_probability(X[i], self.means[cls][i], self.std[cls][i]))
        self.class_prob = {cls: math.e ** self.class_prob[cls] for cls in self.class_prob}
        return self.class_prob

    def predict(self, X):
        ''' This funtion predicts the class of a sample '''
        pred = []
        for x in X:
            pred_class = None
            max_prob = 0
            for cls, prob in self.predict_proba(x).items():
                if prob > max_prob:
                    max_prob = prob
                    pred_class = cls
            pred.append(pred_class)
        return pred

def normalize(X, axis=-1, order=2):

    l2 = np.atleast_1d(np.linalg.norm(X, order, axis))
    l2[l2 == 0] = 1
    return X / np.expand_dims(l2, axis)

def check(s1, s2):
    n = len(s1)
    n2 = len(s2)
    tp = 0
    np = 0
    for i in range(len(s1)):
        if s1[i] == s2[i]:
            tp = tp + 1
        else:
            np = np + 1
    acc = tp / n
    print(acc)
    return acc, tp, np



def main():
    data = pd.read_csv('train.csv')

    data, indexes = univariateSelection(data, 41)

    x1 = data.drop(data.columns[-1], axis=1)
    y1 = data.iloc[:, -1]

    scale = preprocessing.minmax_scale(x1, feature_range=(0.1, 1.1))


    x2 = np.array(scale)
    y2 = np.array(y1)


    test = pd.read_csv('test.csv')
    test = test[indexes]
    x_test = test.drop(test.columns[-1], axis=1)
    y_test = test.iloc[:, -1]
    x_test = np.array(x_test)
    y_test = np.array(y_test)
    x_test = normalize(x_test)




    nb = gaussClf()
    nb.fit(x2,y2)
    final = nb.predict(x_test)
    acc = check(final, y_test)

    print(acc)






