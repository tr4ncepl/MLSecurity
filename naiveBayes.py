import numpy as np
import math
import pandas as pd
from sklearn import datasets
from sklearn.model_selection import train_test_split


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
        exponent = math.exp(-((x - mean) ** 2 / (2 * stdev ** 2)))
        return (1 / (math.sqrt(2 * math.pi) * stdev)) * exponent

    def predict_proba(self, X):
        ''' This function predicts the probability for every class '''
        self.class_prob = {cls: math.log1p(self.class_freq[cls]) for cls in self.classes}
        for cls in self.classes:
            for i in range(len(self.means)):
                print(X[i])
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


iris = datasets.load_iris()
X = iris.data # we only take the first two features.
y = iris.target

data = pd.read_csv('data.csv')

x1 = data[['atr5', 'atr6', 'atr1', 'atr23', 'atr33', 'atr32', 'atr10', 'atr24']]
y1 = data['class']

x2 = np.array(x1)
y2 = np.array(y1)

X_train,X_test,y_train,y_test=train_test_split(x2,y2,test_size=0.2)




nb = gaussClf()
nb.fit(X_train,y_train)
final = nb.predict(X_test)

def check(s1, s2):
    n = len(s1)
    n2 = len(s2)
    if n == n2:
        print("Y")
    else:
        print("WTF")

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

acc = check(final,y_test )

print(final)
print(y_test)




