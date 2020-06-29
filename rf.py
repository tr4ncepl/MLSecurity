import math
import progressbar
import pandas as pd
import numpy as np
import matplotlib as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.cm import get_cmap
import matplotlib.cm as cmx
import matplotlib.colors as colors
from sklearn.model_selection import GridSearchCV

from featureSelection import *
import time
from sklearn.metrics import *
from sklearn.ensemble import RandomForestClassifier



def get_random_subsets(X, y, n_subsets, replacements=True):
    """ Return random subsets (with replacements) of the data """
    n_samples = np.shape(X)[0]
    # Concatenate x and y and do a random shuffle
    X_y = np.concatenate((X, y.values.reshape((1, len(y))).T), axis=1)
    np.random.shuffle(X_y)
    subsets = []

    # Uses 50% of training samples without replacements
    subsample_size = int(n_samples // 2)
    if replacements:
        subsample_size = n_samples  # 100% with replacements

    for _ in range(n_subsets):
        idx = np.random.choice(
            range(n_samples),
            size=np.shape(range(subsample_size)),
            replace=replacements)
        X = X_y[idx][:, :-1]
        y = X_y[idx][:, -1]
        subsets.append([X, y])
    return subsets


def calculate_entropy(y):
    """ Calculate the entropy of label array y """
    log2 = lambda x: math.log(x) / math.log(2)
    unique_labels = np.unique(y)
    entropy = 0
    for label in unique_labels:
        count = len(y[y == label])
        p = count / len(y)
        entropy += -p * log2(p)
    return entropy


class DecisionNode():
    def __init__(self, feature_i=None, threshold=None,
                 value=None, true_branch=None, false_branch=None):
        self.feature_i = feature_i  # Index for the feature that is tested
        self.threshold = threshold  # Threshold value for feature
        self.value = value  # Value if the node is a leaf in the tree
        self.true_branch = true_branch  # 'Left' subtree
        self.false_branch = false_branch  # 'Right' subtree


def divide_on_feature(X, feature_i, threshold):
    split_func = None
    if isinstance(threshold, int) or isinstance(threshold, float):
        split_func = lambda sample: sample[feature_i] >= threshold
    else:
        split_func = lambda sample: sample[feature_i] == threshold

    X_1 = np.array([sample for sample in X if split_func(sample)])
    X_2 = np.array([sample for sample in X if not split_func(sample)])

    return np.array([X_1, X_2])


class DecisionTree(object):
    def __init__(self, min_samples_split=2, min_impurity=1e-7,
                 max_depth=float("inf"), loss=None):
        self.root = None
        self.min_samples_split = min_samples_split
        self.min_impurity = min_impurity
        self.max_depth = max_depth
        self._impurity_calculation = None
        self._leaf_value_calculation = None
        self.one_dim = None
        self.loss = loss

    def fit(self, X, y, loss=None):
        self.one_dim = len(np.shape(y)) == 1
        self.root = self._build_tree(X, y)
        self.loss = None

    def _build_tree(self, X, y, current_depth=0):
        largest_impurity = 0
        best_criteria = None  # Feature index and threshold
        best_sets = None  # Subsets of the data
        if len(np.shape(y)) == 1:
            y = np.expand_dims(y, axis=1)

        Xy = np.concatenate((X, y), axis=1)

        n_samples, n_features = np.shape(X)
        if n_samples >= self.min_samples_split and current_depth <= self.max_depth:

            for feature_i in range(n_features):
                feature_values = np.expand_dims(X[:, feature_i], axis=1)
                unique_values = np.unique(feature_values)


                for threshold in unique_values:

                    Xy1, Xy2 = divide_on_feature(Xy, feature_i, threshold)

                    if len(Xy1) > 0 and len(Xy2) > 0:
                        y1 = Xy1[:, n_features:]
                        y2 = Xy2[:, n_features:]



                        impurity = self._impurity_calculation(y, y1, y2)
                        if impurity > largest_impurity:
                            largest_impurity = impurity
                            best_criteria = {"feature_i": feature_i, "threshold": threshold}
                            best_sets = {
                                "leftX": Xy1[:, :n_features],  # X of left subtree
                                "lefty": Xy1[:, n_features:],  # y of left subtree
                                "rightX": Xy2[:, :n_features],  # X of right subtree
                                "righty": Xy2[:, n_features:]  # y of right subtree
                            }

        if largest_impurity > self.min_impurity:
            true_branch = self._build_tree(best_sets["leftX"], best_sets["lefty"], current_depth + 1)
            false_branch = self._build_tree(best_sets["rightX"], best_sets["righty"], current_depth + 1)
            return DecisionNode(feature_i=best_criteria["feature_i"], threshold=best_criteria[
                "threshold"], true_branch=true_branch, false_branch=false_branch)


        leaf_value = self._leaf_value_calculation(y)


        return DecisionNode(value=leaf_value)

    def predict_value(self, x, tree=None):
        if tree is None:
            tree = self.root

        if tree.value is not None:
            return tree.value

        feature_value = x[tree.feature_i]

        branch = tree.false_branch
        if isinstance(feature_value, int) or isinstance(feature_value, float):
            if feature_value >= tree.threshold:
                branch = tree.true_branch
        elif feature_value == tree.threshold:
            branch = tree.true_branch
        return self.predict_value(x, branch)

    def predict(self, X):
        y_pred = [self.predict_value(sample) for sample in X]
        return y_pred

    def print_tree(self, tree=None, indent=" "):
        if not tree:
            tree = self.root
        if tree.value is not None:
            print(tree.value)
        else:
            print("%s:%s? " % (tree.feature_i, tree.threshold))
            print("%sT->" % (indent), end="")
            self.print_tree(tree.true_branch, indent + indent)
            print("%sF->" % (indent), end="")
            self.print_tree(tree.false_branch, indent + indent)


class ClassificationTree(DecisionTree):

    def _calculate_information_gain(self, y, y1, y2):
        # Calculate information gain
        p = len(y1) / len(y)
        entropy = calculate_entropy(y)
        info_gain = entropy - p * calculate_entropy(y1) - (1 - p) * calculate_entropy(y2)

        return info_gain

    def _majority_vote(self, y):
        most_common = None
        max_count = 0
        for label in np.unique(y):
            count = len(y[y == label])
            if count > max_count:
                most_common = label
                max_count = count
        return most_common

    def fit(self, X, y):
        self._impurity_calculation = self._calculate_information_gain
        self._leaf_value_calculation = self._majority_vote
        super(ClassificationTree, self).fit(X, y)


class RandomForestClasifier():

    def __init__(self, n_estimators=100, max_features=None, min_samples_split=2,
                 min_gain=0, max_depth=float("inf")):
        self.n_estimators = n_estimators  # Number of trees
        self.max_features = max_features  # Maxmimum number of features per tree
        self.min_samples_split = min_samples_split
        self.min_gain = min_gain  # Minimum information gain req. to continue
        self.max_depth = max_depth  # Maximum depth for tree


        self.trees = []
        for _ in range(n_estimators):
            self.trees.append(
                ClassificationTree(
                    min_samples_split=self.min_samples_split,
                    min_impurity=min_gain,
                    max_depth=self.max_depth))


    def fit(self, X, y):
        n_features = np.shape(X)[1]
        if not self.max_features:
            self.max_features = int(math.sqrt(n_features))
        subsets = get_random_subsets(X, y, self.n_estimators)
        for i in (range(self.n_estimators)):
            X_subset, y_subset = subsets[i]
            idx = np.random.choice(range(n_features), size=self.max_features, replace=True)
            self.trees[i].feature_indices = idx
            X_subset = X_subset[:, idx]
            self.trees[i].fit(X_subset, y_subset)

    def predict(self, X):
        y_preds = np.empty((X.shape[0], len(self.trees)))
        for i, tree in enumerate(self.trees):
            idx = tree.feature_indices
            prediction = tree.predict(X[:, idx])

            y_preds[:, i] = prediction

        y_pred = []
        for sample_predictions in y_preds:
            y_pred.append(np.bincount(sample_predictions.astype('int')).argmax())
        return y_pred


def accuracy_score(y_true, y_pred):
    accuracy = np.sum(y_true == y_pred, axis=0) / len(y_true)
    return accuracy

def start(t, n,trees, depth,split):
    train = pd.read_csv('train.csv')
    if depth=="inf":
        dep=None
        print("Y")
    else:
        dep = int(depth)
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
    train_labels = train.iloc[:, -1]


    train_data = dane.drop(dane.columns[-1], axis=1)

    t0 = time.time()
    clf = RandomForestClassifier(criterion="entropy", max_features="sqrt", n_estimators=trees,max_depth=dep,min_samples_leaf=1,min_samples_split=split)
    clf.fit(train_data,train_labels)
    predictions = clf.predict(test_data)
    t1=time.time()
    total = t1 - t0
    accu = accuracy_score(test_labels, predictions)
    prec = precision_score(test_labels, predictions, average='macro')
    rec = recall_score(test_labels, predictions, average='macro')
    f1 = f1_score(test_labels, predictions, average='macro')
    print("Accuracy: ", accu)
    print("Precision: ", prec)
    print("Recall", rec)
    print("F1 Score:", f1)
    print("Total time : ", total)

    final = "Accuracy:   " + str(accu) + "\nPrecision:   " + str(prec) + "\nRecall:   " + str(
        rec) + "\nF1 Score:  " + str(f1) + "\n" + "Total time:   " + str(total)
    return final


