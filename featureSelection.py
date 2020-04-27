import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, RFE, SelectFromModel
from sklearn.feature_selection import chi2
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import ExtraTreesClassifier
import matplotlib.pyplot as plt

pd.set_option('display.float_format', lambda x: '%.3f' % x)
np.set_printoptions(suppress=True)
data = pd.read_csv("train.csv")


def univariateSelection(data, n):
    X = data.iloc[:, 0:41]
    y = data.iloc[:, -1]
    bestfeatures = SelectKBest(score_func=chi2, k=41)
    fit = bestfeatures.fit(X, y)
    dfscores = pd.DataFrame(fit.scores_)
    dfcolumns = pd.DataFrame(X.columns)

    featureScores = pd.concat([dfcolumns, dfscores], axis=1)
    featureScores.columns = ['Specs', 'Score']
    featureScores = featureScores.sort_values(["Score"], ascending=False)
    selected = featureScores["Specs"].head(n)

    indexes = []
    for row in selected:
        indexes.append(row)
    indexes.append("class")

    df = data[indexes]

    return df, indexes


def rge(data, n):
    X = data.iloc[:, 0:41]
    y = data.iloc[:, -1]

    model = RandomForestClassifier(random_state=100, n_estimators=50)

    sel_rfe = RFE(estimator=model, step=1)

    x_train_rfe = sel_rfe.fit_transform(X, y)

    # print(sel_rfe.get_support())

    dfcolumns = pd.DataFrame(X.columns)
    dfscores = pd.DataFrame(sel_rfe.ranking_)
    featureScores = pd.concat([dfcolumns, dfscores], axis=1)
    featureScores.columns = ['Specs', 'Score']
    featureScores = featureScores.sort_values(["Score"])

    selected = featureScores["Specs"].head(n)

    indexes = []
    for row in selected:
        indexes.append(row)
    indexes.append("class")

    print(indexes)

    df = data[indexes]

    return df, indexes


def SFM(data):
    X = data.iloc[:, 0:41]
    y = data.iloc[:, -1]

    model = RandomForestClassifier(random_state=100,
                                   n_estimators=50)

    model.fit(X, y)

    sel_model = SelectFromModel(estimator=model, threshold='mean')

    sel_model.fit(X, y)
    print(sel_model.transform(X))


def featureImportance(data,n):
    X = data.iloc[:, 0:41]
    y = data.iloc[:, -1]

    model = ExtraTreesClassifier()
    model.fit(X, y)

    scores = model.feature_importances_
    dfscores = pd.DataFrame(model.feature_importances_)
    dfcolumns = pd.DataFrame(X.columns)
    feat_importances = pd.concat([dfcolumns, dfscores], axis=1)
    feat_importances.columns = ['Specs', 'Score']
    feat_importances = feat_importances.sort_values(["Score"], ascending=False)

    selected = feat_importances["Specs"].head(n)

    indexes = []
    for row in selected:
        indexes.append(row)
    indexes.append("class")


    df = data[indexes]

    return df, indexes



