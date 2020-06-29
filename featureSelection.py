import pandas as pd
import numpy as np
from sklearn.inspection import permutation_importance
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, RFE, SelectFromModel, RFECV
from sklearn.feature_selection import chi2
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import ExtraTreesClassifier
import matplotlib.pyplot as plt
from boruta import BorutaPy
import time

pd.set_option('display.float_format', lambda x: '%.3f' % x)
np.set_printoptions(suppress=True)
data = pd.read_csv("train.csv")


def univariateSelection(data, n):
    X = data.iloc[:, 0:41]
    y = data.iloc[:, -1]
    t0 = time.time()
    bestfeatures = SelectKBest(score_func=chi2, k=10)
    fit = bestfeatures.fit(X, y)
    t1 = time.time()
    total =t1- t0
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
    print("Selected features ",df)
    return df, indexes





def rge(data, n):
    X = data.iloc[:, 0:41]
    y = data.iloc[:, -1]
    print("RFE")
    t0 = time.time()
    model = RandomForestClassifier(random_state=100, n_estimators=50, max_depth=6, )
    sel_rfe = RFECV(estimator=model, step=5)

    sel_rfe = sel_rfe.fit(X, y)
    print(sel_rfe.grid_scores_)
    t1 = time.time()
    total = t1 - t0

    dfcolumns = pd.DataFrame(X.columns)
    print(sel_rfe.ranking_)
    dfscores = pd.DataFrame(sel_rfe.ranking_)
    featureScores = pd.concat([dfcolumns, dfscores], axis=1)
    featureScores.columns = ['Specs', 'Score']
    featureScores = featureScores.sort_values(["Score"])

    selected = featureScores["Specs"].head(n)
    print(selected)
    print("czas", total)

    indexes = []
    for row in selected:
        indexes.append(row)
    indexes.append("class")

    print(indexes)

    df = data[indexes]

    return df, indexes




def boruta(data, n):
    X = data.iloc[:, 0:41].values
    y = data.iloc[:, -1].values
    x1 = data.iloc[:, 0:41]
    y = y.ravel()
    t0 = time.time()
    #rf_boruta = RandomForestClassifier(n_jobs=-1, class_weight='balanced', max_depth=6)
    #boruta = BorutaPy(rf_boruta, n_estimators='auto', verbose=2, max_iter=35)
    #boruta.fit(X, y)
    ranking = ["atr1",'atr23',"atr24","atr25","atr26","atr27","atr28","atr29","atr30","atr31","atr32","atr33","atr34","atr35","atr36","atr37","atr38","atr39","atr22","atr40","atr41","atr10","atr17","atr16","atr2","atr14","atr13","atr12","atr3","atr4","atr5","atr6","atr11"]
    #print(ranking)
    #dfscores = pd.DataFrame(ranking)
    #dfcolumns = pd.DataFrame(x1.columns)
    #feat_importances = pd.concat([dfcolumns, dfscores], axis=1)
    #feat_importances.columns = ['Specs', 'Score']
    #feat_importances = feat_importances.sort_values(["Score"])
    selected = ranking[:n]
    t1 = time.time()
    total = t1 - t0
    print("czas", total)
    #print(feat_importances)
    indexes = []
    for row in selected:
        indexes.append(row)
    indexes.append("class")
    df = data[indexes]
    #print(indexes)
    #print(df)

    return df, indexes


def featureImportance(data, n):
    X = data.iloc[:, 0:41]
    y = data.iloc[:, -1]
    t0 = time.time()
    model = ExtraTreesClassifier(n_jobs=-1, n_estimators=30, random_state=0)
    model.fit(X, y)
    t1 = time.time()
    total = t1-t0
    dfscores = pd.DataFrame(model.feature_importances_)
    dfcolumns = pd.DataFrame(X.columns)
    feat_importances = pd.concat([dfcolumns, dfscores], axis=1)
    feat_importances.columns = ['Specs', 'Score']
    feat_importances = feat_importances.sort_values(["Score"], ascending=False)

    selected = feat_importances["Specs"].head(n)
    print(total)
    indexes = []
    for row in selected:
        indexes.append(row)
    indexes.append("class")

    df = data[indexes]
    print(indexes)
    print(df)

    return df, indexes
