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
data = pd.read_csv("data.csv")




def univariateSeceltion(data):
    X = data.iloc[:,0:41]
    y = data.iloc[:,-1]
    bestfeatures = SelectKBest(score_func=chi2, k=41)
    fit = bestfeatures.fit(X,y)
    dfscores = pd.DataFrame(fit.scores_)
    dfcolumns = pd.DataFrame(X.columns)

    featureScores = pd.concat([dfcolumns,dfscores],axis=1)
    featureScores.columns = ['Specs','Score']
    result = featureScores.nlargest(41,'Score')
    return result





def rge(data):
    X = data.iloc[:, 0:41]
    y = data.iloc[:, -1]

    model = RandomForestClassifier(random_state=100, n_estimators=50)

    sel_rfe = RFE(estimator=model, n_features_to_select=10, step=1)

    x_train_rfe = sel_rfe.fit_transform(X, y)

    print(sel_rfe.get_support())

    print(sel_rfe.ranking_)


def SFM(data):
    X = data.iloc[:, 0:41]
    y = data.iloc[:, -1]
    model = RandomForestClassifier(random_state=100,
                                   n_estimators=50)

    model.fit(X, y)

    print(model.feature_importances_)

    sel_model = SelectFromModel(estimator=model, prefit=True, threshold='mean')

    train_sfm = sel_model.transform(X)

    print(sel_model.get_support())


def featureImportance(data):
    X = data.iloc[:, 0:41]
    y = data.iloc[:, -1]

    model = ExtraTreesClassifier()
    model.fit(X, y)

    print(model.feature_importances_)

    feat_importances = pd.Series(model.feature_importances_, index=X.columns)
    feat_importances.nlargest(18).plot(kind='barh')
    plt.show()

test = univariateSeceltion(data)

print(test)

featureImportance(data)