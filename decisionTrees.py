from sklearn import tree
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score



data = pd.read_csv("train.csv")
test = pd.read_csv('test.csv')
train_labels = data['class']

train_data = data.drop('class', axis =1)

test_labels = test['class']
test_data = test.drop('class',axis=1)


print(test_labels)
print(train_labels)
print(train_data)
print(test_data)


clf = tree.DecisionTreeClassifier(criterion="entropy", random_state=100)
clf = clf.fit(train_data,train_labels)

predictions = clf.predict(test_data)

acc = accuracy_score(test_labels,predictions)

print(acc)




tree.plot_tree(clf.fit(train_data, train_labels))



