import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score


data = pd.read_csv("train.csv")
test = pd.read_csv('test.csv')
train_labels = data['class']

train_data = data.drop('class', axis =1)

test_labels = test['class']
test_data = test.drop('class',axis=1)





model = GaussianNB()

model.fit(train_data,train_labels)

predictions = model.predict(test_data)


score = accuracy_score(test_labels,predictions)
print(score)

