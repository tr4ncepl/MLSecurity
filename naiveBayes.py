from collections import defaultdict
from math import pi
from math import e
import requests
import random
import csv
import re
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

data = pd.read_csv("dane.txt")

labels = np.array(data['class'])

data = data.drop('class', axis=1)

data_list = list(data.columns)

data = np.array(data)

train_data, test_data, train_labels, test_labels = train_test_split(data, labels, test_size=0.25, random_state=42)

rf = RandomForestRegressor(n_estimators=1000, random_state=42)

rf.fit(train_data, train_labels)

predictions = rf.predict(test_data)

a = 0
b=0
errors = abs(predictions - test_labels)

for i in range(len(predictions)):
    if test_labels[i] == predictions[i]:
        a = a + 1
    else:
        b=b+1

acc = a/len(predictions)
print(acc)
