from csv import reader
from math import sqrt
from random import randrange
import pandas as pd


def str_column_to_float(dataset, column):
    for row in dataset:
        row[column] = float(row[column].strip())


# Convert string column to integer
def str_column_to_int(dataset, column):
    class_values = [row[column] for row in dataset]
    unique = set(class_values)
    lookup = dict()
    for i, value in enumerate(unique):
        lookup[value] = i
    for row in dataset:
        row[column] = lookup[row[column]]
    return lookup

def load_csv(filename):
    dataset = list()
    with open(filename, 'r') as file:
        csv_reader = reader (file)
        for row in csv_reader:
            if not row:
                continue
            dataset.append(row)
    return dataset


def dataset_minmax(dataset):
    minmax = list()
    for i in range(len(dataset[0])):
        col_values = [row[i] for row in dataset]
        value_min = min(col_values)
        value_max = max(col_values)
        minmax.append([value_min, value_max])
    return minmax


def normalize_dataset(dataset, minmax):
    for row in dataset:
        for i in range(len(row)):
            row[i] = (row[i] - minmax[i][0]) / (minmax[i][1] - minmax[i][0])


def cross_validation_split(dataaset, n_folds):
    dataset_split = list()
    dataset_copy = list(dataaset)

    fold_size = int(len(dataaset) / n_folds)

    for _ in range (n_folds):
        fold = list()
        while len(fold) <fold_size:
            index = randrange(len(dataset_copy))
            fold.append(dataset_copy.pop(index))
        dataset_split.append(fold)
    return dataset_split

def accuracy_metric(actual, predicted):
    correct = 0
    for i in range(len(actual)):
        if actual[i] == predicted[i]:
            correct +=1
    return correct / float(len(actual)) * 100.0


def evaluate_algorithm(dataset, algorithm, n_folds, *args):
    folds = cross_validation_split(dataset, n_folds)
    scores = list()
    for fold in folds:
        train_set = list(folds)
        train_set.remove(fold)
        train_set = sum(train_set, [])
        test_set = list()

        for row in fold:
            row_copy = list(row)
            test_set.append(row_copy)
            row_copy[-1] = None
        predicted = algorithm(train_set, test_set, *args)
        actual = [row[-1] for row in fold]
        accuracy = accuracy_metric(actual, predicted)
        scores.append(accuracy)
    return scores


def euclidean_distance(row1, row2):
    distance = 0.0
    for i in range(len(row1)-1):
        distance += (row1[i] - row2[i])**2
    return sqrt(distance)


def get_neighbors(train, test_row, num_neighbors):
    distances = list()
    for train_row in train:
        dist = euclidean_distance(test_row, train_row)
        distances.append((train_row,dist))
    distances.sort(key=lambda tup: tup[1])
    neigbors = list()
    for i in range (num_neighbors):
        neigbors.append(distances[i][0])
    return neigbors

def predict_classification(train, test_row, num_neighbors):

    neighbors = get_neighbors(train, test_row, num_neighbors)
    output_values = [row[-1] for row in neighbors]
    prediction = max(set(output_values), key=output_values.count)

    return prediction

def k_nearest_neighbors(train, test, num_neighbors):
    predictions = list()
    for row in test:
        output = predict_classification(train, row, num_neighbors)
        predictions.append(output)
    return predictions




filename = 'train.csv'

dataset = load_csv(filename)

test = pd.read_csv('test.csv')

test_labels = test['class']

test_data = test.drop('class', axis =1)





dataset = load_csv(filename)
for i in range(len(dataset[0])-1):
    str_column_to_float(dataset, i)




# evaluate algorithm


n_folds = 5
num_neighbors =6
scores = evaluate_algorithm(dataset, k_nearest_neighbors, n_folds, num_neighbors)
print('Scores: %s' % scores)
print('Mean Accuracy: %.3f%%' % (sum(scores)/float(len(scores))))

rows = [[0,1,50,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,55,1,0.13,0,0.87,1,0.02,1,0,255,1,0,0.49,0,0,0.03,0,0.2,1], [1,1,61,3,0,15,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,8,0,0.12,1,0.5,1,0,0.75,29,86,0.31,0.17,0.03,0.02,0,0,0.83,0.71], [0,1,20,2,26,157,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,1,1,1,0,0,0,0,1,0,0,52,26,0.5,0.08,0.02,0,0,0,0,0]]
test_data = test_data.values.tolist()

test_labels = test_labels.values.tolist()
print(rows[1])
print(test_data[1])
print(len(rows[1]))
print(len(test_data[1]))



predictions = list()
for i in range(len(test_data)):
    label = predict_classification(dataset, test_data[i], num_neighbors)
    predictions.append(label)

print(test_labels)
print(predictions)

acc = accuracy_metric(test_labels, predictions)

print(acc)


