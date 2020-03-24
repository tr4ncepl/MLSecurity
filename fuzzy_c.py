
import copy
import math
import random
import time
import sys
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import decimal
import pandas as pd
from sklearn.decomposition import PCA
import pylab as pl
import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import StandardScaler, Normalizer

global MAX
MAX = 100.0

global Epsilon
Epsilon = 0.00000001


# 0 - Canadian
# 1 - Rosa
# 2 - Kama


def normalize():
    seeds = pd.read_csv("uczace2.txt")
    randomize = seeds.sample(frac=1).reset_index(drop=True)
    x = pd.DataFrame(randomize, columns=['atr1', 'atr2', 'atr3', 'atr4', 'atr5', 'atr6', 'atr7'])
    y = pd.DataFrame(randomize, columns=['class'])
    # print(x)

    pca = PCA(n_components=2).fit(x)
    pca_2d = pca.transform(x)
    final = pd.DataFrame(pca_2d)
    export_csv = final.to_csv("a.txt", sep=",", index=False, header=False)
    final['klasa'] = y
    for i in range(0, pca_2d.shape[0]):
        if final.klasa[i] == 0:
            c1 = pl.scatter(pca_2d[i, 0], pca_2d[i, 1], c='r', marker='+')
        elif final.klasa[i] == 1:
            c2 = pl.scatter(pca_2d[i, 0], pca_2d[i, 1], c='g', marker='o')
        elif final.klasa[i] == 2:
            c3 = pl.scatter(pca_2d[i, 0], pca_2d[i, 1], c='b', marker='*')
    pl.legend([c1, c2, c3], ['Canadian', 'Rosa ', 'Kama'])
    pl.title('Przypisanie danych do odpowiednich grup według klas')
    pl.savefig("final1.png")
    # pl.show()
    pl.close()
    return pca_2d, final


def import_data(file):

    data = []
    f = open(str(file), 'r')
    for line in f:
        current = line.split(",")  # enter your own delimiter like ","
        for j in range(0, len(current)):
            current[j] = float(current[j])
        data.append(current)
    return data


def imp(file):
    data = []
    cluster_location = []
    f = open(str(file), 'r')
    for line in f:
        current = line.split(",")
        current_dummy = []
        for j in range(0, len(current) - 1):
            current_dummy.append(float(current[j]))
        j += 1
        # print current[j]
        if current[j] == "0\n":
            cluster_location.append(0)
        elif current[j] == "1\n":
            cluster_location.append(1)
        else:
            cluster_location.append(2)
        data.append(current_dummy)
    print("finished importing data")
    return data, cluster_location

def print_matrix(list):

    for i in range(0, len(list)):
        print(list[i])


def end_conditon(U, U_old):

    global Epsilon
    for i in range(0, len(U)):
        for j in range(0, len(U[0])):
            if abs(U[i][j] - U_old[i][j]) > Epsilon:
                return False
    return True


def initialise_U(data, cluster_number):
    global MAX
    U = []
    for i in range(0, len(data)):
        current = []
        rand_sum = 0.0
        for j in range(0, cluster_number):
            dummy = random.randint(1, int(MAX))
            current.append(dummy)
            rand_sum += dummy
        for j in range(0, cluster_number):
            current[j] = current[j] / rand_sum
        U.append(current)
    return U


def distance(point, center):


    if len(point) != len(center):
        return -1
    dummy = 0.0
    for i in range(0, len(point)):
        dummy += abs(point[i] - center[i]) ** 2
    return math.sqrt(dummy)


def normalise_U(U):
    cluster_labels = list()
    n = len(data)
    for i in range(n):
        max_val, idx = max((val, idx) for (idx, val) in enumerate(U[i]))
        cluster_labels.append(idx)
    return cluster_labels


def fuzzy(data, cluster_number, m):


    U = initialise_U(data, cluster_number)

    while (True):

        U_old = copy.deepcopy(U)

        C = []
        for j in range(0, cluster_number):
            current_cluster_center = []
            for i in range(0, len(data[0])):  # this is the number of dimensions
                dummy_sum_num = 0.0
                dummy_sum_dum = 0.0
                for k in range(0, len(data)):
                    dummy_sum_num += (U[k][j] ** m) * data[k][i]
                    dummy_sum_dum += (U[k][j] ** m)
                current_cluster_center.append(dummy_sum_num / dummy_sum_dum)
            C.append(current_cluster_center)




        distance_matrix = []
        for i in range(0, len(data)):
            current = []
            for j in range(0, cluster_number):
                current.append(distance(data[i], C[j]))
            distance_matrix.append(current)



        for j in range(0, cluster_number):
            for i in range(0, len(data)):
                dummy = 0.0
                for k in range(0, cluster_number):
                    dummy += (distance_matrix[i][j] / distance_matrix[i][k]) ** (2 / (m - 1))
                U[i][j] = 1 / dummy

        if end_conditon(U, U_old):
            print("finished clustering")
            break

    old_U = U
    U = normalise_U(U)
    print("normalised U")
    return U, C, old_U

def check(s1, s2):
    n = len(s1)

    tp = 0
    np = 0
    for i in range(len(s1)):
        if s1[i]==s2[i]:
            tp = tp +1
        else:
            np = np +1
    acc = tp/n
    print(acc)
    return acc, tp, np






## main
if __name__ == '__main__':
    # import the data
    wyk,fin = normalize()
    data = import_data('a.txt')
    test, test2 = imp("ucz.txt")

    start = time.time()


    while(True):
        final_location, centers, old = fuzzy(data, 3, 1.3)
        acc, tp, p = check(final_location, fin.klasa)
        if acc>0.8:
            break

    print(', '.join(map(repr,fin.klasa)))
    print(final_location)
    print(old)
    print_matrix(centers)
    print("Accuracy : ", acc)
    print("Correct :", tp, "\nIncorrect :", p)

    centers = np.array(centers)


    for i in range(0, wyk.shape[0]):
        if final_location[i] == 0:
            c1 = pl.scatter(wyk[i, 0], wyk[i, 1], c='r', marker='+')
        elif final_location[i] == 1:
            c2 = pl.scatter(wyk[i, 0], wyk[i, 1], c='g', marker='o')
        elif final_location[i] == 2:
            c3 = pl.scatter(wyk[i, 0], wyk[i, 1], c='b', marker='*')
    pl.legend([c1, c2, c3] ,['Cluster 1', 'Cluster 2', 'Cluster 3'])
    pl.scatter(centers[:, 0], centers[:, 1], c='black', s=200, alpha=0.5)
    pl.title("Efekt grupowania algorytmu C-Means")
    pl.savefig("final2.png")
    # pl.show()
    pl.close()
    print("time elapsed=", time.time() - start)

