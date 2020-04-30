import randomForest
import naiveBayes
import featureSelection
import kNearestNeighbors
import nb3
import multilayerPerceptor
import supportVectorMachine
import numpy as np
import pandas as pd


def do_knn():
    print("Please select type of feature selection ")
    print("1. Univariate Selection")
    print("2. RFE ")
    print("3. Boruta")
    print("4. Feature importance")
    print("0. None")

    type = inputNumber("Your choose  :   ")

    if type == 0:
        numbers = 0
        return type, numbers

    print("Please select numbers of features ")

    numbers = inputNumber("Features to select :   ")

    return type, numbers


def inputNumber(message):
    while True:
        try:
            userInput = int(input(message))
        except ValueError:
            print("Not an integer! Try again")
            continue
        else:
            return userInput
        break


def inputFloat(message):
    while True:
        try:
            userInput = float(input(message))
        except ValueError:
            print("Not an integer! Try again")
            continue
        else:
            return userInput
        break


print("###########Application for testing algorithms###########\n")

print("Please choose one algorithm that you want to test: \n")

print("1. K Nearest Neighbors")
print("2. Naive Bayes")
print("3. Multilayer Perceptron")
print("4. Random Forest")
print("5. Support Vector Machine")
print("0. End program\n\n")

choose = inputNumber("Your choose :  ")
if choose == 1:
    print("######## K Nearest Neighbors algorithm ########\n ")
    t, n = do_knn()
    print("Algorithm properties :\n")
    folds = inputNumber("Number of folds :   ")
    nei = inputNumber("Number of neighbors :   ")
    kNearestNeighbors.main(t, n, folds, nei)
elif choose == 2:
    print("######## Naive Bayes algorithm ########\n ")
    t, n = do_knn()
    nb3.main(t, n)
elif choose == 3:
    print("######## Multilayer Perceptron ######## \n")
    t, n = do_knn()
    print("Algorithm properties :\n")
    layers = inputNumber("Numbers of neurons in layer :   ")
    iter = inputNumber("Numbers of iteration :   ")
    rate = inputFloat("Learning rate :   ")
    multilayerPerceptor.main(t, n, layers, iter, rate)
elif choose == 4:
    print("######## Random Forest algorithm ########\n ")
    t, n = do_knn()
    print("Algorithm properties :\n")
    trees = inputNumber("Number of trees :   ")
    folds = inputNumber("Number of folds :   ")
    deph = inputNumber("Max deph of tree :   ")
    min_size = inputNumber("Min size :    ")

    randomForest.main(t, n, trees, folds, deph, min_size)

elif choose == 5:
    print("######## Support Vector Machine algorithm ######## \n")
    t, n = do_knn()
    print("Algorithm properties :\n")
    c = inputFloat("Give C :   ")
    iter = inputNumber("Number of iterations :   ")
    tol = inputFloat("Tolerance :   ")
    ver = inputNumber("Verbose :   ")
    supportVectorMachine.main(t, n, c, iter, tol, ver)
elif choose == 0:
    print("######## End of program ######## \n")
    exit()
