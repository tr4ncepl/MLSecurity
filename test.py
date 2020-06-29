import pandas as pd
import os
from PyQt5 import QtCore, QtGui, QtWidgets
import kNearestNeighbors as knn
import supportVectorMachine as svm
import nbtest as nb
import multilayerPerceptor as mp
import rf as rf

from PyQt5 import QtCore, QtGui, QtWidgets
module_dir = os.path.dirname(__file__)
file_path = os.path.join(module_dir, 'history.csv')
history = pd.read_csv(file_path, sep=';', error_bad_lines = False)


class PandasModel(QtCore.QAbstractTableModel):
    def __init__(self, data, parent=None):
        QtCore.QAbstractTableModel.__init__(self, parent)
        self._data = data

    def rowCount(self, parent=None):
        return len(self._data.values)

    def columnCount(self, parent=None):
        return self._data.columns.size

    def data(self, index, role=QtCore.Qt.DisplayRole):
        if index.isValid():
            if role == QtCore.Qt.DisplayRole:
                return str(self._data.values[index.row()][index.column()])
        return None

    def headerData(self, col, orientation, role):
        if orientation == QtCore.Qt.Horizontal and role == QtCore.Qt.DisplayRole:
            return self._data.columns[col]
        return None





class Ui_KNNParams(object):
    def setupUi(self, KNNParams):
        KNNParams.setObjectName("KNNParams")
        KNNParams.resize(422, 217)
        self.cpar = QtWidgets.QLineEdit(KNNParams)
        self.cpar.setGeometry(QtCore.QRect(150, 70, 113, 20))
        self.cpar.setObjectName("cpar")
        self.label = QtWidgets.QLabel(KNNParams)
        self.label.setGeometry(QtCore.QRect(70, 70, 71, 20))
        font = QtGui.QFont()
        font.setPointSize(14)
        self.label.setFont(font)
        self.label.setObjectName("label")
        self.nparam = QtWidgets.QLineEdit(KNNParams)
        self.nparam.setGeometry(QtCore.QRect(150, 110, 113, 20))
        self.nparam.setObjectName("nparam")
        self.label_2 = QtWidgets.QLabel(KNNParams)
        self.label_2.setGeometry(QtCore.QRect(30, 100, 121, 41))
        font = QtGui.QFont()
        font.setPointSize(14)
        self.label_2.setFont(font)
        self.label_2.setObjectName("label_2")
        self.pushButton = QtWidgets.QPushButton(KNNParams)
        self.pushButton.setGeometry(QtCore.QRect(290, 70, 81, 61))
        self.pushButton.setObjectName("pushButton")

        self.retranslateUi(KNNParams)
        QtCore.QMetaObject.connectSlotsByName(KNNParams)

    def retranslateUi(self, KNNParams):
        _translate = QtCore.QCoreApplication.translate
        KNNParams.setWindowTitle(_translate("KNNParams", "Form"))
        self.label.setText(_translate("KNNParams", "Folds = "))
        self.label_2.setText(_translate("KNNParams", "Neigbors  = "))
        self.pushButton.setText(_translate("KNNParams", "Submit"))


class Ui_SVMParams(object):
    def setupUi(self, SVMParams):
        SVMParams.setObjectName("SVMParams")
        SVMParams.resize(424, 220)
        self.centralwidget = QtWidgets.QWidget(SVMParams)
        self.centralwidget.setObjectName("centralwidget")
        self.cpar = QtWidgets.QLineEdit(self.centralwidget)
        self.cpar.setGeometry(QtCore.QRect(200, 20, 113, 20))
        self.cpar.setObjectName("cpar")
        self.nparam = QtWidgets.QLineEdit(self.centralwidget)
        self.nparam.setGeometry(QtCore.QRect(200, 60, 113, 20))
        self.nparam.setObjectName("nparam")
        self.tolparam = QtWidgets.QLineEdit(self.centralwidget)
        self.tolparam.setGeometry(QtCore.QRect(200, 100, 113, 20))
        self.tolparam.setObjectName("tolparam")
        self.label = QtWidgets.QLabel(self.centralwidget)
        self.label.setGeometry(QtCore.QRect(140, 20, 47, 13))
        font = QtGui.QFont()
        font.setPointSize(14)
        self.label.setFont(font)
        self.label.setObjectName("label")
        self.label_2 = QtWidgets.QLabel(self.centralwidget)
        self.label_2.setGeometry(QtCore.QRect(60, 50, 121, 41))
        font = QtGui.QFont()
        font.setPointSize(14)
        self.label_2.setFont(font)
        self.label_2.setObjectName("label_2")
        self.label_3 = QtWidgets.QLabel(self.centralwidget)
        self.label_3.setGeometry(QtCore.QRect(70, 90, 121, 41))
        font = QtGui.QFont()
        font.setPointSize(14)
        self.label_3.setFont(font)
        self.label_3.setObjectName("label_3")
        self.pushButton = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton.setGeometry(QtCore.QRect(340, 40, 81, 61))
        self.pushButton.setObjectName("pushButton")
        SVMParams.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(SVMParams)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 424, 21))
        self.menubar.setObjectName("menubar")
        SVMParams.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(SVMParams)
        self.statusbar.setObjectName("statusbar")
        SVMParams.setStatusBar(self.statusbar)

        self.retranslateUi(SVMParams)
        QtCore.QMetaObject.connectSlotsByName(SVMParams)

    def retranslateUi(self, SVMParams):
        _translate = QtCore.QCoreApplication.translate
        SVMParams.setWindowTitle(_translate("SVMParams", "SVM Params"))
        self.label.setText(_translate("SVMParams", "C = "))
        self.label_2.setText(_translate("SVMParams", "N iterations = "))
        self.label_3.setText(_translate("SVMParams", "Tolerance = "))
        self.pushButton.setText(_translate("SVMParams", "Submit"))


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1024, 799)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.startAlg = QtWidgets.QPushButton(self.centralwidget)
        self.startAlg.setGeometry(QtCore.QRect(360, 485, 151, 41))
        self.startAlg.setObjectName("startAlg")
        self.algOutput = QtWidgets.QTextEdit(self.centralwidget)
        self.algOutput.setGeometry(QtCore.QRect(530, 385, 461, 281))
        self.algOutput.setObjectName("algOutput")
        self.algList = QtWidgets.QListWidget(self.centralwidget)
        self.algList.setGeometry(QtCore.QRect(10, 250, 251, 91))
        self.algList.setObjectName("algList")
        item = QtWidgets.QListWidgetItem()
        self.algList.addItem(item)
        item = QtWidgets.QListWidgetItem()
        self.algList.addItem(item)
        item = QtWidgets.QListWidgetItem()
        self.algList.addItem(item)
        item = QtWidgets.QListWidgetItem()
        self.algList.addItem(item)
        item = QtWidgets.QListWidgetItem()
        self.algList.addItem(item)
        self.featList = QtWidgets.QListWidget(self.centralwidget)
        self.featList.setGeometry(QtCore.QRect(10, 40, 251, 91))
        self.featList.setObjectName("featList")
        item = QtWidgets.QListWidgetItem()
        self.featList.addItem(item)
        item = QtWidgets.QListWidgetItem()
        self.featList.addItem(item)
        item = QtWidgets.QListWidgetItem()
        self.featList.addItem(item)
        item = QtWidgets.QListWidgetItem()
        self.featList.addItem(item)
        item = QtWidgets.QListWidgetItem()
        self.featList.addItem(item)
        self.paramsButt = QtWidgets.QPushButton(self.centralwidget)
        self.paramsButt.setGeometry(QtCore.QRect(190, 350, 91, 41))
        self.paramsButt.setObjectName("paramsButt")
        self.algSumm = QtWidgets.QPlainTextEdit(self.centralwidget)
        self.algSumm.setGeometry(QtCore.QRect(20, 410, 281, 291))
        self.algSumm.setObjectName("algSumm")
        self.nFeat = QtWidgets.QLineEdit(self.centralwidget)
        self.nFeat.setGeometry(QtCore.QRect(30, 170, 113, 20))
        self.nFeat.setObjectName("nFeat")
        self.tableView = QtWidgets.QTableView(self.centralwidget)
        self.tableView.setGeometry(QtCore.QRect(300, 30, 691, 261))
        self.tableView.setObjectName("tableView")
        self.label = QtWidgets.QLabel(self.centralwidget)
        self.label.setGeometry(QtCore.QRect(40, 380, 191, 16))
        font = QtGui.QFont()
        font.setPointSize(10)
        self.label.setFont(font)
        self.label.setObjectName("label")
        self.label_2 = QtWidgets.QLabel(self.centralwidget)
        self.label_2.setGeometry(QtCore.QRect(30, 140, 111, 16))
        font = QtGui.QFont()
        font.setPointSize(10)
        self.label_2.setFont(font)
        self.label_2.setObjectName("label_2")
        self.label_3 = QtWidgets.QLabel(self.centralwidget)
        self.label_3.setGeometry(QtCore.QRect(30, 10, 241, 16))
        font = QtGui.QFont()
        font.setPointSize(10)
        self.label_3.setFont(font)
        self.label_3.setObjectName("label_3")
        self.label_4 = QtWidgets.QLabel(self.centralwidget)
        self.label_4.setGeometry(QtCore.QRect(10, 220, 221, 16))
        font = QtGui.QFont()
        font.setPointSize(10)
        self.label_4.setFont(font)
        self.label_4.setObjectName("label_4")
        self.label_5 = QtWidgets.QLabel(self.centralwidget)
        self.label_5.setGeometry(QtCore.QRect(550, 10, 411, 16))
        font = QtGui.QFont()
        font.setPointSize(14)
        self.label_5.setFont(font)
        self.label_5.setObjectName("label_5")
        self.label_6 = QtWidgets.QLabel(self.centralwidget)
        self.label_6.setGeometry(QtCore.QRect(560, 340, 161, 31))
        font = QtGui.QFont()
        font.setPointSize(14)
        self.label_6.setFont(font)
        self.label_6.setObjectName("label_6")
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 1024, 21))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.startAlg.setText(_translate("MainWindow", "Start"))
        self.algOutput.setHtml(_translate("MainWindow", "<!DOCTYPE HTML PUBLIC \"-//W3C//DTD HTML 4.0//EN\" \"http://www.w3.org/TR/REC-html40/strict.dtd\">\n"
"<html><head><meta name=\"qrichtext\" content=\"1\" /><style type=\"text/css\">\n"
"p, li { white-space: pre-wrap; }\n"
"</style></head><body style=\" font-family:\'MS Shell Dlg 2\'; font-size:8.25pt; font-weight:400; font-style:normal;\">\n"
"<p style=\"-qt-paragraph-type:empty; margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><br /></p></body></html>"))
        __sortingEnabled = self.algList.isSortingEnabled()
        self.algList.setSortingEnabled(False)
        item = self.algList.item(0)
        item.setText(_translate("MainWindow", "Support Vector Machine"))
        item = self.algList.item(1)
        item.setText(_translate("MainWindow", "K Nearest Neighbors"))
        item = self.algList.item(2)
        item.setText(_translate("MainWindow", "Naive Bayes"))
        item = self.algList.item(3)
        item.setText(_translate("MainWindow", "Multilayer Perceptron"))
        item = self.algList.item(4)
        item.setText(_translate("MainWindow", "Random Forest"))
        self.algList.setSortingEnabled(__sortingEnabled)
        __sortingEnabled = self.featList.isSortingEnabled()
        self.featList.setSortingEnabled(False)
        item = self.featList.item(1)
        item.setText(_translate("MainWindow", "Univariate Selection"))
        item = self.featList.item(3)
        item.setText(_translate("MainWindow", "Boruta"))
        item = self.featList.item(2)
        item.setText(_translate("MainWindow", "Recursive Feature Elimination"))
        item = self.featList.item(4)
        item.setText(_translate("MainWindow", "Feature Importance"))
        item = self.featList.item(0)
        item.setText(_translate("MainWindow", "None"))
        self.featList.setSortingEnabled(__sortingEnabled)
        self.paramsButt.setText(_translate("MainWindow", "Parameters"))
        self.label.setText(_translate("MainWindow", "Algorithm Settings"))
        self.label_2.setText(_translate("MainWindow", "Number of features"))
        self.label_3.setText(_translate("MainWindow", "Choose Feature Selection algorithm"))
        self.label_4.setText(_translate("MainWindow", "Choose classification algorithm"))
        self.label_5.setText(_translate("MainWindow", "Latest Classifications"))
        self.label_6.setText(_translate("MainWindow", "Algorithm results"))











class Ui_NBParameters(object):
    def setupUi(self, NBParameters):
        NBParameters.setObjectName("NBParameters")
        NBParameters.resize(422, 217)
        self.centralwidget = QtWidgets.QWidget(NBParameters)
        self.centralwidget.setObjectName("centralwidget")
        self.label = QtWidgets.QLabel(self.centralwidget)
        self.label.setGeometry(QtCore.QRect(40, -10, 341, 91))
        font = QtGui.QFont()
        font.setPointSize(22)
        self.label.setFont(font)
        self.label.setObjectName("label")
        self.pushButton = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton.setGeometry(QtCore.QRect(140, 80, 131, 41))
        self.pushButton.setObjectName("pushButton")
        NBParameters.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(NBParameters)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 422, 21))
        self.menubar.setObjectName("menubar")
        NBParameters.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(NBParameters)
        self.statusbar.setObjectName("statusbar")
        NBParameters.setStatusBar(self.statusbar)

        self.retranslateUi(NBParameters)
        QtCore.QMetaObject.connectSlotsByName(NBParameters)

    def retranslateUi(self, NBParameters):
        _translate = QtCore.QCoreApplication.translate
        NBParameters.setWindowTitle(_translate("NBParameters", "Naive Bayes Parameters"))
        self.label.setText(_translate("NBParameters", "No parameters to choose"))
        self.pushButton.setText(_translate("NBParameters", "Back"))


class Ui_MPParams(object):
    def setupUi(self, MPParams):
        MPParams.setObjectName("MPParams")
        MPParams.resize(423, 220)
        self.centralwidget = QtWidgets.QWidget(MPParams)
        self.centralwidget.setObjectName("centralwidget")
        self.nneur = QtWidgets.QLineEdit(self.centralwidget)
        self.nneur.setGeometry(QtCore.QRect(160, 30, 113, 20))
        self.nneur.setObjectName("nneur")
        self.label = QtWidgets.QLabel(self.centralwidget)
        self.label.setGeometry(QtCore.QRect(10, 0, 141, 71))
        font = QtGui.QFont()
        font.setPointSize(10)
        font.setBold(True)
        font.setWeight(75)
        self.label.setFont(font)
        self.label.setObjectName("label")
        self.pushButton = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton.setGeometry(QtCore.QRect(300, 50, 81, 61))
        self.pushButton.setObjectName("pushButton")
        self.nit = QtWidgets.QLineEdit(self.centralwidget)
        self.nit.setGeometry(QtCore.QRect(160, 70, 113, 20))
        self.nit.setObjectName("nit")
        self.err = QtWidgets.QLineEdit(self.centralwidget)
        self.err.setGeometry(QtCore.QRect(160, 110, 113, 20))
        self.err.setObjectName("err")
        self.label_2 = QtWidgets.QLabel(self.centralwidget)
        self.label_2.setGeometry(QtCore.QRect(0, 60, 151, 41))
        font = QtGui.QFont()
        font.setPointSize(10)
        font.setBold(True)
        font.setWeight(75)
        self.label_2.setFont(font)
        self.label_2.setObjectName("label_2")
        self.label_3 = QtWidgets.QLabel(self.centralwidget)
        self.label_3.setGeometry(QtCore.QRect(70, 100, 121, 41))
        font = QtGui.QFont()
        font.setPointSize(10)
        font.setBold(True)
        font.setWeight(75)
        self.label_3.setFont(font)
        self.label_3.setObjectName("label_3")
        MPParams.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MPParams)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 423, 21))
        self.menubar.setObjectName("menubar")
        MPParams.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MPParams)
        self.statusbar.setObjectName("statusbar")
        MPParams.setStatusBar(self.statusbar)

        self.retranslateUi(MPParams)
        QtCore.QMetaObject.connectSlotsByName(MPParams)

    def retranslateUi(self, MPParams):
        _translate = QtCore.QCoreApplication.translate
        MPParams.setWindowTitle(_translate("MPParams", "MainWindow"))
        self.label.setText(_translate("MPParams", "Number of neurons = "))
        self.pushButton.setText(_translate("MPParams", "Submit"))
        self.label_2.setText(_translate("MPParams", "Number of iterations = "))
        self.label_3.setText(_translate("MPParams", "Error rate = "))


class Ui_RFParams(object):
    def setupUi(self, RFParams):
        RFParams.setObjectName("RFParams")
        RFParams.resize(426, 218)
        self.centralwidget = QtWidgets.QWidget(RFParams)
        self.centralwidget.setObjectName("centralwidget")
        self.label_2 = QtWidgets.QLabel(self.centralwidget)
        self.label_2.setGeometry(QtCore.QRect(40, 70, 151, 41))
        font = QtGui.QFont()
        font.setPointSize(10)
        font.setBold(True)
        font.setWeight(75)
        self.label_2.setFont(font)
        self.label_2.setObjectName("label_2")
        self.min = QtWidgets.QLineEdit(self.centralwidget)
        self.min.setGeometry(QtCore.QRect(190, 120, 113, 20))
        self.min.setObjectName("min")
        self.deph = QtWidgets.QLineEdit(self.centralwidget)
        self.deph.setGeometry(QtCore.QRect(190, 80, 113, 20))
        self.deph.setObjectName("deph")
        self.label = QtWidgets.QLabel(self.centralwidget)
        self.label.setGeometry(QtCore.QRect(50, 10, 141, 71))
        font = QtGui.QFont()
        font.setPointSize(10)
        font.setBold(True)
        font.setWeight(75)
        self.label.setFont(font)
        self.label.setObjectName("label")
        self.trees = QtWidgets.QLineEdit(self.centralwidget)
        self.trees.setGeometry(QtCore.QRect(190, 40, 113, 20))
        self.trees.setObjectName("trees")
        self.pushButton = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton.setGeometry(QtCore.QRect(330, 60, 81, 61))
        self.pushButton.setObjectName("pushButton")
        self.label_3 = QtWidgets.QLabel(self.centralwidget)
        self.label_3.setGeometry(QtCore.QRect(80, 110, 121, 41))
        font = QtGui.QFont()
        font.setPointSize(10)
        font.setBold(True)
        font.setWeight(75)
        self.label_3.setFont(font)
        self.label_3.setObjectName("label_3")
        RFParams.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(RFParams)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 426, 21))
        self.menubar.setObjectName("menubar")
        RFParams.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(RFParams)
        self.statusbar.setObjectName("statusbar")
        RFParams.setStatusBar(self.statusbar)

        self.retranslateUi(RFParams)
        QtCore.QMetaObject.connectSlotsByName(RFParams)

    def retranslateUi(self, RFParams):
        _translate = QtCore.QCoreApplication.translate
        RFParams.setWindowTitle(_translate("RFParams", "MainWindow"))
        self.label_2.setText(_translate("RFParams", "Max deph of tree = "))
        self.label.setText(_translate("RFParams", "Number of trees = "))
        self.pushButton.setText(_translate("RFParams", "Submit"))
        self.label_3.setText(_translate("RFParams", "Min_Split = "))


class SVMParams(QtWidgets.QMainWindow, Ui_SVMParams):
    def __init__(self):
        super(SVMParams, self).__init__()
        self.setupUi(self)


class KNNParams(QtWidgets.QMainWindow, Ui_KNNParams):
    def __init__(self):
        super(KNNParams, self).__init__()
        self.setupUi(self)


class NBParams(QtWidgets.QMainWindow, Ui_NBParameters):
    def __init__(self):
        super(NBParams, self).__init__()
        self.setupUi(self)


class MPParams(QtWidgets.QMainWindow, Ui_MPParams):
    def __init__(self):
        super(MPParams, self).__init__()
        self.setupUi(self)


class RFParams(QtWidgets.QMainWindow, Ui_RFParams):
    def __init__(self):
        super(RFParams, self).__init__()
        self.setupUi(self)


class MainWindow(QtWidgets.QMainWindow, Ui_MainWindow):
    def __init__(self):
        super(MainWindow, self).__init__()
        self.setupUi(self)
        model = PandasModel(history)
        self.tableView.setModel(model)
        self.paramsButt.clicked.connect(self.pressedParam)
        self.startAlg.clicked.connect(self.classify)
        self.algList.clicked.connect(self.pressedParam)

    def pressedParam(self):
        type = self.algList.currentItem().text()
        if (type == "Support Vector Machine"):
            self.window = SVMParams()
            self.window.setupUi(self.window)
            self.window.show()
            self.window.pushButton.clicked.connect(self.passSVM)
            self.alg = 1


        elif (type == "K Nearest Neighbors"):
            self.window = KNNParams()
            self.window.setupUi(self.window)
            self.window.show()
            self.window.pushButton.clicked.connect(self.passKNN)
        elif (type == "Naive Bayes"):
            self.window = NBParams()
            self.window.setupUi(self.window)
            self.window.show()
            self.window.pushButton.clicked.connect(self.passNB)
        elif (type == "Multilayer Perceptron"):
            self.window = MPParams()
            self.window.setupUi(self.window)
            self.window.show()
            self.window.pushButton.clicked.connect(self.passMP)
        elif (type == "Random Forest"):
            self.window = RFParams()
            self.window.setupUi(self.window)
            self.window.show()
            self.window.pushButton.clicked.connect(self.passRF)

    def passSVM(self):
        self.c = self.window.cpar.text()
        self.n = self.window.nparam.text()
        self.tol = self.window.tolparam.text()
        self.nf = self.nFeat.text()
        self.feat_type = self.featList.currentRow()
        feat_name = self.featList.currentItem().text()
        self.window.hide()
        if (self.feat_type == 0):
            self.nf = 41
        params = "Selected algorithm: Support Vector Machine\n" + "Feature Selection: " + feat_name + "\nwith selected features: " + str(
            self.nf) + \
                 "\n\nAlgorithm parameters:\n" + "C = " + str(self.c) + "\nIterations = " + str(
            self.n) + "\nTolerance = " + str(self.tol)
        self.algSumm.clear()
        self.algSumm.insertPlainText(params)

    def passKNN(self):
        self.alg = 2
        self.c = self.window.cpar.text()
        self.n = self.window.nparam.text()
        self.nf = self.nFeat.text()
        self.feat_type = self.featList.currentRow()
        feat_name = self.featList.currentItem().text()
        self.window.hide()
        if (self.feat_type == 0):
            self.nf = 41
        params = "Selected algorithm: K Nearest Neighbors\n" + "Feature Selection: " + feat_name + "\nwith selected features: " + str(
            self.nf) + \
                 "\n\nAlgorithm parameters:\n" + "Folds = " + str(self.c) + "\nNeighbors = " + str(self.n)
        self.algSumm.clear()
        self.algSumm.insertPlainText(params)

    def passNB(self):
        self.alg = 3
        self.nf = self.nFeat.text()
        self.feat_type = self.featList.currentRow()
        feat_name = self.featList.currentItem().text()
        self.window.hide()
        if (self.feat_type == 0):
            self.nf = 41
        params = "Selected algorithm: Naive Bayes\n" + "Feature Selection: " + feat_name + "\nwith selected features: " + str(
            self.nf) + \
                 "\n\nAlgorithm parameters:\n" + "No parameters available for this algorithm"
        self.algSumm.clear()
        self.algSumm.insertPlainText(params)

    def passMP(self):
        self.alg = 4
        self.neurons = self.window.nneur.text()
        self.iter = self.window.nit.text()
        self.error = self.window.err.text()
        self.nf = self.nFeat.text()
        self.feat_type = self.featList.currentRow()
        feat_name = self.featList.currentItem().text()
        self.window.hide()
        if (self.feat_type == 0):
            self.nf = 41
        params = "Selected algorithm: Multilayer Perceptron\n" + "Feature Selection: " + feat_name + "\nwith selected features: " + str(
            self.nf) + \
                 "\n\nAlgorithm parameters:\n" + "Neurons in hidden layer = " + str(
            self.neurons) + "\nNumber of iterations = " + str(self.iter) + "\nError rate = " + str(self.error)
        self.algSumm.clear()
        self.algSumm.insertPlainText(params)

    def passRF(self):
        self.alg = 5
        self.trs = self.window.trees.text()
        self.dph = self.window.deph.text()
        self.gain = self.window.min.text()
        self.nf = self.nFeat.text()
        self.feat_type = self.featList.currentRow()
        feat_name = self.featList.currentItem().text()
        self.window.hide()
        if (self.feat_type == 0):
            self.nf = 41
        params = "Selected algorithm: Random Forest\n" + "Feature Selection: " + feat_name + "\nwith selected features: " + str(
            self.nf) + \
                 "\n\nAlgorithm parameters:\n" + "Trees in forest= " + str(
            self.trs) + "\nDepth of each tree = " + str(self.dph) + "\nMinimum gain to split = " + str(self.gain)
        self.algSumm.clear()
        self.algSumm.insertPlainText(params)

    def classify(self):
        if (self.alg == 1):
            final = svm.start(int(self.feat_type), int(self.nf), int(self.c), int(self.n, ), float(self.tol), 2)
            self.algOutput.clear()
            self.algOutput.insertPlainText(final)
        elif (self.alg == 2):
            final = knn.start(int(self.feat_type), int(self.nf), int(self.c), int(self.n))
            self.algOutput.clear()
            self.algOutput.insertPlainText(final)
        elif (self.alg == 3):
            final = nb.start(int(self.feat_type), int(self.nf))
            self.algOutput.clear()
            self.algOutput.insertPlainText(final)
        elif (self.alg == 4):
            final = mp.start(int(self.feat_type), int(self.nf), int(self.neurons), int(self.iter), float(self.error))
            self.algOutput.clear()
            self.algOutput.insertPlainText(final)
        elif (self.alg == 5):
            final = rf.start(int(self.feat_type), int(self.nf), int(self.trs), str(self.dph),int(self.gain) )
            self.algOutput.clear()
            self.algOutput.insertPlainText(final)


if __name__ == "__main__":
    import sys

    app = QtWidgets.QApplication(sys.argv)
    w = MainWindow()
    w.show()
    sys.exit(app.exec_())
