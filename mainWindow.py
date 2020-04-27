import randomForest as rf

from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(800, 600)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.selection_box = QtWidgets.QComboBox(self.centralwidget)
        self.selection_box.setGeometry(QtCore.QRect(30, 100, 171, 21))
        self.selection_box.setObjectName("selection_box")
        self.selection_box.addItem("")
        self.selection_box.addItem("")
        self.selection_box.addItem("")
        self.selection_box.addItem("")

        self.feature_box = QtWidgets.QComboBox(self.centralwidget)
        self.feature_box.setGeometry(QtCore.QRect(30, 160, 69, 22))
        self.feature_box.setObjectName("feature_box")
        self.feature_box.addItem("")
        self.feature_box.addItem("")
        self.feature_box.addItem("")
        self.feature_box.addItem("")
        self.feature_box.addItem("")

        self.algorithms_box = QtWidgets.QComboBox(self.centralwidget)
        self.algorithms_box.setGeometry(QtCore.QRect(30, 40, 171, 22))
        self.algorithms_box.setObjectName("algorithms_box")
        self.algorithms_box.addItem("")
        self.algorithms_box.addItem("")
        self.algorithms_box.addItem("")
        self.algorithms_box.addItem("")
        self.algorithms_box.addItem("")

        self.button = QtWidgets.QPushButton(self.centralwidget)
        self.button.setGeometry(QtCore.QRect(30, 230, 75, 23))
        self.button.setObjectName("button")

        self.button.clicked.connect(self.pressed)

        self.plainTextEdit = QtWidgets.QPlainTextEdit(self.centralwidget)
        self.plainTextEdit.setGeometry(QtCore.QRect(250, 90, 491, 461))
        self.plainTextEdit.setObjectName("plainTextEdit")
        summary = "TEST XD"
        self.plainTextEdit.insertPlainText(summary)

        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 800, 21))
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
        self.selection_box.setItemText(0, _translate("MainWindow", "Univariate Selection"))
        self.selection_box.setItemText(1, _translate("MainWindow", "None"))
        self.selection_box.setItemText(2, _translate("MainWindow", "Recursive Feature Elimination"))
        self.selection_box.setItemText(3, _translate("MainWindow", "Feature Importance"))
        self.feature_box.setItemText(0, _translate("MainWindow", "8"))
        self.feature_box.setItemText(1, _translate("MainWindow", "12"))
        self.feature_box.setItemText(2, _translate("MainWindow", "16"))
        self.feature_box.setItemText(3, _translate("MainWindow", "20"))
        self.feature_box.setItemText(4, _translate("MainWindow", "Wszystkie"))
        self.algorithms_box.setItemText(0, _translate("MainWindow", "Support Vector Machine"))
        self.algorithms_box.setItemText(1, _translate("MainWindow", "Random Forest"))
        self.algorithms_box.setItemText(2, _translate("MainWindow", "Naive Bayes"))
        self.algorithms_box.setItemText(3, _translate("MainWindow", "K Nearest Neighbors"))
        self.algorithms_box.setItemText(4, _translate("MainWindow", "Multilayer Perceptor"))
        self.button.setText(_translate("MainWindow", "Train"))

    def pressed(self):
        type = self.algorithms_box.currentIndex()
        test = rf.main()
        xd = test.toString()
        self.plainTextEdit.insertPlainText(xd)




if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())
