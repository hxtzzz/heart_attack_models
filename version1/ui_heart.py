# -*- coding: utf-8 -*-

################################################################################
## Form generated from reading UI file 'newAttack.ui'
##
## Created by: Qt User Interface Compiler version 5.15.2
##
## WARNING! All changes made in this file will be lost when recompiling UI file!
################################################################################

from PySide2.QtCore import *
from PySide2.QtGui import *
from PySide2.QtWidgets import *


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        if not MainWindow.objectName():
            MainWindow.setObjectName(u"MainWindow")
        MainWindow.resize(910, 749)
        MainWindow.setStyleSheet(u"*{\n"
"font-size:15px;\n"
"font-family:arial;\n"
"}")
        MainWindow.setLocale(QLocale(QLocale.English, QLocale.UnitedKingdom))
        self.centralwidget = QWidget(MainWindow)
        self.centralwidget.setObjectName(u"centralwidget")
        self.verticalLayout_2 = QVBoxLayout(self.centralwidget)
        self.verticalLayout_2.setObjectName(u"verticalLayout_2")
        self.tabWidget = QTabWidget(self.centralwidget)
        self.tabWidget.setObjectName(u"tabWidget")
        self.tab = QWidget()
        self.tab.setObjectName(u"tab")
        self.tab.setStyleSheet(u"QPushButton:hover { color: red }")
        self.verticalLayout = QVBoxLayout(self.tab)
        self.verticalLayout.setObjectName(u"verticalLayout")
        self.horizontalLayout = QHBoxLayout()
        self.horizontalLayout.setObjectName(u"horizontalLayout")
        self.widget = QWidget(self.tab)
        self.widget.setObjectName(u"widget")
        self.gridLayout = QGridLayout(self.widget)
        self.gridLayout.setObjectName(u"gridLayout")
        self.line_2 = QFrame(self.widget)
        self.line_2.setObjectName(u"line_2")
        self.line_2.setFrameShape(QFrame.HLine)
        self.line_2.setFrameShadow(QFrame.Sunken)

        self.gridLayout.addWidget(self.line_2, 2, 0, 1, 3)

        self.spinAge = QSpinBox(self.widget)
        self.spinAge.setObjectName(u"spinAge")
        sizePolicy = QSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.spinAge.sizePolicy().hasHeightForWidth())
        self.spinAge.setSizePolicy(sizePolicy)
        self.spinAge.setMaximum(120)

        self.gridLayout.addWidget(self.spinAge, 0, 2, 1, 1)

        self.label_3 = QLabel(self.widget)
        self.label_3.setObjectName(u"label_3")

        self.gridLayout.addWidget(self.label_3, 3, 0, 1, 2)

        self.label_4 = QLabel(self.widget)
        self.label_4.setObjectName(u"label_4")
        sizePolicy1 = QSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)
        sizePolicy1.setHorizontalStretch(0)
        sizePolicy1.setVerticalStretch(0)
        sizePolicy1.setHeightForWidth(self.label_4.sizePolicy().hasHeightForWidth())
        self.label_4.setSizePolicy(sizePolicy1)

        self.gridLayout.addWidget(self.label_4, 4, 0, 1, 3)

        self.t120 = QRadioButton(self.widget)
        self.buttonGroupbs = QButtonGroup(MainWindow)
        self.buttonGroupbs.setObjectName(u"buttonGroupbs")
        self.buttonGroupbs.addButton(self.t120)
        self.t120.setObjectName(u"t120")
        sizePolicy.setHeightForWidth(self.t120.sizePolicy().hasHeightForWidth())
        self.t120.setSizePolicy(sizePolicy)

        self.gridLayout.addWidget(self.t120, 9, 1, 1, 1)

        self.man = QRadioButton(self.widget)
        self.buttonGroupSex = QButtonGroup(MainWindow)
        self.buttonGroupSex.setObjectName(u"buttonGroupSex")
        self.buttonGroupSex.addButton(self.man)
        self.man.setObjectName(u"man")
        sizePolicy.setHeightForWidth(self.man.sizePolicy().hasHeightForWidth())
        self.man.setSizePolicy(sizePolicy)

        self.gridLayout.addWidget(self.man, 1, 1, 1, 1)

        self.spinBp = QSpinBox(self.widget)
        self.spinBp.setObjectName(u"spinBp")
        sizePolicy.setHeightForWidth(self.spinBp.sizePolicy().hasHeightForWidth())
        self.spinBp.setSizePolicy(sizePolicy)
        self.spinBp.setMaximum(200)

        self.gridLayout.addWidget(self.spinBp, 7, 2, 1, 1)

        self.label_7 = QLabel(self.widget)
        self.label_7.setObjectName(u"label_7")
        sizePolicy2 = QSizePolicy(QSizePolicy.Minimum, QSizePolicy.Fixed)
        sizePolicy2.setHorizontalStretch(0)
        sizePolicy2.setVerticalStretch(0)
        sizePolicy2.setHeightForWidth(self.label_7.sizePolicy().hasHeightForWidth())
        self.label_7.setSizePolicy(sizePolicy2)

        self.gridLayout.addWidget(self.label_7, 8, 0, 1, 1)

        self.label_6 = QLabel(self.widget)
        self.label_6.setObjectName(u"label_6")

        self.gridLayout.addWidget(self.label_6, 7, 0, 1, 2)

        self.spinChol = QSpinBox(self.widget)
        self.spinChol.setObjectName(u"spinChol")
        sizePolicy.setHeightForWidth(self.spinChol.sizePolicy().hasHeightForWidth())
        self.spinChol.setSizePolicy(sizePolicy)
        self.spinChol.setMaximum(600)

        self.gridLayout.addWidget(self.spinChol, 8, 2, 1, 1)

        self.f120 = QRadioButton(self.widget)
        self.buttonGroupbs.addButton(self.f120)
        self.f120.setObjectName(u"f120")

        self.gridLayout.addWidget(self.f120, 9, 2, 1, 1)

        self.label_5 = QLabel(self.widget)
        self.label_5.setObjectName(u"label_5")
        sizePolicy1.setHeightForWidth(self.label_5.sizePolicy().hasHeightForWidth())
        self.label_5.setSizePolicy(sizePolicy1)

        self.gridLayout.addWidget(self.label_5, 5, 0, 1, 3)

        self.label_8 = QLabel(self.widget)
        self.label_8.setObjectName(u"label_8")
        sizePolicy1.setHeightForWidth(self.label_8.sizePolicy().hasHeightForWidth())
        self.label_8.setSizePolicy(sizePolicy1)
        self.label_8.setMinimumSize(QSize(0, 0))

        self.gridLayout.addWidget(self.label_8, 9, 0, 1, 1)

        self.woman = QRadioButton(self.widget)
        self.buttonGroupSex.addButton(self.woman)
        self.woman.setObjectName(u"woman")

        self.gridLayout.addWidget(self.woman, 1, 2, 1, 1)

        self.label_2 = QLabel(self.widget)
        self.label_2.setObjectName(u"label_2")

        self.gridLayout.addWidget(self.label_2, 1, 0, 1, 1)

        self.label = QLabel(self.widget)
        self.label.setObjectName(u"label")

        self.gridLayout.addWidget(self.label, 0, 0, 1, 1)

        self.line = QFrame(self.widget)
        self.line.setObjectName(u"line")
        sizePolicy2.setHeightForWidth(self.line.sizePolicy().hasHeightForWidth())
        self.line.setSizePolicy(sizePolicy2)
        self.line.setFrameShape(QFrame.HLine)
        self.line.setFrameShadow(QFrame.Sunken)

        self.gridLayout.addWidget(self.line, 6, 0, 1, 3)

        self.spinCP = QSpinBox(self.widget)
        self.spinCP.setObjectName(u"spinCP")
        sizePolicy.setHeightForWidth(self.spinCP.sizePolicy().hasHeightForWidth())
        self.spinCP.setSizePolicy(sizePolicy)
        self.spinCP.setMinimumSize(QSize(50, 0))
        self.spinCP.setMinimum(0)
        self.spinCP.setMaximum(3)
        self.spinCP.setValue(0)

        self.gridLayout.addWidget(self.spinCP, 3, 2, 1, 1)


        self.horizontalLayout.addWidget(self.widget)

        self.line_4 = QFrame(self.tab)
        self.line_4.setObjectName(u"line_4")
        self.line_4.setFrameShape(QFrame.VLine)
        self.line_4.setFrameShadow(QFrame.Sunken)

        self.horizontalLayout.addWidget(self.line_4)

        self.widget_2 = QWidget(self.tab)
        self.widget_2.setObjectName(u"widget_2")
        self.gridLayout_2 = QGridLayout(self.widget_2)
        self.gridLayout_2.setObjectName(u"gridLayout_2")
        self.spinBoxThall = QSpinBox(self.widget_2)
        self.spinBoxThall.setObjectName(u"spinBoxThall")
        self.spinBoxThall.setMaximum(20)

        self.gridLayout_2.addWidget(self.spinBoxThall, 9, 2, 1, 1)

        self.line_3 = QFrame(self.widget_2)
        self.line_3.setObjectName(u"line_3")
        sizePolicy2.setHeightForWidth(self.line_3.sizePolicy().hasHeightForWidth())
        self.line_3.setSizePolicy(sizePolicy2)
        self.line_3.setFrameShape(QFrame.HLine)
        self.line_3.setFrameShadow(QFrame.Sunken)

        self.gridLayout_2.addWidget(self.line_3, 3, 0, 1, 3)

        self.spinHeartRate = QSpinBox(self.widget_2)
        self.spinHeartRate.setObjectName(u"spinHeartRate")
        self.spinHeartRate.setMaximum(250)

        self.gridLayout_2.addWidget(self.spinHeartRate, 4, 2, 1, 1)

        self.spinSlope = QSpinBox(self.widget_2)
        self.spinSlope.setObjectName(u"spinSlope")
        self.spinSlope.setMaximum(20)

        self.gridLayout_2.addWidget(self.spinSlope, 7, 2, 1, 1)

        self.label_9 = QLabel(self.widget_2)
        self.label_9.setObjectName(u"label_9")

        self.gridLayout_2.addWidget(self.label_9, 0, 0, 1, 2)

        self.label_13 = QLabel(self.widget_2)
        self.label_13.setObjectName(u"label_13")

        self.gridLayout_2.addWidget(self.label_13, 5, 0, 1, 1)

        self.doubleSpinBox = QDoubleSpinBox(self.widget_2)
        self.doubleSpinBox.setObjectName(u"doubleSpinBox")
        self.doubleSpinBox.setDecimals(1)
        self.doubleSpinBox.setMaximum(5.000000000000000)
        self.doubleSpinBox.setSingleStep(0.100000000000000)

        self.gridLayout_2.addWidget(self.doubleSpinBox, 6, 2, 1, 1)

        self.label_17 = QLabel(self.widget_2)
        self.label_17.setObjectName(u"label_17")

        self.gridLayout_2.addWidget(self.label_17, 9, 0, 1, 1)

        self.yes = QRadioButton(self.widget_2)
        self.buttonGroupexng = QButtonGroup(MainWindow)
        self.buttonGroupexng.setObjectName(u"buttonGroupexng")
        self.buttonGroupexng.addButton(self.yes)
        self.yes.setObjectName(u"yes")
        sizePolicy.setHeightForWidth(self.yes.sizePolicy().hasHeightForWidth())
        self.yes.setSizePolicy(sizePolicy)

        self.gridLayout_2.addWidget(self.yes, 5, 1, 1, 1)

        self.label_16 = QLabel(self.widget_2)
        self.label_16.setObjectName(u"label_16")

        self.gridLayout_2.addWidget(self.label_16, 8, 0, 1, 1)

        self.spinRestecg = QSpinBox(self.widget_2)
        self.spinRestecg.setObjectName(u"spinRestecg")
        sizePolicy1.setHeightForWidth(self.spinRestecg.sizePolicy().hasHeightForWidth())
        self.spinRestecg.setSizePolicy(sizePolicy1)
        self.spinRestecg.setMaximum(2)

        self.gridLayout_2.addWidget(self.spinRestecg, 0, 2, 1, 1)

        self.label_15 = QLabel(self.widget_2)
        self.label_15.setObjectName(u"label_15")

        self.gridLayout_2.addWidget(self.label_15, 7, 0, 1, 1)

        self.label_11 = QLabel(self.widget_2)
        self.label_11.setObjectName(u"label_11")
        sizePolicy1.setHeightForWidth(self.label_11.sizePolicy().hasHeightForWidth())
        self.label_11.setSizePolicy(sizePolicy1)

        self.gridLayout_2.addWidget(self.label_11, 2, 0, 1, 2)

        self.spinBoxCaa = QSpinBox(self.widget_2)
        self.spinBoxCaa.setObjectName(u"spinBoxCaa")
        self.spinBoxCaa.setMaximum(20)

        self.gridLayout_2.addWidget(self.spinBoxCaa, 8, 2, 1, 1)

        self.label_10 = QLabel(self.widget_2)
        self.label_10.setObjectName(u"label_10")
        sizePolicy1.setHeightForWidth(self.label_10.sizePolicy().hasHeightForWidth())
        self.label_10.setSizePolicy(sizePolicy1)

        self.gridLayout_2.addWidget(self.label_10, 1, 0, 1, 2)

        self.label_12 = QLabel(self.widget_2)
        self.label_12.setObjectName(u"label_12")

        self.gridLayout_2.addWidget(self.label_12, 4, 0, 1, 2)

        self.label_14 = QLabel(self.widget_2)
        self.label_14.setObjectName(u"label_14")

        self.gridLayout_2.addWidget(self.label_14, 6, 0, 1, 1)

        self.no = QRadioButton(self.widget_2)
        self.buttonGroupexng.addButton(self.no)
        self.no.setObjectName(u"no")
        sizePolicy.setHeightForWidth(self.no.sizePolicy().hasHeightForWidth())
        self.no.setSizePolicy(sizePolicy)

        self.gridLayout_2.addWidget(self.no, 5, 2, 1, 1)


        self.horizontalLayout.addWidget(self.widget_2)


        self.verticalLayout.addLayout(self.horizontalLayout)

        self.pushButton = QPushButton(self.tab)
        self.pushButton.setObjectName(u"pushButton")
        sizePolicy.setHeightForWidth(self.pushButton.sizePolicy().hasHeightForWidth())
        self.pushButton.setSizePolicy(sizePolicy)

        self.verticalLayout.addWidget(self.pushButton)

        self.tabWidget.addTab(self.tab, "")
        self.tab_2 = QWidget()
        self.tab_2.setObjectName(u"tab_2")
        self.tabWidget.addTab(self.tab_2, "")

        self.verticalLayout_2.addWidget(self.tabWidget)

        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QMenuBar(MainWindow)
        self.menubar.setObjectName(u"menubar")
        self.menubar.setGeometry(QRect(0, 0, 910, 23))
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QStatusBar(MainWindow)
        self.statusbar.setObjectName(u"statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)

        self.tabWidget.setCurrentIndex(0)


        QMetaObject.connectSlotsByName(MainWindow)
    # setupUi

    def retranslateUi(self, MainWindow):
        MainWindow.setWindowTitle(QCoreApplication.translate("MainWindow", u"heart attack models", None))
        self.label_3.setText(QCoreApplication.translate("MainWindow", u"Chest pain type", None))
        self.label_4.setText(QCoreApplication.translate("MainWindow", u"0 = Typical Angina, 1 = Atypical Angina", None))
        self.t120.setText(QCoreApplication.translate("MainWindow", u"true", None))
        self.man.setText(QCoreApplication.translate("MainWindow", u"man", None))
        self.label_7.setText(QCoreApplication.translate("MainWindow", u"Cholestoral in mg/dl", None))
        self.label_6.setText(QCoreApplication.translate("MainWindow", u"Resting blood pressure (in Hg mm)", None))
        self.f120.setText(QCoreApplication.translate("MainWindow", u"false", None))
        self.label_5.setText(QCoreApplication.translate("MainWindow", u"2 = Non-anginal Pain, 3 = Asymptomatic ", None))
        self.label_8.setText(QCoreApplication.translate("MainWindow", u"fasting blood sugar > 120 mg/dl", None))
        self.woman.setText(QCoreApplication.translate("MainWindow", u"woman", None))
        self.label_2.setText(QCoreApplication.translate("MainWindow", u"sex", None))
        self.label.setText(QCoreApplication.translate("MainWindow", u"age", None))
        self.label_9.setText(QCoreApplication.translate("MainWindow", u"Resting electrocardiographic results", None))
        self.label_13.setText(QCoreApplication.translate("MainWindow", u"Exercise induced angina", None))
        self.label_17.setText(QCoreApplication.translate("MainWindow", u"Thalium Stress Test result", None))
        self.yes.setText(QCoreApplication.translate("MainWindow", u"yes", None))
        self.label_16.setText(QCoreApplication.translate("MainWindow", u"Number of major vessels", None))
        self.label_15.setText(QCoreApplication.translate("MainWindow", u"slope", None))
        self.label_11.setText(QCoreApplication.translate("MainWindow", u"2 = Left ventricular hypertrophy", None))
        self.label_10.setText(QCoreApplication.translate("MainWindow", u"0 = Normal, 1 = ST-T wave abnormality", None))
        self.label_12.setText(QCoreApplication.translate("MainWindow", u"Maximum heart rate achieved", None))
        self.label_14.setText(QCoreApplication.translate("MainWindow", u"Previous peak ", None))
        self.no.setText(QCoreApplication.translate("MainWindow", u"no", None))
        self.pushButton.setText(QCoreApplication.translate("MainWindow", u"PREDICT NOW", None))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tab), QCoreApplication.translate("MainWindow", u"heart attack prediction", None))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tab_2), QCoreApplication.translate("MainWindow", u"Tab 2", None))
    # retranslateUi

