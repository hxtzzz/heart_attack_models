import sys
import os
from PySide2 import QtWidgets
from PySide2.QtWidgets import QMessageBox
from PySide2.QtGui import QIcon
from PySide2.QtUiTools import QUiLoader
import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler
import pandas as pd


# network class
class BinaryNetwork(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.l1 = nn.Linear(input_size, 32)
        self.l2 = nn.Linear(32, 16)
        self.l3 = nn.Linear(16, 8)
        self.out = nn.Linear(8, output_size)

    def forward(self, x):
        x = self.l1(x)
        x = F.relu(x)
        x = self.l2(x)
        x = F.relu(x)
        x = self.l3(x)
        x = F.relu(x)
        x = self.out(x)
        return torch.sigmoid(x)  # scaling values between 0 and 1


class MainWindow:
    def __init__(self):
        self.sex = -1
        self.bs = -1
        self.exng = -1
        self.output = -1
        self.ui = QUiLoader().load('newAttack.ui')
        self.ui.tabWidget.removeTab(1)
        # signal and slot
        self.ui.buttonGroupSex.buttonClicked.connect(self.handleSex)
        self.ui.buttonGroupbs.buttonClicked.connect(self.handleBs)
        self.ui.buttonGroupexng.buttonClicked.connect(self.handleExng)
        self.ui.pushButton.clicked.connect(self.predict)
        self.ui.actionCopyRight.triggered.connect(lambda: self.copyright())
        self.ui.actionCheck.triggered.connect(lambda: self.check())
        self.ui.actionSave.triggered.connect(lambda: self.save())
        self.ui.actionEnglish.triggered.connect(lambda: self.help())
        self.ui.actionChinese.triggered.connect(lambda: self.helpChinese())
        self.ui.actionUpload.triggered.connect(lambda: self.upload())

    def handleSex(self):
        if self.ui.buttonGroupSex.checkedButton().text() == "man":
            self.sex = 1
        else:
            self.sex = 0

    def handleBs(self):
        if self.ui.buttonGroupbs.checkedButton().text() == "t120":
            # print(">120")
            self.bs = 1
        else:
            self.bs = 0

    def handleExng(self):
        if self.ui.buttonGroupexng.checkedButton().text() == "yes":
            # print("yes")
            self.exng = 1
        else:
            self.exng = 0

    def copyright(self):
        QMessageBox.about(self.ui,
                          'CopyRight',
                          f'''From NewCastle University
                          \nauthor: Hu Xiaotian'''
                          )

    def check(self):
        if self.output == -1:
            QMessageBox.warning(self.ui, "Warning", "Please complete the prediction", QMessageBox.StandardButton.Ok)
        else:
            chekcBox = QMessageBox.question(self.ui, "Check", "Is this prediction line with your situation?",
                                            QMessageBox.Yes | QMessageBox.No)
            if chekcBox == QMessageBox.No:
                if self.output == 0:
                    self.checkedoutput = 1
                else:
                    self.checkedoutput = 0
            if self.checkedoutput == 0:
                res = "less chance of heart attack"
            else:
                res = "more chance of heart attack"
            QMessageBox.about(self.ui,
                              'Success',
                              f'''Your prediction has been checked to:
                                           \n{res}'''
                              )

    def save(self):
        if self.output == -1:
            QMessageBox.warning(self.ui, "Warning", "Please complete the prediction at first", QMessageBox.StandardButton.Ok)
        else:
            npout = np.array([[self.checkedoutput]])
            self.whole = np.concatenate((self.input, npout), axis=1)
            print(self.whole)
            # filePath, _ = QFileDialog.getSaveFileName(
            #     self.ui,
            #     "保存文件",
            #     r"d:\\",
            #     "json类型 (*.json)"
            # )
            df = pd.DataFrame(self.whole)
            file_name = 'prediction.csv'
            if not os.path.exists(file_name):
                # if not exists
                df.to_csv("prediction.csv", header=0, index=False)
            else:
                # add mode of pandas
                df.to_csv('prediction.csv', mode='a', header=0, index=False)
            QMessageBox.about(self.ui,
                              'Success',
                              f'''Your prediction has been saved.
                                \n Prediction.csv is stored in the installation path'''
                              )

    def help(self):
        os.system("notepad.exe help.txt")

    def helpChinese(self):
        os.system("notepad.exe helpChinese.txt")

    def upload(self):
        QMessageBox.about(self.ui,
                          'Thank you',
                          f'''You can send the .csv file to author:
                          \n Email: hxtx@outlook.com'''
                          )

    def predict(self):
        sex = self.sex
        bs = self.bs
        exng = self.exng
        age = self.ui.spinAge.value()
        cp = self.ui.spinCP.value()
        bp = self.ui.spinBp.value()
        chol = self.ui.spinChol.value()
        restecg = self.ui.spinRestecg.value()
        thalachh = self.ui.spinHeartRate.value()
        oldpeak = self.ui.doubleSpinBox.value()
        slope = self.ui.spinSlope.value()
        caa = self.ui.spinBoxCaa.value()
        thall = self.ui.spinBoxThall.value()
        # scale input values as the models.
        if sex == -1 or bs == -1 or exng == -1 or bp == 0 or chol == 0 or oldpeak == 0 or thalachh == 0:
            QMessageBox.warning(self.ui, "Warning", "Please input all the information", QMessageBox.StandardButton.Ok)
        else:
            scaler = pickle.load(open('scaler.pkl', 'rb'))
            x = np.array([[age, sex, cp, bp, chol, bs, restecg, thalachh, exng, oldpeak, slope, caa, thall]])
            self.input = x
            x = scaler.transform(x)
            x = torch.from_numpy(x).float()
            new_m = torch.load('net092.pt')
            # call evaluate mode
            new_m.eval()
            predict = new_m(x)
            p = torch.round(predict)
            predict = predict.squeeze()
            p = p.squeeze()
            self.output = p.detach().numpy()
            self.checkedoutput = self.output
            #  print(p)
            if p == 0:
                res = "less chance of heart attack"
            else:
                res = "more chance of heart attack, be careful"
            QMessageBox.about(self.ui,
                              'Prediction',
                              f'''Your prediction chance：{predict}
                                \n{res}'''
                              )


if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    app.setWindowIcon(QIcon('heartlogo.png'))
    window = MainWindow()
    window.ui.show()
    sys.exit(app.exec_())
