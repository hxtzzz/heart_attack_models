import sys
from PySide2 import QtWidgets
from PySide2.QtWidgets import QMessageBox
from PySide2.QtGui import QIcon
from PySide2.QtUiTools import QUiLoader
# from ui_heart import Ui_MainWindow
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
        self.ui = QUiLoader().load('newAttack.ui')
        # signal and slot
        self.ui.buttonGroupSex.buttonClicked.connect(self.handleSex)
        self.ui.buttonGroupbs.buttonClicked.connect(self.handleBs)
        self.ui.buttonGroupexng.buttonClicked.connect(self.handleExng)
        self.ui.pushButton.clicked.connect(self.predict)

    def handleSex(self):
        if self.ui.buttonGroupSex.checkedButton().text() == "man":
            # print("man")
            self.sex = 1
        else:
            # print("woman")
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

    def predict(self):
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
        heart_data = 'heart.csv'
        heart_df = pd.read_csv(heart_data)
        X = heart_df.drop('output', axis=1).to_numpy()
        scaler = StandardScaler().fit(X)
        x = np.array([[age, self.sex, cp, bp, chol, self.bs, restecg, thalachh, self.exng, oldpeak, slope, caa, thall]])
       # print(x)
        x = scaler.transform(x)
       # print(x)
        x = torch.from_numpy(x).float()
       # print(x)
        new_m = torch.load('rnn1.pt')
        predict = new_m(x)
        p = torch.round(predict)
      #  print(predict)
      #  print(p)
        p = p.squeeze()
      #  print(p)
        if p == 0:
            res = "less chance of heart attack"
        else:
            res = "more chance of heart attack"
        QMessageBox.about(self.ui,
                          'Prediction',
                          f'''Your prediction：\n{res}'''
                          )


if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    app.setWindowIcon(QIcon('heartlogo.png'))
    window = MainWindow()
    window.ui.show()
    sys.exit(app.exec_())
