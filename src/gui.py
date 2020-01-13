from PyQt5.QtGui import *
from PyQt5.QtWidgets import (QMainWindow, QWidget, QPushButton, QVBoxLayout, QApplication, QLabel,
                             QGridLayout, QHBoxLayout, QComboBox, QCalendarWidget, QListWidget,
                             QSlider, QMessageBox, QRadioButton, QButtonGroup, QDialog, QFileDialog)
from PyQt5.QtCore import *
import threading
import traceback, sys
from tkinter.filedialog import askdirectory
from tkinter import Tk, messagebox
import os
from image_processing_utils import run_one_brain

class StartWindow(QMainWindow):

    def __init__(self):
        super().__init__()
        self.root_dir = "/mnt/4T/brain_imgs/"
        self.save_dir = "/mnt/4T/brain_result/"
        self.script_dir = "/home/silasi/ANTs/Scripts"

        self.widget_main = QWidget()
        self.layout_main = QGridLayout()
        self.widget_main.setLayout(self.layout_main)

        self.allList = QListWidget()
        self.analyseList = QListWidget()
        self.btnAdd = QPushButton(">>")
        self.btnAdd.clicked.connect(self.add)

        self.buttonGroupWrite = QButtonGroup(self.widget_main)
        self.radio_write = QRadioButton("Write a sumarry")
        self.radio_show = QRadioButton("Show the result (For debugging)")
        self.buttonGroupWrite.addButton(self.radio_write)
        self.buttonGroupWrite.addButton(self.radio_show)
        self.radio_write.setChecked(True)

        self.buttonGroupAutoSeg = QButtonGroup(self.widget_main)
        self.radio_auto = QRadioButton("Auto Segmentation")
        self.radio_man = QRadioButton("Use existing manual label")
        self.radio_auto.setChecked(True)

        self.labelRootDir = QLabel("Select root directory containing all the brain folders")
        self.btnRootDir = QPushButton('Select')
        self.btnRootDir.clicked.connect(self.select_rootDir)
        self.lineRootDir = QLabel(self.root_dir)

        self.labelSaveDir = QLabel("Select directory to save results")
        self.btnSaveDir = QPushButton('Select')
        self.btnSaveDir.clicked.connect(self.select_saveDir)
        self.lineSaveDir = QLabel(self.save_dir)

        self.labelScriptDir = QLabel("Select ANTs script directory")
        self.btnScriptDir = QPushButton('Select')
        self.btnScriptDir.clicked.connect(self.select_scriptDir)
        self.lineScriptDir = QLabel(self.script_dir)

        self.btnRun = QPushButton("Analyse (Will not response for a while)")
        self.btnRun.clicked.connect(self.analyse)

        self.layout_main.addWidget(self.labelRootDir, 0, 0)
        self.layout_main.addWidget(self.lineRootDir, 1, 0)
        self.layout_main.addWidget(self.btnRootDir, 1, 1)

        self.layout_main.addWidget(self.labelSaveDir, 2, 0)
        self.layout_main.addWidget(self.lineSaveDir, 3, 0)
        self.layout_main.addWidget(self.btnSaveDir, 3, 1)

        self.layout_main.addWidget(self.labelScriptDir, 4, 0)
        self.layout_main.addWidget(self.lineScriptDir, 5, 0)
        self.layout_main.addWidget(self.btnScriptDir, 5, 1)

        self.layout_main.addWidget(self.radio_show, 6, 0)
        self.layout_main.addWidget(self.radio_write, 6, 1)

        self.layout_main.addWidget(self.radio_auto, 7, 0)
        self.layout_main.addWidget(self.radio_man, 7, 1)

        self.layout_main.addWidget(self.allList, 8, 0)
        self.layout_main.addWidget(self.btnAdd, 8, 1)
        self.layout_main.addWidget(self.analyseList, 8, 2)

        self.layout_main.addWidget(self.btnRun, 9, 0)

        self.setCentralWidget(self.widget_main)
        self.update_list()
        self.show()

    def update_list(self):
        for item in os.listdir(self.root_dir):
            self.allList.addItem(os.path.join(self.root_dir, item))

    def add(self):
        for item in self.allList.selectedItems():
            if len(self.analyseList.findItems(item.text(), Qt.MatchExactly)) == 0:
                self.analyseList.addItem(item.text())

    def choosePath(self):
        root = Tk()
        root.withdraw()
        result = askdirectory(initialdir="/", title="Select root directory containing all cage folders")
        return result

    def select_rootDir(self):
        self.root_dir = self.choosePath()
        if type(self.root_dir) is str:
            self.lineRootDir.setText(self.root_dir)
            self.allList.clear()
            self.analyseList.clear()
            self.update_list()

    def select_saveDir(self):
        self.save_dir = self.choosePath()
        if type(self.save_dir) is str:
            self.lineSaveDir.setText(self.save_dir)

    def select_scriptDir(self):
        self.script_dir = self.choosePath()
        if type(self.script_dir) is str:
            self.lineScriptDir.setText(self.script_dir)

    def analyse(self):
        for brain_dir in [str(self.analyseList.item(i).text()) for i in range(self.analyseList.count())]:
            run_one_brain(brain_dir, self.save_dir, True, True, self.script_dir, True, self.radio_write.isChecked(),
                          self.radio_show.isChecked(), False, False, self.radio_auto.isChecked())

if __name__ == '__main__':
    app = QApplication([])
    start_window = StartWindow()
    start_window.show()
    app.exit(app.exec_())



