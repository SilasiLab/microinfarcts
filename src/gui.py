from PyQt5.QtGui import *
from PyQt5.QtWidgets import (QMainWindow, QWidget, QPushButton, QVBoxLayout, QApplication, QLabel,
                             QGridLayout, QHBoxLayout, QComboBox, QCalendarWidget, QListWidget,
                             QSlider, QMessageBox, QRadioButton, QButtonGroup, QDialog, QFileDialog,
                             QLineEdit)
from PyQt5.QtCore import *
import threading
import traceback, sys
from tkinter.filedialog import askdirectory
from tkinter import Tk, messagebox
import os
from image_processing_utils import run_one_brain

class StartWindow(QMainWindow):
    """
    Main entrance of the whole process.
    """
    def __init__(self):
        super().__init__()
        self.root_dir = os.getcwd()
        self.save_dir = os.getcwd()
        self.script_dir = "/home/silasi/ANTs/Scripts"
        # self.script_dir = os.getcwd()

        self.widget_main = QWidget()
        self.layout_main = QGridLayout()
        self.widget_main.setLayout(self.layout_main)

        self.allList = QListWidget()
        self.analyseList = QListWidget()
        self.btnAdd = QPushButton(">>")
        self.btnAdd.clicked.connect(self.add)

        self.buttonGroupWrite = QButtonGroup(self.widget_main)
        self.radio_write = QRadioButton("Write a sumarry")
        self.radio_show = QRadioButton("Show results visually")
        self.buttonGroupWrite.addButton(self.radio_write)
        self.buttonGroupWrite.addButton(self.radio_show)
        self.radio_write.setChecked(True)

        self.buttonGroupAutoSeg = QButtonGroup(self.widget_main)
        self.radio_auto = QRadioButton("Automatically detect microspheres")
        self.radio_man = QRadioButton(" Count microspheres manually")
        self.radio_auto.setChecked(True)

        self.labelnote = QLabel("*make sure images are saved in the correct directory path (i.e. images/brain1/raw/)")
        self.labelnote2 = QLabel("*Select which analysis step you wish to perform")
        self.labelnote3 = QLabel("*Select the root directory")

        self.labelRootDir = QLabel("Select root directory containing serial brain images. (i.e. images/)")
        self.btnRootDir = QPushButton('Select')
        self.btnRootDir.clicked.connect(self.select_rootDir)
        self.lineRootDir = QLabel(self.root_dir)

        self.labelSaveDir = QLabel("Select directory to save results (different from root directory)")
        self.btnSaveDir = QPushButton('Select')
        self.btnSaveDir.clicked.connect(self.select_saveDir)
        self.lineSaveDir = QLabel(self.save_dir)

        self.labelScriptDir = QLabel("Select ANTs script directory")
        self.btnScriptDir = QPushButton('Select')
        self.btnScriptDir.clicked.connect(self.select_scriptDir)
        self.lineScriptDir = QLabel(self.script_dir)

        self.btnRun = QPushButton("Analyse (Will not response for a while)")
        self.btnRun.clicked.connect(self.analyse)

        self.thickness_label = QLabel(self)
        self.thickness_label.setText('                                                                       Section thickness:')
        self.thickness_combo = QComboBox(self)
        self.thickness_combo.addItem("25")
        self.thickness_combo.addItem("50")
        self.thickness_combo.addItem("100")

        self.layout_main.addWidget(self.labelnote, 0, 0)
        self.layout_main.addWidget(self.labelRootDir, 1, 0)
        self.layout_main.addWidget(self.lineRootDir, 2, 0)
        self.layout_main.addWidget(self.btnRootDir, 2, 1)

        self.layout_main.addWidget(self.labelSaveDir, 3, 0)
        self.layout_main.addWidget(self.lineSaveDir, 4, 0)
        self.layout_main.addWidget(self.btnSaveDir, 4, 1)

        self.layout_main.addWidget(self.labelScriptDir, 5, 0)
        self.layout_main.addWidget(self.lineScriptDir, 6, 0)
        self.layout_main.addWidget(self.btnScriptDir, 6, 1)

        self.layout_main.addWidget(self.labelnote2, 7, 0)
        self.layout_main.addWidget(self.radio_show, 8, 0)
        self.layout_main.addWidget(self.radio_write, 8, 1)

        self.layout_main.addWidget(self.radio_auto, 9, 0)
        self.layout_main.addWidget(self.radio_man, 9, 1)

        self.layout_main.addWidget(self.labelnote3, 10, 0)
        self.layout_main.addWidget(self.allList, 11, 0)
        self.layout_main.addWidget(self.btnAdd, 11, 1)
        self.layout_main.addWidget(self.analyseList, 11, 2)

        self.layout_main.addWidget(self.thickness_label, 12, 0)

        self.layout_main.addWidget(self.thickness_combo, 12, 1)
        self.layout_main.addWidget(self.btnRun, 12, 2)

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
        self.thickness = int(str(self.thickness_combo.currentText()))
        print(self.thickness)
        for brain_dir in [str(self.analyseList.item(i).text()) for i in range(self.analyseList.count())]:
            run_one_brain(brain_dir, self.save_dir, True, True, self.script_dir, True, self.radio_write.isChecked(),
                          self.radio_show.isChecked(), False, False, self.radio_auto.isChecked(),
                          section_thickness=self.thickness)




