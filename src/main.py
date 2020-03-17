from image_processing_utils import *
import argparse
from gui import *



app = QApplication([])
start_window = StartWindow()
start_window.show()
app.exit(app.exec_())