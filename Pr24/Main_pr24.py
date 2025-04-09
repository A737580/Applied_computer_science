import sys
from PySide6.QtWidgets import (QApplication, QMainWindow, QTabWidget)
from widgets.BioQRApp import BioQRApp
from widgets.LsbEmbeddingWidget import LsbEmbeddingWidget

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Formation and study of hidden encrypted barcode")  
        tab_widget = QTabWidget(self)
        self.setCentralWidget(tab_widget)
        
        self.bioQrApp_tab = BioQRApp()
        self.lsbEmbedding_tab = LsbEmbeddingWidget(self)

        tab_widget.addTab(self.bioQrApp_tab, "Bio QR-code Encoder/Decoder/Visualizer (PAP)")
        tab_widget.addTab(self.lsbEmbedding_tab, "LSB Embedding")

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MainWindow()
    window.setGeometry(100, 100, 1050, 700)
    window.show()
    sys.exit(app.exec())