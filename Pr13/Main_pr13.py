import sys
from PySide6.QtWidgets import (QApplication, QMainWindow)
from Pr13.gui.gui import PainterAttributionApp

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Атрибуция картин по автору")  
        self.setGeometry(100, 100, 1150, 800) 
        
        self.main_widget = PainterAttributionApp()
        self.setCentralWidget(self.main_widget)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    
    window = MainWindow()
    window.show()
    sys.exit(app.exec())