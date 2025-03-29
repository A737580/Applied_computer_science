import sys
from PySide6.QtWidgets import (QApplication, QMainWindow, QTabWidget)
from widgets.templateMatchingWidget import TemplateMatchingWidget
from widgets.violaJonesWidget import ViolaJonesWidget
from widgets.symmetryLinesWidget import SymmetryLinesWidget 
from widgets.templateMakerWidget import TemplateMaker

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Методы OpenCV с PySide6")
        tab_widget = QTabWidget(self)
        self.setCentralWidget(tab_widget)
        self.template_matching_tab = TemplateMatchingWidget()
        tab_widget.addTab(self.template_matching_tab, "Template Matching")
        self.template_matching_tab = ViolaJonesWidget()
        tab_widget.addTab(self.template_matching_tab, "Viola Jones")
        self.template_matching_tab = SymmetryLinesWidget()
        tab_widget.addTab(self.template_matching_tab, "Symmetry Lines")
        self.template_maker = TemplateMaker()
        tab_widget.addTab(self.template_maker, "Template maker")

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MainWindow()
    window.resize(600, 700)
    window.show()
    sys.exit(app.exec())