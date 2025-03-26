from PySide6.QtWidgets import QLabel
from PySide6.QtGui import QPainter, QPen
from PySide6.QtCore import Qt, QRect, Signal

class ImageLabel(QLabel):
    mouse_release_completed = Signal()
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAlignment(Qt.AlignCenter)
        self._pixmap = None
        self.start_point = None
        self.end_point = None
        self.drawing = False
        self.selection_rect = None

    def setPixmap(self, pixmap):
        self._pixmap = pixmap
        self.selection_rect = None
        super().setPixmap(pixmap)

    def mousePressEvent(self, event):
        if self._pixmap is None:
            return
        if event.button() == Qt.LeftButton:
            self.start_point = event.pos()
            self.end_point = event.pos()
            self.drawing = True
            self.selection_rect = QRect(self.start_point, self.end_point)
            self.update()

    def mouseMoveEvent(self, event):
        if self.drawing:
            self.end_point = event.pos()
            self.selection_rect = QRect(self.start_point, self.end_point).normalized()
            self.update()

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton and self.drawing:
            self.end_point = event.pos()
            self.selection_rect = QRect(self.start_point, self.end_point).normalized()
            self.drawing = False
            self.update()
            self.mouse_release_completed.emit()

    def paintEvent(self, event):
        super().paintEvent(event)
        if self.selection_rect:
            painter = QPainter(self)
            pen = QPen(Qt.red, 2, Qt.SolidLine)
            painter.setPen(pen)
            painter.drawRect(self.selection_rect)

    def getSelectionRect(self):
        return self.selection_rect
