from PyQt5.QtWidgets import QWidget
from PyQt5.QtGui import QPainter, QImage, QPen
from PyQt5.QtCore import Qt, QPoint

class DrawingCanvas(QWidget):
    def __init__(self, parent=None):
        super(DrawingCanvas, self).__init__(parent)

        self.image = QImage(480, 480, QImage.Format_RGB32)
        self.image.fill(Qt.white)

        self.drawing = False
        self.brushSize = 5
        self.brushColor = Qt.black
        self.lastPoint = QPoint()

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.drawing = True
            self.lastPoint = event.pos()
        elif event.button() == Qt.RightButton:
            self.clearCanvas()

    def mouseMoveEvent(self, event):
        if (event.buttons() & Qt.LeftButton) & self.drawing:
            painter = QPainter(self.image)
            painter.setPen(QPen(self.brushColor, self.brushSize, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin))
            painter.drawLine(self.lastPoint, event.pos())
            self.lastPoint = event.pos()
            self.update()

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.drawing = False

    def paintEvent(self, event):
        canvasPainter = QPainter(self)
        canvasPainter.drawImage(self.rect(), self.image, self.image.rect())

    def clearCanvas(self):
        self.image.fill(Qt.white)
        self.update()

    def saveImage(self, fileName, fileFormat):
        self.image.save(fileName, fileFormat)

    def saveChangedImage(self, fileName, fileFormat, newWidth, newHeight):
        resizedImage = self.image.scaled(newWidth, newHeight, Qt.IgnoreAspectRatio, Qt.SmoothTransformation)
        resizedImage.save(fileName, fileFormat)