from PyQt5 import QtCore
from PyQt5.QtWidgets import QLabel, QLineEdit, QVBoxLayout, QHBoxLayout, QComboBox, QPushButton, QWidget, QFileDialog, \
    QMessageBox, QApplication
from PyQt5.QtGui import QPixmap, QFont
from PIL import Image
import numpy as np
from .DrawingApp import DrawingCanvas
from .ProcessingApp import Processor
from .Steklov import InverseIterationMethod

class MainApp(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        self.setGeometry(0, 0, 1000, 500)
        self.setWindowTitle("Ввод данных")

        hbox = QHBoxLayout(self)

        self.left_label = QLabel(self)
        self.left_label.setStyleSheet("background-color: white")
        self.left_label.setFixedSize(500, 500)

        left_vbox = QVBoxLayout(self.left_label)

        font = QFont()
        font.setPointSize(14)

        label1 = QLabel("Количество точек по оси X:")
        label1.setFont(font)
        left_vbox.addWidget(label1)
        self.points_x_line_edit = QLineEdit()
        self.points_x_line_edit.setFont(font)
        self.points_x_line_edit.setFixedHeight(40)
        left_vbox.addWidget(self.points_x_line_edit)

        label2 = QLabel("Количество точек по оси Y:")
        label2.setFont(font)
        left_vbox.addWidget(label2)
        self.points_y_line_edit = QLineEdit()
        self.points_y_line_edit.setFont(font)
        self.points_y_line_edit.setFixedHeight(40)
        left_vbox.addWidget(self.points_y_line_edit)

        label3 = QLabel("Шаг интегрирования:")
        label3.setFont(font)
        left_vbox.addWidget(label3)
        self.integration_step_line_edit = QLineEdit()
        self.integration_step_line_edit.setFont(font)
        self.integration_step_line_edit.setFixedHeight(40)
        left_vbox.addWidget(self.integration_step_line_edit)

        label5 = QLabel("Количество знаков после запятой:")
        label5.setFont(font)
        left_vbox.addWidget(label5)
        self.calc_acc = QLineEdit()
        self.calc_acc.setFont(font)
        self.calc_acc.setFixedHeight(40)
        left_vbox.addWidget(self.calc_acc)

        label4 = QLabel("Количество функций:")
        label4.setFont(font)
        left_vbox.addWidget(label4)
        self.functions_combo_box = QComboBox()
        self.functions_combo_box.setFont(font)
        self.functions_combo_box.setFixedHeight(40)

        for i in range(1, 17):
            self.functions_combo_box.addItem(str(i))
        left_vbox.addWidget(self.functions_combo_box)

        self.load_file_button = QPushButton("Загрузить файл")
        self.load_file_button.setFixedSize(480, 40)
        self.load_file_button.clicked.connect(self.loadFile)
        self.load_file_button.setStyleSheet("background-color: blue; color: white; font-size: 20px;")
        left_vbox.addWidget(self.load_file_button)

        delete_button = QPushButton("Удалить файл")
        delete_button.setFixedSize(480, 40)
        delete_button.clicked.connect(self.deleteFile)
        delete_button.setStyleSheet("background-color: blue; color: white; font-size: 20px;")
        left_vbox.addWidget(delete_button)

        self.finish_button = QPushButton("Обработать")
        self.finish_button.setFixedSize(480, 40)
        self.finish_button.clicked.connect(self.finish)
        self.finish_button.setStyleSheet("background-color: purple; color: white; font-size: 20px;")
        left_vbox.addWidget(self.finish_button)

        self.right_label = QLabel(self)
        pixmap = QPixmap('Pictures/Background/background.png')
        self.right_label.setPixmap(pixmap)
        self.right_label.setFixedSize(500, 500)


        right_vbox = QVBoxLayout(self.right_label)

        self.canvas = DrawingCanvas()
        right_vbox.addWidget(self.canvas)

        hbox.addWidget(self.left_label)
        hbox.addWidget(self.right_label)

        self.setLayout(hbox)

        self.file_path = None

    def loadFile(self):
        file_name, _ = QFileDialog.getOpenFileName(self, "Загрузить файл", "", "All Files (*.*)")
        if file_name:
            self.file_path = file_name
            self.load_file_button.setStyleSheet("background-color: green; color: white; font-size: 20px;")

    def deleteFile(self):
        self.file_path = None
        self.load_file_button.setStyleSheet("background-color: blue; color: white; font-size: 20px;")

    def finishDrawing(self, x, y, canv):
        canv.saveImage("Pictures/Area/Original_image.png", "png")

        canv.saveChangedImage("Pictures/Area/Appropriate_image.png", "png", x, y)

        self.saveTxt('Pictures/Area/Appropriate_image.png')

    def saveTxt(self, image_path):
        img = Image.open(image_path).convert('L')
        data = np.array(img)

        threshold = 254
        fn = lambda x: 0 if x > threshold else 1
        binary_data = np.array([[fn(pixel) for pixel in row] for row in data])

        with open('Data/Image_Matrix.txt', 'w') as text_file:
            for row in binary_data:
                line = ''.join(str(pixel) for pixel in row)
                text_file.write(line + '\n')


    def finish(self):
        self.finish_button.setStyleSheet("background-color: green; color: white; font-size: 20px;")
        self.finish_button.setText("Обработка данных...")
        QApplication.processEvents()
        try:
            h = float(self.integration_step_line_edit.text())
            n = int(self.functions_combo_box.currentText())
            calc_acc = int(self.calc_acc.text())
            if not self.file_path:
                x = int(self.points_x_line_edit.text())
                y = int(self.points_y_line_edit.text())
                self.finishDrawing(x, y, self.canvas)
                self.file_path = "Data/Image_Matrix.txt"

            Processor(self.file_path)
            self.deleteFile()

            self.Numerical_Method = InverseIterationMethod('Data/area.txt', h, n, calc_acc)
            self.Numerical_Method.Print()
            self.showPictureSelection(n)

        except ValueError as e:
            QMessageBox.warning(self, "Ошибка", "Были введены некорректные данные")

        self.finish_button.setStyleSheet("background-color: purple; color: white; font-size: 20px;")
        self.finish_button.setText("Обработать")

    def showPictureSelection(self, N):
        self.picture_window = QWidget()
        self.picture_window.setGeometry(100, 100, 800, 400)
        self.picture_window.setWindowTitle("Результаты")
        self.picture_window.setStyleSheet("background-color: white;")

        hbox = QHBoxLayout(self.picture_window)

        self.function_selection_panel = QWidget()
        self.function_selection_layout = QVBoxLayout(
            self.function_selection_panel)

        font = QFont()
        font.setPointSize(14)

        label = QLabel("Выберите номер функции:")
        label.setFont(font)

        self.function_selection_layout.addWidget(label)

        self.function_combo_box = QComboBox()
        self.function_combo_box.setFont(font)
        self.function_combo_box.setFixedHeight(40)
        self.function_combo_box.addItem("-")
        for i in range(1, N + 1):
            self.function_combo_box.addItem(str(i))
        self.function_selection_layout.addWidget(self.function_combo_box)

        label1 = QLabel("Выберите тип графика:")
        label1.setFont(font)

        self.function_selection_layout.addWidget(label1)

        self.function_graph = QComboBox()
        self.function_graph.setFont(font)
        self.function_graph.addItem("3D график функции")
        self.function_graph.addItem("Тепловая карта")
        self.function_graph.addItem("Контурная карта")
        self.function_graph.setFixedHeight(40)
        self.function_selection_layout.addWidget(self.function_graph)


        self.function_combo_box.currentIndexChanged.connect(self.updatePicture)
        self.function_graph.currentIndexChanged.connect(self.updatePicture)

        self.function_selection_panel.setLayout(self.function_selection_layout)
        self.function_selection_panel.setStyleSheet("background-color: white;")
        self.function_selection_layout.setAlignment(QtCore.Qt.AlignCenter)

        hbox.addWidget(self.function_selection_panel)

        self.picture_label = QLabel(self.picture_window)
        self.picture_label.setFixedSize(500, 500)
        hbox.addWidget(self.picture_label)

        self.picture_window.setLayout(hbox)
        self.picture_window.show()

    def updatePicture(self):
        if self.function_combo_box.currentText() != '-':
            index = int(self.function_combo_box.currentText())
            option = self.function_graph.currentText()
            if option == "3D график функции":
                self.Numerical_Method.printOneFunction(index - 1)
            elif option == "Тепловая карта":
                self.Numerical_Method.printHeatMap(index - 1)
            else:
                self.Numerical_Method.printCounterMap(index - 1)
            picture_path = f"Pictures/Results/Eigenfunction.png"
            pixmap = QPixmap(picture_path)
            self.picture_label.setPixmap(pixmap)