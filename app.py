# -*- coding: utf-8 -*-
"""
Created on Fri May 17 17:17:04 2024

"""

import sys
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QLineEdit, QLabel, QPushButton, QHBoxLayout, QSpacerItem, QSizePolicy
from PyQt5.QtGui import QPixmap, QPalette, QBrush, QImage
from PyQt5.QtCore import Qt
from textprocessing import clean_text
from transformers import BertTokenizer, BertForSequenceClassification
from bertTraining import recommend_car


class SimpleApp(QWidget):
    def __init__(self):
        super().__init__()
        self.model_dir = "saved_model"
        self.initUI()
        
        
    def load_model_and_tokenizer(self, model_dir):
        model = BertForSequenceClassification.from_pretrained(model_dir)
        tokenizer = BertTokenizer.from_pretrained(model_dir)
        return model, tokenizer
    
    
    def recommend(self, clean_text):
        model, tokenizer = self.load_model_and_tokenizer(self.model_dir)
        return recommend_car(clean_text, model, tokenizer)
        

    def initUI(self):
        self.setWindowTitle('Recommendation App')
        self.setGeometry(100, 100, 400, 300)  # Set window size
        
        # Set background image
        self.set_background_image('aiStockPhoto.png')
        
        # Create layout
        main_layout = QVBoxLayout()
        input_layout = QHBoxLayout()
        output_layout = QHBoxLayout()
        # Create widgets
        self.input_field = QLineEdit(self)
        self.output_label = QLabel('', self)
        self.process_button = QPushButton('Recommend', self)
        # Set styles for input field and output label to appear as a box
        # Set fixed sizes
        self.input_field.setFixedSize(300, 40)
        self.output_label.setFixedSize(300, 40)
        
        # Set styles for input field and output label to appear as a box
        self.input_field.setStyleSheet("""
            QLineEdit {
                border: 2px solid black;
                padding: 10px;
                background-color: white;
            }
        """)
        self.output_label.setStyleSheet("""
            QLabel {
                border: 2px solid black;
                padding: 10px;
                background-color: white;
            }
        """)
        
        # Add widgets to horizontal layouts
        input_layout.addItem(QSpacerItem(0, 0, QSizePolicy.Expanding, QSizePolicy.Minimum))
        input_layout.addWidget(self.input_field)
        input_layout.addItem(QSpacerItem(0, 0, QSizePolicy.Expanding, QSizePolicy.Minimum))
        
        output_layout.addItem(QSpacerItem(0, 0, QSizePolicy.Expanding, QSizePolicy.Minimum))
        output_layout.addWidget(self.output_label)
        output_layout.addItem(QSpacerItem(0, 0, QSizePolicy.Expanding, QSizePolicy.Minimum))
        
        # Add horizontal layouts and button to main layout
        main_layout.addItem(QSpacerItem(0, 0, QSizePolicy.Minimum, QSizePolicy.Expanding))
        main_layout.addLayout(input_layout)
        main_layout.addWidget(self.process_button, alignment=Qt.AlignCenter)
        main_layout.addLayout(output_layout)
        main_layout.addItem(QSpacerItem(0, 0, QSizePolicy.Minimum, QSizePolicy.Expanding))
        
        # Set the layout for the main window
        self.setLayout(main_layout)
        
        # Connect the button click event to the handler
        self.process_button.clicked.connect(self.process_input)
        
    def set_background_image(self, image_path):
        oImage = QImage(image_path)
        sImage = oImage.scaled(self.size(), Qt.KeepAspectRatioByExpanding, Qt.SmoothTransformation)
        palette = QPalette()
        palette.setBrush(QPalette.Window, QBrush(sImage))
        self.setPalette(palette)
        
    def resizeEvent(self, event):
        self.set_background_image('aiStockPhoto.png')
        super().resizeEvent(event)
        
    def process_input(self):
        input_text = self.input_field.text()
        output_text = self.my_processing_function(input_text)
        self.output_label.setText(output_text)
        
    def my_processing_function(self, text):
        text = clean_text(text)
        text = self.recommend(text)
        text = text.replace("_", " ")
        return f"Recommended car: {text}"

if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = SimpleApp()
    ex.show()
    sys.exit(app.exec_())