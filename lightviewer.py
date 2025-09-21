import sys
import os
import numpy as np
from astropy.io import fits
from PyQt5.QtWidgets import (QApplication, QMainWindow, QLabel, QMessageBox, 
                             QWidget, QFileDialog, QAction, QMenu)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPixmap, QImage, QKeySequence

class LightViewer(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("LightViewer - Drag & Drop Image")
        self.setFixedSize(800, 600)
        
        # Store current image data
        self.current_file_path = None
        self.current_fits_data = None
        self.is_auto_stretched = False  # Default to linear
        
        # Central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # Image label
        self.image_label = QLabel(central_widget)
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setStyleSheet("QLabel { background-color: white; }")
        self.image_label.setText("Drag & drop an image or FITS file here\nOr use File → Open")
        self.image_label.setGeometry(0, 0, 800, 600)
        
        # Enable drag and drop
        self.setAcceptDrops(True)
        
        # Create menu bar
        self.create_menu()

    def create_menu(self):
        menubar = self.menuBar()
        
        # File menu
        file_menu = menubar.addMenu('File')
        
        # Open action
        open_action = QAction('Open', self)
        open_action.setShortcut('Ctrl+O')
        open_action.triggered.connect(self.open_file)
        file_menu.addAction(open_action)
        
        # Auto-stretch toggle action
        self.stretch_action = QAction('Auto Stretch (Ctrl+A)', self)
        self.stretch_action.setShortcut('Ctrl+A')
        self.stretch_action.setCheckable(True)
        self.stretch_action.setChecked(False)  # Default to unchecked (linear)
        self.stretch_action.triggered.connect(self.toggle_auto_stretch)
        file_menu.addAction(self.stretch_action)
        
        # Exit action
        exit_action = QAction('Exit', self)
        exit_action.setShortcut('Ctrl+Q')
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)

    def toggle_auto_stretch(self):
        self.is_auto_stretched = self.stretch_action.isChecked()
        if self.current_fits_data is not None:
            self.display_fits_data()

    def open_file(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Open Image or FITS File",
            "",
            "Image Files (*.png *.jpg *.jpeg *.bmp *.tiff *.tif);;FITS Files (*.fits *.fit *.fts);;All Files (*)"
        )
        
        if file_path:
            self.load_image(file_path)

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dropEvent(self, event):
        urls = event.mimeData().urls()
        if urls:
            file_path = urls[0].toLocalFile()
            self.load_image(file_path)

    def load_image(self, file_path):
        if not os.path.exists(file_path):
            QMessageBox.warning(self, "Error", "File not found!")
            return
            
        self.current_file_path = file_path
        file_ext = os.path.splitext(file_path)[1].lower()
        
        try:
            if file_ext in ['.fits', '.fit', '.fts']:
                self.load_fits_file(file_path)
            else:
                self.load_regular_image(file_path)
                
            self.setWindowTitle(f"LightViewer - {os.path.basename(file_path)}")
            
        except Exception as e:
            QMessageBox.warning(self, "Error", f"Failed to load file: {str(e)}")

    def load_regular_image(self, file_path):
        # Clear FITS data when loading regular image
        self.current_fits_data = None
        self.stretch_action.setEnabled(False)
        
        pixmap = QPixmap(file_path)
        if pixmap.isNull():
            raise ValueError("Unsupported image format")
            
        # Scale to fit while maintaining aspect ratio
        scaled_pixmap = pixmap.scaled(
            self.image_label.width(),
            self.image_label.height(),
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation
        )
        self.image_label.setPixmap(scaled_pixmap)
        self.image_label.setText("")

    def load_fits_file(self, file_path):
        # Read FITS file using astropy
        with fits.open(file_path) as hdul:
            data = hdul[0].data
            if data is None:
                raise ValueError("No image data found in FITS file")
                
            # Handle 2D and 3D data (take first channel if 3D)
            if data.ndim == 3:
                data = data[0]  # Take first channel
            elif data.ndim != 2:
                raise ValueError("Only 2D images are supported")
            
            # Store the FITS data for later processing
            self.current_fits_data = data
            self.stretch_action.setEnabled(True)
            
            # Display the data with linear scaling (default)
            self.is_auto_stretched = False
            self.stretch_action.setChecked(False)
            self.display_fits_data()

    def display_fits_data(self):
        if self.current_fits_data is None:
            return
            
        # Convert to QImage
        if self.is_auto_stretched:
            image = self.array_to_qimage_auto_stretch(self.current_fits_data)
        else:
            image = self.array_to_qimage_linear(self.current_fits_data)
            
        pixmap = QPixmap.fromImage(image)
        
        # Scale to fit
        scaled_pixmap = pixmap.scaled(
            self.image_label.width(),
            self.image_label.height(),
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation
        )
        self.image_label.setPixmap(scaled_pixmap)
        self.image_label.setText("")  # No text overlay

    def array_to_qimage_auto_stretch(self, array):
        """Apply auto-stretch (percentile-based contrast enhancement)"""
        array = array.astype(np.float32)
        
        # Use percentiles for better contrast (avoid outliers)
        p_low, p_high = np.percentile(array, [2, 98])
        
        if p_low == p_high:
            p_high = p_low + 1.0
            
        # Clip and normalize
        clipped = np.clip(array, p_low, p_high)
        normalized = (clipped - p_low) / (p_high - p_low) * 255
        normalized = np.clip(normalized, 0, 255).astype(np.uint8)
        
        # Create QImage
        height, width = array.shape
        bytes_per_line = width
        image = QImage(normalized.data, width, height, bytes_per_line, QImage.Format_Grayscale8)
        
        return image.copy()

    def array_to_qimage_linear(self, array):
        """Linear scaling without auto-stretch"""
        array = array.astype(np.float32)
        min_val, max_val = np.min(array), np.max(array)
        
        if min_val == max_val:
            max_val = min_val + 1.0
            
        normalized = (array - min_val) / (max_val - min_val) * 255
        normalized = np.clip(normalized, 0, 255).astype(np.uint8)
        
        # Create QImage
        height, width = array.shape
        bytes_per_line = width
        image = QImage(normalized.data, width, height, bytes_per_line, QImage.Format_Grayscale8)
        
        return image.copy()

    def keyPressEvent(self, event):
        """Handle keyboard shortcuts"""
        if event.key() == Qt.Key_A and event.modifiers() == Qt.ControlModifier:
            if self.current_fits_data is not None:
                self.is_auto_stretched = not self.is_auto_stretched
                self.stretch_action.setChecked(self.is_auto_stretched)
                self.display_fits_data()
        else:
            super().keyPressEvent(event)

def main():
    app = QApplication(sys.argv)
    viewer = LightViewer()
    viewer.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()