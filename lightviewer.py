import sys
import os
import numpy as np
from astropy.io import fits
from xisf import XISF
from PyQt5.QtWidgets import (QApplication, QMainWindow, QLabel, QMessageBox, 
                             QWidget, QFileDialog, QAction, QMenu)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPixmap, QImage, QKeySequence

import cv2

class LightViewer(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("LightViewer - Drag & Drop Image")
        self.setFixedSize(800, 600)
        
        # Store current image data
        self.current_file_path = None
        self.current_fits_data = None
        self.current_image_metadata = None
        self.is_color_image = False
        self.is_auto_stretched = False  # Default to linear
        
        # Central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # Image label
        self.image_label = QLabel(central_widget)
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setStyleSheet("QLabel { background-color: white; }")
        self.image_label.setText("Drag & drop an image or FITS/XISF file here\nOr use File → Open")
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
            "Open Image File",
            "",
            "Astronomy Images (*.fits *.fit *.fts *.xisf);;"
            "Image Files (*.png *.jpg *.jpeg *.bmp *.tiff *.tif);;"
            "All Files (*)"
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
            if file_ext in ['.fits', '.fit', '.fts', '.xisf']:
                self.load_astronomy_file(file_path)
            else:
                self.load_regular_image(file_path)
                
            self.setWindowTitle(f"LightViewer - {os.path.basename(file_path)}")
            
        except Exception as e:
            QMessageBox.warning(self, "Error", f"Failed to load file: {str(e)}")

    def load_regular_image(self, file_path):
        # Clear FITS data when loading regular image
        self.current_fits_data = None
        self.current_image_metadata = None
        self.is_color_image = False
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

    def load_astronomy_file(self, file_path):
        file_ext = os.path.splitext(file_path)[1].lower()
        
        print(f"Loading astronomy file: {file_path}")
        print(f"File extension: {file_ext}")
        
        if file_ext == '.xisf':
            # Read XISF file
            xisf = XISF(file_path)
            data = xisf.read_image(0)  # Read first image
            self.current_image_metadata = xisf.get_images_metadata()[0] if xisf.get_images_metadata() else None
            print(f"XISF metadata: {self.current_image_metadata}")
        else:
            # Read FITS file using astropy
            with fits.open(file_path) as hdul:
                data = hdul[0].data
                if data is None:
                    raise ValueError("No image data found in file")
                self.current_image_metadata = {
                    'FITSKeywords': dict(hdul[0].header)
                }
            print(f"FITS data shape: {data.shape}, dtype: {data.dtype}")
        
        # Handle different data types and dimensions
        self.is_color_image = False
        
        print(f"Raw data shape: {data.shape}, dtype: {data.dtype}")
        
        # Check if this is a Bayer pattern image that needs debayering
        needs_debayering = self.needs_debayering()
        print(f"Needs debayering: {needs_debayering}")
        
        if needs_debayering:
            print("Attempting to debayer image...")
            # Debayer the image
            data = self.debayer_image(data)
            self.is_color_image = True
            print(f"Debayered data shape: {data.shape}, dtype: {data.dtype}")
        elif data.ndim == 3:
            # For color images, check if it's RGB
            if data.shape[2] == 3:  # RGB
                self.is_color_image = True
                print("RGB color image detected")
                # Keep as color image
            else:
                data = data[0]  # Take first channel for non-RGB 3D images
                print(f"3D non-RGB image, taking first channel: {data.shape}")
        elif data.ndim != 2:
            raise ValueError("Only 2D images or 3D RGB images are supported")
        else:
            print("2D grayscale image detected")
        
        # Store the original data with its native type
        self.current_fits_data = data
        self.stretch_action.setEnabled(True)
        
        # Display the data with linear scaling (default)
        self.is_auto_stretched = False
        self.stretch_action.setChecked(False)
        self.display_fits_data()

    def needs_debayering(self):
        """Check if the image needs debayering based on metadata"""
        if not self.current_image_metadata or 'FITSKeywords' not in self.current_image_metadata:
            print("No metadata or FITSKeywords found")
            return False
            
        fits_keywords = self.current_image_metadata['FITSKeywords']
        print(f"Available FITS keywords: {list(fits_keywords.keys())}")
        
        # Check if this is a Bayer pattern image
        has_bayer_pattern = any(key in fits_keywords for key in ['BAYERPAT', 'BAYER'])
        is_color_space_gray = self.current_image_metadata.get('colorSpace', '').lower() == 'gray'
        
        print(f"Has Bayer pattern: {has_bayer_pattern}")
        print(f"Is color space gray: {is_color_space_gray}")
        
        return has_bayer_pattern and is_color_space_gray

    def debayer_image(self, data):
        """Debayer a Bayer pattern image using OpenCV"""
        print(f"Debayering image with shape: {data.shape}, dtype: {data.dtype}")
        
        # Get Bayer pattern from metadata
        fits_keywords = self.current_image_metadata['FITSKeywords']
        bayer_pattern = None
        
        # Try to get Bayer pattern from various possible keyword names
        for key in ['BAYERPAT', 'BAYER']:
            if key in fits_keywords:
                bayer_pattern = fits_keywords[key][0]['value'] if isinstance(fits_keywords[key], list) else fits_keywords[key]
                print(f"Found Bayer pattern: {bayer_pattern} from key: {key}")
                break
        
        if not bayer_pattern:
            print("No Bayer pattern found in metadata")
            return data
            
        # Map Bayer pattern to OpenCV constant
        pattern_map = {
            'RGGB': cv2.COLOR_BayerRG2BGR,
            'BGGR': cv2.COLOR_BayerBG2BGR,
            'GRBG': cv2.COLOR_BayerGR2BGR,
            'GBRG': cv2.COLOR_BayerGB2BGR
        }
        
        if bayer_pattern not in pattern_map:
            print(f"Unsupported Bayer pattern: {bayer_pattern}")
            return data
            
        print(f"Using OpenCV pattern: {pattern_map[bayer_pattern]} for {bayer_pattern}")
        
        # Convert to 8-bit for debayering
        if data.dtype != np.uint8:
            print(f"Converting from {data.dtype} to uint8 for debayering")
            # Scale to 8-bit
            min_val = np.min(data)
            max_val = np.max(data)
            print(f"Min: {min_val}, Max: {max_val}")
            if min_val == max_val:
                data_8bit = np.zeros_like(data, dtype=np.uint8)
            else:
                data_8bit = ((data.astype(np.float32) - min_val) / (max_val - min_val) * 255).astype(np.uint8)
        else:
            data_8bit = data
            print("Data already uint8, no conversion needed")
        
        # Debayer the image
        print("Performing debayering...")
        debayered = cv2.cvtColor(data_8bit, pattern_map[bayer_pattern])
        print(f"Debayered shape: {debayered.shape}, dtype: {debayered.dtype}")
        
        return debayered

    def display_fits_data(self):
        if self.current_fits_data is None:
            print("No data to display")
            return
            
        print(f"Displaying data - shape: {self.current_fits_data.shape}, dtype: {self.current_fits_data.dtype}, is_color: {self.is_color_image}")
        
        # Convert to QImage
        if self.is_color_image:
            print("Converting color image")
            if self.is_auto_stretched:
                image = self.color_array_to_qimage_auto_stretch(self.current_fits_data)
            else:
                image = self.color_array_to_qimage_linear(self.current_fits_data)
        else:
            print("Converting grayscale image")
            if self.is_auto_stretched:
                image = self.array_to_qimage_auto_stretch(self.current_fits_data)
            else:
                image = self.array_to_qimage_linear(self.current_fits_data)
            
        print(f"QImage created - size: {image.width()}x{image.height()}, format: {image.format()}")
        
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
        print("Display completed")

    def array_to_qimage_auto_stretch(self, array):
        """Apply auto-stretch (percentile-based contrast enhancement) for grayscale"""
        # Convert to float32 for processing, preserving the original data
        array_float = array.astype(np.float32)
        
        # Use percentiles for better contrast (avoid outliers)
        p_low, p_high = np.percentile(array_float, [2, 98])
        
        if p_low == p_high:
            p_high = p_low + 1.0
            
        # Clip and normalize
        clipped = np.clip(array_float, p_low, p_high)
        normalized = (clipped - p_low) / (p_high - p_low) * 255
        normalized = np.clip(normalized, 0, 255).astype(np.uint8)
        
        # Create QImage - convert to bytes first
        height, width = array.shape
        bytes_per_line = width
        # Convert to contiguous array and then to bytes
        normalized_contiguous = np.ascontiguousarray(normalized)
        image_data = normalized_contiguous.tobytes()
        image = QImage(image_data, width, height, bytes_per_line, QImage.Format_Grayscale8)
        
        return image.copy()

    def array_to_qimage_linear(self, array):
        """Linear scaling without auto-stretch - handle different data types properly for grayscale"""
        # Handle different data types
        if array.dtype == np.uint16:
            # For 16-bit data, scale to 8-bit
            normalized = (array.astype(np.float32) / 65535.0 * 255).astype(np.uint8)
        elif array.dtype == np.uint32:
            # For 32-bit data, scale to 8-bit
            normalized = (array.astype(np.float32) / 4294967295.0 * 255).astype(np.uint8)
        elif array.dtype == np.float32 or array.dtype == np.float64:
            # For float data, handle both positive and negative values
            min_val = np.min(array)
            max_val = np.max(array)
            
            if min_val == max_val:
                max_val = min_val + 1.0
                
            # Scale to 0-1 range first, then to 0-255
            if min_val < 0:
                # Handle negative values by shifting to positive
                array_shifted = array - min_val
                normalized = (array_shifted / (max_val - min_val) * 255).astype(np.uint8)
            else:
                normalized = ((array - min_val) / (max_val - min_val) * 255).astype(np.uint8)
        else:
            # For other types (uint8, int16, etc.), use min-max scaling
            min_val = np.min(array)
            max_val = np.max(array)
            
            if min_val == max_val:
                max_val = min_val + 1.0
                
            normalized = ((array.astype(np.float32) - min_val) / (max_val - min_val) * 255).astype(np.uint8)
        
        # Create QImage - convert to bytes first
        height, width = array.shape
        bytes_per_line = width
        # Convert to contiguous array and then to bytes
        normalized_contiguous = np.ascontiguousarray(normalized)
        image_data = normalized_contiguous.tobytes()
        image = QImage(image_data, width, height, bytes_per_line, QImage.Format_Grayscale8)
        
        return image.copy()

    def color_array_to_qimage_auto_stretch(self, array):
        """Apply auto-stretch (percentile-based contrast enhancement) for color images"""
        # Convert to float32 for processing
        array_float = array.astype(np.float32)
        
        # Use percentiles for better contrast (avoid outliers)
        p_low = np.percentile(array_float, 2, axis=(0, 1))
        p_high = np.percentile(array_float, 98, axis=(0, 1))
        
        # Avoid division by zero
        for i in range(3):
            if p_low[i] == p_high[i]:
                p_high[i] = p_low[i] + 1.0
        
        # Clip and normalize each channel
        clipped = np.clip(array_float, p_low, p_high)
        normalized = (clipped - p_low) / (p_high - p_low) * 255
        normalized = np.clip(normalized, 0, 255).astype(np.uint8)
        
        # Create QImage - convert to bytes first
        height, width, _ = array.shape
        bytes_per_line = 3 * width
        # Convert to contiguous array and then to bytes
        normalized_contiguous = np.ascontiguousarray(normalized)
        image_data = normalized_contiguous.tobytes()
        image = QImage(image_data, width, height, bytes_per_line, QImage.Format_RGB888)
        
        return image.copy()

    def color_array_to_qimage_linear(self, array):
        """Linear scaling without auto-stretch for color images"""
        # Handle different data types
        if array.dtype == np.uint16:
            # For 16-bit data, scale to 8-bit
            normalized = (array.astype(np.float32) / 65535.0 * 255).astype(np.uint8)
        elif array.dtype == np.uint32:
            # For 32-bit data, scale to 8-bit
            normalized = (array.astype(np.float32) / 4294967295.0 * 255).astype(np.uint8)
        elif array.dtype == np.float32 or array.dtype == np.float64:
            # For float data, handle both positive and negative values
            min_val = np.min(array, axis=(0, 1))
            max_val = np.max(array, axis=(0, 1))
            
            # Avoid division by zero
            for i in range(3):
                if min_val[i] == max_val[i]:
                    max_val[i] = min_val[i] + 1.0
                    
            # Scale to 0-1 range first, then to 0-255
            if np.any(min_val < 0):
                # Handle negative values by shifting to positive
                array_shifted = array - min_val
                normalized = (array_shifted / (max_val - min_val) * 255).astype(np.uint8)
            else:
                normalized = ((array - min_val) / (max_val - min_val) * 255).astype(np.uint8)
        else:
            # For other types (uint8, int16, etc.), use min-max scaling
            min_val = np.min(array, axis=(0, 1))
            max_val = np.max(array, axis=(0, 1))
            
            # Avoid division by zero
            for i in range(3):
                if min_val[i] == max_val[i]:
                    max_val[i] = min_val[i] + 1.0
                    
            normalized = ((array.astype(np.float32) - min_val) / (max_val - min_val) * 255).astype(np.uint8)
        
        # Create QImage - convert to bytes first
        height, width, _ = array.shape
        bytes_per_line = 3 * width
        # Convert to contiguous array and then to bytes
        normalized_contiguous = np.ascontiguousarray(normalized)
        image_data = normalized_contiguous.tobytes()
        image = QImage(image_data, width, height, bytes_per_line, QImage.Format_RGB888)
        
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