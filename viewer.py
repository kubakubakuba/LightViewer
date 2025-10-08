import os
import numpy as np
from astropy.io import fits
from xisf import XISF
from PyQt5.QtWidgets import (QMainWindow, QLabel, QMessageBox, 
							 QWidget, QFileDialog, QAction, QMenu, QVBoxLayout)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPixmap, QImage, QKeySequence
import cv2

from image_processing import *

DEBUG = False

def dbgprint(t):
	if DEBUG:
		print(t)

class LightViewer(QMainWindow):
	def __init__(self):
		super().__init__()
		self.setWindowTitle("LightViewer - Drag & Drop Image")
		self.setMinimumSize(800, 600)
		
		# Store current image data
		self.current_file_path = None
		self.current_fits_data = None
		self.current_image_metadata = None
		self.is_color_image = False
		self.is_auto_stretched = False
		self.brightness_factor = 1.0
		self.original_pixmap = None
		
		# Central widget with dark theme
		central_widget = QWidget()
		central_widget.setStyleSheet("QWidget { background-color: #121212; }")
		self.setCentralWidget(central_widget)
		
		# Layout for central widget
		layout = QVBoxLayout(central_widget)
		layout.setContentsMargins(0, 0, 0, 0)
		
		# Image label
		self.image_label = QLabel()
		self.image_label.setAlignment(Qt.AlignCenter)
		self.image_label.setStyleSheet("QLabel { background-color: #121212; color: #cccccc; }")
		self.image_label.setText("Drag & drop an image or FITS/XISF file here\nOr use File → Open")
		layout.addWidget(self.image_label)
		
		# Enable drag and drop
		self.setAcceptDrops(True)
		
		# Create menu bar with dark theme
		self.create_menu()

	def create_menu(self):
		menubar = self.menuBar()
		menubar.setStyleSheet("""
			QMenuBar {
				background-color: #1e1e1e;
				color: #cccccc;
				border: none;
			}
			QMenuBar::item:selected {
				background-color: #2d2d2d;
			}
			QMenu {
				background-color: #1e1e1e;
				color: #cccccc;
				border: 1px solid #333;
			}
			QMenu::item:selected {
				background-color: #2d2d2d;
			}
		""")
		
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
		self.stretch_action.setChecked(False)
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

	def wheelEvent(self, event):
		"""Handle mouse wheel for brightness adjustment with Ctrl modifier"""
		if event.modifiers() == Qt.ControlModifier:
			delta = event.angleDelta().y()
			if delta > 0:
				self.brightness_factor += 0.05  # Increase brightness
			else:
				self.brightness_factor = max(0.0, self.brightness_factor - 0.05)  # Decrease brightness
			self.display_fits_data()
			event.accept()
		else:
			super().wheelEvent(event)

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
		self.is_auto_stretched = False
		self.brightness_factor = 1.0
		self.stretch_action.setEnabled(False)
		
		pixmap = QPixmap(file_path)
		if pixmap.isNull():
			raise ValueError("Unsupported image format")
		
		# Convert QPixmap to a NumPy array for consistent processing
		image = pixmap.toImage()
		
		# Determine if the image is grayscale or color
		if image.hasAlphaChannel() or (image.format() in [QImage.Format_RGB32, QImage.Format_ARGB32, QImage.Format_ARGB32_Premultiplied]):
			image = image.convertToFormat(QImage.Format_RGB888)
			self.is_color_image = True
		elif image.format() in [QImage.Format_Grayscale8, QImage.Format_Grayscale16]:
			self.is_color_image = False
		else:
			image = image.convertToFormat(QImage.Format_RGB888)
			self.is_color_image = True
		
		# Get the raw bytes from QImage
		image_bytes = image.constBits().asbytes()
		
		if self.is_color_image:
			# Reshape bytes to a NumPy array with 3 channels
			data = np.frombuffer(image_bytes, np.uint8).reshape(image.height(), image.width(), 3)
			# Convert to grayscale for preview consistency
			self.current_fits_data = cv2.cvtColor(data, cv2.COLOR_RGB2GRAY)
		else:
			# Reshape bytes to a NumPy array with 1 channel
			self.current_fits_data = np.frombuffer(image_bytes, np.uint8).reshape(image.height(), image.width())

		self.stretch_action.setEnabled(True)
		self.is_auto_stretched = True
		self.stretch_action.setChecked(True)
		
		self.display_fits_data()

	def resizeEvent(self, event):
		"""Handle window resize events"""
		super().resizeEvent(event)
		if self.original_pixmap is not None:
			self.resize_image()

	def resize_image(self):
		"""Resize the image to fit the current window size"""
		if self.original_pixmap is None:
			return
			
		# Scale to fit while maintaining aspect ratio
		scaled_pixmap = self.original_pixmap.scaled(
			self.image_label.width(),
			self.image_label.height(),
			Qt.KeepAspectRatio,
			Qt.SmoothTransformation
		)
		self.image_label.setPixmap(scaled_pixmap)
		self.image_label.setText("")

	def load_astronomy_file(self, file_path):
		file_ext = os.path.splitext(file_path)[1].lower()
		
		dbgprint(f"Loading astronomy file: {file_path}")
		dbgprint(f"File extension: {file_ext}")
		
		if file_ext == '.xisf':
			# Read XISF file
			xisf = XISF(file_path)
			data = xisf.read_image(0)
			self.current_image_metadata = xisf.get_images_metadata()[0] if xisf.get_images_metadata() else None
			dbgprint(f"XISF metadata: {self.current_image_metadata}")
		else:
			# Read FITS file using astropy
			with fits.open(file_path) as hdul:
				data = hdul[0].data
				if data is None:
					raise ValueError("No image data found in file")
				self.current_image_metadata = {
					'FITSKeywords': dict(hdul[0].header)
				}
			dbgprint(f"FITS data shape: {data.shape}, dtype: {data.dtype}")
		
		# Handle different data types and dimensions
		self.is_color_image = False
		
		dbgprint(f"Raw data shape: {data.shape}, dtype: {data.dtype}")
		
		# Check if this is a Bayer pattern image that needs debayering
		needs_debayering = self.needs_debayering()
		dbgprint(f"Needs debayering: {needs_debayering}")
		
		if needs_debayering:
			dbgprint("Attempting to debayer image...")
			# Debayer the image
			data = self.debayer_image(data)
			self.is_color_image = True
			dbgprint(f"Debayered data shape: {data.shape}, dtype: {data.dtype}")
		elif data.ndim == 3:
			# For color images, check if it's RGB
			if data.shape[2] == 3:
				self.is_color_image = True
				dbgprint("RGB color image detected")
				# Keep as color image
			else:
				data = data[0]
				dbgprint(f"3D non-RGB image, taking first channel: {data.shape}")
		elif data.ndim != 2:
			raise ValueError("Only 2D images or 3D RGB images are supported")
		else:
			dbgprint("2D grayscale image detected")
		
		# Store the original, full-resolution data
		self.current_fits_data = data
		self.stretch_action.setEnabled(True)
		self.is_auto_stretched = True
		self.stretch_action.setChecked(True)
		self.brightness_factor = 1.0 # Reset brightness
		
		self.display_fits_data()

	def needs_debayering(self):
		"""Check if the image needs debayering based on metadata"""
		if not self.current_image_metadata or 'FITSKeywords' not in self.current_image_metadata:
			dbgprint("No metadata or FITSKeywords found")
			return False
			
		fits_keywords = self.current_image_metadata['FITSKeywords']
		dbgprint(f"Available FITS keywords: {list(fits_keywords.keys())}")
		
		# Check if this is a Bayer pattern image
		has_bayer_pattern = any(key in fits_keywords for key in ['BAYERPAT', 'BAYER'])
		is_color_space_gray = self.current_image_metadata.get('colorSpace', '').lower() == 'gray'
		
		dbgprint(f"Has Bayer pattern: {has_bayer_pattern}")
		dbgprint(f"Is color space gray: {is_color_space_gray}")
		
		return has_bayer_pattern and is_color_space_gray

	def debayer_image(self, data):
		"""Debayer a Bayer pattern image using OpenCV"""
		dbgprint(f"Debayering image with shape: {data.shape}, dtype: {data.dtype}")
		
		# Get Bayer pattern from metadata
		fits_keywords = self.current_image_metadata['FITSKeywords']
		bayer_pattern = None
		
		# Try to get Bayer pattern from various possible keyword names
		for key in ['BAYERPAT', 'BAYER']:
			if key in fits_keywords:
				bayer_pattern = fits_keywords[key][0]['value'] if isinstance(fits_keywords[key], list) else fits_keywords[key]
				dbgprint(f"Found Bayer pattern: {bayer_pattern} from key: {key}")
				break
		
		if not bayer_pattern:
			dbgprint("No Bayer pattern found in metadata")
			return data
			
		# Map Bayer pattern to OpenCV constant
		pattern_map = {
			'RGGB': cv2.COLOR_BayerRG2BGR,
			'BGGR': cv2.COLOR_BayerBG2BGR,
			'GRBG': cv2.COLOR_BayerGR2BGR,
			'GBRG': cv2.COLOR_BayerGB2BGR
		}
		
		if bayer_pattern not in pattern_map:
			dbgprint(f"Unsupported Bayer pattern: {bayer_pattern}")
			return data
			
		dbgprint(f"Using OpenCV pattern: {pattern_map[bayer_pattern]} for {bayer_pattern}")
		
		# Convert to 8-bit for debayering
		if data.dtype != np.uint8:
			dbgprint(f"Converting from {data.dtype} to uint8 for debayering")
			# Scale to 8-bit
			min_val = np.min(data)
			max_val = np.max(data)
			dbgprint(f"Min: {min_val}, Max: {max_val}")
			if min_val == max_val:
				data_8bit = np.zeros_like(data, dtype=np.uint8)
			else:
				data_8bit = ((data.astype(np.float32) - min_val) / (max_val - min_val) * 255).astype(np.uint8)
		else:
			data_8bit = data
			dbgprint("Data already uint8, no conversion needed")
		
		# Debayer the image
		dbgprint("Performing debayering...")
		debayered = cv2.cvtColor(data_8bit, pattern_map[bayer_pattern])
		dbgprint(f"Debayered shape: {debayered.shape}, dtype: {debayered.dtype}")
		
		return debayered

	def display_fits_data(self):
		if self.current_fits_data is None:
			dbgprint("No data to display")
			return
		
		processed_array = None
		
		# Choose the appropriate conversion function based on image type and stretch mode
		if self.is_color_image:
			if self.is_auto_stretched:
				# Use auto-stretch to get a good grayscale preview
				processed_array = color_array_to_qimage_grayscale(self.current_fits_data, self.brightness_factor)
				dbgprint("Using auto-stretch for grayscale preview")
			else:
				# Use a linear stretch for a basic grayscale preview
				processed_array = color_array_to_qimage_linear(self.current_fits_data, self.brightness_factor)
				# The linear stretch on a color image is likely to be all black, so we need a proper linear grayscale conversion
				# Let's use the same color_array_to_qimage_grayscale with a linear scale instead of percentile
				
				# The simplest way to show a linear representation is to convert to grayscale first
				grayscale_array = 0.2989 * self.current_fits_data[:, :, 0] + 0.5870 * self.current_fits_data[:, :, 1] + 0.1140 * self.current_fits_data[:, :, 2]
				processed_array = array_to_qimage_linear(grayscale_array, self.brightness_factor)
				dbgprint("Using linear stretch for grayscale preview")
		else:
			if self.is_auto_stretched:
				# Use auto-stretch
				processed_array = array_to_qimage_auto_stretch(self.current_fits_data, self.brightness_factor)
			else:
				# Use linear stretch
				processed_array = array_to_qimage_linear(self.current_fits_data, self.brightness_factor)
		
		# Downsize the stretched image for faster previewing
		max_size = 1024
		height, width = processed_array.shape[0], processed_array.shape[1]
		
		if height > max_size or width > max_size:
			scale_factor = max_size / max(height, width)
			new_width = int(width * scale_factor)
			new_height = int(height * scale_factor)
			downsized_array = cv2.resize(processed_array, (new_width, new_height), interpolation=cv2.INTER_AREA)
			dbgprint(f"Resizing stretched image to {new_width}x{new_height} for display")
		else:
			downsized_array = processed_array
		
		# Create QImage from the downsized array
		height, width = downsized_array.shape[0], downsized_array.shape[1]
		bytes_per_line = downsized_array.shape[1]
		
		if downsized_array.ndim == 3:
			image = QImage(downsized_array.tobytes(), width, height, downsized_array.shape[2] * width, QImage.Format_RGB888)
		else:
			image = QImage(downsized_array.tobytes(), width, height, bytes_per_line, QImage.Format_Grayscale8)

		dbgprint(f"QImage created - size: {image.width()}x{image.height()}, format: {image.format()}")
		
		# Store the original pixmap for resizing
		self.original_pixmap = QPixmap.fromImage(image)
		
		# Scale to fit
		self.resize_image()

	def keyPressEvent(self, event):
		"""Handle keyboard shortcuts"""
		if event.key() == Qt.Key_A and event.modifiers() == Qt.ControlModifier:
			if self.current_fits_data is not None:
				# Toggle auto stretch
				self.is_auto_stretched = not self.is_auto_stretched
				self.stretch_action.setChecked(self.is_auto_stretched)
				self.display_fits_data()
		else:
			super().keyPressEvent(event)