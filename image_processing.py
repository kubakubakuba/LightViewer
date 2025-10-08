import numpy as np
from PyQt5.QtGui import QImage

def array_to_qimage_auto_stretch(array, brightness_factor=1.0):
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
	normalized = np.clip(normalized * brightness_factor, 0, 255).astype(np.uint8)
	
	return normalized

def array_to_qimage_linear(array, brightness_factor=1.0):
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
	
	normalized = np.clip(normalized.astype(np.float32) * brightness_factor, 0, 255).astype(np.uint8)
	
	return normalized

def color_array_to_qimage_auto_stretch(array, brightness_factor=1.0):
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
	normalized = np.clip(normalized * brightness_factor, 0, 255).astype(np.uint8)
	
	return normalized

def color_array_to_qimage_linear(array, brightness_factor=1.0):
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
	
	normalized = np.clip(normalized.astype(np.float32) * brightness_factor, 0, 255).astype(np.uint8)
	
	return normalized

def color_array_to_qimage_grayscale(array, brightness_factor=1.0):
	"""Convert a color array to a grayscale QImage with auto-stretch"""
	# Convert to float32 for processing
	array_float = array.astype(np.float32)
	
	# Convert to grayscale using a standard luminance formula
	grayscale_array = 0.2989 * array_float[:, :, 0] + 0.5870 * array_float[:, :, 1] + 0.1140 * array_float[:, :, 2]
	
	# Apply auto-stretch using percentiles
	p_low, p_high = np.percentile(grayscale_array, [2, 98])
	
	if p_low == p_high:
		p_high = p_low + 1.0
		
	# Clip and normalize
	clipped = np.clip(grayscale_array, p_low, p_high)
	normalized = (clipped - p_low) / (p_high - p_low) * 255
	normalized = np.clip(normalized * brightness_factor, 0, 255).astype(np.uint8)
	
	return normalized