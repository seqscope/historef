import cv2
import numpy as np

def intensity_cut(image, threshold=0.9):
    
    # Convert the image to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray_array = np.array(gray_image)

    # Calculate the histogram of the grayscale image and plot
    histogram, bin_edges = np.histogram(gray_array, bins=256, range=(0, 255))
    
    # Find the intensity value where the cumulative count first exceeds or equals the 90% threshold
    cumulative_histogram = np.cumsum(histogram)
    total_pixels = cumulative_histogram[-1]
    top_10_percent_intensity = np.argmax(cumulative_histogram >= total_pixels * threshold )

    # Adjust Min-Max to highlight top 10% pixels and black out down 90% pixels
    contrast_adjusted_array = gray_array - top_10_percent_intensity
    contrast_adjusted_array[contrast_adjusted_array > 255 - top_10_percent_intensity] = 0
    contrast_adjusted_array = contrast_adjusted_array * (255 / np.max(contrast_adjusted_array))
    contrast_adjusted_array = np.clip(contrast_adjusted_array, 0, 255).astype(np.uint8)
    
    return contrast_adjusted_array
