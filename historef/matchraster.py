import subprocess
import tempfile
from pathlib import Path

import numpy as np
import cv2
from rasterio.control import GroundControlPoint


def find_best_transform(A, B, transforms, blur=1):
    """
    Finds the translation vector that minimizes the difference between A and the translated B.

    Args:
        A: The first image array.
        B: The second image array.
        transform: A list of transform matrices

    Returns:
        The transform matrix that minimizes the difference, 
    """
    
    min_error = np.inf
    best_tf = None
    best_idx = None
    best_B = None
    errors = []

    cv2.imwrite("../sample/A_stretch_blur.png", A)

    for idx, tf in enumerate(transforms):
        if idx % 500 == 0: print(idx) 
            
        error = error_raster_tf(A, B, tf) 
        errors.append(error)
        
        if error < min_error:
            min_error = error
            best_tf = tf
            best_idx = idx
            best_B = None   ## remove
    
    print(f"Best Transform: {best_idx} ({min_error})")
    return best_tf, best_idx, best_B, errors


def error_raster_tf(A, B, tf):
    B_tf =  warp_affine(B, tf, A)
    error = np.sum(np.abs(A - B_tf)) / A.size
    return error


def warp_affine(input, tf, reference):
    input_tf = cv2.warpAffine(input, tf[:2, :], (reference.shape[1], reference.shape[0]))
    return input_tf


def error_raster(A, B):
    error = np.sum(np.abs(A - B)) / A.size
    return error


def preprocess_image(image, channel=None, xy_swap=False, blur=None, gamma=None):
        
    # Check if a specific channel is selected
    if channel is not None:
        if channel < image.shape[2]:
            gray_image = image[:, :, channel]
        else:
            raise ValueError("Invalid channel number")
    else:
        # Check if the image is grayscale already
        if len(image.shape) == 2:
            gray_image = image
        else:
            # Convert the image to grayscale using all channel information
            gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Check if blur is specified and apply blur
    if blur is not None:
        if blur > 0:
            gray_image = cv2.GaussianBlur(gray_image, (blur, blur), 0)
        else:
            raise ValueError("Blur value must be greater than 0")

    if gamma is not None:
        gray_image = wb_level(gray_image, gamma=gamma)
       
    # Check if xy_swap is True and transpose the image
    if xy_swap:
        gray_image = np.transpose(gray_image)

    return gray_image


def wb_level(image, in_black=0, in_white=255, gamma=1, out_black=0, out_white=255):
    image = np.clip( (image - in_black) / (in_white - in_black), 0, 255)
    image = ( image ** (1/gamma) ) *  (out_white - out_black) + out_black
    image = np.clip(image, 0, 255).astype(np.uint8)
    return image


def gcps_from_pairs(match_pairs, rasterio=False, xy_swap=False, y_flip=False):
    gcps = []
    for p in match_pairs:
        target_x = p[2][1] if xy_swap else p[2][0]
        target_y = p[2][0] if xy_swap else p[2][1]
        target_y = -target_y if y_flip else target_y
        gcp = (p[0][1], p[0][0], target_x, target_y)
        if rasterio:
            gcps.append(GroundControlPoint(*gcp))
        else:
            gcps.append(gcp)
    return gcps

    
def execute_gdal_translate(gcps, input_file, output_file):
    # Build the gdal_translate command with -gcp options
    cmd = ['gdal_translate']
    
    for gcp in gcps:
        cmd.extend(['-gcp', str(gcp[0]), str(gcp[1]), str(gcp[2]), str(gcp[3])])
    
    cmd.extend(['-a_srs', 'epsg:3857', input_file, output_file])
    
    try:
        # Execute the command
        subprocess.run(cmd, check=True)
        print("gdal_translate executed successfully.")
    except subprocess.CalledProcessError as e:
        print(f"Error executing gdal_translate: {e}")


def execute_gdalwarp(input_file, output_file):
    # Build the gdal_translate command with -gcp options
    cmd = ['gdalwarp']
    cmd.extend(['-order', '2', '-refine_gcps', '20', '20', input_file, output_file])
    
    try:
        # Execute the command
        print(" ".join(map(str,cmd)))
        subprocess.run(cmd, check=True)
        print("gdal_translate executed successfully.")
    except subprocess.CalledProcessError as e:
        print(f"Error executing gdal_translate: {e}")



def warp_from_gcps(matched_pairs, hnef, alignf):

    gcps = gcps_from_pairs(matched_pairs)
        
    temp_dir = tempfile.TemporaryDirectory()
    temp_dir_path = Path(temp_dir.name)
    translate_file = temp_dir_path / "histology_translated.tif"
    if translate_file.exists():
        translate_file.unlink()

    execute_gdal_translate(gcps, hnef, translate_file)
    execute_gdalwarp(translate_file, alignf)

