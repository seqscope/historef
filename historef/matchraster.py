import subprocess
import random
import tempfile
from pathlib import Path
import os

import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler

import numpy as np
import cv2



def find_best_transform_eff(
    A, B, transforms, output_dir, 
    k=5, eps=100, error_type='sad', force_cluster_id=-1):
    """
    Efficiently finds the translation vector that minimizes the difference between A and the translated B.

    Args:
        A: The first image array.
        B: The second image array.
        transform: A list of transform matrices
        k: number of sample per cluster
        eps: DBSCAN eps param

    Returns:
        The transform matrix that minimizes the difference, 
    """

    ps = ps_tms(transforms)   # transformed points
    clustering, cluster_points = find_representative_points_with_ids(ps, k=k, eps=eps)   
    num_clusters = len(clustering.keys())
    tf_candidates = transforms
    errors_cluster = {0:[]}
    print(f"Number of clusters:{num_clusters}")
    if num_clusters > 0:
        if force_cluster_id >= 0:
            best_cluster = force_cluster_id
            tf_candidates = [transforms[t[0]] for t in cluster_points[best_cluster]]
        else:
            best_cluster, errors_cluster = find_best_cluster(
                A, B, clustering, transforms, 
                output_dir, error_type=error_type)
            tf_candidates = [transforms[t[0]] for t in cluster_points[best_cluster]]
    else:
        tf_candidates = transforms
    best_tf, best_idx, errors  = find_best_transform(A, B, tf_candidates, error_type=error_type)
    e = {'error_all_clusters': errors_cluster, 'errors_best_cluster': errors}
    
    return best_tf, best_idx, e


def ps_tms(tms):
    '''create a set of points transformed by transforms given (0,0,1)'''
    points = []
    for t in tms:
        p = np.dot(t, np.array([0,0,1]).T)
        points.append(p)
    ps = np.vstack(points)[:, :2]
    return ps


def find_representative_points_with_ids(points, k=5, eps=100):
    
    # DBSCAN clustering
    dbscan = DBSCAN(eps=eps, min_samples=5)
    dbscan.fit(points)

    # Extract cluster labels
    labels = dbscan.labels_

    # Create a dictionary to store points and their IDs for each cluster
    cluster_points = {}

    # Iterate through each point and assign it to its cluster along with its ID
    for i, (label, point) in enumerate(zip(labels, points)):
        if label not in cluster_points:
            cluster_points[label] = []
        cluster_points[label].append((i, point))

    num_outliers = 0
    if -1 in cluster_points.keys():
        num_outliers = len(cluster_points[-1])
    print(f"Number of outlier points: {num_outliers}")

    # Create a dictionary to store representative points and their IDs for each cluster
    representative_points = {}

    # Iterate through each cluster and select k random non-outlier points with IDs
    for label, points_list in cluster_points.items():
        if label == -1:
            continue  # Skip outliers
        if len(points_list) >= k:
            random_representatives = random.sample(points_list, k)
        else:
            random_representatives = points_list
        representative_points[label] = [(point_id, point) for point_id, point in random_representatives]

    return representative_points, cluster_points


def find_best_cluster(
        sbcd_lvl, hist_green, clusters, tms, 
        output_dir, error_type='sad'):
    
    min_error = np.inf
    best_cluster = None
    errors = {} 
    print(f"Total {len(clusters.keys())} clusters.")

    for k in clusters.keys(): 
        error = 0
        errors[k] = []
        for item in clusters[k]:
            idx = item[0]
            tf = tms[idx]
            e = error_raster_tf(sbcd_lvl, hist_green, tf, error_type=error_type)
            errors[k].append(e)
            error += e
        mean_error = sum(errors[k])/len(errors[k])
        if mean_error < min_error:
            min_error = mean_error
            best_cluster = k
        # Find the representative item (with minimum error) for the cluster
        min_error_idx = np.argmin(errors[k])  # Get the index of the minimum error in the cluster
        representative_tf = tms[clusters[k][min_error_idx][0]]  # Get the tf of the representative item

        # Ensure the output directory exists
        cluster_output_dir = os.path.join(output_dir, f"clusters/merged_image_cluster_{k}.png")
        os.makedirs(os.path.dirname(cluster_output_dir), exist_ok=True)

        # Write the merged image for the representative item
        write_merged_image(sbcd_lvl, hist_green, representative_tf, cluster_output_dir)


    print(f"Best Cluster ID: {best_cluster}")
    return best_cluster, errors


def find_best_transform(A, B, transforms, error_type='sad'):
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
    errors = []

    for idx, tf in enumerate(transforms):
        if idx % 100 == 0: print(idx) 
            
        error = error_raster_tf(A, B, tf, error_type=error_type) 
        errors.append(error)
        
        if error < min_error:
            min_error = error
            best_tf = tf
            best_idx = idx
    
    print(f"Best Transform: {best_idx} ({min_error})")
    return best_tf, best_idx, errors


def error_raster_tf(A, B, tf, error_type='sad'):
    B_tf =  warp_affine(B, tf, A)
    if error_type == 'sad':
        error = error_raster_sad(A, B_tf)
    else:
        error = error_raster_ccorr(A, B_tf)
    return error


def warp_affine(input, tf, reference):
    input_tf = cv2.warpAffine(input, tf[:2, :], (reference.shape[1], reference.shape[0]))
    return input_tf


def error_raster_sad(A, B):
    error = np.sum(np.abs(A - B)) / A.size
    return error


def error_raster_ccorr(A, B):
    """
    Computes the cross-correlation error between two images.

    Args:
        A: The first image array.
        B: The second image array.

    Returns:
        The cross-correlation error.
    """
    # Ensure the images are the same size
    if A.shape != B.shape:
        raise ValueError("The images must be the same size for cross-correlation.")

    # Normalize the images
    A_normalized = cv2.normalize(A, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    B_normalized = cv2.normalize(B, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)


    # Compute the normalized cross-correlation
    result = cv2.matchTemplate(A_normalized, B_normalized, cv2.TM_CCORR_NORMED)

    # The result is a single value since the images are of the same size
    score = result[0, 0]

    # Convert to an error metric (lower is better)
    error = 1 - score
    return error


def preprocess_image(image, channel=None, xy_swap=False, blur=None, gamma=None, inverse=False):
        
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

    if inverse:
        gray_image = 255 - gray_image

    return gray_image


def wb_level(image, in_black=0, in_white=255, gamma=1, out_black=0, out_white=255):
    image = np.clip( (image - in_black) / (in_white - in_black), 0, 255)
    image = ( image ** (1/gamma) ) *  (out_white - out_black) + out_black
    image = np.clip(image, 0, 255).astype(np.uint8)
    return image


def gcps_from_pairs(match_pairs, xy_swap=False, y_flip=False):
    gcps = []
    for p in match_pairs:
        target_x = p[2][1] if xy_swap else p[2][0]
        target_y = p[2][0] if xy_swap else p[2][1]
        target_y = -target_y if y_flip else target_y
        
        gcp = (p[0][1], p[0][0], target_x, target_y)
        gcps.append(gcp)
    return gcps

    
def execute_gdal_translate(gcps, input_file, output_file, simg=''):
    # Build the gdal_translate command with -gcp options
    if simg:
        cmd = ['singularity', 'exec', simg, 'gdal_translate']
    else:
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


def execute_gdalwarp(input_file, output_file, simg=''):
    # Build the gdal_translate command with -gcp options
    if simg:
        cmd = ['singularity', 'exec', simg, 'gdalwarp']
    else:
        cmd = ['gdalwarp']

    cmd.extend(['-order', '2', '-refine_gcps', '20', '20', input_file, output_file])
    
    try:
        # Execute the command
        print(" ".join(map(str,cmd)))
        subprocess.run(cmd, check=True)
        print("gdal_translate executed successfully.")
    except subprocess.CalledProcessError as e:
        print(f"Error executing gdal_translate: {e}")



def warp_from_gcps(matched_pairs, hnef, alignf, simg=''):

    gcps = gcps_from_pairs(matched_pairs)
        
    temp_dir = tempfile.TemporaryDirectory()
    temp_dir_path = Path(temp_dir.name)
    translate_file = temp_dir_path / "histology_translated.tif"
    if translate_file.exists():
        translate_file.unlink()

    execute_gdal_translate(gcps, hnef, translate_file, simg)
    execute_gdalwarp(translate_file, alignf, simg)


def write_merged_image(nge_raster, hne_raster, tf, output_path=None):
    hne_tf = cv2.warpAffine(
        hne_raster, tf[:2, :], 
        (nge_raster.shape[1], nge_raster.shape[0]))
    zeros = np.zeros(nge_raster.shape[:2], dtype="uint8")
    merged = cv2.merge([
        zeros,
        hne_tf, 
        nge_raster 
    ])
    if output_path:
        cv2.imwrite(str(output_path), merged)
    else:
        plt.figure(figsize=(10,10))
        plt.imshow(merged, cmap='Greys')


