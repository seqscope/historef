import numpy as np
from math import atan2, degrees, radians, cos, sin, sqrt

import cv2
import matplotlib.pyplot as plt
from scipy.spatial import KDTree
from scipy.spatial.distance import euclidean
from scipy.signal import find_peaks
from scipy.stats import gaussian_kde
from scipy.optimize import minimize_scalar, minimize
from scipy.ndimage import affine_transform


def find_overlapping_transform(A, B, rotation, scale_factor, max_nearest=300):
    '''identify transform that transformed B is within A'''
    # Calculate the bounding box for A just once
    bbox_A = bbox(A, 100)
    avg_distances = []

    overlapping_transforms = []
    for point_B in B:
        for point_A in A:
            tm = transform_points_matrix(point_B, point_A, rotation, scale_factor)
            tf_B = apply_transform_points(B, tm)
            bbox_B = bbox(tf_B)
    
            # Check if the bounding box of B is contained within the bounding box of A
            if is_contained(bbox_B, bbox_A):
                distance = mean_nearest_distance(A, tf_B)
                if distance < max_nearest:
                    avg_distances.append(distance)
                    overlapping_transforms.append(tm)
    print(f"Number of transforms: {len(overlapping_transforms)}")
    return overlapping_transforms, avg_distances


def apply_transform_points(points, transformation_matrix):
    points_array = np.hstack([points, np.ones((len(points), 1))]) 
    points_transformed = np.dot(transformation_matrix, points_array.T).T
    points_transformed = points_transformed[:, :2]
    return points_transformed


def transform_points_matrix(source_point, target_point, rotation_angle, scale_factor):
    """
    construct a 3x3 transformation matrix.

    Parameters:
    source_point (tuple): The source 2D point to be moved to the origin.
    target_point (tuple): The target 2D point where the origin will be moved after transformations.
    rotation_angle (float): The rotation angle in degrees.
    scale_factor (float): The scale factor for scaling the points.

    Returns:
    transformation matrix 
    """

    # Translation matrix to move source point to the origin
    translation_to_origin = np.array([
        [1, 0, -source_point[0]],
        [0, 1, -source_point[1]],
        [0, 0, 1]
    ])

    # Rotation matrix
    angle_rad = np.radians(rotation_angle)
    rotation_matrix = np.array([
        [np.cos(angle_rad), -np.sin(angle_rad), 0],
        [np.sin(angle_rad), np.cos(angle_rad), 0],
        [0, 0, 1]
    ])

    # Scaling matrix
    scaling_matrix = np.array([
        [scale_factor, 0, 0],
        [0, scale_factor, 0],
        [0, 0, 1]
    ])

    # Translation matrix to move origin to the target point
    translation_to_target = np.array([
        [1, 0, target_point[0]],
        [0, 1, target_point[1]],
        [0, 0, 1]
    ])

    # Combine all transformations into a single transformation matrix
    transformation_matrix = np.dot(np.dot(np.dot(translation_to_target, scaling_matrix), rotation_matrix), translation_to_origin)

    return transformation_matrix


def bbox(points, padding=0):
    x_coords, y_coords = zip(*points)
    min_x, min_y, max_x, max_y = min(x_coords), min(y_coords), max(x_coords), max(y_coords)
    return min_x - padding, min_y - padding, max_x + padding, max_y + padding


def is_contained(box1, box2):
    # Check if box1 is contained within box2
    return box1[0] >= box2[0] and box1[1] >= box2[1] and box1[2] <= box2[2] and box1[3] <= box2[3]


def euclidean_distance(point1, point2):
        return sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2)


def mean_nearest_distance(points_A, points_B):
    total_distance = 0
    for point_b in points_B:
        distances = [euclidean_distance(point_b, point_a) for point_a in points_A]
        min_distance = min(distances)
        total_distance += min_distance
    mean_distance = total_distance / len(points_B)
    return mean_distance


def find_nearest_pairs(points_A, points_B, transform, threshold=100):
    '''
    return the pairs matching points, find nearest point_A after transforming point_B
    the distance should be less than threshold

    '''
    # returns source (x,y), tansformd (x,y), target (x,y), nearest_distance
    transformed_points_B = apply_transform_points(points_B, transform)

    # Find the nearest pairs
    nearest_pairs = []
    for idx, point_B in enumerate(points_B):
        point_Bt = transformed_points_B[idx]
        nearest_point = None
        min_distance = float('inf')
        for point_A in points_A:
            dist = euclidean_distance(point_A, point_Bt)
            if dist < min_distance:
                min_distance = dist
                nearest_point = point_A
        if min_distance < threshold:
            nearest_pairs.append((tuple(point_B), tuple(point_Bt), nearest_point, min_distance))

    return nearest_pairs


