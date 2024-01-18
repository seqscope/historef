import numpy as np
from math import atan2, degrees, sqrt

import matplotlib.pyplot as plt
from scipy.spatial import KDTree
from scipy.signal import find_peaks
from scipy.optimize import minimize_scalar, minimize


def construct_grid_graph(center_tsv, angle_allowance=5):
    v, e = construct_nearest_graph(center_tsv)
    # edge: (p1, p2, angle)
    peak_angles = find_peak_bearings([edge[2] for edge in e]) 
    grid_edges = []
    for edge in e:
        bearing = edge[2]
        if any(bearing_difference(bearing, peak_angle) <= angle_allowance for peak_angle in peak_angles):
            grid_edges.append(edge)
    return v, grid_edges


def identify_scale_factor(edges_1, edges_2):
    '''
    returns ratio of short side grid lengths from two edge sets 
    Usage: 
        scale_factor, xgrid_sbcd, xgrid_hist = identify_scale_factor(e_sbcd, e_hist)
    '''
    
    xgrid_1 = min(find_grid_lengths(edges_1))
    xgrid_2 = min(find_grid_lengths(edges_2))
    scale_factor = xgrid_1 /  xgrid_2
    print("xgrid_1 / xgrid_2::", f"{xgrid_1} / {xgrid_2} = {scale_factor}")
    return (scale_factor, xgrid_1, xgrid_2)


def identify_rotation_angle(edges):
    '''returns an angle that minimize the error to the cardinal angles'''
    angles = calculate_bearing_angles(edges)
    rotation_angle = find_optimal_rotation(angles)
    print("Rotation Angle", rotation_angle)
    return rotation_angle


def construct_nearest_graph(center_points):
    points = [(p[0], p[1]) for p in center_points]
    tree = KDTree(points)
    num_neighbors = 4  # including itself

    edges = []
    for point in points:
        distances, indices = tree.query(point, k=num_neighbors)
        for i in indices[1:]:
            bearing, length = calculate_bearing_length(point, points[i])
            edges.append((point, points[i], bearing, length))
    return points, edges


def roundTo90(theta):
    """Round an angle to the nearest multiple of 90 degrees."""
    return round(theta / 90) * 90


def error_function_angle(alpha, angles):
    """Compute the sum of squared errors from the nearest cardinal direction for each angle."""
    total_error = 0
    for theta in angles:
        new_theta = (theta + alpha) % 360
        error = (roundTo90(new_theta) - new_theta) ** 2
        total_error += error
    return total_error


def find_optimal_rotation(angles):
    """Find the optimal rotation angle for a list of angles."""
    result = minimize_scalar(error_function_angle, args=(angles,))
    return result.x


def calculate_bearing_length(point1, point2):
    dx = point2[0] - point1[0]
    dy = point2[1] - point1[1]
    bearing = np.degrees(np.arctan2(dy, dx)) % 180
    length = sqrt(dx*dx + dy*dy)
    return bearing, length


def calculate_bearing_angles(edges):
    """Calculate the bearing angles of edges."""
    angles = []
    for edge in edges:
        point0 = edge[0]
        point1 = edge[1]
        dx = point0[0] - point1[0]
        dy = point0[1] - point1[1]
        bearing = atan2(dy, dx)
        angles.append(degrees(bearing) % 180)  # Bearing angle mod 180
    return angles


def find_peak_bearings(angles):
    """find peak bearing to identify the orientation of the grid"""
    angle1 = round(find_optimal_rotation(angles),3)
    angle2 = angle1 + 90
    return [angle1, angle2]


def bearing_difference(angle1, angle2):
    return min((angle1 - angle2) % 180, (angle2 - angle1) % 180)


def find_grid_lengths(edges, top_n=2):
    '''histogram based method'''
    lengths = [e[3] for e in edges]
    # Create a histogram
    hist, bin_edges = np.histogram(lengths, bins=1000, range=(100, 1100))
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    
    # Find peaks
    peaks, _ = find_peaks(hist, prominence=10)
    
    # Sort peaks by their frequency (height in the histogram)
    sorted_peak_indices = np.argsort(hist[peaks])[::-1]  # Sort in descending order
    
    # Get the x-values (edge lengths) of the top peaks
    top_peak_lengths = bin_centers[peaks][sorted_peak_indices]
    top_peak_lengths = top_peak_lengths[:top_n]

    return top_peak_lengths


def plot_edges(vertices, edges, output_file=None):
    plt.figure(figsize=(10, 10))
    x = [p[0] for p in vertices]
    y = [p[1] for p in vertices]
    plt.scatter(x, y, c='blue', label='Points')

    for edge in edges:
        point0 = edge[0]
        point1 = edge[1]
        plt.plot([point0[0], point1[0]], [point0[1], point1[1]], 'r-')
     
    plt.xlabel('x')
    plt.ylabel('y')
    plt.gca().set_aspect('equal', adjustable='box')  # Setting the aspect ratio to 1:1
    plt.legend()
    plt.xlim([-1000, 12000])
    plt.ylim([-1000, 12000])

    
    if output_file is not None:
        plt.savefig(output_file)
    else:
        plt.show()

