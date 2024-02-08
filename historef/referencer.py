import argparse
from pathlib import Path

import numpy as np
import cv2
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

from historef.center import get_fiducial_mark_centers
from historef.grid import \
    construct_grid_graph, \
    identify_scale_factor, \
    identify_rotation_angle,\
    plot_edges
from historef.transform import \
    find_overlapping_transform, \
    apply_transform_points,\
    find_nearest_pairs
from historef.util import intensity_cut
from historef.matchraster import \
    find_best_transform_eff, preprocess_image,\
    warp_from_gcps


def main():
    parser = argparse.ArgumentParser(description='Process some files.')
    parser.add_argument('--nge', required=True, type=str, help='Path to the NGE image')
    parser.add_argument('--hne', required=True, type=str, help='Path to the H&E image')
    parser.add_argument('--aligned', required=True, type=str, help='Path to the aligned image')

    # Add arguments for each parameter in the params dictionary
    parser.add_argument('--nge_xy_swap', type=lambda x: (str(x).lower() == 'true'), help='Override nge_xy_swap value')
    parser.add_argument('--nge_center_channel', type=int, help='Override nge_center_channel value')
    parser.add_argument('--nge_center_template', type=str, help='Override nge_center_template value')
    parser.add_argument('--nge_center_min_dist', type=int, help='Override nge_center_min_dist value')
    parser.add_argument('--hne_xy_swap', type=lambda x: (str(x).lower() == 'true'), help='Override hne_xy_swap value')
    parser.add_argument('--hne_center_channel', type=int, help='Override hne_center_channel value')
    parser.add_argument('--hne_center_template', type=str, help='Override hne_center_template value')
    parser.add_argument('--hne_center_min_dist', type=int, help='Override hne_center_min_dist value')
    parser.add_argument('--matching_max_nearest', type=float, help='Override matching_max_nearest value')
    parser.add_argument('--nge_raster_channel', type=int, help='Override nge_raster_channel value')
    parser.add_argument('--nge_raster_blur', type=int, help='Override nge_raster_blur value')
    parser.add_argument('--nge_raster_gamma', type=float, help='Override nge_raster_gamma value')
    parser.add_argument('--hne_raster_channel', type=int, help='Override hne_raster_channel value')
    parser.add_argument('--hne_raster_blur', type=int, help='Override hne_raster_blur value')
    parser.add_argument('--hne_raster_gamma', type=float, help='Override hne_raster_gamma value')
    parser.add_argument('--matched_pair_max_distance', type=int, help='Override matched_pair_max_distance value')
    parser.add_argument('--sample_per_cluster', type=int, help='Override number of sampled transform per cluster')
    parser.add_argument('--force_cluster_id', type=int, help='Override best cluster id')
    parser.add_argument('--error_type', type=str, help='error type. ccorr or sad')

    args = parser.parse_args()
 
    params = {
        'nge_xy_swap': True,
        'nge_center_channel': 0,
        'nge_center_template': 'sbcd',
        'nge_center_min_dist': 200,
        'hne_xy_swap': True,
        'hne_center_channel': 0,
        'hne_center_template': 'HnE_121',
        'hne_center_min_dist': 300,
        'matching_max_nearest': 0.33,
        'nge_raster_channel':1,
        'nge_raster_blur':5,
        'nge_raster_gamma':2,
        'hne_raster_channel':1,
        'hne_raster_blur':5,
        'hne_raster_gamma':3,
        'matched_pair_max_distance': 100,
        'sample_per_cluster': 5,
        'force_cluster_id': -1,
        'error_type': 'sad',
    }

    # Update params with any provided arguments
    for key in params:
        if getattr(args, key, None) is not None:
            params[key] = getattr(args, key)
    
    nge_path = Path(args.nge)
    hne_path = Path(args.hne)
    aligned_parent_dir = Path(args.aligned).parent

    if check_file_paths(nge_path, hne_path, aligned_parent_dir):
        print("All files and directories exist. Proceeding with process.")
        process(str(nge_path), str(hne_path), args.aligned, params)
    else:
        print("One or more files/directories do not exist. Exiting.")


def check_file_paths(nge_path, hne_path, aligned_parent_dir):
    """Check if the provided file paths and directories exist."""
    if not nge_path.exists():
        print(f"Error: The file {nge_path} does not exist.")
        return False
    if not hne_path.exists():
        print(f"Error: The file {hne_path} does not exist.")
        return False
    if not aligned_parent_dir.exists():
        print(f"Warning: The directory {aligned_parent_dir} does not exist. Creating it.")
        aligned_parent_dir.mkdir(parents=True, exist_ok=True)
    return True


def process(ngef, hnef, alignf, params):
   
    output_dir = Path(alignf).parent

    im_nge_raw = cv2.imread(ngef, cv2.IMREAD_COLOR)
    im_nge = preprocess_image(
        im_nge_raw, 
        channel=params['nge_center_channel'], 
        xy_swap=params['nge_xy_swap'])
    circles_nge, _ = get_fiducial_mark_centers(
        im_nge, 
        params['nge_center_template'], 
        min_dist=params['nge_center_min_dist'])
    v_nge, e_nge = construct_grid_graph(circles_nge)
    plot_edges(v_nge, e_nge, output_dir / "nge_graph.png")

    im_hne_raw = cv2.imread(hnef, cv2.IMREAD_COLOR)
    im_hne = intensity_cut(im_hne_raw)
    im_hne = preprocess_image(
        im_hne, 
        xy_swap=params['hne_xy_swap'])
    circles_hne, _ = get_fiducial_mark_centers(
        im_hne, 
        params['hne_center_template'],
        min_dist=params['hne_center_min_dist'])
    v_hne, e_hne = construct_grid_graph(circles_hne)
    plot_edges(v_hne, e_hne, output_dir / "hne_graph.png")

    scale_factor, xgrid_nge, xgrid_hne = \
        identify_scale_factor(e_nge, e_hne)
    rotation = identify_rotation_angle(e_hne)
    print(f"Scale Factor: {scale_factor}, Rotation Angle: {rotation}")

    tms, avg_distances = find_overlapping_transform(
        v_nge, v_hne, 
        rotation, scale_factor, 
        max_nearest=params['matching_max_nearest']*xgrid_nge)
    plot_edges(
        apply_transform_points(v_hne, tms[0]), e_nge, 
        output_dir / "matching_sample.png")

    nge_raster = preprocess_image(
        im_nge_raw, xy_swap=params['nge_xy_swap'], 
        channel=params['nge_raster_channel'], 
        blur=params['nge_raster_blur'], 
        gamma=params['nge_raster_gamma'])
    cv2.imwrite(str(output_dir / "nge_preprocessed.png"), nge_raster)
    
    hne_raster = preprocess_image(
        im_hne_raw, xy_swap=params['hne_xy_swap'], 
        channel=params['hne_raster_channel'], 
        blur=params['hne_raster_blur'], 
        gamma=params['hne_raster_gamma'])
    cv2.imwrite(str(output_dir / "hne_preprocessed.png"), hne_raster)

    best_tf, best_idx, errors  = \
        find_best_transform_eff(
            nge_raster, hne_raster, tms, 
            k=params['sample_per_cluster'], 
            force_cluster_id=params['force_cluster_id'],
            error_type=params['error_type'])
    print(f"Best Transform (ID {best_idx}):", best_tf)
    print("errors:", errors)
    plot_errors(
        errors['error_all_clusters'], 
        errors['errors_best_cluster'], 
        output_path= output_dir / "raster_diff_errors.png")
    write_merged_image(nge_raster, hne_raster, best_tf, output_dir / "best_merged_image.png") 

    matched_pairs = find_nearest_pairs(
        v_nge, v_hne, best_tf,                          #type:ignore
        threshold=params['matched_pair_max_distance'])  #type:ignore
    warp_from_gcps(matched_pairs, hnef, alignf)
    plot_edges(
        apply_transform_points(v_hne, best_tf), e_nge, 
        output_dir / "best_matched_pair.png")


def write_merged_image(nge_raster, hne_raster, tf, output_path=None):
    hne_tf = cv2.warpAffine(
        hne_raster, tf[:2, :], 
        (nge_raster.shape[1], nge_raster.shape[0]))
    zeros = np.zeros(nge_raster.shape[:2], dtype="uint8")
    merged = cv2.merge([nge_raster, hne_tf , zeros])
    if output_path:
        cv2.imwrite(str(output_path), merged)
    else:
        plt.figure(figsize=(10,10))
        plt.imshow(merged, cmap='Greys')


# Extracting cluster IDs and corresponding errors
def plot_errors(clusters, comparison=None, dot_size=3, output_path=None):
    # Preparing data for plotting
    x_values = []
    y_values = []
    for cluster_id, errors in clusters.items():
        for error in errors:
            x_values.append(cluster_id)
            y_values.append(error)

    # Plotting cluster errors
    plt.figure(figsize=(6, 6))
    plt.scatter(
        x_values, y_values, 
        s=dot_size, label='Cluster Errors')

    if comparison:
        # Determining the x-coordinate for the comparison values
        comparison_x = max(clusters.keys()) + 1
        # Plotting comparison values
        plt.scatter(
            [comparison_x] * len(comparison), comparison, 
            color='red', s=dot_size,  label='Best Cluster')

    # Plot labels and title
    plt.xlabel('Cluster ID')
    plt.ylabel('Raster Differences')
    plt.gca().xaxis.set_major_locator(MultipleLocator(1))
    plt.title('Comparison of Errors Between Clusters')
    plt.legend()
    if output_path:
        plt.savefig(output_path)
    else:
        plt.show()


if __name__=='__main__':
    main()
