import cv2

from historef.center import get_fiducial_mark_centers
from historef.grid import \
    construct_grid_graph, \
    identify_scale_factor,  \
    identify_rotation_angle 
from historef.transform import \
    find_overlapping_transform, \
    apply_transform_points, \
    find_nearest_pairs
from historef.util import intensity_cut
from historef.matchraster import \
    find_best_transform, preprocess_image,\
    gcps_from_pairs, execute_gdal_translate, execute_gdalwarp, \
    warp_from_gcps


def main():
    pass



def process(ngef, hnef, alignf):
    
    params = {
        'nge_xy_swap': True,
        'nge_center_channel': 0,
        'nge_center_template': 'sbcd',
        'nge_center_min_dist': 200,
        'hne_xy_swap': True,
        'hne_center_channel': 0,
        'hne_center_template': 'HnE',
        'hne_center_min_dist': 300,
        'matching_max_nearest': 0.33,
        'nge_raster_channel':1,
        'nge_raster_blur':5,
        'nge_raster_gamma':2,
        'hne_raster_channel':1,
        'hne_raster_blur':5,
        'hne_raster_gamma':2,


    }

    im_nge_raw = cv2.imread(ngef, cv2.IMREAD_COLOR)
    im_nge = preprocess_image(im_nge_raw, channel=0, xy_swap=True)
    circles_nge, _ = get_fiducial_mark_centers(im_nge, 'sbcd', min_dist = 200)
    v_nge, e_nge = construct_grid_graph(circles_nge)

    im_hne_raw = cv2.imread(hnef, cv2.IMREAD_COLOR)
    im_hne = intensity_cut(im_hne_raw)
    im_hne = preprocess_image(im_hne, xy_swap=True)
    circles_hne, _ = get_fiducial_mark_centers(im_hne, 'HnE', min_dist = 300)
    v_hne, e_hne = construct_grid_graph(circles_hne)

    scale_factor, xgrid_nge, xgrid_hne = identify_scale_factor(e_nge, e_hne)
    rotation = identify_rotation_angle(e_hne)

    tms, avg_distances = find_overlapping_transform(
        v_nge, v_hne, 
        rotation, scale_factor, 
        max_nearest=0.33*xgrid_nge)

    nge_raster = preprocess_image(
        im_nge_raw, xy_swap=True, 
        channel=1, blur=5, gamma=2)
    
    hne_raster = preprocess_image(
        im_hne_raw, xy_swap=True, 
        channel=1, blur=5, gamma=2)

    best_tf, best_idx, best_B, errors  = \
        find_best_transform(nge_raster, hne_raster, tms)

    matched_pairs = find_nearest_pairs(v_nge, v_hne, tms[best_idx], threshold=100)  #type:ignore
    warp_from_gcps(matched_pairs, hnef, alignf)



if __name__=='__main__':
    main()
