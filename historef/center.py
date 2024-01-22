import os, sys, random, math
import importlib.resources as pkg_resources

import cv2
import numpy as np
from scipy import stats
from scipy.signal import convolve2d


def get_fiducial_mark_centers(im_sbcd, im_tmpl_option='sbcd', max_circles=384, min_dist=150, min_intensity=50, square_size=5):
    """Detect fiducial marks in a given image and a template. 
    The image and the template should be grayscale images.

    Usage:
    centers, im_matched = get_fiducial_mark_centers(im_nge, im_nge_tmpl, min_dist = 200)

    Parameters:
    - im_sbcd: a large grayscale image, either histology or NGE
    - im_tmpl_option: a small grayscale image of a single template of fiducial mark
      select 'sbcd'|'HnE' (default: 'sbcd')
    - max_circles: maximum number of fiducial marks to be detected
    - min_dist: minimum distance between fiducial marks
    - min_intensity: minimum intensity of fiducial marks after 2D convolution
    - square_size: size of the square to calculate the average intensity via convolution

    Returns
    - list of list 
        x : x-coordinate of detected centers
        y : y-coordinate of detected centers
        filter : PASS or FAIL, based on the current filtering criteria.
        intensity : Mean intensity values at the center after template matching and convolution
        pixels_in : Number of pixels "inside" the region proximal to the center (higher the better)
        pixels_out : Number of pixels "outside" the region proximal to the center (lower the better)
    """

    tmplf = pkg_resources.files('historef') / f"template/fiducial_mark.{im_tmpl_option}.png"
    im_tmpl = cv2.imread(str(tmplf), cv2.IMREAD_GRAYSCALE)

    ## default settings not exposed to users:
    method = cv2.TM_CCOEFF_NORMED     ## template matching method
    quantile_thres = 0.999            ## quantile threshold to extract the top 0.1% of the matched values
    quantile_high_thres = 0.9999      ## quantile threshold to extract the top 0.001% of the matched values

    ## get the input image size
    (tmpl_height, tmpl_width) = im_tmpl.shape
    (sbcd_height, sbcd_width) = im_sbcd.shape

    ## 1. perform template matching to identify possible locations of fiducial marks
    ## Perform template matching in correlation coefficient scale
    im_matched = cv2.matchTemplate(im_sbcd, im_tmpl, method)
    ## 2. Zero out values below top 0.1% threshold
    im_matched_thres = np.quantile(im_matched, quantile_thres)
    im_matched_thres_high = np.quantile(im_matched, quantile_high_thres)
    im_matched[im_matched < im_matched_thres] = 0
    ## Normalize the image to 0-255
    im_matched = cv2.normalize(im_matched, dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    ## the image dimension should be (sbcd_width - tmpl_width + 1, sbcd_height - tmpl_height + 1)
    (matched_height, matched_width) = im_matched.shape

    ## 3. perform a 2D convolution to pinpoint the center of fiducial marks
    intensity_map = np.zeros_like(im_matched, dtype=np.float32)
    kernel = np.ones((square_size, square_size), dtype=np.float32) / (square_size ** 2)
    intensity_map    = convolve2d(im_matched, kernel, mode='valid', boundary='fill', fillvalue=0)
    ## the image dimension should be (sbcd_width - tmpl_width - square_size + 1, sbcd_height - tmpl_height - square_size + 1)
    (intensity_height, intensity_width) = intensity_map.shape

    ## 4. Iteratively identify top N brightest squares using greedy algorithm
    brightest_squares = []
    offset_h = square_size // 2 + tmpl_height // 2
    offset_w = square_size // 2 + tmpl_width // 2
    n_detected = 0
    n_skipped = 0
    while len(brightest_squares) < max_circles:
        # Find the brightest location
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(intensity_map)
        if max_val < min_intensity:
            # No more bright squares to find
            break

        cx = max_loc[0] + offset_w  ## x coordinate of the center in the original image
        cy = max_loc[1] + offset_h  ## y coordinate of the center in the original image
        mx = max_loc[0] + (square_size // 2)  ## x coordinate of the center in the matched image
        my = max_loc[1] + (square_size // 2)  ## y coordinate of the center in the matched image

        ## take 51x51 pixels around the center to make sure that the center is not a false positive
        ex_h = 50
        ex_w = 50

        ## Extract the 51x51 pixels around the center
        im_neighbor = im_matched[(my-ex_h):(my+ex_h+1), (mx-ex_w):(mx+ex_w+1)]

        ## take 9x9 squares to ignore in false positive detection 
        sm_h = tmpl_height // 10
        sm_w = tmpl_width // 10

        ## Threshold the pixels to ignore the center
        thres_neighbor = im_matched_thres_high  ## intensity threshold to consider as false positives
        thres_out_pixels = ex_h * ex_w // 50            ## the number of pixels to be detected outside the center regions to be considered as false positives 
        thres_in_pixels  = (sm_h * sm_w) // 3

        pixels_in  = (im_neighbor[(ex_h-sm_h):(ex_h+sm_h+1), (ex_w-sm_w):(ex_w+sm_w+1)] > thres_neighbor).sum()
        pixels_all = (im_neighbor > thres_neighbor).sum()
        pixels_out = pixels_all - pixels_in

        ## 5. check whether the number of pixels above the threshold is larger than the threshold
        if ( pixels_out > thres_out_pixels or pixels_in < thres_in_pixels ): ## this is not a genuine fiducial mark
            # print("Skipped a fiducial mark at (%d, %d) with (%d, %d)" % (cx, cy, pixels_in, pixels_out), file=sys.stderr)
            brightest_squares.append([cx, cy, "FAIL", max_val, pixels_in, pixels_out])
            n_skipped += 1
        else:
#           print("Detected a fiducial mark at (%d, %d) with (%d, %d)" % (cx, cy, pixels_in, pixels_out), file=sys.stderr)
            brightest_squares.append([cx, cy, "PASS", max_val, pixels_in, pixels_out])
            n_detected += 1

        # 6. Zero out a (min_dist x min_dist) area around the found square to a low value to avoid close picks
        x_start = max(0, max_loc[0] - min_dist)
        y_start = max(0, max_loc[1] - min_dist)
        x_end = min(matched_width, max_loc[0] + min_dist)
        y_end = min(matched_height, max_loc[1] + min_dist)
        intensity_map[y_start:y_end, x_start:x_end] = 0

    print("detected %d fiducial marks and skipped %d" % (n_detected, n_skipped), file=sys.stderr)

    ## sort the coordinates by x and y
    brightest_squares.sort(key=lambda x: x[1])
    brightest_squares.sort(key=lambda x: x[0])

    return brightest_squares, im_matched



def get_fiducial_mark_centers_hough(im_sbcd, im_tmpl_option='sbcd', th=0.999, param1=500, param2=10, minRadius=2, maxRadius=15):
 
    tmplf = pkg_resources.files('historef') / f"template/fiducial_mark.{im_tmpl_option}.png"
    print(tmplf)
    im_tmpl = cv2.imread(str(tmplf), cv2.IMREAD_GRAYSCALE)

    method = cv2.TM_CCOEFF_NORMED     ## template matching method
    
    res = cv2.matchTemplate(im_sbcd, im_tmpl, method)
    threshold = np.quantile(res, th)  
    res[res<threshold] = 0
    im_circle = 255 * res
    im_circle = np.clip(im_circle, 0, 255)  # Ensure values are within [0, 255]
    im_circle = im_circle.astype(np.uint8)  # Convert to np.uint8
    
    circles = cv2.HoughCircles(
        im_circle, cv2.HOUGH_GRADIENT, 1, 30,
        param1, param2, minRadius, maxRadius)
    return circles.tolist()[0], res, im_circle

