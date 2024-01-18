from historef.center import * 
from historef.grid import * 
import cv2


def test_construct_nearest_graph():
    ngef  = "tests/resource/test_center.png"   ## nbcd-nmatch-nge 3way image
    im_nge = cv2.imread(ngef, cv2.IMREAD_COLOR)
    im_nge = im_nge[:,:,0] ## use only blue channel

    centers, im_matched = get_fiducial_mark_centers(im_nge, 'sbcd', min_dist = 200)
    v, e = construct_grid_graph(centers)

    assert len(v) == 36


##################
# test with real data
##################

def test_construct_nearest_graph_sge():
    ngef  = "sample/sge.png"   ## nbcd-nmatch-nge 3way image
    im_nge = cv2.imread(ngef, cv2.IMREAD_COLOR)
    im_nge = im_nge[:,:,0] ## use only blue channel

    centers, im_matched = get_fiducial_mark_centers(im_nge, 'sbcd', min_dist = 200)
    v, e = construct_grid_graph(centers)
    plot_edges(v, e, "sample/sge-grid.png")

def test_construct_nearest_graph_histology():
    ngef  = "sample/histology.tif"   ## nbcd-nmatch-nge 3way image
    im_nge = cv2.imread(ngef, cv2.IMREAD_COLOR)
    im_nge = im_nge[:,:,0] ## use only blue channel

    centers, im_matched = get_fiducial_mark_centers(im_nge, 'HnE', min_dist = 300)
    v, e = construct_grid_graph(centers)
    plot_edges(v, e, "sample/histology-grid.png")
