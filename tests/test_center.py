from historef.center import * 
import cv2


def test_get_fiducial():
    ngef  = "tests/resource/test_center.png"   ## nbcd-nmatch-nge 3way image
    im_nge = cv2.imread(ngef, cv2.IMREAD_COLOR)
    im_nge = im_nge[:,:,0] ## use only blue channel

    centers, im_matched = get_fiducial_mark_centers(im_nge, 'sbcd', min_dist = 200)
    assert len(centers)==36
