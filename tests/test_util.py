from historef.util import * 
import cv2


def test_intensity_cut():
    image  = "tests/resource/histology.tif"   
    image = cv2.imread(image, cv2.IMREAD_COLOR)
    image_cut = intensity_cut(image)
    print(image_cut.shape)
    assert image_cut.shape == (1929, 2311)  # just fake assert. turn on the line below to do integrate test 
    # cv2.imwrite("tests/resource/histology-cut.tif", image_cut) 
