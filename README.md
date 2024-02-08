## historef

Automated referencing tool between between Seq-Scope sequence data and H&E histology image.

### installation

Install using the released package

- https://github.com/seqscope/historef/releases/


### usage

Download sample datasets
```
$ wget https://historef-sample-data.s3.amazonaws.com/sample/b08c/histology.tif
$ wget https://historef-sample-data.s3.amazonaws.com/sample/b08c/transcript.png
```

Then run the referencer.
```
$ python -m historef.referencer \
--nge transcript.png \
--hne histology.tif \
--aligned output/histology_aligned.tif
```

It will generated referenced geotiff under output folder with several reports. The geotiff will be XY-swapped against transcript to address Y-axis flip.


### how it works

historef aligns a histological image to a given NGE image by matching fiducial marks that are visible from both of the images. The best matching is the matching that minimizes the difference between the two images. 


First, historef identifies fiducial marks both from the NGE image 
and the histological images using template matching. We use template matching 
since the size of fiducial marks in the images are fixed. Highlighting top 10% pixels in histology helps identifying fiducial marks. From the identified fiducial marks, we pinpoint the center of each fiducial mark using 2d convolution. After this, we get two sets of center points of fiducial marks: one for the 
histology (H), the other for the NGE image (N).

Then, we need to find a transform t such that tH is overlapped with N. To find such transform, we need to find scale factor (S), rotation angle (R), and translation vector (T).

To find a scale factor and rotation angle, we construct a rectangular lattice for each points sets. The scale factor is inferred from comparing the short-side of the two lattices. The rotation angle is inferred from calculating the bearing of the lattices.


The candidate translation vectors are listed by (all SRH) X (all N). We filter out vectors such that bounding box of TSRH is not within bounding box of N. Since many of the translation vectors are parallel and hence redundant, we group the translation vectors into clusters using DBSCAN.

We sample several translation vectors from each cluster and calculate difference between transformed histology and NGE image. The difference are measured using cross correlation between the two. We can find the best transform from the cluster with highest cross correlation.

Instead of directly transform the histology using the best transform, we use the transform to match fiducial marks between two images. We match points in H and points in N if the transformed point H and its nearest point N are within 1/3 of lattice distance. These matches are the 'ground control point (GCP)' pairs for the alignment.

We are going to find the final transform using these GCPs.Since we usually have plenty of GCPs, we warp the histology using second order polynomial. And we refine the transform by removing a GCP if error of the pair is higher than given threshold.






