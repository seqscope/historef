#!/bin/bash

# Check if exactly three arguments are given
if [ "$#" -ne 3 ]; then
    echo "Usage: $0 <reference.png> <input.tif> <output.tif>"
    exit 1
fi

# Convert relative paths to absolute paths
REFERENCE_IMAGE=$(realpath "$1")
INPUT_TIF=$(realpath "$2")
OUTPUT_TIF=$(realpath "$3")

echo $REFERENCE_IMAGE
# Step 1: Run gdalinfo to get information about the image
INFO=$(gdalinfo "$REFERENCE_IMAGE" 2>&1)


# Debugging: Print the gdalinfo output
# echo "gdalinfo output:"
# echo "$INFO"

# Step 2: Extract width and height from the output
if [[ $INFO =~ Size\ is\ ([0-9]+),\ ([0-9]+) ]]; then
    WIDTH=${BASH_REMATCH[1]}
    HEIGHT=${BASH_REMATCH[2]}
    echo "Extracted dimensions: WIDTH=${WIDTH}, HEIGHT=${HEIGHT}"
else
    echo "Failed to extract image dimensions."
    exit 1
fi

# Step 3: Run gdalwarp with the extracted width and height
gdalwarp \
  "$INPUT_TIF" "$OUTPUT_TIF" -ct "+proj=pipeline +step +proj=axisswap +order=2,-1" \
  -overwrite \
  -te 0 -$HEIGHT $WIDTH 0 -ts $WIDTH $HEIGHT

echo "gdalwarp command executed with dimensions: width=${WIDTH}, height=${HEIGHT}"
