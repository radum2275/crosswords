#!/bin/bash

# Input argument: rbs file

bn=`basename -s .rbs $1`

# STEP 4
# Create the grid image
cp $bn.xml ../grid-images2
cd ../grid-images2
./create-grid-image.bash $bn.xml
#rm $bn.xml
cd ../careuri-definitii-4
mv ../grid-images2/$bn.png .
