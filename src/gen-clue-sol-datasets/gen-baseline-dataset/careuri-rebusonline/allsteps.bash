#!/bin/bash

# Input argument: rbs file

bn=`basename -s .rbs $1`

dos2unix $1

# STEP 1
# Create .xml file
python2.7 rbs2xml.py $1 > $bn.xml

# STEP 2
# Create the jsx pack
./create-jsx-pack.bash $bn.xml

