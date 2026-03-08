#!/bin/bash

# Input argument: xml file
# Create the html file and the js file

bn=`basename -s .xml $1`

# STEP 1
# Create .js file
python3 xml2jsx.py $1 > ./js/$bn.js

# STEP 2    
# Create the html file
cp sample-index.html $bn.html
sed -i "s/sample-index.js/$bn.js/g" $bn.html
