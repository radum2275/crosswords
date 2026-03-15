#!/bin/bash

# Input argument: xml file
# Create the html file and the js file

bn=`basename -s .html $1`

# Create the html file
cp sample-index.html $bn.html
sed -i "s/sample-index.js/$bn.js/g" $bn.html
