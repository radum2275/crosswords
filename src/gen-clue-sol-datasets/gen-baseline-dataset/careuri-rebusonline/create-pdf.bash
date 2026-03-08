#!/bin/bash

file=$1
fname=$(basename $file)
fbname=${fname%.*}
#echo $fbname

python2.7 xml2pdf.py $file > $fbname.tex
pdflatex $fbname.tex
mv $fbname.pdf pdf/
rm -f *aux *tex *log
