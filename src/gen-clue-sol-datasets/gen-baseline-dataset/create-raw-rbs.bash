#!/bin/bash

TXTDIR=txt
RBSDIR=rbs
DOCDIR=doc

rm -f "$TXTDIR" "$RBSDIR" "$DOCDIR"
mkdir "$TXTDIR"
mkdir "$RBSDIR"
mkdir "$DOCDIR"

shopt -s nullglob

for i in "$DOCDIR"/*.doc; do 
    basename=$(basename "$i")
    filename=$(echo "$basename" | cut -d'.' -f1)
    #echo "$filename"
    antiword "$i" > "$TXTDIR/$filename.txt"
done

for i in "$TXTDIR"/*.txt ; do 
    basename=$(basename "$i")
    filename=$(echo "$basename" | cut -d'.' -f1)
    python txt2rbs.py "$i" > "$RBSDIR/$filename.rbs"
done
