#!/bin/bash
for f in *.*; do
    ext="${f##*.}"
    base="${f%.*}"
    newbase="${base//./_}"
    newbase="${newbase// /_}"
    newname="$newbase.$ext"
    if [[ "$f" != "$newname" ]]; then
        mv -- "$f" "$newname"
    fi
done