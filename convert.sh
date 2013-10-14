#!/bin/sh

NEW="new_$1"
rm -rf $NEW
mkdir $NEW

for i in `ls -1 $1`
do
	convert $1/$i -adaptive-resize 200x200\> -size 200x200 xc:white +swap -gravity center -composite $NEW/$i
done
