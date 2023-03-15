#!bin/sh
cd /home/litianyi/data/EgoMotion/bvh/
for file in ./*
do
    if test -f $file
    then
        if [ "${file##*.}"x = "bvh"x ]
        then
            bvh-converter $file
        fi
    fi
    if test -d $file
    then
        echo $file 是目录
    fi
done