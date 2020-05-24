#!/bin/bash


echo "total, fps, resize, blurring, luminance, unblurring, sobel, refinement" > data.txt
for i in 080 280 480 680 880; do
    for n in 1 2 3 4 5; do
        ./comic-upscaler 2 $i.bmp out >> data.txt
    done
done