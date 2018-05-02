#!/bin/bash

echo "Hello World"
raspistill -o demo/input/image.jpg
python demo/objectSegmentation.py ./demo/input ./demo/output
python chain_coding.py demo/output/segimage.jpg

