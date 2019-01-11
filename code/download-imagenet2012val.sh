#!/bin/bash

mkdir ../data/test_data
echo "Downloading ILSVRC2012 dataset"
wget -O ../data/test_data/ILSVRC2012_img_val.tar http://www.image-net.org/challenges/LSVRC/2012/nnoupb/ILSVRC2012_img_val.tar
tar -xvf ../data/test_data/ILSVRC2012_img_val.tar -C ../data/test_data
