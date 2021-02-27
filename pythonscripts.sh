#!/bin/bash
date 

python imageRegistration_gpu.py -g 0 -i /discus/rashid/ImageRegistration/Data/drone_images_tif/set_1/ -o registration_output/g1/ -r -p > output_gpu_1.csv & 
python imageRegistration_gpu.py -g 1 -i /discus/rashid/ImageRegistration/Data/drone_images_tif/set_2/ -o registration_output/g2/ -r -p > output_gpu_2.csv &
python imageRegistration_gpu.py -g 2 -i /discus/rashid/ImageRegistration/Data/drone_images_tif/set_3/ -o registration_output/g3/ -r -p > output_gpu_3.csv &
#python imageRegistration_gpu.py -g 3 -i /discus/rashid/ImageRegistration/Data/drone_images_tif/set_4/ -o registration_output/g4/ -r -p > output_gpu_4.csv &

wait;wait;wait;
#wait;

date
