#!/bin/bash
date

python imageRegistration_gpu.py -g 0 -i /u1/rashid/ImageRegistration/Data/drone_images_tif/set_1/ -o registration_output/g1_local/ -r -p > output_gpu_1.csv &
python imageRegistration_gpu.py -g 1 -i /u1/rashid/ImageRegistration/Data/drone_images_tif/set_2/ -o registration_output/g2_local/ -r -p > output_gpu_2.csv &
python imageRegistration_gpu.py -g 2 -i /u1/rashid/ImageRegistration/Data/drone_images_tif/set_3/ -o registration_output/g3_local/ -r -p > output_gpu_3.csv &
#python imageRegistration_gpu.py -g 3 -i /u1/rashid/ImageRegistration/Data/drone_images_tif/set_4/ -o registration_output/g4_local/ -r -p > output_gpu_4.csv &

wait;wait;wait;
#wait;

date
