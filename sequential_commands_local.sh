#!/bin/bash

python imageRegistration_cpu.py -i /u1/rashid/ImageRegistration/Data/drone_images_tif/full_set/ -o registration_output/ -r -p > output_full_local.csv
python imageRegistration_cpu.py -i /u1/rashid/ImageRegistration/Data/drone_images_tif/set_1/ -o registration_output/ -r -p > output_25_local.csv
python imageRegistration_cpu.py -i /u1/rashid/ImageRegistration/Data/drone_images_tif/set_50/ -o registration_output/ -r -p > output_50_local.csv
python imageRegistration_cpu.py -i /u1/rashid/ImageRegistration/Data/drone_images_tif/set_75/ -o registration_output/ -r -p > output_75_local.csv
python imageRegistration_gpu.py -g 0 -i /u1/rashid/ImageRegistration/Data/drone_images_tif/full_set/ -o registration_output/ -r -p > output_full_gpu_local.csv
python imageRegistration_gpu.py -g 0 -i /u1/rashid/ImageRegistration/Data/drone_images_tif/set_1/ -o registration_output/ -r -p > output_25_gpu_local.csv
python imageRegistration_gpu.py -g 0 -i /u1/rashid/ImageRegistration/Data/drone_images_tif/set_50/ -o registration_output/ -r -p > output_50_gpu_local.csv
python imageRegistration_gpu.py -g 0 -i /u1/rashid/ImageRegistration/Data/drone_images_tif/set_75/ -o registration_output/ -r -p > output_75_gpu_local.csv


