#!/bin/bash

python imageRegistration_cpu.py -i /discus/rashid/ImageRegistration/Data/drone_images_tif/full_set/ -o registration_output/ -r -p > output_full.csv
python imageRegistration_cpu.py -i /discus/rashid/ImageRegistration/Data/drone_images_tif/set_1/ -o registration_output/ -r -p > output_25.csv
python imageRegistration_cpu.py -i /discus/rashid/ImageRegistration/Data/drone_images_tif/set_50/ -o registration_output/ -r -p > output_50.csv
python imageRegistration_cpu.py -i /discus/rashid/ImageRegistration/Data/drone_images_tif/set_75/ -o registration_output/ -r -p > output_75.csv
python imageRegistration_gpu.py -g 0 -i /discus/rashid/ImageRegistration/Data/drone_images_tif/full_set/ -o registration_output/ -r -p > output_full_gpu.csv
python imageRegistration_gpu.py -g 0 -i /discus/rashid/ImageRegistration/Data/drone_images_tif/set_1/ -o registration_output/ -r -p > output_25_gpu.csv
python imageRegistration_gpu.py -g 0 -i /discus/rashid/ImageRegistration/Data/drone_images_tif/set_50/ -o registration_output/ -r -p > output_50_gpu.csv
python imageRegistration_gpu.py -g 0 -i /discus/rashid/ImageRegistration/Data/drone_images_tif/set_75/ -o registration_output/ -r -p > output_75_gpu.csv
