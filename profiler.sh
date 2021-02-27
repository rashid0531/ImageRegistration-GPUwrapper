#!/bin/bash

#nvprof --devices 0 --profile-child-processes ./imageRegistration_gpu.py -g 0 -i /discus/rashid/ImageRegistration/Data/drone_images_tif/set_1/ -o /u1/rashid/ImageRegistration/Code/rashid_attempt_1/registration_output/ -r -p > output_gpu_profile_time.txt

nvprof --print-summary-per-gpu ./imageRegistration_gpu.py -g 0 -i /discus/rashid/ImageRegistration/Data/drone_images_tif/set_1/ -o /u1/rashid/ImageRegistration/Code/rashid_attempt_1/registration_output/ -r -p > output_gpu_profile_time.txt



