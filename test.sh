#!/bin/bash

filename="www"
COUNT=0
for i in {1..2}
do
COUNT=$(($COUNT+1))
R=$(($RANDOM%4))

   echo "hi $R $filename$COUNT"
   python imageRegistration_gpu.py -g $R -i ~/ImageRegistration/Data/drone_images_tif/drone_images_tif_1/ -o registration_output/ -r -p > output$COUNT.txt
done
