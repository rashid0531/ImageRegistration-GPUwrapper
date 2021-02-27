#!/bin/bash
sum=0
while read -r line
do
    (( sum += line ))
done < output_gpu_1.csv
echo $sum
