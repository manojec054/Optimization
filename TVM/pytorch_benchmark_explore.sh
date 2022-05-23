#!/bin/bash

for sh in 16 64 256 800 1600 3200
    do 
    nvprof --profile-from-start off --print-gpu-summary  python pytorch_benchmark_explore.py --shape $sh
    done
