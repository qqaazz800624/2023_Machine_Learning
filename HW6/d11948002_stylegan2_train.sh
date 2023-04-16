#!/bin/bash

export CUDA_VISIBLE_DEVICES=0

stylegan2_pytorch --data /neodata/ML/hw6_dataset/faces \
                  --num-train-steps 35000 \
                  --image-size 64 