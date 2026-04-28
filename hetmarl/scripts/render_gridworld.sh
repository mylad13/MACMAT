#!/bin/sh
env="GridWorld"

seed_max=1

echo "env is ${env}"
for seed in `seq ${seed_max}`
do
 CUDA_VISIBLE_DEVICES=0 python render/render_gridworld.py
done
