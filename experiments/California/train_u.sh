#!/bin/bash

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
export PYTHONPATH=$ROOT
cd $ROOT

python run_c.py \
  --cpan_path ./dataset/California/pan_lat_lon.npy \
  --cera_path ./dataset/California/era_lat_lon.npy \
  --csta_path ./dataset/California/obs_lat_lon.npy \
  --save_path ./save/California \
  --data_path ./dataset/California \
  --adj_path ./dataset/California/sensor_graph/adj_mat.pkl \
  --device cuda:4 \
  --batch_size 32 \
  --k 2 \
  --area California \
  --target 0 \
  --seq_len 8 \
  --pre_len 8 \
  --d_model 64 \
  --lr 0.001