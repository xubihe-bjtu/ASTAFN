#!/bin/bash

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
export PYTHONPATH=$ROOT
cd $ROOT

python run_g.py \
  --inference \
  --cpan_path ./dataset/GUANGDONG/pan_lat_lon.npy \
  --cera_path ./dataset/GUANGDONG/era_lat_lon.npy \
  --csta_path ./dataset/GUANGDONG/obs_lat_lon.npy \
  --save_path ./save/GUANGDONG \
  --data_path ./dataset/GUANGDONG \
  --adj_path ./dataset/GUANGDONG/sensor_graph/adj_mat.pkl \
  --device cuda:0 \
  --batch_size 64 \
  --k 9 \
  --num_layer 4 \
  --area GUANGDONG \
  --target 2 \
  --seq_len 8 \
  --pre_len 8 \
  --d_model 128 \
  --dropout 0.1 \
  --epochs 40 \
  --lr 0.01