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
  --device cuda:1 \
  --batch_size 32 \
  --k 2 \
  --num_layer 8 \
  --area GUANGDONG \
  --target 3 \
  --seq_len 8 \
  --pre_len 8 \
  --d_model 128 \
  --epochs 100 \
  --lr 0.001