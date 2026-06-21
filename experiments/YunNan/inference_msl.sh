#!/bin/bash

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
export PYTHONPATH=$ROOT
cd $ROOT

python run_y.py \
  --inference \
  --cpan_path ./dataset/YUNNAN/pan_lat_lon.npy \
  --cera_path ./dataset/YUNNAN/era_lat_lon.npy \
  --csta_path ./dataset/YUNNAN/obs_lat_lon.npy \
  --save_path ./save/YUNNAN \
  --data_path ./dataset/YUNNAN \
  --adj_path ./dataset/YUNNAN/sensor_graph/adj_mat.pkl \
  --device cuda:4 \
  --batch_size 32 \
  --k 2 \
  --num_layer 4 \
  --area YUNNAN \
  --target 2 \
  --seq_len 8 \
  --pre_len 8 \
  --d_model 128 \
  --epochs 100 \
  --lr 0.001