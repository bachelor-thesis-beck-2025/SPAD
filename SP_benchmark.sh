#!/bin/bash


PYTHONPATH='.':$PYTHONPATH \
python main.py \
--config configs/pipelines/test/SP_benchmark.yml \
--dataset.folder /mnt/md0/beck/datasets/benchmarks/faceforensics_benchmark_images \
--num_workers 8 \
--network.checkpoint '/home/beck/repos/SPAD/results/checkpoints/net-best_LFW.ckpt'
