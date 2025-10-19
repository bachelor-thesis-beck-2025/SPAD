#!/bin/bash
# sh scripts/ood/vos/cifar10_test_vos.sh

PYTHONPATH='.':$PYTHONPATH \
python main.py \
--config configs/datasets/SP_LFW.yml \
configs/pipelines/test/SP_test.yml \
--num_workers 8 \
--network.checkpoint 'results/X_sep_SP_SP_LFW_e5_lr0.0002_wd5e-06_m0/net-best_epoch4_batch316acc0.9903.ckpt' 
