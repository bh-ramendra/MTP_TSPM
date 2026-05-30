#!/usr/bin/env bash
/DATA/ramendra_2511ai39/miniconda3/envs/avqa/bin/python -u main_train.py --Temp_Selection --top_k 10 --Spatio_Perception --batch-size 8 --epochs 30 --lr 1e-4 --num_workers 4 --gpu 1 --checkpoint TSPM_local --model_save_dir models
