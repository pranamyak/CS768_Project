#!/bin/bash

CUDA_VISIBLE_DEVICES=3 python3 main_imp.py --config 'configs/superpixels_graph_classification_GCN_kuzu_100k.json' --dataset kuzu_biased_0.9 --data_dir '/mnt/nas/pkulkarni/debiasGNN_DisC/Disc_source_code/data/' --seed 31 --mask_epochs 200 --swap_epochs 100 --lambda_swap 10 --use_mask 1 --q 0.7 --lambda_dis 1 --out_dir "output_GCN_kuzu_0.9_q_0.7_lambda_swap_10" 

CUDA_VISIBLE_DEVICES=3 python3 main_imp.py --config 'configs/superpixels_graph_classification_GIN_kuzu_100k.json' --dataset kuzu_biased_0.9 --data_dir '/mnt/nas/pkulkarni/debiasGNN_DisC/Disc_source_code/data/' --seed 31 --mask_epochs 200 --swap_epochs 100 --lambda_swap 10 --use_mask 1 --q 0.7 --lambda_dis 1 --out_dir "output_GIN_kuzu_0.9_q_0.7_lambda_swap_10" 

CUDA_VISIBLE_DEVICES=3 python3 main_imp.py --config 'configs/superpixels_graph_classification_GCNII_kuzu_100k.json' --dataset kuzu_biased_0.9 --data_dir '/mnt/nas/pkulkarni/debiasGNN_DisC/Disc_source_code/data/' --seed 31 --mask_epochs 200 --swap_epochs 100 --lambda_swap 10 --use_mask 1 --q 0.7 --lambda_dis 1 --out_dir "output_GCNII_kuzu_0.9_q_0.7_lambda_swap_10" 

CUDA_VISIBLE_DEVICES=3 python3 main_imp.py --config 'configs/superpixels_graph_classification_GCN_kuzu_100k.json' --dataset kuzu_biased_0.95 --data_dir '/mnt/nas/pkulkarni/debiasGNN_DisC/Disc_source_code/data/' --seed 31 --mask_epochs 200 --swap_epochs 100 --lambda_swap 10 --use_mask 1 --q 0.7 --lambda_dis 1 --out_dir "output_GCN_kuzu_0.95_q_0.7_lambda_swap_10" 

CUDA_VISIBLE_DEVICES=3 python3 main_imp.py --config 'configs/superpixels_graph_classification_GIN_kuzu_100k.json' --dataset kuzu_biased_0.95 --data_dir '/mnt/nas/pkulkarni/debiasGNN_DisC/Disc_source_code/data/' --seed 31 --mask_epochs 200 --swap_epochs 100 --lambda_swap 10 --use_mask 1 --q 0.7 --lambda_dis 1 --out_dir "output_GIN_kuzu_0.95_q_0.7_lambda_swap_10" 

CUDA_VISIBLE_DEVICES=3 python3 main_imp.py --config 'configs/superpixels_graph_classification_GCNII_kuzu_100k.json' --dataset kuzu_biased_0.95 --data_dir '/mnt/nas/pkulkarni/debiasGNN_DisC/Disc_source_code/data/' --seed 31 --mask_epochs 200 --swap_epochs 100 --lambda_swap 10 --use_mask 1 --q 0.7 --lambda_dis 1 --out_dir "output_GCNII_kuzu_0.95_q_0.7_lambda_swap_10" 

CUDA_VISIBLE_DEVICES=3 python3 main_imp.py --config 'configs/superpixels_graph_classification_GCN_kuzu_100k.json' --dataset kuzu_biased_0.8 --data_dir '/mnt/nas/pkulkarni/debiasGNN_DisC/Disc_source_code/data/' --seed 31 --mask_epochs 200 --swap_epochs 100 --lambda_swap 10 --use_mask 1 --q 0.7 --lambda_dis 1 --out_dir "output_GCN_kuzu_0.8_q_0.7_lambda_swap_10" 

CUDA_VISIBLE_DEVICES=3 python3 main_imp.py --config 'configs/superpixels_graph_classification_GIN_kuzu_100k.json' --dataset kuzu_biased_0.8 --data_dir '/mnt/nas/pkulkarni/debiasGNN_DisC/Disc_source_code/data/' --seed 31 --mask_epochs 200 --swap_epochs 100 --lambda_swap 10 --use_mask 1 --q 0.7 --lambda_dis 1 --out_dir "output_GIN_kuzu_0.8_q_0.7_lambda_swap_10" 

CUDA_VISIBLE_DEVICES=3 python3 main_imp.py --config 'configs/superpixels_graph_classification_GCNII_kuzu_100k.json' --dataset kuzu_biased_0.8 --data_dir '/mnt/nas/pkulkarni/debiasGNN_DisC/Disc_source_code/data/' --seed 31 --mask_epochs 200 --swap_epochs 100 --lambda_swap 10 --use_mask 1 --q 0.7 --lambda_dis 1 --out_dir "output_GCNII_kuzu_0.8_q_0.7_lambda_swap_10" 