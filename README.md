# DisC
Source code for "NeurIPS2022-Debiasing Graph Neural Networks via Learning Disentangled Causal Substructure"

paper: https://arxiv.org/pdf/2209.14107.pdf

![image](https://github.com/googlebaba/DisC/blob/main/framework.png)

                                                             The framework of DisC
# Contact
Shaohua Fan, Email:fanshaohua@bupt.edu.cn

# Datasets 
Datasets used for Table 1: https://drive.google.com/file/d/1pv_cFKYJxXpT4qJ6jgvNn17MIovZUrhA/view?usp=sharing

Unseen test set for Table 2: https://drive.google.com/file/d/18LE0RnUBksGHsbO0lFtEC0O4jiO7B9_J/view?usp=sharing  # f[0] is the unbiased test set

# Requirements
pip -r requirements.txt

# Running the model
DisC_GCN 

python Disc_gcn_run.py

DisC_Gin

python Disc_gin_run.py

DisC_Gcnii

python Disc_gcnii_run.py

Can also use the command for running the desired model for the required dataset:
CUDA_VISIBLE_DEVICES=3 python3 main_imp.py --config 'configs/superpixels_graph_classification_GCN_MNIST_100k.json' --dataset MNIST_75sp_0.9 --data_dir '/mnt/nas/pkulkarni/debiasGNN_DisC/Disc_source_code/data/' --seed 31 --mask_epochs 200 --swap_epochs 100 --lambda_swap 10 --use_mask 1 --q 0.7 --lambda_dis 1 --out_dir "output_GCN_MNIST_75sp_0.9_q_0.7_lambda_swap_10" 