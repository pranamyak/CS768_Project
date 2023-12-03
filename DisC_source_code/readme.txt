# first install environment for GNN and DisC

pip install -r requirements.txt
# baseline for base models: GCN, GIN, GCNII
# replace dataname in each .sh
sh gcn_run.sh
sh gin_ruh.sh
sh gcnii_run.sh


# for DisC model

# DisC_GCN
sh DisC_gcn_run.sh

# DisC_GIN
sh DisC_gin_run.sh

# DisC_GCNII
sh DisC_gcnii_run.sh


# DIR
cd DIR
# install environment for DIR
pip install -r requirements.txt
# replace corresponding data name in DIR/datasets/mnistsp_dataset.py line 46 and 56. The dataset names are in /data/DIR_data 
python3 -m train.mnistsp_dir
