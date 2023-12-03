GPU=$1
seed0=31
seed1=32
seed2=33
seed3=34
code=main_imp.py 

dataset=MNIST_75sp_0.8 # MNIST_75sp_0.9 MNIST_75sp_0.95 fashion_0.8 fashion_0.9 fashion_0.95 kuzu_0.8 kuzu_0.9 kuzu_0.95
all_epochs=200
str1="output_GIN_"$dataset
tmux new -s $str1 -d
tmux send-keys "source activate benchmark_gnn" C-m #replace benchmark_gnn with your environment name

tmux send-keys "
CUDA_VISIBLE_DEVICES=0 \
python -u $code --config 'configs/superpixels_graph_classification_GIN_MNIST_100k.json' \
--dataset $dataset \
--seed $seed0  \
--mask_epochs $all_epochs \
--out_dir $str1 &

CUDA_VISIBLE_DEVICES=1 \
python -u $code --config 'configs/superpixels_graph_classification_GIN_MNIST_100k.json' \
--dataset $dataset \
--seed $seed1  \
--mask_epochs $all_epochs \
--out_dir $str1 &

CUDA_VISIBLE_DEVICES=2 \
python -u $code --config 'configs/superpixels_graph_classification_GIN_MNIST_100k.json' \
--dataset $dataset \
--seed $seed2  \
--mask_epochs $all_epochs \
--out_dir $str1 &
 
CUDA_VISIBLE_DEVICES=3 \
python -u $code --config 'configs/superpixels_graph_classification_GIN_MNIST_100k.json' \
--dataset $dataset \
--seed $seed3  \
--mask_epochs $all_epochs \
--out_dir $str1 &
 

wait" C-m
