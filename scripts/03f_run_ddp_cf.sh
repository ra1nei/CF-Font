set -x

size=64
item=100
k=240
basis_n=10
data=/kaggle/working/CF-Font/my_data/train

my_load_model="/kaggle/input/cf-font-ckpt/model_100.ckpt"

model_name="CF_finetune_from_model_100"

base_idxs="basis/PATH_TO_YOUR_BASIS_ID.txt" 
base_ws="basis/PATH_TO_YOUR_BASIS_WS.pth"

# -----------------------------

# Chú ý: Nếu bạn chỉ có 1 GPU (Kaggle thường là 1), hãy sửa nproc_per_node=1
CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch \
    --nproc_per_node=1 --use_env main.py \
    --content_fusion \
    --img_size ${size} \
    --data_path ${data} \
    --lr 1e-4 \
    --output_k ${k} \
    --batch_size 16 \
    --iters 1000 \
    --epoch 200 \
    --val_num 10 \
    --baseline_idx 0 \
    --save_path output/models \
    --load_model ${my_load_model} \
    --base_idxs ${base_idxs} --base_ws ${base_ws} \
    --ddp \
    --no_val \
    --wdl --w_wdl 0.01