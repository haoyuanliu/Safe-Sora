export CUDA_VISIBLE_DEVICES=1

GPU_COUNT=$(echo $CUDA_VISIBLE_DEVICES | awk -F',' '{print NF}')

if [ "$GPU_COUNT" -gt 1 ]; then
    echo "Using DDP training with GPUs: $CUDA_VISIBLE_DEVICES"
    torchrun \
    --nproc_per_node=$GPU_COUNT \
    --master_port=12357 \
    train_qipange_onlybit_conv_without_wm_loss.py --use_ddp
else
    echo "Using single-GPU training with GPU: $CUDA_VISIBLE_DEVICES"
    python train_qipange_onlybit_conv_without_wm_loss.py
fi
