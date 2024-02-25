#!/bin/bash
base_dir="datasets_face/test_data_demo"
# Iterate over each test image in the base directory
for folder in "$base_dir"/*; do
    img_path=$(basename "$folder")  # Extract the test image
    index="${img_path%.*}"
    CUDA_VISIBLE_DEVICES=0 accelerate launch \
        --machine_rank 0 \
        --num_machines 1 \
        --main_process_port 11135 \
        --num_processes 1 \
        --gpu_ids 0\
        train.py \
        --face_img_path=datasets_face/test_data_demo/${img_path}\
        --output_dir="experiments512/save_${index}" \
        --resolution=512 \
        --train_batch_size=1 \
        --checkpointing_steps=50 \
        --gradient_accumulation_steps=1 \
        --seed=42\
        --learning_rate=5e-5\
        --l_hair_diff_lambda=0.1     
done