#!/bin/bash

ngpu=1
save_dir="eval_results/"
global_record_file="eval_results/eval_record_collection_0514_darth.csv"
model="meta-llama/Llama-2-7b-hf"
selected_subjects="all"

cd ../../
export CUDA_VISIBLE_DEVICES=0

python evaluate_from_local.py \
                 --selected_subjects $selected_subjects \
                 --ngpu $ngpu \
                 --save_dir $save_dir \
                 --model $model \
                 --global_record_file $global_record_file




