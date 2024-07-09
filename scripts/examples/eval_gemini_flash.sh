#!/bin/bash

cd ../../

python evaluate_from_api.py \
                 --model_name gemini-1.5-flash-latest \
                 --output_dir eval_results \
                 --assigned_subjects all
