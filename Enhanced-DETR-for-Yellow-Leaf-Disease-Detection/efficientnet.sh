#!/usr/bin/env bash

set -x

python -u main.py \
    --output_dir "exps/effi-v2S_ddetr" \
    --backbone "efficientnet" \
    --batch_size 2