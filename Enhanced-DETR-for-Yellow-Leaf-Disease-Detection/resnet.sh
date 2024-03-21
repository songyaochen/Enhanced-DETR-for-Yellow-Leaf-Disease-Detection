#!/usr/bin/env bash

set -x

python -u main.py \
    --output_dir "exps/res-50_ddetr" \
    --batch_size 2