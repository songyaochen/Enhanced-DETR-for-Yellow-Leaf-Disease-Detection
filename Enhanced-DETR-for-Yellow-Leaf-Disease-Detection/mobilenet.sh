#!/usr/bin/env bash

set -x


python -u main.py \
    --output_dir "exps/mb-v3L_ddetr" \
    --backbone "mobilenet" \
    --batch_size 2