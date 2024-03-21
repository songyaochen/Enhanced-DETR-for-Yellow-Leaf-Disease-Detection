# Requirements
- Python version: 3.8 (tested)
- CUDA version: 11.7 (tested)
- CUDA version: 11.5 (this version maybe conflict with  build MSDeformAttn)

# Set up environment
## create conda environment

```bash
conda create -n env_name python=3.8
```

## activate conda environment
```bash
conda activate env_name
```

## install dependency
```bash
pip install -r requirements.txt
```

## install MSDeformAttn package (required)

```bash
pip install models/ops
```

# Download coco dataset
- http://images.cocodataset.org/zips/train2017.zip
- http://images.cocodataset.org/zips/val2017.zip
- http://images.cocodataset.org/annotations/annotations_trainval2017.zip

After download unzip all in **path/to/coco**\
Folder structure should look like this

```
path/to/coco/
├── train2017/
├── val2017/
└── annotations/
    ├── instances_train2017.json
    └── instances_val2017.json
```

# Training
```bash
bash train_efficientnet.sh
```

content of **train_efficientnet.sh**
```
#!/usr/bin/env bash

set -x

python -u main.py \
    --output_dir "exps/effi-v2S_ddetr" \
    --backbone "efficientnet" \
    --batch_size 2
```
>**Note**\
Use arg **--coco_path "path/to/coco"** to update coco path\
**path/to/coco** is coco dataset directory in previous step

# Download model weight
download weight folder and move it inside repo folder (same level as main.py)

https://drive.google.com/drive/folders/1F4WzkFU0agqYoJIAwQ3FMZNb7-PQbuI1?usp=drive_link



In each model there are 3 files
| File Name | Description |
| --- | --- |
| `checkpoint0049.pth` | model weight |
| `log.txt` | evaluation at each epoch on COCO train dataset |
| `train.log` | log during training |

# Inference
- Inference a image
```
python inference.py --backbone name_backbone  --inf_path path/to/image.png
```
- Inference a folder with multiple images
```
python inference.py --backbone name_backbone  --folder --inf_path path/to/folder
```
>**Note**\
replace name_backbone with one of the following:

| name_backbone | Model using |
| --- | --- |
| resnet50 | res-50_ddetr |
| efficientnet | effi-v2S_ddetr |
| mobilenet | mb-v3L_ddetr |
| swin | swin-T_ddetr |

# Inference with GUI
- Run backend using flask (This will load 4 model and 4 debug model. Config file `backend.py` if needed)
```
python backend
```
- Run fronend using streamlit
```
streamlit run frontend.py  
```

Video demo: https://www.youtube.com/watch?v=8VNTvP90f5Y
# Run bench mark on google colab
https://colab.research.google.com/drive/1rxxCjrq1dEC8e7-bGUoSTMd1CkZ4CGwR


# Explain file
## Python file
| File name | Description |
| --- | --- |
| main.py | train and evaluate model |
| inference.py | running inference on images |
| frontend.py | web UI to inference image using streamlit|
| backend.py  | host a backend using Flask |
| draw_coco.py | draw lablel of COCO train dataset|
| draw_comparision.py | combine COCO label and result of detection as an image | 
| benchmark.py| benchmark to know the FPS of a model |
| debug.py| Show attention weight of encoder and decoder |
| extract_train_time.py | extract training time of file `weight/train.log`  |
| inspect_eval.py | get AP of each class at an evaluation |
| visualize.py | plot loss and AP of the training process |

## Bash file
| File name | Description |
| --- | --- |
| benchmark_resnet.sh | example of running `benchmark.py` |   
| train_dist*.sh | train model distributively on 2 GPU |
| train_*.sh | train model |
| view.sh | view the last line of `weight/train.log` |
