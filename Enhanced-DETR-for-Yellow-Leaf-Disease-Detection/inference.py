import datasets.transforms as T
from models import build_model
import argparse
import torch
from models.deformable_detr import PostProcess
import matplotlib.pyplot as plt
import cv2
import numpy as np
from main import get_args_parser
import random
from PIL import Image
import sys
import os
# from torchinfo import summary

model_name_dict = {
    "resnet50": "res-50_ddetr",
    "mobilenet": "mb-v3L_ddetr",
    "efficientnet": "effi-v2S_ddetr",
    "swin": "swin-T_ddetr"
}

CLASSES = [
 '1', '2', '3', '4', '5', '6'
]


def make_colors():
    seed = 2001
    random.seed(seed)
    return [[random.randint(0, 255) for _ in range(3)]
            for _ in range(len(CLASSES))]


def plot_one_box(img, box, color, label=None, line_thickness=3):
    # Plots one bounding box on image img
    tl = line_thickness
    color = color or [random.randint(0, 255) for _ in range(3)]
    c1, c2 = (box[0], box[1]), (box[2], box[3])
    cv2.rectangle(img, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
    if label:
        tf = max(tl - 1, 1)  # font thickness
        t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
        c2 = c1[0] + t_size[0], c1[1] + t_size[1] + 5
        cv2.rectangle(img, c1, c2, color, -1, cv2.LINE_AA)  # filled
        cv2.putText(img, label, (c1[0], c1[1] + t_size[1] + 2), 0, tl / 3,
                    [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)


def init_inference_transform():
    """ inspired by "from datasets.coco import make_coco_transforms" """
    normalize = T.Compose([
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    return T.Compose([
        T.RandomResize([800], max_size=1333),
        normalize,
    ])


def inference_one_image(model, device, img_path):
    # Read the input image using PIL (RGB) cause transform is designed for PIL
    img = Image.open(img_path).convert("RGB")

    transform = init_inference_transform()
    tensor, _ = transform(image=img, target=None)
    tensor_list = tensor.unsqueeze(0)

    # Convert to CV2 image
    img_cv2 = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

    inputs = tensor_list.to(device)
    raw_output = model(inputs)
    # DEBUG
    # summary(model, input_size=(1, 3, 800, 600), depth=100)

    postprocess = PostProcess()
    target_sizes = torch.tensor(
        [[img_cv2.shape[0], img_cv2.shape[1]]]).to(device)
    output = postprocess(raw_output, target_sizes)

    scores = output[0]["scores"].cpu().tolist()
    scores = [round(score, 2) for score in scores]
    labels = output[0]["labels"].cpu().tolist()
    boxes = output[0]["boxes"].int().cpu().numpy().tolist()

    colors = make_colors()

    for s, l, b in zip(scores, labels, boxes):
        if s >= 0.5:
            plot_one_box(img_cv2, box=b, color=colors[l], label=str(
                CLASSES[l]) + " " + str(s))

    return img_cv2


if __name__ == "__main__":
    # set up args parser
    parser = argparse.ArgumentParser(
        'Deformable DETR training and evaluation script', parents=[get_args_parser()])

    parser.add_argument('--folder', action='store_true',
                        help='Use this flag to process a folder of images')
    parser.add_argument('--inf_path', type=str, default='input.png',
                        help='Path to the input image or folder of images for inference')

    args = parser.parse_args()

    model_path = os.path.join(
        ".", "weight", model_name_dict[args.backbone], "checkpoint0049.pth")

    # intialize model
    model, _, _ = build_model(args)
    model.to(args.device)
    model.eval()

    ckpt = torch.load(model_path, map_location=lambda storage, loc: storage)
    model.load_state_dict(ckpt['model'])

    if args.folder:
        # Get a list of all image files in the folder
        img_files = [os.path.join(args.inf_path, f) for f in os.listdir(
            args.inf_path) if f.endswith('.png') or f.endswith('.jpeg') or f.endswith('.jpg')]

        output_folder = os.path.join(
            ".", "output_img", model_name_dict[args.backbone])
        os.makedirs(output_folder, exist_ok=True)

        # Process each image file in the folder
        for img_file in img_files:
            print(img_file)

            # Perform object detection on the image
            result_img = inference_one_image(model, args.device, img_file)

            # Save the image with predicted bounding boxes
            output_path = os.path.join(
                output_folder, os.path.basename(img_file))
            cv2.imwrite(output_path, result_img)
    else:
        # Perform object detection on a single image
        result_img = inference_one_image(model, args.device, args.inf_path)

        # Save the image with predicted bounding boxes
        output_path = "output.png"
        cv2.imwrite(output_path, result_img)
