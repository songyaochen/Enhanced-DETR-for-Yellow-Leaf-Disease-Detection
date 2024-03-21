# import cv2
# import json
# import base64
import numpy as np
from flask import Flask, request
from flask_cors import CORS, cross_origin
# from src.status_code import STATUS_CODE
from utils import decode_pil_img, encode_image
import argparse
from main import get_args_parser
import os
from models import build_model
from models.deformable_detr_debug import build as build_model_debug
import torch
from inference import init_inference_transform, plot_one_box, make_colors
from models.deformable_detr import PostProcess
import cv2
import random
from draw_comparision import write_text_to_image_comparision
from debug import show_encoder_result, show_decoder_result
import time

CLASSES = [
    'N/A', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 'stop sign',
    'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A', 'N/A',
    'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
    'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
    'bottle', 'N/A', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
    'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
    'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table',
    'N/A', 'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
    'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A', 'book',
    'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]


def load_model(debug=False):
    model_dict = {
        "res-50_ddetr": "resnet50",
        "mb-v3L_ddetr": "mobilenet",
        "effi-v2S_ddetr": "efficientnet",
        "swin-T_ddetr": "swin"
    }

    # if debug:
    #     del model_dict["swin-T_ddetr"]

    parser = argparse.ArgumentParser(
        'Deformable DETR training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()

    for k in model_dict.keys():
        backbone = model_dict[k]
        model_path = os.path.join(
            ".", "weight", k, "checkpoint0049.pth")

        # intialize model
        args.backbone = backbone
        if debug:
            model, _, _ = build_model_debug(args)
            hooks = [
                model.backbone[-2].register_forward_hook(
                    lambda self, input, output: (
                        conv_features.append(output)
                    )
                ),
                model.transformer.encoder.layers[-1].self_attn.register_forward_hook(
                    lambda self, input, output: (
                        enc_attn_weights.append(output[1])
                    )
                ),
                model.transformer.decoder.layers[-1].cross_attn.register_forward_hook(
                    lambda self, input, output: (
                        dec_attn_weights.append(output[1])
                    )
                ),
            ]
        else:
            model, _, _ = build_model(args)

        model.to(args.device)
        model.eval()
        ckpt = torch.load(
            model_path, map_location=lambda storage, loc: storage)
        model.load_state_dict(ckpt['model'])

        model_dict[k] = model
    return model_dict


conv_features, enc_attn_weights, dec_attn_weights = [], [], []
model_dict = load_model()
model_debug_dict = load_model(debug=True)
transform = init_inference_transform()
postprocess = PostProcess()
colors = make_colors()
device = "cuda"


app = Flask(__name__)
cors = CORS(app, resources={r"/app/*": {"origins": "*"}})


def draw_image(img_pil, raw_output, threshold):
    img_cv2 = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
    target_sizes = torch.tensor(
        [[img_cv2.shape[0], img_cv2.shape[1]]]).to(device)
    output = postprocess(raw_output, target_sizes)

    scores = output[0]["scores"].cpu().tolist()
    scores = [round(score, 2) for score in scores]
    labels = output[0]["labels"].cpu().tolist()
    boxes = output[0]["boxes"].int().cpu().numpy().tolist()

    list_json = []
    for s, l, b in zip(scores, labels, boxes):
        if s >= threshold:
            plot_one_box(img_cv2, box=b, color=colors[l], label=str(
                CLASSES[l]) + " " + str(s))
            list_json.append({"label": CLASSES[l], "score": s, "box": b})

    return img_cv2, list_json


@torch.no_grad()
def inference(DOC_BYTE, model_name, threshold):
    img_pil = decode_pil_img(DOC_BYTE)
    tensor, _ = transform(image=img_pil, target=None)
    print(tensor.shape)
    tensor_list = tensor.unsqueeze(0).to(device)

    start_time = time.time()
    raw_output = model_dict[model_name](tensor_list)
    end_time = time.time()
    elapsed_time = end_time - start_time
    img_cv2, list_json = draw_image(img_pil, raw_output, threshold)

    return img_cv2, str(round(elapsed_time, 2)) + " secconds", list_json


@torch.no_grad()
def inference_with_debug(DOC_BYTE, idxs, model_name, threshold):
    img_pil = decode_pil_img(DOC_BYTE)
    tensor, _ = transform(image=img_pil, target=None)
    tensor_list = tensor.unsqueeze(0).to(device)

    global conv_features, enc_attn_weights, dec_attn_weights
    conv_features, enc_attn_weights, dec_attn_weights = [], [], []

    raw_output = model_debug_dict[model_name](tensor_list)

    print("+" * 10)
    conv_features = conv_features[0]
    print(conv_features.keys())
    enc_attn_weights = enc_attn_weights[0].to('cpu')
    dec_attn_weights = dec_attn_weights[0].to('cpu')

    img_cv2, _ = draw_image(img_pil, raw_output, threshold)

    if model_name == "swin-T_ddetr":
        return img_cv2, None, None

    img_encoder = show_encoder_result(
        img_pil, idxs, conv_features, enc_attn_weights)
    img_decoder = show_decoder_result(
        img_pil, raw_output, conv_features, dec_attn_weights, threshold)

    return img_cv2, img_encoder, img_decoder


@app.route("/app/detect", methods=["POST"])
@cross_origin(supports_credentials=True)
def detect():
    try:
        message = "success"
        return_DOC = []
        attention = {"encoder": {},
                     "decoder": {}}
        request_data = request.get_json(force=True)
        model_name = request_data.get("model_name")
        threshold = request_data.get("threshold")
        points = request_data.get("points")
        DOC_BYTE = request_data.get("document")
        step_by_step = request_data.get("step_by_step")
        runtime = {}
        result = {}
        result_json = []
        if model_name == "all models":
            if step_by_step:
                for k in model_debug_dict.keys():
                    img_cv2, img_encoder, img_decoder = inference_with_debug(
                        DOC_BYTE, points, k, threshold)
                    result[k] = img_cv2
                    attention["encoder"][k] = encode_image(
                        img_encoder, flag=True)
                    attention["decoder"][k] = encode_image(
                        img_decoder, flag=True)

            else:
                for k in model_dict.keys():
                    result[k], runtime[k], _ = inference(
                        DOC_BYTE, k, threshold)

            write_text_to_image_comparision(result)
            img1, img2, img3, img4 = result.values()
            left_col = cv2.vconcat([img1, img3])
            right_col = cv2.vconcat([img2, img4])
            result = cv2.hconcat([left_col, right_col])
        else:
            result, runtime, result_json = inference(
                DOC_BYTE, model_name, threshold)
        return_DOC = encode_image(result)
    # except Exception as e:
    #     message = "Internal Server Error"
    #     print(e)
    finally:
        # Generate response data
        response_data = {
            "message": message,
            "attention": attention,
            "result": return_DOC,
            "step_by_step": step_by_step,
            "run_time": runtime
        }
        if result_json:
            response_data["result_json"] = result_json

    return response_data


if __name__ == "__main__":
    # warm up
    with torch.no_grad():
        for model in model_dict.values():
            tensor = torch.randn(1, 3, 800, 800).to(device)
            model(tensor)
    # Start server
    app.run(debug=False, host="0.0.0.0", port=4500)
