import datasets.transforms as T
from models.deformable_detr_debug import build
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
import matplotlib.colors as mcolors

model_name_dict = {
    "resnet50": "res-50_ddetr",
    "mobilenet": "mb-v3L_ddetr",
    "efficientnet": "effi-v2S_ddetr",
    "swin": "swin-T_ddetr"
}

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


def plot_feauture_map(img, keep, conv_features, dec_attn_weights, bboxes_scaled):
    # get the feature map shape
    h, w = conv_features['0'].tensors.shape[-2:]

    # print(dec_attn_weights.shape)
    # print(keep.shape)

    fig, axs = plt.subplots(ncols=len(bboxes_scaled), nrows=2, figsize=(22, 7))
    for idx, ax_i, (xmin, ymin, xmax, ymax) in zip(keep, axs.T.reshape(-1, 2), bboxes_scaled):
        # ax = ax_i[0]
        # ax.imshow(dec_attn_weights[0, idx].view(h, w))
        # ax.axis('off')
        # ax.set_title(f'query id: {idx.item()}')
        # ax = ax_i[1]
        # ax.imshow(img)
        # ax.add_patch(plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin,
        #                            fill=False, color='blue', linewidth=3))
        # ax.axis('off')
        # ax.set_title(CLASSES[probas[idx].argmax()])
        pass
    fig.tight_layout()
    # fig.savefig('feature_map.png')


# for output bounding box post-processing
def box_cxcywh_to_xyxy(x):
    x_c, y_c, w, h = x.unbind(1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
         (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=1)


def rescale_bboxes(out_bbox, size):
    img_w, img_h = size
    b = box_cxcywh_to_xyxy(out_bbox)
    b = b * torch.tensor([img_w, img_h, img_w, img_h],
                         dtype=torch.float32, device="cuda")
    return b.cpu()


def show_encoder_result(img, idxs, conv_features, enc_attn_weights):
    # output of the CNN
    # happy code
    temp = list(conv_features.keys())[0]  # '0' for other or '3' for swin
    f_map = conv_features[temp]
    print("Encoder attention:      ", enc_attn_weights[0].shape)
    print("Feature map:            ", f_map.tensors.shape)

    # get the HxW shape of the feature maps of the CNN
    shape = f_map.tensors.shape[-2:]
    # and reshape the self-attention to a more interpretable shape
    sattn = enc_attn_weights[0].reshape(shape + shape)
    print("Reshaped self-attention:", sattn.shape)

    # downsampling factor for the CNN, is 32 for DETR and 16 for DETR DC5
    fact = 32

    # let's select 4 reference points for visualization

    num_row = 2
    num_col = (len(idxs) + 1) // num_row

    # here we create the canvas
    fig = plt.figure(constrained_layout=True, figsize=(25 * 0.7, 8.5 * 0.7))
    # and we add one plot per reference point

    gs = fig.add_gridspec(2, 2 + num_col)

    # and now let's add the central image, with the reference points as red circles
    fcenter_ax = fig.add_subplot(gs[:, 0:2])
    fcenter_ax.imshow(img)
    for (x, y) in idxs:
        fcenter_ax.add_patch(plt.Circle(
            (x, y), fact // 2, color='r'))
        fcenter_ax.axis('off')

    axs = []
    for i in range(len(idxs)):
        r, c = i // num_col, i % num_col
        axs.append(fig.add_subplot(gs[r, 2 + c]))

    # for each one of the reference points, let's plot the self-attention
    # for that point
    for idx_o, ax in zip(idxs, axs):
        idx = (idx_o[0] // fact, idx_o[1] // fact)
        ax.imshow(sattn[..., idx[1], idx[0]],
                  cmap='cividis', interpolation='nearest')
        ax.axis('off')
        ax.set_title(f'{idx_o}')

    # fig.savefig('feature_map2.png')

    # convert the figure to a numpy array
    canvas = fig.canvas
    canvas.draw()
    w, h = canvas.get_width_height()
    img_res = np.frombuffer(canvas.tostring_rgb(),
                            dtype='uint8').reshape(h, w, 3)

    return img_res


def show_decoder_result(img, outputs, conv_features, dec_attn_weights, threshold):
    print("--------")

    probas = outputs['pred_logits'].sigmoid()[0].to('cpu')
    keep = probas.max(-1).values > threshold

    print("pred logit shape", outputs["pred_logits"].shape)
    print("probas shape", probas.shape)
    print("keep shape", keep.shape)

    # print(probas.max(-1).values)
    # print(probas.sigmoid().max(-1).values)

    # convert boxes from [0; 1] to image scales
    bboxes_scaled = rescale_bboxes(
        outputs['pred_boxes'][0, keep], img.size)
    # get the feature map shape
    print("NUM BOX", len(bboxes_scaled))

    # happy code
    temp = list(conv_features.keys())[0]  # '0' for other or '3' for swin
    h, w = conv_features[temp].tensors.shape[-2:]

    print("len box", len(bboxes_scaled))
    fig, axs = plt.subplots(
        ncols=len(bboxes_scaled), nrows=2, figsize=(22, 7))

    # Define the blue color map ranging from blue to white
    cmap = mcolors.LinearSegmentedColormap.from_list(
        'blue_scale', [(0, 'white'), (1, 'black')])

    for idx, ax_i, (xmin, ymin, xmax, ymax) in zip(keep.nonzero(), axs.T.reshape(-1, 2), bboxes_scaled):
        ax = ax_i[0]
        ax.imshow(dec_attn_weights[0, idx].view(h, w), cmap=cmap)
        ax.patch.set_edgecolor('red')  # Add black border
        ax.patch.set_linewidth(2)  # Set border width
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(f'query id: {idx.item()}')

        ax = ax_i[1]
        ax.imshow(img)
        ax.add_patch(plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin,
                                   fill=False, color='blue', linewidth=3))
        ax.axis('off')
        ax.set_title(CLASSES[probas[idx].argmax()])
    fig.tight_layout()
    # fig.savefig('feature_map.png')

    # convert the figure to a numpy array
    canvas = fig.canvas
    canvas.draw()
    w, h = canvas.get_width_height()
    img_res = np.frombuffer(canvas.tostring_rgb(),
                            dtype='uint8').reshape(h, w, 3)

    return img_res


@torch.no_grad()
def inference_one_image(model, device, img_path):
    # Read the input image using PIL (RGB) cause transform is designed for PIL
    img = Image.open(img_path).convert("RGB")

    transform = init_inference_transform()
    tensor, _ = transform(image=img, target=None)
    tensor_list = tensor.unsqueeze(0)

    # Convert to CV2 image
    # img_cv2 = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

    # use lists to store the outputs via up-values
    conv_features, enc_attn_weights, dec_attn_weights = [], [], []

    hooks = [
        model.backbone[-2].register_forward_hook(
            lambda self, input, output: conv_features.append(output)
        ),
        model.transformer.encoder.layers[-1].self_attn.register_forward_hook(
            lambda self, input, output: enc_attn_weights.append(
                output[1])
        ),
        model.transformer.decoder.layers[-1].cross_attn.register_forward_hook(
            lambda self, input, output: dec_attn_weights.append(
                output[1])
        ),
    ]

    inputs = tensor_list.to(device)
    outputs = model(inputs)

    # for hook in hooks:
    #     hook.remove()

    # don't need the list anymore
    conv_features = conv_features[0]
    enc_attn_weights = enc_attn_weights[0].to('cpu')
    dec_attn_weights = dec_attn_weights[0].to('cpu')

    # print(conv_features)
    print("enc_attn_weights", enc_attn_weights.shape)
    print("dec_attn_weights", dec_attn_weights.shape)

    # print(model.transformer.encoder.layers[-1])
    # print(model.transformer.decoder.layers[-1].cross_attn)
    idxs = [(200, 200), (280, 400), (200, 600), (440, 500),
            (200, 200)]
    show_encoder_result(img, idxs, conv_features, enc_attn_weights)
    show_decoder_result(img, outputs, conv_features,
                        dec_attn_weights, threshold=0.5)

    return 1


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
    model, _, _ = build(args)
    model.to(args.device)

    ckpt = torch.load(model_path, map_location=lambda storage, loc: storage)
    model.load_state_dict(ckpt['model'])
    model.eval()

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
