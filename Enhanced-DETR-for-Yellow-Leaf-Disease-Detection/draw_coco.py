from pycocotools.coco import COCO
from inference import plot_one_box, make_colors, CLASSES
import cv2
import numpy as np
import os
import time


# define the path to the COCO JSON file
ann_file = '../coco-dataset/annotations/instances_val2017.json'

# create a COCO object to load the annotations
coco = COCO(ann_file)

for i in coco.imgToAnns:
    print(i)
    break


colors = make_colors()
# Convert to CV2 image

coco_path = '../coco-dataset/val2017'

img_files = [os.path.join(coco_path, f) for f in os.listdir(
    coco_path) if f.endswith('.png') or f.endswith('.jpeg') or f.endswith('.jpg')]

for img_file in img_files:
    # img_file = "../coco-dataset/val2017/000000454661.jpg"
    img = cv2.imread(img_file)
    file_name = os.path.basename(img_file)
    file_name_without_extension = os.path.splitext(file_name)[0]

    img_id = int(file_name_without_extension)

    ann_ids = coco.getAnnIds(imgIds=img_id, iscrowd=False)
    anns = coco.loadAnns(ann_ids)
    for ann in anns:
        x1, y1, w, h = ann["bbox"]
        x2, y2 = x1 + w, y1 + h
        b = (x1, y1, x2, y2)
        b = tuple(int(x) for x in b)
        l = ann["category_id"]
        plot_one_box(img, box=b, color=colors[l], label=str(CLASSES[l]))

    output_folder = os.path.join(
        ".", "output_img", "coco_val")
    os.makedirs(output_folder, exist_ok=True)

    output_path = os.path.join(output_folder, file_name)

    cv2.imwrite(output_path, img)

# print(img.size)
# img_cv2 = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)


# for ann in annotations:
#     l = ann["category_id"]
#     b = ann["bbox"]
#     plot_one_box(img_cv2, box=b, color=colors[l], label=str(CLASSES[l]))

# output_path = "coco_output.png"
# cv2.imwrite(output_path, img_cv2)
