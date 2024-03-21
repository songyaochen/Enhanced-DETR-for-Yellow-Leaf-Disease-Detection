import cv2
import os
import time

BORDER_SIZE = 50
TEXT_COLOR = (0, 0, 0)  # Black color
TEXT_FONT = cv2.FONT_HERSHEY_SIMPLEX
TEXT_SCALE = 1
TEXT_THICKNESS = 2
TEXT_PADDING = 10  # Padding around the text


def write_text_to_image_comparision(dict_text_image: dict):
    """
    "caption of image": img,...
    visualize: [[0,1],
                [2,3]]
    """

    for idx, text in enumerate(dict_text_image.keys()):
        text_size = cv2.getTextSize(
            text, TEXT_FONT, TEXT_SCALE, TEXT_THICKNESS)[0]
        # Center the text horizontally
        text_x = int((dict_text_image[text].shape[1] - text_size[0]) / 2)
        if idx < 2:
            dict_text_image[text] = cv2.copyMakeBorder(dict_text_image[text], BORDER_SIZE, 0, 0, 0,
                                                       cv2.BORDER_CONSTANT, value=(255, 255, 255))
            text_y = BORDER_SIZE - TEXT_PADDING
        else:
            dict_text_image[text] = cv2.copyMakeBorder(dict_text_image[text], 0, BORDER_SIZE, 0, 0,
                                                       cv2.BORDER_CONSTANT, value=(255, 255, 255))
            text_y = dict_text_image[text].shape[0] - \
                BORDER_SIZE + text_size[1] + TEXT_PADDING

        cv2.putText(dict_text_image[text], text, (text_x, text_y), TEXT_FONT,
                    TEXT_SCALE, TEXT_COLOR, TEXT_THICKNESS)


if __name__ == "__main__":
    coco_path = './output_img/coco_val'

    img_files = [os.path.join(coco_path, f) for f in os.listdir(
        coco_path) if f.endswith('.png') or f.endswith('.jpeg') or f.endswith('.jpg')]

    for img_file in img_files:

        # write text to coco image
        img_coco = cv2.imread(img_file)
        img_coco = cv2.resize(
            img_coco, (img_coco.shape[1] * 2, img_coco.shape[0] * 2))
        img_coco = cv2.copyMakeBorder(img_coco, BORDER_SIZE, BORDER_SIZE, 0, 0,
                                      cv2.BORDER_CONSTANT, value=(255, 255, 255))

        text = "coco_val"
        text_size = cv2.getTextSize(
            text, TEXT_FONT, TEXT_SCALE, TEXT_THICKNESS)[0]
        # Center the text horizontally
        text_x = int((img_coco.shape[1] - text_size[0]) / 2)
        text_y = BORDER_SIZE - TEXT_PADDING
        cv2.putText(img_coco, text, (text_x, text_y), TEXT_FONT,
                    TEXT_SCALE, TEXT_COLOR, TEXT_THICKNESS)

        # write text to other images
        dict_text_image = {"res-50_ddetr": None,
                           "mb-v3L_ddetr": None,
                           "effi-v2S_ddetr": None,
                           "swin-T_ddetr": None}

        # read image
        for k in dict_text_image.keys():
            dict_text_image[k] = cv2.imread(img_file.replace('coco_val', k))
        write_text_to_image_comparision(dict_text_image)

        img1, img2, img3, img4 = dict_text_image.values()
        left_col = cv2.vconcat([img1, img3])
        right_col = cv2.vconcat([img2, img4])
        result = cv2.hconcat([img_coco, left_col, right_col])

        # write output image
        file_name = os.path.basename(img_file)
        print(file_name)
        output_folder = os.path.join(
            ".", "output_img", "comparision")
        os.makedirs(output_folder, exist_ok=True)

        output_path = os.path.join(output_folder, file_name)

        cv2.imwrite(output_path, result)
