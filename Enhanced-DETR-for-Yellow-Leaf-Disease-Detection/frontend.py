
import inspect
import os
import sys

current_dir = os.path.dirname(os.path.abspath(
    inspect.getfile(inspect.currentframe())))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

import base64
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import cv2
from io import StringIO
import json
import requests
import streamlit as st
from utils import decode_img, encode_image
from utils import get_ellipse_coords
from streamlit_image_coordinates import streamlit_image_coordinates
import torchvision.transforms.functional as F
from datasets.transforms import get_size


st.set_page_config(layout="wide")

st.markdown("<h1 style='text-align: center; color: red;'>OBJECT DETECTION WITH TRANSFORMER</h1>",
            unsafe_allow_html=True)

if 'button' not in st.session_state:
    st.session_state["button"] = False

if "points" not in st.session_state:
    st.session_state["points"] = []


def reset_point():
    st.session_state["points"] = []


with st.sidebar:
    threshold = st.slider('Confidence threshold', 0.0, 1.0, 0.5)
    add_radio = st.radio(label="Choose model",
                         options=("res-50_ddetr",
                                  "mb-v3L_ddetr",
                                  "effi-v2S_ddetr",
                                  "swin-T_ddetr",
                                  "all models"))
    uploaded_file = st.file_uploader("Choose image file.", type=[
        'png', 'jpg', 'jpeg', 'PNG', 'JPEG'], accept_multiple_files=False, on_change=reset_point)


font = ImageFont.truetype("arial.ttf", 24)


def recognize(url="http://192.168.1.4:4500/app/detect", body={}):
    step_by_step = False
    if add_radio == "res-50_ddetr":
        st.markdown("""<p style="font-size:20px">resnet_50_deformable_detr</p>""",
                    unsafe_allow_html=True)
    elif add_radio == "mb-v3L_ddetr":
        st.markdown("""<p style="font-size:20px">mobilenet_v3_deformable_detr</p>""",
                    unsafe_allow_html=True)
    elif add_radio == "effi-v2S_ddetr":
        st.markdown("""<p style="font-size:20px">efficientnet_v2s_deformable_detr</p>""",
                    unsafe_allow_html=True)
    elif add_radio == "swin-T_ddetr":
        st.markdown("""<p style="font-size:20px">swin_transformer_t_deformable_detr</p>""",
                    unsafe_allow_html=True)
    elif add_radio == "all models":
        st.markdown("""<p style="font-size:20px">all models</p>""",
                    unsafe_allow_html=True)

    if add_radio == "all models":
        step_by_step = st.checkbox("Step by step")

    st.markdown(
        """<hr style="height:5px;border:none;color:#345678;background-color:#345678;" /> """, unsafe_allow_html=True)
    col1, col2 = st.columns(2)

    ready_to_recognize = True
    image_base64 = []
    global uploaded_file
    if uploaded_file != None:
        image_pil = Image.open(uploaded_file)
        size = get_size(image_pil.size, 800, 1333)
        image_pil = F.resize(image_pil, size)

        image = np.array(image_pil)
        h, w, _ = image.shape
        scale_factor = 800 / w
        h_show = int(h * scale_factor)
        w_show = int(w * scale_factor)

        # output
        with col2:
            output_title_placeholder = st.empty()
            output_placeholder = st.empty()

        if not step_by_step:
            with col1:
                st.markdown("**Input:**")
                st.image(image, width=w_show)
                st.write("image shape", image.shape)

        else:
            ready_to_recognize = False
            with col1:
                st.markdown("**Input:**")
                draw = ImageDraw.Draw(image_pil)
                value = streamlit_image_coordinates(
                    image_pil, key=None, width=w_show, height=h_show)
                st.write("image shape", image.shape)

                if value is not None:
                    point = int(
                        value["x"] // scale_factor), int(value["y"] // scale_factor)
                    if point not in st.session_state["points"]:
                        st.session_state["points"].append(point)

                # Draw an ellipse at each coordinate in points
                for point in st.session_state["points"]:
                    coords = get_ellipse_coords(point)
                    draw.ellipse(coords, fill="red")
                    text_coords = (coords[0] - 10, coords[1] - 35)
                    # draw the text
                    draw.text(text_coords, str(point), fill="red", font=font)
            with col2:
                st.markdown(
                    "**Click on input image to choose attention point:**")
                image_placeholder = st.empty()
                image_placeholder.image(image_pil, width=w_show)

                col2_1, col2_2 = st.columns(2)
                with col2_1:
                    finish_button = st.button("Finish select point")
                with col2_2:
                    clear_button = st.button("Clear attention points")

                if finish_button:
                    ready_to_recognize = True

                if clear_button:
                    st.session_state["points"] = []
                    image_placeholder.image(image, width=w_show)

                st.markdown(st.session_state["points"])

        image_base64 = encode_image(image)

        if ready_to_recognize:
            body = {
                "document": image_base64,
                "threshold": threshold,
                "model_name": add_radio,
                "step_by_step": step_by_step,
                "points": list(st.session_state["points"])
            }
            body = json.dumps(body)
            data = requests.post(url, data=body)
            uploaded_file = None
            return data.text, output_title_placeholder, output_placeholder
    return None, None, None


if add_radio in ["res-50_ddetr", "mb-v3L_ddetr", "effi-v2S_ddetr", "swin-T_ddetr", "all models"]:
    data, output_title_placeholder, output_placeholder = recognize()
    if data is not None:
        data = json.loads(data)

        if data["step_by_step"]:

            st.markdown(
                """<hr style="height:5px;border:none;color:#345678;background-color:#345678;" /> """, unsafe_allow_html=True)
            st.markdown("**Encoder:**")

            encoder = data["attention"]["encoder"]
            for k, v in encoder.items():
                if k == "swin-T_ddetr":
                    continue

                st.text(k)
                st.image(decode_img(v))

            st.markdown(
                """<hr style="height:5px;border:none;color:#345678;background-color:#345678;" /> """, unsafe_allow_html=True)
            st.markdown("**Decoder:**")

            decoder = data["attention"]["decoder"]
            for k, v in decoder.items():
                if k == "swin-T_ddetr":
                    continue

                st.text(k)
                st.image(decode_img(v))

        if len(data['result']) > 0:
            result = decode_img(data['result'])
            if add_radio == "all models":
                st.markdown(
                    """<hr style="height:5px;border:none;color:#345678;background-color:#345678;" /> """, unsafe_allow_html=True)
                st.markdown("**Output:**")
                st.image(result, width=900)
            else:
                output_title_placeholder.markdown("**Output:**")
                output_placeholder.image(result, width=800)

        return_json = {}
        return_json["message"] = data['message']
        return_json["run time"] = data["run_time"]

        if "result_json" in data.keys():
            return_json["result json"] = data["result_json"]

        st.code(json.dumps(return_json, indent=2))
