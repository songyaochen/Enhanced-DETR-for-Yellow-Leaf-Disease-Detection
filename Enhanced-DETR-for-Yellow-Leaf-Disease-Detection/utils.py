import numpy as np
import cv2
import base64
from PIL import Image
import io


def encode_image(image, flag=False):
    """
    input: cv2 image
    output: base64 encoded image
    """
    if image is None:
        return "None"
    image = np.array(image)
    if not flag:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    _, im_arr = cv2.imencode('.jpg', image)
    im_bytes = im_arr.tobytes()
    b64_string = base64.b64encode(im_bytes)
    image_base64 = b64_string.decode("utf-8")
    return image_base64


def decode_img(img_base64):
    """
    input: base64 encoded image
    output: cv2 image
    """
    if img_base64 is None:
        return []

    img = img_base64.encode()
    img = base64.b64decode(img)
    img = np.frombuffer(img, dtype=np.uint8)
    img = cv2.imdecode(img, flags=cv2.IMREAD_COLOR)

    return img


def decode_pil_img(img_base64):
    """
    input: base64 encoded image
    output: PIL image
    """
    img = img_base64.encode()
    img = base64.b64decode(img)
    img = io.BytesIO(img)
    img = Image.open(img)
    return img


def check_valid_img(img, min_height=50, min_width=50):
    """
    input: cv2 image
    output: bool (image is valid or not)
    """
    is_numpy = isinstance(img, np.ndarray)
    is_color = img.ndim == 3
    is_uint8 = img.dtype == 'uint8'
    # is_large = img.shape[0] >= min_height and img.shape[1] >= min_width

    is_valid = is_numpy and is_color and is_uint8
    if not is_valid:
        print('is_numpy: {}, is_color: {}, is_uint8: {}'.format(
            is_numpy, is_color, is_uint8))
    return is_valid


def get_ellipse_coords(point):
    center = point
    radius = 10
    return (
        center[0] - radius,
        center[1] - radius,
        center[0] + radius,
        center[1] + radius,
    )
