import cv2
import numpy as np


# code from github/ultralytics/yolov3/utils
def plot_one_box(x, img, color=None, label: str = None, line_thickness=None):
    # Plots one bounding box on image img
    tl = line_thickness or round(0.001 * (img.shape[0] + img.shape[1]) / 2) + 1  # line/font thickness
    # color = color or [random.randint(0, 255) for _ in range(3)]
    color = color or [0, 0, 255]
    c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
    cv2.rectangle(img, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
    if label:
        tf = max(tl - 1, 1)  # font thickness
        t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
        cv2.rectangle(img, c1, c2, color, -1, cv2.LINE_AA)  # filled
        cv2.putText(img, label, (c1[0], c1[1] - 2), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)


def plot_center_text(x, img, label: str, color=None):
    tl = round(0.001 * (img.shape[0] + img.shape[1]) / 2) + 1
    tf = max(tl - 1, 1)  # font thickness
    c1, c2 = (int((x[0] + x[2]) / 2), int((x[1] + x[3]) / 2)), (int(x[2]), int(x[3]))
    cv2.putText(img, label, (c1[0], c1[1] - 2), 0, tl / 3, [0, 0, 255], thickness=tf, lineType=cv2.LINE_AA)


def plot_frame_info(img, label, color=None):
    color = color or [0, 255, 0]
    tl = round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1
    tf = max(tl - 1, 1)
    cv2.putText(img, label, (tl*3, tl*10), 0, tl / 3, color, thickness=tf, lineType=cv2.LINE_AA)


def plot_tp_box(xyxy, img, fill, color='g'):
    # color must be an integer in range 1~3 or a character in 'bgr'
    if type(color) is str:
        color_tuple = 'bgr'
        color_idx = color_tuple.find(color)
    elif color not in range(3):
        print("[WARNING] INVALID CHANNEL VALUE/TYPE WARNING FROM FUNC \'tp_rectangle\', ASSIGN IT AS DEFAULT VALUE (1)")
        color_idx = 1
    else:
        color_idx = color

    xyxy = [int(coord) for coord in xyxy]

    xx = xyxy[::2]
    yy = xyxy[1::2]

    xx = [max(min(x, img.shape[1]), 0) for x in xx]
    yy = [max(min(y, img.shape[0]), 0) for y in yy]

    x1, x2 = xx
    y1, y2 = yy
    rect = np.ones((y2 - y1, x2 - x1), dtype=np.uint8) * 255
    img[y1:y2, x1:x2, color_idx] = cv2.addWeighted(img[y1:y2, x1:x2, color_idx], fill, rect, 0.5, 1.0)
