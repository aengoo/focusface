import math


def box_adapt(box: list, rat=1.):
    w, h = (box[2] - box[0], box[3] - box[1])
    if rat != 1.:
        pad = [- ((w * rat) - w) / 2, - ((h * rat) - h) / 2, ((w * rat) - w) / 2, ((h * rat) - h) / 2]
        new_box = [(xy + pad[idx]) for idx, xy in enumerate(box[:4])] + box[4:]
    else:
        new_box = [xy for idx, xy in enumerate(box[:4])] + box[4:]
    return new_box


def get_box_diagonal(xyxy):
    w, h = (xyxy[2] - xyxy[0], xyxy[3] - xyxy[1])
    return math.sqrt((w**2) + (h**2))

