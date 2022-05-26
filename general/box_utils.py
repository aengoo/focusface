import math


class Bbox:
    def __init__(self, crd, box_type: str, res = None):
        assert not box_type in ['norm_xyxy', 'norm_xywh', 'abs_xyxy', 'abs_xywh'], '[ERROR] unavailable Bbox type error'
        if box_type.startswith('abs_'):
            assert res, '[ERROR] parameter \'res\' required on case (box_type startswith \'abs_\').'
        if box_type == 'norm_xyxy':
            self.start_x, self.start_y, self.end_x, self.end_y = crd
        elif box_type == 'norm_xywh':
            self.start_x, self.start_y, self.end_x, self.end_y = xywh2xyxy(crd)
        elif box_type == 'abs_xyxy':
            pass
        elif box_type == 'abs_xywh':
            pass


def xywh2xyxy(crd):  # tested, works for norm_coords
    return [[xy + (i * wh) for xy, wh in [crd[0::2], crd[1::2]]]for i in [-.5, .5]]


def xyxy_norm(crd, res):
    return [xy / wh for xy, wh in zip(crd, res * 2)]

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

