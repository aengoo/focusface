from .models import EmbeddingLoader, FaceComparer
from .utils.plot_utils import *

AVAILABLE_RESOLUTIONS = {
    'FHD': (1080, 1920, 3),
    'HD': (720, 1280, 2),
    'sHD': (360, 640, 1),
    'VGA': (480, 640, 1),
}


def box_adapt(boxes: np.ndarray, res: tuple, rat=1.):
    res_boxes = boxes[:, :4] * np.array(res[:2][::-1] + res[:2][::-1])
    boxes_shape = res_boxes[:, 2:] - res_boxes[:, :2]
    boxes_center = (boxes_shape / 2) + res_boxes[:, :2]
    pad_boxes_rad = (boxes_shape * rat if rat != 1. else boxes_shape) / 2.
    pad_res_boxes = np.concatenate((boxes_center - pad_boxes_rad, boxes_center + pad_boxes_rad, boxes[:, 4:]), axis=1, dtype='float32')
    return pad_res_boxes


class Identifier:
    def __init__(self, embed_db_path: str, n: int, idt_res: str, box_ratio: float, is_eval=False, model: str = 'small'):
        self._ebd_loader = EmbeddingLoader(embed_db_path, idt_model=model, n_faces=n)
        self.ebd_dict = self._ebd_loader.ebd_dict
        self.comparer = FaceComparer(self.ebd_dict, idt_model=model)

        self.faces = []
        self.res = AVAILABLE_RESOLUTIONS[idt_res]

        self.box_ratio = box_ratio
        self.is_eval = is_eval

    def run(self, img_idt, boxes):
        idt_boxes = []
        adapted_boxes = box_adapt(boxes, self.res, self.box_ratio)
        for tmp_box in adapted_boxes:
            # 좌표형식 xyxy
            tid = -1

            if len(tmp_box) > 5:
                tid = int(tmp_box[5])

            box = [int(b) for b in tmp_box[:4]]
            score = tmp_box[4]

            if box[1] < 0:
                box[1] = 0
            if box[0] < 0:
                box[0] = 0

            cropped = img_idt[box[1]:box[3], box[0]:box[2]]

            rs_x = 80


            face_id, face_dist, face_std_score = self.comparer.compare_face(cv2.resize(cropped, dsize=(rs_x, int(rs_x*((box[3]-box[1])/(box[2]-box[0]))))) if box[2]-box[0] > rs_x else cropped, get_score=True)
            # face_name, face_dist, face_std_score = self.encoder.match_face(cropped, get_score=True)
            # TODO: 인코딩 과정에서 해상도가 너무 크면 부하가 너무 크게 걸려서..
            idt_boxes.append([box, score, tid, face_id, face_dist, face_std_score])

        if self.is_eval:
            return idt_boxes

        else:
            # tracking not applied
            [plot_center_text(box[0], img_idt, label=format(box[1], '.4f')) for box in idt_boxes]
            [plot_one_box(box[0], img_idt, label=str(box[3])) for box in idt_boxes]
            return img_idt

    def get_df(self):
        return self._ebd_loader.df
