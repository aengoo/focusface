from .models import EmbeddingLoader, FaceComparer
import numpy as np
import cv2

AVAILABLE_RESOLUTIONS = {
    'FHD': (1080, 1920, 3),
    'HD': (720, 1280, 2),
    'sHD': (360, 640, 1),
    'VGA': (480, 640, 1),
}


# 탐지 박스 크기 비율을 조정합니다.
def box_adapt(boxes: np.ndarray, res: tuple, rat=1.):
    res_boxes = boxes[:, :4] * np.array(res[:2][::-1] + res[:2][::-1])
    boxes_shape = res_boxes[:, 2:] - res_boxes[:, :2]
    boxes_center = (boxes_shape / 2) + res_boxes[:, :2]
    pad_boxes_rad = (boxes_shape * rat if rat != 1. else boxes_shape) / 2.
    pad_res_boxes = np.concatenate((boxes_center - pad_boxes_rad, boxes_center + pad_boxes_rad, boxes[:, 4:]),
                                   axis=1, dtype='float32')
    return pad_res_boxes


class Identifier:
    """
    주어진 얼굴 영역 영상을 식별하는 객체입니다.
    동작에 관한 자세한 설명은 각 클래스 함수 주석을 참고해주시기 바랍니다.
    """
    def __init__(self, embed_db_path: str, n: int, idt_res: str, box_ratio: float, model: str = 'small'):
        """
        얼굴 식별을 위한 각 모듈을 미리 로드합니다.
        EmbeddingLoader는 DB에서 미리 추출된 얼굴 특징 임베딩을 로드합니다.
        추출되어 있는 임베딩이 존재하지 않으면 생성하는 작업을 수행합니다.
        FaceComparer는 입력되는 매 프레임에서 입력되는 얼굴 프레임이 DB의 얼굴 특징과 비교하여 결론을 도출합니다.
        얼굴 특징 추출을 위한 인코더가 로드됩니다.
        입력 영상의 지나친 해상도는 매우 큰 지연시간을 야기하기 때문에 최대 해상도 제한을 둡니다.

        :param embed_db_path:  임베딩 DB CSV 파일 경로
        :param n: n_faces 옵션, 허수를 포함한 총 DB 사이즈를 결정합니다.
        :param idt_res: 식별시 활용할 프레임 해상도를 결정합니다.
        :param box_ratio: 탐지 박스 확장 면적 비율을 결정합니다. 1.3이 가장 적절한 것으로 실험적으로 확인했습니다.
        :param model: small or large, 얼굴 인코더의 크기입니다. 랜드마크의 갯수와 CNN 모델 사이즈에 차이가 있습니다. 성능에 큰 영향을 미치지 않습니다.
        """

        # 임베딩 벡터 DB를 로드합니다.
        self._ebd_loader = EmbeddingLoader(embed_db_path, idt_model=model, n_faces=n)

        self.ebd_dict = self._ebd_loader.ebd_dict
        self.comparer = FaceComparer(self.ebd_dict, idt_model=model)

        self.faces = []
        self.res = AVAILABLE_RESOLUTIONS[idt_res]
        self.box_ratio = box_ratio

    def run(self, img_idt, boxes):
        """
        단일 얼굴 영역 이미지를 식별합니다.
        :param img_idt: 얼굴 영역 이미지
        :param boxes: 탐지 박스 좌표로 구성된 iterable 객체
        :return: 식별 결과를 반환합니다.
        """
        idt_boxes = []
        adapted_boxes = box_adapt(boxes, self.res, self.box_ratio)
        for tmp_box in adapted_boxes:
            # 좌표형식 xyxy
            tid = -1  # tracking ID

            if len(tmp_box) > 5:
                tid = int(tmp_box[5])  # 입력받은 탐지 박스 길이가 5 이상이면 인덱스 5에 tracking ID가 포함된 것으로 간주합니다.

            box = [int(b) for b in tmp_box[:4]]  # 박스 좌표를 정수형으로 변환합니다.
            score = tmp_box[4]

            if box[1] < 0:
                box[1] = 0
            if box[0] < 0:
                box[0] = 0

            cropped = img_idt[box[1]:box[3], box[0]:box[2]]

            rs_x = 80
            # 얼굴 영역 이미지 사이즈를 약간 조정합니다. 최대 크기를 80으로 조정합니다.
            # 해당 얼굴과 DB를 대조하여 가장 유사항 얼굴 ID, 벡터거리, 표준점수를 반환받습니다.
            face_id, face_dist, face_std_score = self.comparer.compare_face(
                cv2.resize(cropped, dsize=(rs_x, int(rs_x*((box[3]-box[1])/(box[2]-box[0])))))
                if box[2]-box[0] > rs_x else cropped, get_score=True)
            idt_boxes.append([box, score, tid, face_id, face_dist, face_std_score])  # 식별 결과 반환값 목록
        return idt_boxes

    def get_df(self):
        """

        :return:
        """
        return self._ebd_loader.df
