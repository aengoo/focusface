import face_recognition
import numpy as np


class FaceComparer:
    """
    포착된 얼굴 임베딩 벡터를 DB와 비교하여 벡터간 거리와 그 표준점수를 측정합니다.
    """
    def __init__(self, ebd_dict: dict, idt_model: str = 'small', tolerance=0.6):
        """
        얼굴 비교 객체를 초기화합니다.
        :param ebd_dict: DB로부터 읽어들인 임베딩 데이터를 입력받습니다.
        :param idt_model: small or large, 얼굴 인코더의 크기입니다. 랜드마크의 갯수와 CNN 모델 사이즈에 차이가 있습니다. 성능에 큰 영향을 미치지 않습니다.
        :param tolerance: 얼굴 임베딩 벡터 간의 거리가 이 torlerance 이상인 경우 걸러냅니다.
        """
        self.ebd_dict = ebd_dict
        self.model = idt_model
        self.tolerance = tolerance

    def compare_face(self, face_img: np.ndarray, get_score: bool = False):
        """
        입력된 얼굴 이미지로부터 임베딩 벡터를 추출하고, 이를 DB와 대조하여 벡터간 거리와 그 표준 점수를 측정하여 반환합니다.
        :param face_img: 얼굴 영역 이미지를 입력받습니다.
        :param get_score: 표준 점수를 반환할지 결정합니다.
        :return: 가장 가까운 얼굴의 ID와 그 얼굴과의 벡터 거리를 반환합니다. 선택적으로 표준점수를 반환합니다.
        """
        face_id = '-'  # 얼굴 ID 초기화, 식별 불가능 할 경우 이 문자로 표기됩니다.
        match_distance = self.tolerance
        standard_seed = 0.  # 표준 점수 시드 초기화

        """
        OPEN_CV는 기본 세팅에서 BGR 순으로 이미지를 읽습니다.
        face_recognition 라이브러리는 이를 역순으로 뒤집어 RGB로 읽습니다.
        이 경우에는 이미 읽은 이미지의 일부 영역이므로 뒤집어 주었습니다.
        추후 수정시 염두해야 합니다.
        """
        face_embeddings = face_recognition.face_encodings(face_img[:, :, ::-1], model=self.model)

        if face_embeddings:  # 입력된 얼굴영역 임베딩이 추출되었으면..
            face_ebd = face_embeddings[0]  # 무조건 리스트에 싸여 반환되므로, 0인덱싱하여 뽑아냅니다.
            # DB의 벡터와 유클리드 거리를 계산합니다. 이 결과 배열의 길이는 n_faces와 같습니다.
            face_distances = face_recognition.face_distance(list(self.ebd_dict.values()), face_ebd)

            # tolerance에 따라 너무 거리가 먼 것들을 걸러냅니다.
            matches = list(face_distances <= self.tolerance)
            if face_distances.shape[0]:
                best_match_index = np.argmin(face_distances)  # 제일 가까운 임베딩 벡터를 선정해둡니다.
                match_distance = face_distances[best_match_index]  # 제일 가까운 임베딩 벡터와의 거리를 측정합니다.
                if get_score:  # 표준 점수를 구해야 하면,
                    # 표준 점수 계산 과정
                    score = self.tolerance - match_distance
                    scores = self.tolerance - face_distances
                    standard_seed = (score - np.average(scores)) / np.std(scores)
                if matches[best_match_index]:
                    face_id = list(self.ebd_dict.keys())[best_match_index]  # 가장 가까운 벡터의 얼굴 ID를 확인합니다.
        if get_score:
            return face_id, match_distance, standard_seed
        else:
            return face_id, match_distance
