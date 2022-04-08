import face_recognition
import numpy as np


class FaceComparer:
    def __init__(self, ebd_dict: dict, idt_model: str = 'small', tolerance=0.6):
        self.ebd_dict = ebd_dict
        self.model = idt_model
        self.tolerance = tolerance

    def compare_face(self, face_img: np.ndarray, get_score: bool = False):
        face_id = '-'
        match_distance = self.tolerance
        standard_seed = 0.

        face_embeddings = face_recognition.face_encodings(face_img[:, :, ::-1], model=self.model)
        if face_embeddings:
            face_ebd = face_embeddings[0]
            face_distances = face_recognition.face_distance(list(self.ebd_dict.values()), face_ebd)
            matches = list(face_distances <= self.tolerance)
            if face_distances.shape[0]:
                best_match_index = np.argmin(face_distances)
                match_distance = face_distances[best_match_index]
                if get_score:
                    score = self.tolerance - match_distance
                    scores = self.tolerance - face_distances
                    standard_seed = (score - np.average(scores)) / np.std(scores)
                    # standard_seed.clip(-3., 3.)
                if matches[best_match_index]:
                    face_id = list(self.ebd_dict.keys())[best_match_index]
        if get_score:
            return face_id, match_distance, standard_seed
        else:
            return face_id, match_distance
