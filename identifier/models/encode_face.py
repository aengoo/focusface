import face_recognition
import cv2
import numpy as np
import os
from itertools import combinations
import random
import copy


def random_combination(iterable, r):
    pool = tuple(iterable)
    n = len(pool)
    indices = sorted(random.sample(range(n), r))
    return tuple(pool[i] for i in indices)


class EncodeFace:
    def __init__(self, target_path, other_path, n=0, model: str = 'small', tolerance=0.6):
        self.model = model

        self.target_encodings, self.target_names = self.encode_all(target_path)
        self.other_encodings = self.encode_all(other_path)[0]

        self.n = n
        self.tolerance = tolerance
        self.temp_encodings = []
        self.temp_names = []

    def encode_all(self, path):
        face_encodings = []
        face_names = []
        face_dirs = os.listdir(path)
        for face_dir in face_dirs:
            path_face_dir = os.path.join(path, face_dir)
            # face_encodings.update({face_dir: []})
            for face_img in os.listdir(path_face_dir):
                face_names.append(face_dir)
                # img = face_recognition.load_image_file(str(os.path.join(path_face_dir, face_img)))
                img = face_recognition.load_image_file(os.path.join(path_face_dir, face_img))
                encodings = face_recognition.face_encodings(img, model=self.model)
                if len(encodings):
                    encoding = encodings[0]
                    face_encodings.append(encoding)
        return face_encodings, face_names

    def set_random_encodings(self, target_name):
        encodings = list(random_combination(self.other_encodings, self.n - 1))
        names = ['KFACE' + str(i).zfill(4) for i in range(self.n - 1)]
        self.temp_encodings = encodings + [self.target_encodings[self.target_names.index(target_name)]]
        self.temp_names = names + [target_name]

    def set_all_random_encodings(self):
        encodings = list(random_combination(self.other_encodings, len(self.target_names)))
        names = ['KFACE' + str(i).zfill(4) for i in range(len(self.target_names))]
        self.temp_encodings = encodings + self.target_encodings
        self.temp_names = names + self.target_names

    def get_embed(self, face):
        return face_recognition.face_encodings(face[:, :, ::-1], model=self.model)

    def match_face(self, face: np.ndarray, get_score=False):
        # print(face.shape)
        name = "-"
        match_distance = self.tolerance
        standard_seed = 0.  # TODO
        face_encodings = face_recognition.face_encodings(face[:, :, ::-1], model=self.model)
        if len(face_encodings):
            face_encoding = face_encodings[0]
            matches = face_recognition.compare_faces(self.temp_encodings, face_encoding, tolerance=self.tolerance)

            face_distances = face_recognition.face_distance(self.temp_encodings, face_encoding)
            # print(face_distances)
            if face_distances.shape[0]:
                best_match_index = np.argmin(face_distances)
                match_distance = face_distances[best_match_index]
                if get_score:
                    score = self.tolerance - match_distance
                    scores = self.tolerance - face_distances
                    standard_seed = (score - np.average(scores)) / np.std(scores)
                    # standard_seed.clip(-3., 3.)
                if matches[best_match_index]:
                    name = self.temp_names[best_match_index]
        if get_score:
            return name, match_distance, standard_seed
        else:
            return name, match_distance

    def get_faces_cnt(self):
        if self.n:
            return self.n
        else:
            return len(self.temp_names)

    def get_faces_names(self):
        return self.temp_names
