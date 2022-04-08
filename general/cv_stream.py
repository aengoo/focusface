from abc import *
import cv2
import numpy as np


class StreamOut(metaclass=ABCMeta):
    @abstractmethod
    def push_info(self, info: dict):
        pass

    @abstractmethod
    def push_frame(self, img: np.ndarray):
        pass

    @abstractmethod
    def push_face(self, img: np.ndarray):
        pass


class CvStreamOut(StreamOut):
    def __init__(self):
        super().__init__()

    def push_info(self, info: dict):
        print(f'[IDENTIFIED] {info["NAME"]}({info["ID"]}) is identified.')

    def push_frame(self, img: np.ndarray):
        cv2.imshow('test', img)
        if cv2.waitKey(1) == ord('q'):
            raise StopIteration

    def push_face(self, img: np.ndarray):
        pass


if __name__ == "__main__":
    print(CvStreamOut())
