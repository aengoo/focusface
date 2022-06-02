from abc import *
import cv2
import numpy as np


class StreamOut(metaclass=ABCMeta):
    @abstractmethod
    def push_info(self, info: dict):
        pass

    @abstractmethod
    def push_frame(self, img: np.ndarray):
        return True  # keep iteration if True

    @abstractmethod
    def push_face(self, img: np.ndarray, idt):
        pass


class CvStreamOut(StreamOut):
    def __init__(self):
        super().__init__()

    def push_info(self, info: dict):
        print(f'[IDENTIFIED] {info["NAME"]}({info["ID"]}) is identified.')

    def push_frame(self, img: np.ndarray):
        cv2.imshow('test', img)
        if cv2.waitKey(1) == ord('q'):
            cv2.destroyAllWindows()
            return False
        return True

    def push_face(self, img: np.ndarray, idt):
        pass


if __name__ == "__main__":
    print(CvStreamOut())
