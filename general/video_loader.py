from abc import *
from .configs import *
import cv2 as cv


class Loader(metaclass=ABCMeta):
    @abstractmethod
    def get_frame(self):
        pass


class StreamLoader(Loader):
    def __init__(self, vid_res, downsample):
        self._cap = cv.VideoCapture(0)
        self._cap.set(cv.CAP_PROP_BUFFERSIZE, 3)
        self._cap.set(cv.CAP_PROP_FPS, 30)  # TODO: 체크할것

        if vid_res == "adaptive":
            if downsample != 1:
                print("[Alert] opt[--vid-res] is \'adaptive\', so opt[down] is disabled.")
        else:
            self._cap.set(cv.CAP_PROP_FRAME_WIDTH, AVAILABLE_RESOLUTIONS[vid_res][1])
            self._cap.set(cv.CAP_PROP_FRAME_HEIGHT, AVAILABLE_RESOLUTIONS[vid_res][0])

    def get_frame(self):
        return self._cap.read()
