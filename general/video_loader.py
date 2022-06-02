from abc import *
from .configs import *
import cv2 as cv


class Loader(metaclass=ABCMeta):
    """
    다양한 입력 옵션에 대응하기 위한 추상 클래스입니다.
    현재 버전에서는 로컬캠 연결 모드만 구현되어 있습니다.
    """
    @abstractmethod
    def get_frame(self):
        """
        :return: 단일 프레임을 반환합니다.
        """
        pass


class StreamLoader(Loader):
    def __init__(self, vid_res, downsample):
        """
        비디오 캡쳐 장치로부터 동영상을 읽어들이기 위한 OPEN_CV 객체를 세팅합니다.
        :param vid_res: 동영상 프레임의 해상도를 설정합니다. 캠의 세팅을 변경하는 것이 아니라 직접 다운샘플링을 수행합니다.
        :param downsample: 다운샘플링 옵션이 문제없이 설정되어 있는지 검사합니다.
        """
        self._cap = cv.VideoCapture(0)  # 0번 카메라
        self._cap.set(cv.CAP_PROP_BUFFERSIZE, 3)
        self._cap.set(cv.CAP_PROP_FPS, 30)  # 30 FPS로 고정

        """
        video 해상도가 [적응적; adaptive]이면 캠의 기본 세팅에 따라 프레임을 읽습니다. 
        문제를 예측할 수 없기 때문에, 이 경우는 탐지 프로세스의 다운샘플링 옵션이 비활성화됩니다. 
        """
        if vid_res == "adaptive":
            if downsample != 1:
                print("[Alert] opt[--vid-res] is \'adaptive\', so opt[down] is disabled.")
        else:
            # 해상도를 세팅합니다.
            self._cap.set(cv.CAP_PROP_FRAME_WIDTH, AVAILABLE_RESOLUTIONS[vid_res][1])
            self._cap.set(cv.CAP_PROP_FRAME_HEIGHT, AVAILABLE_RESOLUTIONS[vid_res][0])

    def get_frame(self):
        """
        :return: 단일 프레임을 반환합니다.
        """
        return self._cap.read()
