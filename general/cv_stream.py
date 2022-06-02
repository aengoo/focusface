from abc import *
import cv2
import numpy as np


# 출력을 위한 추상 객체
class StreamOut(metaclass=ABCMeta):
    """
    다양한 옵션에 대응하여 결과 출력을 위한 추상 클래스입니다.
    현재 구현되어 있는 모든 출력 객체 클래스는 이 추상 객체의 형식을 준수합니다.
    """

    @abstractmethod
    def push_info(self, info: dict):
        """
        식별이 완료된 얼굴(인물)에 대한 정보를 받아 출력합니다.
        :param info: dict타입 객체, 인물에 대한 정보 데이터
        :return: 반환값이 없습니다.
        """
        pass

    @abstractmethod
    def push_frame(self, img: np.ndarray):
        """
        탐지/식별 정보가 기록된 프레임 영상을 받아 출력합니다.
        :param img: numpy array 객체, 가공된 프레임 영상
        :return: 출력을 계속할 지 여부를 선택
        """
        return True  # keep iteration if True

    @abstractmethod
    def push_face(self, img: np.ndarray, idt):
        """
        식별이 완료된 얼굴 영역 이미지를 받아 출력합니다.
        :param img: numpy array 객체, 얼굴 영역 이미지
        :param idt: 얼굴(인물) ID
        :return: 반환값이 없습니다.
        """
        pass


class CvStreamOut(StreamOut):
    def __init__(self):
        super().__init__()

    def push_info(self, info: dict):
        """
        간단한 인물 정보를 콘솔에 출력합니다.
        :param info: dict타입 객체, 인물에 대한 정보 데이터
        :return: 반환값이 없습니다.
        """
        print(f'[IDENTIFIED] {info["NAME"]}({info["ID"]}) is identified.')

    def push_frame(self, img: np.ndarray):
        """
        OPEN_CV 윈도우로 프레임을 출력합니다. 해당 윈도우에 키보드 입력으로 Q 또는 q를 입력하면 정지합니다.
        이 로직은 GUI에서도 동작합니다. 자세한 동작 방식은 pyqt_stream_server.py를 참고하시기 바랍니다.
        :param img: numpy array 객체, 가공된 프레임 영상
        :return: 출력을 계속할 지 여부를 선택
        """
        cv2.imshow('test', img)
        if cv2.waitKey(1) == ord('q') or cv2.waitKey(1) == ord('Q'):
            cv2.destroyAllWindows()  # CV 윈도우를 닫습니다.
            return False  # stop iteration if False
        return True  # keep iteration if True

    def push_face(self, img: np.ndarray, idt):
        """
        이 함수는 OPEN_CV 출력 모드에서는 작동하지 않습니다.
        :param img: numpy array 객체, 얼굴 영역 이미지
        :param idt: 얼굴(인물) ID
        :return: 반환값이 없습니다.
        """
        pass


if __name__ == "__main__":
    print(CvStreamOut())  # 테스트
