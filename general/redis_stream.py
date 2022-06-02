import base64
import datetime
import json

from .cv_stream import *
import redis


class RedisStreamOut(StreamOut):
    def __init__(self, port: int):
        """
        Redis 객체 생성을 위해 생성자를 오버라이딩합니다.
        :param port: Redis 포트입니다. 이는 웹서버 코드의 Redis 포트와 일치해야 합니다.
        """
        super().__init__()
        self.rd = redis.StrictRedis(host='localhost', port=port, db=0)  # Redis는 프로세스간 통신을 위해서만 사용됩니다.
        self.rd.flushdb()  # db 초기화
        self.push_idx = 0
        self.det_dict_simple = {}
        self.det_dict_detail = {}

    def push_info(self, info: dict):
        """
        식별이 완료된 얼굴(인물)에 대한 정보를 받아 출력합니다.
        두가지 방식으로 나눠 출력합니다.
        [SIMPLE]: INDEX, DETECTED, ID_PHOTO, ID, NAME for simple
        [DETAIL]: INDEX, ID, NAME, BIRTH, RAP, DETAIL, DATETIME for Detail
        :param info: dict타입 객체, 인물에 대한 정보 데이터
        :return: 반환값이 없습니다.
        """
        # INDEX, DETECTED, ID_PHOTO, ID, NAME for simple
        # INDEX, ID, NAME, BIRTH, RAP, DETAIL, DATETIME for Detail
        simple = {'INDEX': str(self.push_idx),
                  'DETECTED': '',
                  'ID_PHOTO': '',
                  'ID': info['ID'],
                  'NAME': info['NAME']
                  }
        detail = {'INDEX': str(self.push_idx),
                  'ID': info['ID'],
                  'NAME': info['NAME'],
                  'BIRTH': info['BIRTH'],
                  'RAP': info['RAP'],
                  'DETAIL': info['DETAIL'],
                  'DATETIME': datetime.datetime.now().strftime("%Y.%m.%d %H:%M:%S")
                  }
        self.det_dict_simple.update({str(self.push_idx): simple})
        self.det_dict_detail.update({str(self.push_idx): detail})
        json_dict_simple = json.dumps(self.det_dict_simple, ensure_ascii=False).encode('utf-8')
        json_dict_detail = json.dumps(self.det_dict_detail, ensure_ascii=False).encode('utf-8')
        self.rd.set("dict_simple", json_dict_simple)
        self.rd.set("dict_detail", json_dict_detail)
        self.push_idx += 1
        print(detail)

    def push_frame(self, img: np.ndarray):
        """
        탐지/식별 정보가 기록된 프레임 영상을 받아 출력합니다.
        이미지를 버퍼 데이터로 인코딩하여 redis에 입력합니다.
        :param img: numpy array 객체, 가공된 프레임 영상
        :return: REDIS 출력 모드에서는 반환값이 없습니다. 항상 계속 출력합니다.
        """
        retval, buffer = cv2.imencode('.jpg', img)
        fr = base64.b64encode(buffer)
        fr = fr.decode('utf-8')
        encoded = json.dumps(fr).encode('utf-8')  # Object of type 'bytes' is not JSON serializable
        self.rd.lpush("vid", encoded)
        self.rd.ltrim("vid", 0, 29)
        return True

    def push_face(self, img: np.ndarray, idt):
        """
        식별이 완료된 얼굴 영역 이미지를 받아 출력합니다.
        이미지를 버퍼 데이터로 인코딩하여 redis에 입력합니다.
        :param img: numpy array 객체, 얼굴 영역 이미지
        :param idt: 얼굴(인물) ID
        :return: 반환값이 없습니다.
        """
        retval, buffer = cv2.imencode('.jpg', img)
        fr = base64.b64encode(buffer)
        fr = fr.decode('utf-8')
        encoded = json.dumps(fr).encode('utf-8')  # Object of type 'bytes' is not JSON serializable
        self.rd.lpush("det" + idt, encoded)
        self.rd.ltrim("det" + idt, 0, 29)