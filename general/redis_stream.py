import base64
import datetime
import json

from .cv_stream import *
import redis


class RedisStreamOut(StreamOut):
    def __init__(self, port: int):
        super().__init__()
        self.rd = redis.StrictRedis(host='localhost', port=port, db=0)
        self.rd.flushdb()  # db 초기화
        self.push_idx = 0
        self.det_dict_simple = {}
        self.det_dict_detail = {}

    def push_info(self, info: dict):
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
        retval, buffer = cv2.imencode('.jpg', img)
        fr = base64.b64encode(buffer)
        fr = fr.decode('utf-8')
        encoded = json.dumps(fr).encode('utf-8')  # Object of type 'bytes' is not JSON serializable
        self.rd.lpush("vid", encoded)
        self.rd.ltrim("vid", 0, 29)
        return True

    def push_face(self, img: np.ndarray, idt):
        retval, buffer = cv2.imencode('.jpg', img)
        fr = base64.b64encode(buffer)
        fr = fr.decode('utf-8')
        encoded = json.dumps(fr).encode('utf-8')  # Object of type 'bytes' is not JSON serializable
        self.rd.lpush("det" + idt, encoded)
        self.rd.ltrim("det" + idt, 0, 29)