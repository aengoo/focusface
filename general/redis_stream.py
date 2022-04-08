from .cv_stream import *
import redis


class RedisStreamOut(StreamOut):
    def __init__(self, port: int):
        super().__init__()
        self.rd = redis.StrictRedis(host='localhost', port=port, db=0)
        self.rd.flushdb()  # db 초기화

    def push_info(self, info: dict):
        pass

    def push_frame(self, img: np.ndarray):
        pass

    def push_face(self, img: np.ndarray):
        pass
