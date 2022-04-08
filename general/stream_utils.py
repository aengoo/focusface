import redis
import json


def lpush_frame(img, rdo: redis.StrictRedis, key):
    img = json.dumps(img).encode('utf-8')  #Object of type 'bytes' is not JSON serializable
    rdo.lpush(key, img)
    rdo.ltrim(key, 0, 29)