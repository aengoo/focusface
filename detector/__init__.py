from .models import cfg_re50, cfg_mnet, PriorBox
from .models.retinaface import *
from .models.utils import *
import torch.backends.cudnn as cudnn
import numpy as np
import os

"""
프로그램이 로드될 때 torch 및 cudnn 설정을 최적화합니다. 
$ cudnn.benchmark = True
이는 고정 입력 해상도 조건에서 연산을 가속합니다. 
"""

# torch, cudnn settings
torch.set_grad_enabled(False)
cudnn.benchmark = True


class Detector:
    """
    주어진 영상에서 얼굴 영역을 탐지하는 객체입니다.
    동작에 관한 자세한 주석은 각 클래스 함수를 참고해주시기 바랍니다.
    """

    def __init__(self, weight_path: str, model: str = 're50', conf_thresh: float = 0.5):
        """
        얼굴 탐지(Face Localization)을 위한 모델 로드를 수행합니다.
        동영상이 입력되기 전에 미리 생성해두어야 하며, 싱글톤과 유사하게 고유객체로 하나만 생성하는 것을 권장합니다.
        추후 싱글톤 적용이 가능합니다.
        백본 네트워크로 ResNet-50 또는 MobileNet0.25x를 활용합니다.

        ResNet-50 모델 기반 탐지기는 더 많은 VRAM 및 GPU 연산량을 요구하지만 약간 더 높은 탐지 정확도를 보입니다.
        MobileNet0.25x 모델은 매우 적은 파라미터를 사용하는 모델입니다.
        모델 깊이가 얕기 때문에 탐지 프로세스만 두고 보면 상대적으로 빠르지만, 전체 시스템 측면에서의 속도 차이는 경미합니다.
        VRAM 사양이 제한되는 상황에서 고려할 수 있는 선택지입니다.

        객체 신뢰도 점수(confidence score)는 탐지된 얼굴 객체가 실제 얼굴일 가능성을 예측한 수치입니다.
        통계적인 결과와는 관계가 없으며, 확률과는 성격이 다른 수치입니다.
        일반적으로 0.95 이상으로 얼굴 영역을 예측합니다.
        이 수치는 카메라와 얼굴의 거리, 얼굴의 방향, 장애물에 의해 낮아질 수 있습니다.

        이 객체에서 미리 설정되는 파라미터는 다음과 같습니다.

        :param weight_path: 모델 로드를 위한 가중치 파일 경로를 입력해야 합니다.
        :param model: 백본 네트워크 모델을 설정합니다. 're50' 또는 'mnet'으로 설정할 수 있습니다. 기본 설정은 're50'으로 되어 있습니다.
        :param conf_thresh: 탐지 얼굴 객체 선별을 위한 신뢰도 하한 임계값을 설정합니다. 기본 설정은 0.5로 설정되어 있습니다.
        """

        if model == 're50':
            self.cfg = cfg_re50
        elif model == 'mnet':
            self.cfg = cfg_mnet

        # net and model
        self.net = RetinaFace(cfg=self.cfg, phase='test')
        self.net = load_model(self.net, os.path.join(weight_path), False)
        self.net.eval()
        self.device = torch.device("cuda")
        self.net = self.net.to(self.device)
        self.conf_thresh = conf_thresh

    def run(self, img_tensor, vectorized=True, nms_thr: float = 0.4):
        """
`       각 프레임에 대해 객체 탐지를 수행하는 함수입니다.

        :param img_tensor: np.ndarray type의 CV_RGB 인덱싱 이미지를 입력합니다.
        :param vectorized: 계산량이 많고, 지연시간에 큰 영향을 주는 Anchor Box 처리를 병렬화할 지 설정합니다.
        :param nms_thr: 비최대억제(NMS)를 위한 임계값을 설정합니다. 0.4~0.5로 설정하는 것을 권장합니다.
        :return: 탐지된 객체들을 numpy 배열 객체로 반환합니다. 형태는 다음과 같습니다. (x, 5)
        """
        img = np.float32(img_tensor)
        img -= (104, 117, 123)
        img = img.transpose(2, 0, 1)
        img = torch.from_numpy(img).unsqueeze(0)
        img = img.to(self.device)
        loc, conf, landms = self.net(img)
        priorbox = PriorBox(self.cfg, image_size=tuple(img_tensor.shape[:2]))
        if vectorized:
            priors = priorbox.vectorized_forward()
        else:
            priors = priorbox.forward()
        priors = priors.to(self.device)
        prior_data = priors.data
        boxes = decode(loc.data.squeeze(0), prior_data, self.cfg['variance'])
        boxes = boxes.cpu().numpy()
        scores = conf.squeeze(0).data.cpu().numpy()[:, 1]
        inds = np.where(scores > self.conf_thresh)[0]
        boxes = boxes[inds]
        scores = scores[inds]
        order = scores.argsort()[::-1]
        boxes = boxes[order]
        scores = scores[order]
        dets = np.hstack((boxes, scores[:, np.newaxis])).astype(np.float32, copy=False)
        keep = py_cpu_nms(dets, nms_thr)
        dets = dets[keep, :]
        return dets
