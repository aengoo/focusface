from .models import cfg_re50, cfg_mnet, PriorBox
from .models.retinaface import *
from .models.utils import *
import torch.backends.cudnn as cudnn
import numpy as np
import os

# torch, cudnn settings
torch.set_grad_enabled(False)
cudnn.benchmark = True


class Detector:
    def __init__(self, weight_path: str, model: str = 're50', conf_thresh: float = 0.5):
        # configure backbone network
        """
        model : re50 or mnet
        """

        if model == 're50':
            self.cfg = cfg_re50
        elif model == 'mnet':
            self.cfg = cfg_mnet

        # net and model
        self.net = RetinaFace(cfg=self.cfg, phase='test')
        self.net = load_model(self.net, os.path.join(weight_path), False)
        self.net.eval()
        # print('Finished loading model!')
        # print(net)
        self.device = torch.device("cuda")
        self.net = self.net.to(self.device)
        self.conf_thresh = conf_thresh

    def run(self, img_tensor, vectorized=True, nms_thr: float = 0.4):
        img = np.float32(img_tensor)
        # im_height, im_width = img.shape[:2]
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
        # boxes = boxes * scale
        boxes = boxes.cpu().numpy()
        scores = conf.squeeze(0).data.cpu().numpy()[:, 1]
        # landms = decode_landm(landms.data.squeeze(0), prior_data, self.cfg['variance'])
        # landms = landms.cpu().numpy()

        # ignore low scores
        inds = np.where(scores > self.conf_thresh)[0]
        boxes = boxes[inds]
        # landms = landms[inds]
        scores = scores[inds]

        # keep top-K before NMS
        order = scores.argsort()[::-1]
        boxes = boxes[order]
        # landms = landms[order]
        scores = scores[order]

        # do NMS
        dets = np.hstack((boxes, scores[:, np.newaxis])).astype(np.float32, copy=False)
        keep = py_cpu_nms(dets, nms_thr)
        dets = dets[keep, :]
        # landms = landms[keep]

        # dets = np.concatenate((dets, landms), axis=1)

        return dets  # returns only coordinates and score, no landmarks
