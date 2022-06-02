
from __future__ import print_function
import argparse

from detector import Detector
from identifier import Identifier, plot_tp_box, plot_one_box
from tracker import Sort, score_board
from general import *

import numpy as np
import os
import copy
import cv2 as cv
import time


class StreamServer:
    def __init__(self, opt):
        print(opt)
        self.det = Detector(weight_path=os.path.join(opt.data, opt.det_weight),
                            model=opt.det_model,
                            conf_thresh=opt.conf_thresh)
        self.idt = Identifier(embed_db_path=os.path.join(opt.data, opt.suspect_db),
                              n=opt.n_faces,
                              idt_res=opt.vid_res,
                              box_ratio=opt.box_ratio,
                              is_eval=True,
                              model=opt.idt_model)
        self.fdf = self.idt.get_df()  # Face Data Frame
        self.trk = Sort(max_age=3, min_hits=1, iou_threshold=opt.iou_thresh)
        self.frame_loader = StreamLoader(opt.vid_res, opt.down)
        self.spt_entry = score_board.SuspectEntry(opt.insense)

        self.stream_out = None
        if opt.output == 'opencv':
            self.stream_out = CvStreamOut()
        elif opt.output == 'redis':
            self.stream_out = RedisStreamOut(opt.redis_port)
        self.opt = opt

    def run(self):
        ret, img_raw = self.frame_loader.get_frame()
        if ret:
            if self.opt.down != 1:
                img_det = cv.resize(img_raw, [i // self.opt.down for i in AVAILABLE_RESOLUTIONS[self.opt.vid_res][:2][::-1]])
            else:
                img_det = copy.deepcopy(img_raw)

            boxes = np.clip(self.det.run(img_det), 0., 1.)
            identified = self.idt.run(img_raw, boxes)
            tracked = self.trk.update(boxes)
            track_identified = self.idt.run(img_raw, tracked)

            img_push = copy.deepcopy(img_raw)
            for box, score, tid, fid, face_dist, face_std_score in track_identified:
                total_score = ((score + face_std_score + (1 - (face_dist * 1.65))) * (1. / self.opt.criteria)) ** 3
                ret_report, suspect = self.spt_entry.register(score_board.SuspectFace(tid, fid, total_score, box))
                if suspect.is_reported():
                    plot_one_box(box, img_push,
                                 label=self.fdf[self.fdf['ID'] == self.spt_entry.suspect_dict[tid].get_face_id()]["NAME"].values[0])
                else:
                    plot_one_box(box, img_push)
                if ret_report:
                    tmp_box = suspect.get_last_box()
                    self.stream_out.push_face(img_raw[tmp_box[1]:tmp_box[3], tmp_box[0]:tmp_box[2]])
                    self.stream_out.push_info(
                        self.fdf[self.fdf['ID'] == self.spt_entry.suspect_dict[tid].get_face_id()].to_dict('records')[0])
                    suspect.set_reported()

            for box, score, tid, fid, face_dist, face_std_score in identified:
                plot_tp_box(box, img_push, 0.5, 'g')
            return self.stream_out.push_frame(img_push)

        else:
            print("no camera detected")


def get_args():
    parser = argparse.ArgumentParser()
    for k in cfg_opt_dict:
        parser.add_argument('--' + k, type=type(cfg_opt_dict[k]), default=cfg_opt_dict[k], help='')
    return parser.parse_args()


if __name__ == '__main__':
    cmd_opt = get_args()
    server = StreamServer(cmd_opt)
    while True:
        server.run()
