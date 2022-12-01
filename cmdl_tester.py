from __future__ import print_function
import argparse

from detector import Detector
from identifier import Identifier
from tracker import Sort, score_board
from general import *

import numpy as np
import cv2 as cv
import os
import copy
import time
from tqdm import tqdm
"""
# code from github/ultralytics/yolov3/utils
def plot_one_box(x, img, color=None, label: str = None, line_thickness=None):
    
    # 이미지(img)의 xyxy좌표(x)상에 상자를 그립니다.
    
    # Plots one bounding box on image img
    tl = line_thickness or round(0.001 * (img.shape[0] + img.shape[1]) / 2) + 1  # line/font thickness
    # color = color or [random.randint(0, 255) for _ in range(3)]
    color = color or [0, 0, 255]
    c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
    cv.rectangle(img, c1, c2, color, thickness=tl, lineType=cv.LINE_AA)
    if label:
        tf = max(tl - 1, 1)  # font thickness
        t_size = cv.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
        cv.rectangle(img, c1, c2, color, -1, cv.LINE_AA)  # filled
        cv.putText(img, label, (c1[0], c1[1] - 2), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv.LINE_AA)
"""


class TestFocusFace:
    def __init__(self, opt):
        # 얼굴 탐지(face localization)를 위한 객체 로드
        self.det = Detector(weight_path=os.path.join(opt.data, opt.det_weight),
                            model=opt.det_model,
                            conf_thresh=opt.conf_thresh)

        # 얼굴 식별(face identification)을 위한 객체 로드
        # 이 객체 생성 중 얼굴 임베딩 데이터베이스 체크, 해당 클래스 생성자 참고
        self.idt = Identifier(embed_db_path=os.path.join(opt.data, opt.suspect_db),
                              n=opt.n_faces,
                              idt_res=opt.vid_res,
                              box_ratio=opt.box_ratio,
                              model=opt.idt_model)

        # 식별기 객체로부터 DB 데이터프레임 획득
        self.fdf = self.idt.get_df()  # Face Data Frame

        # 객체 추적 세팅, 탐지 상실시 [max_age]만큼 프레임 유지, 최소 [min_hits] 프레임만큼 탐지되면 추적 목록에 추가.
        # iou_threshold는 추적 시작을 위한 민감도의 역수, 너무 낮추면 '밀집한 다중 얼굴 객체' 탐지시 비정상적인 추적 발생
        self.trk = Sort(max_age=3, min_hits=1, iou_threshold=opt.iou_thresh)

        # 입력 동영상 로더. 입력원 변경시(ip cam or video file) 해당 클래스에 기능 추가 필요
        # self.frame_loader = StreamLoader(opt.vid_res, opt.down)

        # 용의자 엔트리, 각 모듈의 예측 결과를 취합하여 최종 예측 도출
        self.spt_entry = score_board.SuspectEntry(opt.insense)
        """
        # 출력 방식 선택
        self.stream_out = None
        if opt.output == 'strm':
            self.stream_out = CvStreamOut()
        elif opt.output == 'file':
            self.stream_out = CvFileOut(os.path.join(opt.data, opt.out_path))
        """
        # 옵션 저장
        self.opt = opt
        self.time_dict = {
            'down': [],
            'det': [],
            'trk': [],
            'idt': [],
            'score': []
        }


    def run(self, img_raw, now_frame, log_path):
        # start = time.time()  # 프레임 타이머, 처리시간 측정 시작
        # ret, img_raw = self.frame_loader.get_frame()  # 단일 프레임 획득
        t_down = time.time()
        img_det = cv.resize(img_raw, tuple([i // self.opt.down for i in AVAILABLE_RESOLUTIONS[self.opt.vid_res][:2][::-1]]))
        self.time_dict['down'].append(time.time() - t_down)
        t_det = time.time()
        boxes = np.clip(self.det.run(img_det), 0., 1.)  # 탐지 수행, 그 결과인 박스 좌표 범위가 영상 밖으로 벗어나지 않도록 조정
        self.time_dict['det'].append(time.time() - t_det)
        t_trk = time.time()
        tracked = self.trk.update(boxes)  # 추적 수행, 탐지 결과에 트래킹_ID 부여
        self.time_dict['trk'].append(time.time() - t_trk)
        t_idt = time.time()
        track_identified = self.idt.run(img_raw, tracked)  # 추적된 객체 식별
        self.time_dict['idt'].append(time.time() - t_idt)
        # img_push = copy.deepcopy(img_raw)  # 출력할 프레임 복제
        t_score = time.time()
        for box, score, tid, fid, face_dist, face_std_score in track_identified:
            # 종합 점수 = (표준점수, 얼굴신뢰도, 식별점수)를 특정 임계값을 기준으로 세제곱 양극화
            with open(str(log_path) + '.txt', 'a') as f:
                f.write(f'{now_frame:d}, {str(tid):s}, {str(fid):s}, {score:.6f}, {face_dist:.6f}, {face_std_score:.6f}, {box[0]:d}, {box[1]:d}, {box[2]:d}, {box[3]:d}\n')

            total_score = ((score + face_std_score + (1 - (face_dist * 1.65))) * (1. / self.opt.criteria)) ** 3

            # 용의자 엔트리에 종합점수를 비롯한 정보 축적
            lets_report, suspect = self.spt_entry.register(score_board.SuspectFace(tid, fid, total_score, box))

            # 누적된 종합 점수가 임계점(insense) 돌파시 보고_플래그(lets_report) 활성화
            """
            if suspect.is_reported():  # 용의자가 이미 보고된 상태면(예측 모델이 이미 확신했으면)
                # 예측 라벨을 포함하여 출력용 프레임에 표기
                plot_one_box(box, img_push,
                             label=self.fdf[self.fdf['ID'] ==
                                            self.spt_entry.suspect_dict[tid].get_face_id()]["NAME"].values[0])
            else:
                # 박스만 표기
                plot_one_box(box, img_push)
            """
            if lets_report:  # 보고_플래그(lets_report) 활성화?
                # tmp_box = suspect.get_last_box()  # 확신한 시점의 용의자 얼굴 영역 크롭을 위한 박스 좌표
                # face_id = self.spt_entry.suspect_dict[tid].get_face_id()  # 얼굴 ID

                # 용의자 얼굴 영역 이미지 출력, 현재는 Redis out 모드에서만 유효 동작
                # self.stream_out.push_face(img_raw[tmp_box[1]:tmp_box[3], tmp_box[0]:tmp_box[2]], face_id)

                # 용의자 신상 정보 출력
                # self.stream_out.push_info(self.fdf[self.fdf['ID'] == face_id].to_dict('records')[0])
                suspect.set_reported()  # 해당 용의자의 상태를 '이미 보고됨'으로 수정
        self.time_dict['score'].append(time.time() - t_score)

        # end = time.time()
        # process FPS :  1./self.stream_out.push_frame(img_push), end-start
        # 출력 상태 여부, 프레임당 처리 시간 반환. 출력상태 여부는 OPEN_CV 출력시 종료를 위해 사용됨
        # return self.stream_out.push_frame(img_push), end-start


def get_args():
    parser = argparse.ArgumentParser()
    for k in cfg_opt_dict:
        parser.add_argument('--' + k, type=type(cfg_opt_dict[k]), default=cfg_opt_dict[k], help='')
    return parser.parse_args()


if __name__ == '__main__':
    cmd_opt = get_args()
    tester = TestFocusFace(cmd_opt)
    vid_list = os.listdir(os.path.join(cmd_opt.data, cmd_opt.vid_path))
    # label_list = os.listdir(cmd_opt.data, cmd_opt.label_path)
    # trimmed_label_list = [label_name.replace('.txt', '') for label_name in label_list]

    for vid_idx, vid_name in enumerate(vid_list):
        save_path = os.path.join(cmd_opt.data, cmd_opt.out_path, vid_name)
        stream = cv.VideoCapture(os.path.join(cmd_opt.data, cmd_opt.vid_path, vid_name))
        total_frame = stream.get(cv.CAP_PROP_FRAME_COUNT)
        frame_idx = 0
        with tqdm(total=total_frame) as pbar:
            while frame_idx <= total_frame:
                ret, frame = stream.read()
                if ret:
                    tester.run(frame, frame_idx, save_path)
                    frame_idx += 1
                    pbar.update(1)
                else:
                    total_frame -= 1
        for k in tester.time_dict.keys():
            print(f'{k:s} | mean:', np.mean(np.array(tester.time_dict[k])), '| std:', np.std(np.array(tester.time_dict[k])))
        stream.release()
