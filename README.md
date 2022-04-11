# focusface

```
■ RTCDS_Project
├　■ data
┃　├　■ weights
┃　┃　├　□ mobilenet0.25_Final.pth
┃　┃　├　□ mobilenetV1X0.25_pretrain.tar
┃　┃　└　□ Resnet50_Final.pth
┃　┃　
┃　└　■ target
┃　　　├　★ suspect_db_example.csv
┃　　　├　□ [target_faces_dir]
┃　　　┃　├　□ [face_name_dir_001]
┃　　　┃　┃　├　□ [face_name].jpg	
┃　　　┃　┃　├　☆ [face_name].[model].ebd
┃　　　┃　┃　└　☆ [face_name].[model].checksum
┃　　　┃　┃
┃　　　┃　├　□ [face_name_dir_002]
┃　　　┃　：
┃　　　┃　├　□ [face_name_dir_003]
┃　　　┃　└　□ [face_name_dir_004]
┃　　　┃
┃　　　└　□ [sample_faces_dir]
┃　　　
└　■ focusface
　　├　general
　　┃　└　★ config.py
　　├　cmdl_stream_server.py
　　├　pyqt_stream_server.py
　　└　focusface.ui
```

1. RTCDS_Project의 기본 디렉토리(■로 표기)를 구성합니다.
2. 필요한 데이터(□로 표기)를 다운로드합니다. 
   - `../data/target` : [download](https://drive.google.com/file/d/1lTmbSY6Ksne23LCK46bbkepJRYAX2p6w/view?usp=sharing) (unavailable for general user)
   - `../data/weights` : [official](https://drive.google.com/drive/folders/1oZRSG0ZegbVkVwUd8wUIQx8W7yfZ_ki1) / [our](https://drive.google.com/file/d/1rmGkQ11o7kn1Rdp5AKmRwbXI5v-kDmXb/view?usp=sharing)

3. 사전 설정(★로 표기)을 확인하고 수정합니다. csv 파일 생성은 focusface/identifier/models/embedding_loader.py를 참고할 수 있습니다.

4. 실행에 필요한 환경 설정을 마치고 `****`_stream_server.py 중 하나를 실행합니다.



------

## Installation

```
https://github.com/biubug6/Pytorch_Retinaface  # for face detection
https://github.com/ageitgey/face_recognition  # for face identification
https://github.com/abewley/sort  # for face tracking
```

### Requirements

- Python 3.3+, but **anaconda Python 3.8+ recommended**
- dilb(cmake) with [Windows](https://github.com/ageitgey/face_recognition/issues/175#issue-257710508), [macOS, Linux(Ubuntu)](https://gist.github.com/ageitgey/629d75c1baac34dfa5ca2a1928a7aeaf)
- tested on Pytorch version 1.7.0+ and torchvision 0.5.0+ [(pytorch versions)](https://pytorch.org/get-started/previous-versions/)

### Installation Manual

1. Create conda environment and Set python version

```
$ conda create -n [env_name] python=3.8
```

2. Install cmake and dlib

```
Windows : https://github.com/ageitgey/face_recognition/issues/175#issue-257710508
osX(macOS), Linux : https://gist.github.com/ageitgey/629d75c1baac34dfa5ca2a1928a7aeaf
```

3. Activate conda environment and Install Pytorch

```
# https://pytorch.org/get-started/previous-versions/

(example of windows, torch 1.7.0, cudatoolkit 10.1)
$ conda install pytorch==1.7.0 torchvision==0.8.0 torchaudio==0.7.0 cudatoolkit=10.1 -c pytorch
```

4. Install other python libraries

```
$ pip install -r requirements.txt
```

------

## Loading Processes

1. `./detector/init.py` Detector를 로드합니다.
   - 고정 해상도 입력 설정 및 비훈련 모드 활성화
   - 입력된 파라미터에 맞게 네트워크(`re50` or `mnet`)를 장치(`cpu` or `cuda`)에 로드합니다.
2. `./identifier/init.py` Identifier를 로드합니다.
   - `embedding_loader.py` 에서 DB(`*.csv`)를 읽어 타겟의 얼굴 임베딩을 로드합니다.
   - 임베딩을 읽는 과정에서 자동으로 `sha256 hash`를 대조하여 무결성 검사를 실시합니다. 
   - 무결성 검사 결과 결함이 있는 안면 임베딩은 다시 생성합니다.
   - DB 로드 결과를 FaceComparer 객체에 전달하고 로드를 완료합니다.

3. `SORT` 로드 및 `cv2.VideoCapture`를 설정합니다.

------

## References

- [RetinaFace (mxnet)](https://github.com/deepinsight/insightface/tree/master/RetinaFace)
- [RetinaFace (pytorch)](https://github.com/biubug6/Pytorch_Retinaface)
- [FaceBoxes](https://github.com/zisianw/FaceBoxes.PyTorch)
- [face_recognition](https://github.com/ageitgey/face_recognition)
- [SORT](https://github.com/abewley/sort)

```
@inproceedings{deng2020retinaface,
title={Retinaface: Single-shot multi-level face localisation in the wild},
author={Deng, Jiankang and Guo, Jia and Ververas, Evangelos and Kotsia, Irene and Zafeiriou, Stefanos},
booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
pages={5203--5212},
year={2020}
}

@Article{dlib09,
author = {Davis E. King},
title = {Dlib-ml: A Machine Learning Toolkit},
journal = {Journal of Machine Learning Research},
year = {2009},
volume = {10},
pages = {1755-1758},
}
```

