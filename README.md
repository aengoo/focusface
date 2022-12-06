DEMO Video `VideoWriter`: https://drive.google.com/file/d/150ASIsI86AY0qonqZwaCsHAXlMc3fEGG/preview

DEMO Video `VideoCapture`: https://drive.google.com/file/d/1VrlquwaTCqS9IKKojJHTFl3oX4jcBZgc/preview

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

1. Construct the directory path of RTCDS_Project (dirs marked with char '■' are required to be initilized with user's needs).
2. Download the required data (marked with char '□')
   - `../data/target` : [download](https://drive.google.com/file/d/1lTmbSY6Ksne23LCK46bbkepJRYAX2p6w/view?usp=sharing) (unavailable for general user)
   - `../data/weights` : [official](https://drive.google.com/drive/folders/1oZRSG0ZegbVkVwUd8wUIQx8W7yfZ_ki1) / [our](https://drive.google.com/file/d/1rmGkQ11o7kn1Rdp5AKmRwbXI5v-kDmXb/view?usp=sharing)

3. Check and modify configurations (marked with ★). You can refer to `focusface/identifier/models/embedding_loader.py` to create a initial csv file.

4. After installation and setting required for execution, run one of `****`_stream_server.py.


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

1. Load Detector by importing module `detector`
   - Set fixed input resolution and disable training mode (set grad false)
   - Loads the network (`re50` or `mnet`) onto the device (`cpu` or `cuda`) according to the entered parameters.
2. Load Identifier by importing module `identifier`
   - In `embedding_loader.py`, read the DB (`*.csv`) to load the target's face embedding.
   - In the process of reading the embedding, an integrity check is performed by automatically matching `sha256 hash`.
   - Regenerate face embeddings that are defective as a result of the integrity check.
   - Pass the DB load result to the FaceComparer object and complete the load.
3. Load `SORT` and set `cv2.VideoCapture`.

------

## Arguments

refer to `/general/configs.py`

```
# at general/configs.py
# default value

cfg_opt_dict = {
    'data': '../data',
    'vid-res': 'adaptive',
    'det-model': 're50',
    'det-weight': 'weights/Resnet50_Final.pth',
    'box-ratio': 1.30,
    'down': 2,
    'conf-thresh': 0.50,
    'suspect-db': 'target/suspect_db_example.csv',
    's-faces': 'target/faces-400',
    'n-faces': 20,
    'idt-model': 'small',
    'iou-thresh': 0.30,
    'insense': 15,
    'criteria': 4.50,
    'redis-port': 6379,
    'output': 'opencv'
}
```

------

## Citation

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

