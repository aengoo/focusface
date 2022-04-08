# focusface

https://github.com/biubug6/Pytorch_Retinaface

https://github.com/ageitgey/face_recognition

https://github.com/abewley/sort

```
■RTCDS_Project
├　■data
┃　├　■weights
┃　┃　├　□mobilenet0.25_Final.pth
┃　┃　├　□mobilenetV1X0.25_pretrain.tar
┃　┃　└　□Resnet50_Final.pth
┃　┃　
┃　└　■target
┃　　　├　★suspect_db_example.csv
┃　　　├　□[target_faces_dir]
┃　　　┃　├　□[face_name_dir_001]
┃　　　┃　┃　├　□[face_name].jpg	
┃　　　┃　┃　├　☆[face_name].[model].ebd
┃　　　┃　┃　└　☆[face_name].[model].checksum
┃　　　┃　┃
┃　　　┃　├　□[face_name_dir_002]
┃　　　┃　：
┃　　　┃　├　□[face_name_dir_003]
┃　　　┃　└　□[face_name_dir_004]
┃　　　┃
┃　　　└　□[sample_faces_dir]
┃　　　
└　■focusface
　　├　general
　　┃　└　★config.py
　　├　cmdl_stream_server.py
　　├　pyqt_stream_server.py
　　└　focusface.ui
```

1. RTCDS_Project의 기본 디렉토리(■로 표기)를 구성합니다.
2. 필요한 데이터(□로 표기)를 다운로드합니다. 
3. 사전 설정(★로 표기)을 확인하고 수정합니다. csv 파일 생성은 focusface/identifier/models/embedding_loader.py를 참고할 수 있습니다.
4. 실행에 필요한 환경 설정을 마치고 stream_server.py 중 하나를 실행합니다.