AVAILABLE_RESOLUTIONS = {
    'FHD': (1080, 1920, 3),
    'HD': (720, 1280, 2),
    'sHD': (360, 640, 1),
    'VGA': (480, 640, 1),
}

cfg_opt_dict = {
    'data': '../data',
    'vid-res': 'HD',
    'det-model': 're50',
    'det-weight': 'weights/Resnet50_Final.pth',
    'box-ratio': 1.30,
    'down': 4,
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

qt2opt_vid_res = {
    'adaptive (initialized by data)': 'adaptive',
    'VGA (640x480)': 'VGA',
    'sHD (640x360)': 'sHD',
    'HD (1280x720)': 'HD',
    'FHD (1920x1080)': 'FHD'
}

qt2opt_det_model = {
    'Resnet50_pretrained_RetinaFace': 're50',
    'mobilenet0.25_pretrained_RetinaFace': 'mnet'
}

qt2opt_idt_model = {
    'Small (using 5 Landmarks)': 'small',
    'Large (using 68 Landmarks)': 'large'
}

combo_matcher = {
    'vid-res': qt2opt_vid_res,
    'det-model': qt2opt_det_model,
    'idt-model': qt2opt_idt_model
}