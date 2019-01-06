import sys
import argparse
import os
from yolo import YOLO, detect_video_yolo
from PIL import Image
path = os.path.abspath(__file__)
cwd = os.path.split(path)[0]
if cwd.endswith('apps'):
    os.chdir(cwd[0:-4])
    cwd = os.getcwd()

__author__ = 'WangZe'

yolo = YOLO()
video_path = "D:\\project3_faster_rcnn\\models-master\\research\\hat_dataset\\video_data\\test_video\\1.mp4"
output_image_path = "D:\\project3_faster_rcnn\\models-master\\research\\hat_dataset\\video_data\\test_video\\"

kwargs = dict(frame_num_for_judge=5, use_normalized_coordinates=False,
              overlap_threshold=0.65, min_prob=0.5, min_x=0.05, min_y=0.03, min_area=0.001)
detect_video_yolo(yolo, video_path, output_path="", out_image_path=output_image_path, **kwargs)



