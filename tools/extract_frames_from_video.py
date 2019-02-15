# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import os
path = "D:\\project3_faster_rcnn\\models-master\\research\\hat_dataset\\video_data\\nobody_train_video"

os.chdir(path)

video_list = os.listdir(path)

video_name = video_list[0]

for video_name in video_list:
    
    command = "ffmpeg -i " + video_name + " -r 0.1 -f image2 images/" + video_name[11:14] + "_%03d.jpg"
    
    os.system(command)
