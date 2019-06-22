# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import os
# path = "D:\\project3_faster_rcnn\\models-master\\research\\hat_dataset\\video_data\\nobody_train_video"
path = "D:\\project3_faster_rcnn\\models-master\\research\\hat_detection\\video_data\\negative_samples\\"

os.chdir(path)

video_list = os.listdir(path)

# video_name = video_list[0]

for i, video_name in enumerate(video_list):
    if not os.path.exists('./images'):
        os.mkdir('./images')
    command = "ffmpeg -i " + video_name + " -r 0.1 -f image2 images/" + 'neg_' + str(i) + "_%03d.jpg"
    
    os.system(command)
