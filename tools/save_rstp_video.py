import numpy as np
import cv2
import datetime

# cap = cv2.VideoCapture(0)

# Define the codec and create VideoWriter object


def collect_video():
    cap1 = cv2.VideoCapture(
        "rtsp://61.166.49.242:554/pag://61.166.49.242:7302:017191:0:MAIN:TCP?cnid=1&pnid=1&auth=50&streamform=rtp")
    cap2 = cv2.VideoCapture(
        "rtsp://61.166.49.242:554/pag://61.166.49.242:7302:017192:0:MAIN:TCP?cnid=1&pnid=1&auth=50&streamform=rtp")
    cap3 = cv2.VideoCapture(
        "rtsp://61.166.49.242:554/pag://61.166.49.242:7302:017193:0:MAIN:TCP?cnid=1&pnid=1&auth=50&streamform=rtp")
    fourcc = cv2.VideoWriter_fourcc(*'MP4V')
    while True:
        start_hour = datetime.datetime.now().hour
        day = datetime.datetime.now().day
        minute = datetime.datetime.now().minute
        if start_hour == 11 or start_hour == 17:
            out1 = cv2.VideoWriter(str(day) + str(start_hour) + '-output1.mp4', fourcc, 25.0, (1280, 720))
            out2 = cv2.VideoWriter(str(day) + str(start_hour) + '-output2.mp4', fourcc, 25.0, (1280, 720))
            out3 = cv2.VideoWriter(str(day) + str(start_hour) + '-output3.mp4', fourcc, 25.0, (1280, 720))
            while (datetime.datetime.now().hour - start_hour) < 1:
                ret1, frame1 = cap1.read()
                ret2, frame2 = cap2.read()
                ret3, frame3 = cap3.read()
                out1.write(frame1)
                out2.write(frame2)
                out3.write(frame3)
            out1.release()
            out2.release()
            out3.release()
            print("save finished!")


collect_video()
