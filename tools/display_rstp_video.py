import numpy as np
import cv2
import datetime


def collect_video():
    cap1 = cv2.VideoCapture(
        "rtsp://61.166.49.242:554/pag://61.166.49.242:7302:017191:0:MAIN:TCP?cnid=1&pnid=1&auth=50&streamform=rtp ")
    cap2 = cv2.VideoCapture(
        "rtsp://61.166.49.242:554/pag://61.166.49.242:7302:017192:0:MAIN:TCP?cnid=1&pnid=1&auth=50&streamform=rtp")
    cap3 = cv2.VideoCapture(
        "rtsp://61.166.49.242:554/pag://61.166.49.242:7302:017193:0:MAIN:TCP?cnid=1&pnid=1&auth=50&streamform=rtp")
    while 1:
        ret1, frame1 = cap1.read()
        if ret1:
            cv2.imshow('frame1', frame1)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            break


collect_video()
