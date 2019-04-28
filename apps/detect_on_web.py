import os
from frcnn_detection_web import *
from absl import flags
import configparser
import cv2
from flask import Flask, render_template, Response

__author__ = "WangZe"


flags.DEFINE_string("config_path",
                    "./hat_detection/test_configuration.config",
                    "configurations of yolo/frcnn testing on video")
Flags = flags.FLAGS


# def main(unused_argv):
#     cf = configparser.ConfigParser()
#     cf.read(Flags.config_path)
#
#     detect_rstp_video_frcnn(Flags.config_path)

app = Flask(__name__)


@app.route('/')
def index():
    return render_template('index.html')


def gen():
    # while True:
    # frame = detect_rstp_video_frcnn('./hat_detection/test_configuration.config')
    # # frame = camera.get_frame()
    # return (b'--frame\r\n'
    #        b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

    frame_byte = detect_rstp_video_frcnn('./hat_detection/test_configuration.config')
    # frame = camera.get_frame()
    return frame_byte


@app.route('/video_feed')
def video_feed():
    return Response(gen(), mimetype='multipart/x-mixed-replace; boundary=frame')

# if __name__ == "__main__":
#     tf.app.run()


if __name__ == '__main__':
    # cf = configparser.ConfigParser()
    # cf.read(Flags.config_path)
    # camera = VideoCamera()
    # app.run(host='127.0.0.1', debug=True)
    app.run(host='0.0.0.0', port=5000)
    # app.run(host='192.168.31.204',debug=True, port=5000)
    # app.run(host='0.0.0.0')




