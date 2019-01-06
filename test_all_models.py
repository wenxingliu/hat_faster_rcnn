import os
from yolo import detect_video_yolo, detect_image_yolo, detect_video_list_yolo
from frcnn_detection import *
from absl import flags
import configparser

__author__ = "WangZe"


flags.DEFINE_string("config_path",
                    "D:\\project3_faster_rcnn\\models-master\\research\\hat_dataset\\test_configuration.config",
                    "configurations of yolo/frcnn testing on video")
Flags = flags.FLAGS


def main(unused_argv):
    cf = configparser.ConfigParser()
    cf.read(Flags.config_path)
    if cf.get("model_selection", "model_name") == "yolo":
        detect_video_yolo(Flags.config_path)
        # detect_image_yolo(Flags.config_path)
        # detect_video_list_yolo(Flags.config_path)
        # detect_video_list_frcnn(Flags.config_path)
    elif cf.get("model_selection", "model_name") == "frcnn":
        detect_video_frcnn(Flags.config_path)
        # detect_image_frcnn(Flags.config_path)
        # detect_video_list_frcnn(Flags.config_path)


if __name__ == "__main__":
    tf.app.run()




