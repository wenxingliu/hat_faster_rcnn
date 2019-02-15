import os
from frcnn_detection import *
from absl import flags
import configparser

__author__ = "WangZe"


flags.DEFINE_string("config_path",
                    "./hat_detection/test_configuration.config",
                    "configurations of yolo/frcnn testing on video")
Flags = flags.FLAGS


def main(unused_argv):
    cf = configparser.ConfigParser()
    cf.read(Flags.config_path)

    detect_rstp_video_frcnn(Flags.config_path)


if __name__ == "__main__":
    tf.app.run()




