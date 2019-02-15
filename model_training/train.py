import os, sys
from absl import flags
import configparser
import tensorflow as tf

sys.path.insert(1, 'D:\\project3_faster_rcnn\\models-master\\research\\')
__author__ = "WangZe"

from model_training.train_frcnn import frcnn_train

cf = configparser.ConfigParser()
flags.DEFINE_string("config_path",
                    "./hat_detection/model_training/train_configuration.config",
                    "configurations of frcnn testing on video")
Flags = flags.FLAGS


def main(unused_argv):
    cf = configparser.ConfigParser()
    cf.read(Flags.config_path)
    frcnn_train(Flags.config_path)


if __name__ == "__main__":
    tf.app.run()

