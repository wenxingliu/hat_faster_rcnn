import os
from absl import flags
import configparser
import tensorflow as tf

__author__ = "WangZe"

from hat_dataset_yolo3.train import yolo_train
from hat_dataset.train import frcnn_train

cf = configparser.ConfigParser()
flags.DEFINE_string("config_path",
                    "D:\\project3_faster_rcnn\\models-master\\research\\hat_dataset\\train_configuration.config",
                    "configurations of yolo/frcnn testing on video")
Flags = flags.FLAGS


def main(unused_argv):
    cf = configparser.ConfigParser()
    cf.read(Flags.config_path)
    if cf.get("model_selection", "model_name") == "yolo":
        yolo_train(Flags.config_path)
    elif cf.get("model_selection", "model_name") == "frcnn":
        frcnn_train(Flags.config_path)


if __name__ == "__main__":
    tf.app.run()

