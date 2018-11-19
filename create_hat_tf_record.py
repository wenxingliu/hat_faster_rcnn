# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

r"""Convert the Oxford pet dataset to TFRecord for object_detection.

See: O. M. Parkhi, A. Vedaldi, A. Zisserman, C. V. Jawahar
     Cats and Dogs
     IEEE Conference on Computer Vision and Pattern Recognition, 2012
     http://www.robots.ox.ac.uk/~vgg/data/pets/

Example usage:
    python object_detection/dataset_tools/create_pet_tf_record.py \
        --data_dir=/home/user/pet \
        --output_dir=/home/user/pet/output
"""
import os
import sys
sys.path.insert(1, 'C:\\dev\\models\\research\\')
os.chdir('C:\\dev\\models\\research\\')

from collections import namedtuple
import io
import pandas as pd
from PIL import Image

import tensorflow as tf
from object_detection.utils import dataset_util


def class_text_to_int(row_label):
    if row_label == 'head':
        return 1


def split(df, group):
    data = namedtuple('data', ['filename', 'object'])
    gb = df.groupby(group)
    return [data(filename, gb.get_group(x)) for filename, x in zip(gb.groups.keys(), gb.groups)]


def create_tf_example(group, path):
    with tf.gfile.GFile(os.path.join(path, '{}'.format(group.filename)), 'rb') as fid:
        encoded_jpg = fid.read()
    encoded_jpg_io = io.BytesIO(encoded_jpg)
    image = Image.open(encoded_jpg_io)
    width, height = image.size

    filename = group.filename.encode('utf8')
    image_format = b'jpg'
    xmins = []
    xmaxs = []
    ymins = []
    ymaxs = []
    classes_text = []
    classes = []

    for index, row in group.object.iterrows():
        xmins.append(row['xmin'] / width)
        xmaxs.append(row['xmax'] / width)
        ymins.append(row['ymin'] / height)
        ymaxs.append(row['ymax'] / height)
        classes_text.append(row['class'].encode('utf8'))
        classes.append(class_text_to_int(row['class']))

    tf_example = tf.train.Example(features=tf.train.Features(feature={
        'image/height': dataset_util.int64_feature(height),
        'image/width': dataset_util.int64_feature(width),
        'image/filename': dataset_util.bytes_feature(filename),
        'image/source_id': dataset_util.bytes_feature(filename),
        'image/encoded': dataset_util.bytes_feature(encoded_jpg),
        'image/format': dataset_util.bytes_feature(image_format),
        'image/object/bbox/xmin': dataset_util.float_list_feature(xmins),
        'image/object/bbox/xmax': dataset_util.float_list_feature(xmaxs),
        'image/object/bbox/ymin': dataset_util.float_list_feature(ymins),
        'image/object/bbox/ymax': dataset_util.float_list_feature(ymaxs),
        'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
        'image/object/class/label': dataset_util.int64_list_feature(classes),
    }))
    return tf_example


def main(val=False):
    images_path = "C:\\dev\models\\research\\hat_dataset\\images\\"
    label_csv_path = "C:\\dev\models\\research\\hat_dataset\\train_labels.csv"

    examples = pd.read_csv(label_csv_path)
    num_train = int(len(examples) * 0.8)

    train_output_path = "C:\\dev\models\\research\\hat_dataset\\TFRecords\\train.record"
    writer = tf.python_io.TFRecordWriter(train_output_path)
    grouped = split(examples[:num_train], 'filename')
    for group in grouped:
        tf_example = create_tf_example(group, images_path)
        writer.write(tf_example.SerializeToString())

    writer.close()
    print('Successfully created the TFRecords: {}'.format(train_output_path))

    if val:
        test_output_path = "C:\\dev\models\\research\\hat_dataset\\TFRecords\\test.record"
        writer = tf.python_io.TFRecordWriter(test_output_path)
        grouped = split(examples[num_train:], 'filename')
        for group in grouped:
            tf_example = create_tf_example(group, images_path)
            writer.write(tf_example.SerializeToString())

        writer.close()
        print('Successfully created the TFRecords: {}'.format(test_output_path))


if __name__ == '__main__':
    main()
