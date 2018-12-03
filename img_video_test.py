import os
import sys
sys.path.insert(1, 'D:\\project3_faster_rcnn\\models-master\\research\\')
os.chdir('D:\\project3_faster_rcnn\\models-master\\research\\')

__author__ = 'WangZe'

import numpy as np
import tensorflow as tf
from time import time
from matplotlib import pyplot as plt
import cv2
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util


def run_inference_for_single_image(image, sess, tensor_dict):

    # Get handles to input and output tensors
    image_tensor = tf.get_default_graph().get_tensor_by_name('image_tensor:0')
    # Run inference
    output_dict = sess.run(tensor_dict, feed_dict={image_tensor: np.expand_dims(image, 0)})
    # all outputs are float32 numpy arrays, so convert types as appropriate
    output_dict['num_detections'] = int(output_dict['num_detections'][0])
    output_dict['detection_classes'] = output_dict[
      'detection_classes'][0].astype(np.uint8)
    output_dict['detection_boxes'] = output_dict['detection_boxes'][0]
    output_dict['detection_scores'] = output_dict['detection_scores'][0]
    if 'detection_masks' in output_dict:
        output_dict['detection_masks'] = output_dict['detection_masks'][0]
    return output_dict


def detect_image(path, image_output_path, detection_graph):
    time1 = time()
    with detection_graph.as_default():
        with tf.Session() as sess:
            ops = tf.get_default_graph().get_operations()
            all_tensor_names = {output.name for op in ops for output in op.outputs}
            tensor_dict = {}
            for key in ['num_detections', 'detection_boxes', 'detection_scores', 'detection_classes']:
                tensor_name = key + ':0'
                if tensor_name in all_tensor_names:
                    tensor_dict[key] = tf.get_default_graph().get_tensor_by_name(
                        tensor_name)
            image_paths = [path + name for name in os.listdir(path)]
            for image_path in image_paths:
                # result image with boxes and labels on it.
                image_np = cv2.imread(image_path)
                # Actual detection.
                output_dict = run_inference_for_single_image(image_np, sess, tensor_dict)
                # Visualization of the results of a detection.
                vis_util.visualize_boxes_and_labels_on_image_array(
                    image_np,
                    output_dict['detection_boxes'],
                    output_dict['detection_classes'],
                    output_dict['detection_scores'],
                    category_index,
                    instance_masks=output_dict.get('detection_masks'),
                    use_normalized_coordinates=True,
                    line_thickness=5)
                plt.figure(figsize=(12, 8))
                plt.imshow(image_np)
                plt.savefig(image_output_path + 'test_' + os.path.split(image_path)[1])
    process_time = round((time() - time1) / 60, 2)
    print("test time: ", process_time)


def detect_video(video_path, output_path=""):
    time1 = time()
    vid = cv2.VideoCapture(video_path)
    if not vid.isOpened():
        raise IOError("Couldn't open webcam or video")
    video_FourCC    = int(vid.get(cv2.CAP_PROP_FOURCC))
    video_fps       = vid.get(cv2.CAP_PROP_FPS)
    video_size      = (int(vid.get(cv2.CAP_PROP_FRAME_WIDTH)),
                        int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    isOutput = True if output_path != "" else False
    if isOutput:
        print("!!! TYPE:", type(output_path), type(video_FourCC), type(video_fps), type(video_size))
        out = cv2.VideoWriter(output_path, video_FourCC, video_fps, video_size)
    accum_time = 0
    curr_fps = 0
    fps = "FPS: ??"
    prev_time = time()
    with detection_graph.as_default():
        with tf.Session() as sess:
            ops = tf.get_default_graph().get_operations()
            all_tensor_names = {output.name for op in ops for output in op.outputs}
            tensor_dict = {}
            for key in ['num_detections', 'detection_boxes', 'detection_scores', 'detection_classes']:
                tensor_name = key + ':0'
                if tensor_name in all_tensor_names:
                    tensor_dict[key] = tf.get_default_graph().get_tensor_by_name(
                        tensor_name)
            while True:
                return_value, image_np = vid.read()
                ###################
                # Actual detection.
                output_dict = run_inference_for_single_image(image_np, sess, tensor_dict)
                # Visualization of the results of a detection.
                result = vis_util.visualize_boxes_and_labels_on_image_array(
                    image_np,
                    output_dict['detection_boxes'],
                    output_dict['detection_classes'],
                    output_dict['detection_scores'],
                    category_index,
                    instance_masks=output_dict.get('detection_masks'),
                    use_normalized_coordinates=True,
                    line_thickness=5)
                #######################
                curr_time = time()
                exec_time = curr_time - prev_time
                prev_time = curr_time
                accum_time = accum_time + exec_time
                curr_fps = curr_fps + 1
                if accum_time > 1:
                    accum_time = accum_time - 1
                    fps = "FPS: " + str(curr_fps)
                    curr_fps = 0
                cv2.putText(result, text=fps, org=(3, 15), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                            fontScale=0.50, color=(255, 0, 0), thickness=2)
                cv2.namedWindow("result", cv2.WINDOW_NORMAL)
                cv2.imshow("result", result)
                if isOutput:
                    out.write(result)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
    print("cost time: %.2f min" % ((time()-time1)/60))


models = ['faster_rcnn_resnet50', 'faster_rcnn_resnet101', 'faster_rcnn_inception_v2']
i = 0
PATH_TO_FROZEN_GRAPH = "D:\\project3_faster_rcnn\\models-master\\research\\hat_dataset\\saved_models\\" + models[i] + \
                       "\\frozen_inference_graph.pb"
PATH_TO_LABELS = "D:\\project3_faster_rcnn\\models-master\\research\\hat_dataset\\hat_label_map.pbtxt"

detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.GraphDef()
    with tf.gfile.GFile(PATH_TO_FROZEN_GRAPH, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')
category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS, use_display_name=True)


images_dir = "D:\\project3_faster_rcnn\\models-master\\research\\hat_dataset\\test_image\\"
image_output_path = "D:\\project3_faster_rcnn\\models-master\\research\\hat_dataset\\test_output\\" + models[i]+"\\"
detect_image(images_dir, image_output_path, detection_graph)

video_path = "D:\\project3_faster_rcnn\\models-master\\research\\hat_dataset\\video_data\\test_video\\1.mp4"
output_path = "D:\\project3_faster_rcnn\\models-master\\research\\hat_dataset\\video_data\\test_video\\1_"+models[i]+".mp4"
detect_video(video_path, output_path)
