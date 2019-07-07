import os
import sys
sys.path.insert(1, 'D:\\project3_faster_rcnn\\models-master\\research\\')
os.chdir('D:\\project3_faster_rcnn\\models-master\\research\\')

__author__ = 'WangZe'

import configparser
import numpy as np
import tensorflow as tf
import time
from matplotlib import pyplot as plt
from collections import deque
import cv2
from object_detection.utils import label_map_util
from heuristic_filter import visualize, filtered_box_stage1, filtered_box_stage2


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


def get_detection_graph_and_index(PATH_TO_FROZEN_GRAPH, PATH_TO_LABELS):
    detection_graph = tf.Graph()
    with detection_graph.as_default():
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile(PATH_TO_FROZEN_GRAPH, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')
    category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS, use_display_name=True)
    return detection_graph, category_index


def detect_image_frcnn(config_path):
    cf = configparser.ConfigParser()
    cf.read(config_path)
    frozen_graph_path = cf.get("faster_rcnn_model", "PATH_TO_FROZEN_GRAPH")
    labels_path = cf.get("faster_rcnn_model", "PATH_TO_LABELS")
    detection_graph, category_index = get_detection_graph_and_index(frozen_graph_path, labels_path)

    path = cf.get("test_image_path", "image_path")
    image_output_path = cf.get("test_image_path", "output_image_path")

    use_normalized_coordinates = cf.getboolean("faster_rcnn_model", "use_normalized_coordinates")
    nms_threshold = cf.getfloat("test_video_config", "nms_threshold")
    min_prob = cf.getfloat("test_video_config", "min_prob")
    min_x = cf.getfloat("test_video_config", "min_x")
    min_y = cf.getfloat("test_video_config", "min_y")
    min_area = cf.getfloat("test_video_config", "min_area")

    image_output_path = os.path.abspath(image_output_path) + "\\"
    detection_mode = 'image'
    video_fps = None
    video_size = None
    time1 = time.time()
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
            image_paths = [os.path.abspath(path) + "\\" + name for name in os.listdir(path)]
            for i, image_path in enumerate(image_paths):
                # result image with boxes and labels on it.
                image_np = cv2.imread(image_path)
                # Actual detection.
                output_dict = run_inference_for_single_image(image_np, sess, tensor_dict)
                # Visualization of the results of a detection.
                filtered_outputdict, result = filtered_box_stage1(detection_mode, output_dict, image_path,
                                                                  video_fps, video_size, use_normalized_coordinates,
                                                                  nms_threshold,  min_prob, min_x, min_y, min_area)
                image = visualize(
                    image_np,
                    filtered_outputdict['boxes'],
                    filtered_outputdict['classes'],
                    filtered_outputdict['scores'],
                    category_index,
                    use_normalized_coordinates=True,
                    line_thickness=3,
                )
                no_hat_num = result[-2]
                plt.figure(figsize=(12, 8))
                plt.imshow(image)
                cv2.imwrite(image_output_path + str(i) + '_nohat_nums=' + str(no_hat_num) + ".jpg", image)
    process_time = round((time.time() - time1) / 60, 2)
    print("test time: ", process_time)


def detect_video_frcnn(config_path):
    cf = configparser.ConfigParser()
    cf.read(os.path.abspath(config_path))
    frozen_graph_path = cf.get("faster_rcnn_model", "PATH_TO_FROZEN_GRAPH")
    labels_path = cf.get("faster_rcnn_model", "PATH_TO_LABELS")
    detection_graph, category_index = get_detection_graph_and_index(frozen_graph_path, labels_path)

    video_path = cf.get("frcnn_test_video_path", "ts_video_path")
    # output_path = cf.get("frcnn_test_video_path", "output_image_path")
    out_log_path = cf.get("frcnn_test_video_path", "output_info_path")

    frame_num_for_judge = cf.getint("test_video_config", "frame_num_for_judge")
    use_normalized_coordinates = cf.getboolean("faster_rcnn_model", "use_normalized_coordinates")
    nms_threshold = cf.getfloat("test_video_config", "nms_threshold")
    min_prob = cf.getfloat("test_video_config", "min_prob")
    min_x = cf.getfloat("test_video_config", "min_x")
    min_y = cf.getfloat("test_video_config", "min_y")
    min_area = cf.getfloat("test_video_config", "min_area")
    min_ratio = cf.getfloat("test_video_config", "min_ratio")
    frame_interval = cf.getint("test_video_config", "interval")

    detect_mode = "video"
    time1 = time.time()
    vid = cv2.VideoCapture(video_path)
    if not vid.isOpened():
        raise IOError("Couldn't open webcam or video")
    video_FourCC = int(vid.get(cv2.CAP_PROP_FOURCC))
    video_length = int(vid.get(cv2.CAP_PROP_FRAME_COUNT))
    video_fps = vid.get(cv2.CAP_PROP_FPS)
    video_size = (int(vid.get(cv2.CAP_PROP_FRAME_WIDTH)),
                        int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    # isOutput = True if output_path != "" else False
    # if isOutput:
    #     print("!!! TYPE:", type(output_path), type(video_FourCC), type(video_fps), type(video_size))
    #     out = cv2.VideoWriter(output_path, video_FourCC, video_fps, video_size)
    accum_time = 0
    curr_fps = 0
    prev_time = time.time()
    frame_id = -1
    results = deque()
    frames = deque()
    output_info = []
    video_name = os.path.split(video_path)[1]
    save_path = os.path.join(out_log_path, video_name + "_test")
    if os.path.exists(save_path):
        import shutil
        shutil.rmtree(save_path, ignore_errors=True)
    os.makedirs(save_path)
    logs = open(save_path + "\\" + video_name + "_log.txt", "w")
    logs.write("video name:" + video_name)
    logs.write("\nvideo path:" + os.path.abspath(video_path))
    logs.write("\nvideo fps: " + str(video_fps))
    logs.write("\nvideo length: " + str(int(video_length/video_fps)) + "s\n\n")
    # logs.write("                " + str(int(video_length / video_fps/60)) + "min")
    save_image = True
    detection_mode = 'video'
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
                frame_id += 1
                if frame_id % frame_interval != 0:
                    # if frame_id % 50 > 10:
                    continue
                if return_value:
                    ###################
                    # Actual detection.
                    output_dict = run_inference_for_single_image(image_np, sess, tensor_dict)
                    filtered_outputdict, result = filtered_box_stage1(detection_mode, output_dict, frame_id,
                                                                      video_fps,
                                                                      video_size,
                                                                      use_normalized_coordinates,
                                                                      nms_threshold,
                                                                      min_prob, min_x,
                                                                      min_y, min_area)
                    image = visualize(
                        image_np,
                        filtered_outputdict['boxes'],
                        filtered_outputdict['classes'],
                        filtered_outputdict['scores'],
                        category_index,
                        use_normalized_coordinates,
                        line_thickness=3,
                    )
                    results.append(result)
                    frames.append(image)
                    if len(results) >= frame_num_for_judge:
                        no_hat_time_, no_hat_num, boxes = filtered_box_stage2(results, frame_num_for_judge, min_ratio)
                        # if no_hat_time_==True, no_hat chefs exists.
                        if no_hat_time_ and save_image:
                            print("Found %d chefs without hats " % no_hat_num)
                            cv2.imwrite(save_path + "\\" + video_name + "_" + str(frame_id) + '.jpg', frames[-1])
                        results.popleft()
                        frames.popleft()

                    #######################
                    curr_time = time.time()
                    exec_time = curr_time - prev_time
                    prev_time = curr_time
                    accum_time = accum_time + exec_time
                    curr_fps = curr_fps + 1
                    if accum_time > 1:
                        accum_time = accum_time - 1
                        fps = "FPS: " + str(curr_fps)
                        curr_fps = 0
                    # cv2.putText(image, text=fps, org=(3, 15), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    #             fontScale=0.50, color=(255, 0, 0), thickness=2)
                    # cv2.namedWindow("result", cv2.WINDOW_NORMAL)
                    # cv2.imshow("result", image)
                    # if isOutput:
                    #     out.write(image)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                else:
                    break
    boxes_max = []
    if boxes:
        for i in range(len(boxes)):
            boxes_info = np.array(boxes[i]).max(axis=0)
            boxes_max.append(boxes_info)
    res = dict()
    res["folder"] = os.path.abspath(out_log_path)
    res["box"] = boxes_max
    testing_time = (time.time() - time1) / 60
    logs.write(str(res))
    # logs.write("\ntest time: " + str(round(testing_time, 2)) + "min\n")
    logs.close()
    print("cost time: %.2f min" % testing_time)


def detect_video_list_frcnn(config_path):
    cf = configparser.ConfigParser()
    cf.read(config_path)
    video_list_path = cf.get("frcnn_test_video_path", "video_list_path")
    video_list_paths = [os.path.join(video_list_path, path) for path in os.listdir(video_list_path)]

    frozen_graph_path = cf.get("faster_rcnn_model", "PATH_TO_FROZEN_GRAPH")
    labels_path = cf.get("faster_rcnn_model", "PATH_TO_LABELS")
    detection_graph, category_index = get_detection_graph_and_index(frozen_graph_path, labels_path)

    output_path = cf.get("frcnn_test_video_path", "output_video_path")
    out_log_path = cf.get("frcnn_test_video_path", "out_log_path")

    frame_num_for_judge = cf.getint("test_video_config", "frame_num_for_judge")
    use_normalized_coordinates = cf.getboolean("faster_rcnn_model", "use_normalized_coordinates")
    nms_threshold = cf.getfloat("test_video_config", "nms_threshold")
    min_prob = cf.getfloat("test_video_config", "min_prob")
    min_x = cf.getfloat("test_video_config", "min_x")
    min_y = cf.getfloat("test_video_config", "min_y")
    min_area = cf.getfloat("test_video_config", "min_area")
    min_ratio = cf.getfloat("test_video_config", "min_ratio")
    frame_interval = cf.getint("test_video_config", "interval")

    for video_path in video_list_paths:
        detect_mode = "video"
        time1 = time.time()
        vid = cv2.VideoCapture(video_path)
        if not vid.isOpened():
            raise IOError("Couldn't open webcam or video")
        video_FourCC = int(vid.get(cv2.CAP_PROP_FOURCC))
        video_length = int(vid.get(cv2.CAP_PROP_FRAME_COUNT))
        video_fps = vid.get(cv2.CAP_PROP_FPS)
        video_size = (int(vid.get(cv2.CAP_PROP_FRAME_WIDTH)),
                      int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        isOutput = True if output_path != "" else False
        if isOutput:
            print("!!! TYPE:", type(output_path), type(video_FourCC), type(video_fps), type(video_size))
            out = cv2.VideoWriter(output_path, video_FourCC, video_fps, video_size)
        accum_time = 0
        curr_fps = 0
        fps = "FPS: ??"
        prev_time = time.time()
        frame_id = 0
        results = deque()
        frames = deque()
        output_info = []
        video_name = os.path.split(video_path)[1]
        save_path = os.path.join(out_log_path, video_name + "_test")
        if os.path.exists(save_path):
            import shutil
            shutil.rmtree(save_path, ignore_errors=True)
        os.makedirs(save_path)
        logs = open(save_path + "\\" + video_name + "_log.txt", "w")
        logs.write("video name:" + video_name)
        logs.write("\nvideo path:" + video_path)
        logs.write("\nvideo fps: " + str(video_fps))
        logs.write("\nvideo length: " + str(int(video_length / video_fps)) + "s")
        # logs.write("                " + str(int(video_length / video_fps/60)) + "min")
        save_image = True
        detection_mode = 'video'
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
                    frame_id += 1
                    if frame_id % frame_interval != 0:
                        # if frame_id % 50 > 10:
                        continue
                    if return_value:
                        ###################
                        # Actual detection.
                        output_dict = run_inference_for_single_image(image_np, sess, tensor_dict)
                        filtered_outputdict, result = filtered_box_stage1(detection_mode, output_dict, frame_id,
                                                                          video_fps,
                                                                          video_size,
                                                                          use_normalized_coordinates,
                                                                          nms_threshold,
                                                                          min_prob, min_x,
                                                                          min_y, min_area)
                        image = visualize(
                            image_np,
                            filtered_outputdict['boxes'],
                            filtered_outputdict['classes'],
                            filtered_outputdict['scores'],
                            category_index,
                            use_normalized_coordinates,
                            line_thickness=3,
                        )
                        results.append(result)
                        frames.append(image)
                        if len(results) >= frame_num_for_judge:
                            no_hat_time_, no_hat_num, id = filtered_box_stage2(results, frame_num_for_judge, min_ratio)
                            # if no_hat_time_==True, at this time, there are no_hat chefs
                            if no_hat_time_:
                                output_info.append([no_hat_time_, no_hat_num])
                                print("Found %d chefs without hats " % no_hat_num)
                                cv2.imwrite(save_path + "\\" + video_name + "_" + str(frame_id) + '.jpg', frames[id])
                            results.popleft()
                            frames.popleft()

                        #######################
                        curr_time = time.time()
                        exec_time = curr_time - prev_time
                        prev_time = curr_time
                        accum_time = accum_time + exec_time
                        curr_fps = curr_fps + 1
                        if accum_time > 1:
                            accum_time = accum_time - 1
                            fps = "FPS: " + str(curr_fps)
                            curr_fps = 0
                        cv2.putText(image, text=fps, org=(3, 15), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                                    fontScale=0.50, color=(255, 0, 0), thickness=2)
                        cv2.namedWindow("result", cv2.WINDOW_NORMAL)
                        cv2.imshow("result", image)
                        if isOutput:
                            out.write(image)
                        if cv2.waitKey(1) & 0xFF == ord('q'):
                            break
                    else:
                        break
        testing_time = (time.time() - time1) / 60
        logs.write("\ntest time: " + str(round(testing_time, 2)) + "min\n")
        logs.close()
        print("cost time: %.2f min" % testing_time)


def detect_rstp_video_frcnn_on_web(config_path):
    cf = configparser.ConfigParser()
    cf.read(config_path)
    frozen_graph_path = cf.get("faster_rcnn_model", "PATH_TO_FROZEN_GRAPH")
    labels_path = cf.get("faster_rcnn_model", "PATH_TO_LABELS")
    detection_graph, category_index = get_detection_graph_and_index(frozen_graph_path, labels_path)

    video_rstp_address = cf.get("rstp_video_address", "video_rstp_address1")
    output_path = cf.get("rstp_video_address", "output_video_path")
    out_log_path = cf.get("rstp_video_address", "out_log_path")

    frame_num_for_judge = cf.getint("test_video_config", "frame_num_for_judge")
    use_normalized_coordinates = cf.getboolean("faster_rcnn_model", "use_normalized_coordinates")
    nms_threshold = cf.getfloat("test_video_config", "nms_threshold")
    min_prob = cf.getfloat("test_video_config", "min_prob")
    min_x = cf.getfloat("test_video_config", "min_x")
    min_y = cf.getfloat("test_video_config", "min_y")
    min_area = cf.getfloat("test_video_config", "min_area")
    min_ratio = cf.getfloat("test_video_config", "min_ratio")
    frame_interval = cf.getint("test_video_config", "interval")
    detect_mode = "video"
    time1 = time.time()
    vid = cv2.VideoCapture(video_rstp_address)
    if not vid.isOpened():
        raise IOError("Couldn't open webcam or video")
    video_FourCC = int(vid.get(cv2.CAP_PROP_FOURCC))
    video_fps = vid.get(cv2.CAP_PROP_FPS)
    video_size = (int(vid.get(cv2.CAP_PROP_FRAME_WIDTH)),
                        int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    isOutput = True if output_path != "" else False
    if isOutput:
        print("!!! TYPE:", type(output_path), type(video_FourCC), type(video_fps), type(video_size))
        out = cv2.VideoWriter(output_path, video_FourCC, video_fps, video_size)
    accum_time = 0
    curr_fps = 0
    fps = "FPS: ??"
    prev_time = time.time()
    frame_id = -1
    results = deque()
    frames = deque()
    output_info = []
    video_name = "rstp2"
    save_path = os.path.join(out_log_path, video_name + "_test")
    # if os.path.exists(save_path):
    #     import shutil
    #     shutil.rmtree(save_path, ignore_errors=True)
    #     print("remove dir: " + save_path)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    # os.makedirs(save_path)
    logs = open(save_path + "\\" + video_name + "_log.txt", "w")
    import datetime
    logs.write("detection start time:" + datetime.datetime.now().ctime())
    logs.write("video_rstp_address:" + video_rstp_address)
    logs.write("\nvideo fps: " + str(video_fps))
    save_image = True
    detection_mode = 'video'
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
                frame_id += (frame_id % 5 * 3600) + 1
                if frame_id % frame_interval != 0:
                    continue
                if return_value:
                    ###################
                    # Actual detection.
                    output_dict = run_inference_for_single_image(image_np, sess, tensor_dict)
                    filtered_outputdict, result = filtered_box_stage1(detection_mode, output_dict, frame_id, video_fps,
                                                                      video_size,
                                                                      use_normalized_coordinates,
                                                                      nms_threshold,
                                                                      min_prob, min_x,
                                                                      min_y, min_area)
                    image = visualize(
                            image_np,
                            filtered_outputdict['boxes'],
                            filtered_outputdict['classes'],
                            filtered_outputdict['scores'],
                            category_index,
                            use_normalized_coordinates,
                            line_thickness=3,
                            )
                    results.append(result)
                    frames.append(image)
                    if len(results) >= frame_num_for_judge:
                        no_hat_time_, no_hat_num, id = filtered_box_stage2(results, frame_num_for_judge, min_ratio)
                        # if no_hat_time_==True, at this time, there are no_hat chefs
                        if no_hat_time_:
                            output_info.append([no_hat_time_, no_hat_num])
                            now_time = datetime.datetime.now()
                            date = str(now_time.month) + "_" + str(now_time.day) + "_" + str(now_time.hour)\
                                + "_" + str(now_time.minute) + "_" + str(now_time.second)
                            print("Found %d chefs without hats at " % no_hat_num, now_time.ctime())
                            #cv2.imwrite(save_path + "\\" + video_name + "_time=" +
                            #            date + "_no_hats" + '.jpg', frames[id])
                        results.popleft()
                        frames.popleft()

                    #######################
                    curr_time = time.time()
                    exec_time = curr_time - prev_time
                    prev_time = curr_time
                    accum_time = accum_time + exec_time
                    curr_fps = curr_fps + 1
                    if accum_time > 1:
                        accum_time = accum_time - 1
                        fps = "FPS: " + str(curr_fps)
                        curr_fps = 0
                    """
                    cv2.putText(image, text=fps, org=(3, 15), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                                fontScale=0.50, color=(255, 0, 0), thickness=2)
                    cv2.namedWindow("result", cv2.WINDOW_NORMAL)
                    cv2.imshow("result", image)
                    """
                    # output to show on web
                    ret, jpeg = cv2.imencode('.jpg', image)
                    img_byte = jpeg.tobytes()
                    # yield img_byte
                    out_web = (b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + img_byte + b'\r\n\r\n')
                    yield out_web
                    if isOutput:
                        out.write(image)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                else:
                    break

    testing_time = (time.time() - time1) / 60
    logs.write("\ntest time: " + str(round(testing_time, 2)) + "min\n")
    logs.close()
    print("cost time: %.2f min" % testing_time)

def detect_rstp_video_frcnn(config_path):
    cf = configparser.ConfigParser()
    cf.read(config_path)
    frozen_graph_path = cf.get("faster_rcnn_model", "PATH_TO_FROZEN_GRAPH")
    labels_path = cf.get("faster_rcnn_model", "PATH_TO_LABELS")
    detection_graph, category_index = get_detection_graph_and_index(frozen_graph_path, labels_path)

    video_rstp_address = cf.get("rstp_video_address", "video_rstp_address1")
    output_path = cf.get("rstp_video_address", "output_video_path")
    out_log_path = cf.get("rstp_video_address", "out_log_path")

    frame_num_for_judge = cf.getint("test_video_config", "frame_num_for_judge")
    use_normalized_coordinates = cf.getboolean("faster_rcnn_model", "use_normalized_coordinates")
    nms_threshold = cf.getfloat("test_video_config", "nms_threshold")
    min_prob = cf.getfloat("test_video_config", "min_prob")
    min_x = cf.getfloat("test_video_config", "min_x")
    min_y = cf.getfloat("test_video_config", "min_y")
    min_area = cf.getfloat("test_video_config", "min_area")
    min_ratio = cf.getfloat("test_video_config", "min_ratio")
    frame_interval = cf.getint("test_video_config", "interval")
    detect_mode = "video"
    time1 = time.time()
    vid = cv2.VideoCapture(video_rstp_address)
    if not vid.isOpened():
        raise IOError("Couldn't open webcam or video")
    video_FourCC = int(vid.get(cv2.CAP_PROP_FOURCC))
    video_fps = vid.get(cv2.CAP_PROP_FPS)
    video_size = (int(vid.get(cv2.CAP_PROP_FRAME_WIDTH)),
                        int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    isOutput = True if output_path != "" else False
    if isOutput:
        print("!!! TYPE:", type(output_path), type(video_FourCC), type(video_fps), type(video_size))
        out = cv2.VideoWriter(output_path, video_FourCC, video_fps, video_size)
    accum_time = 0
    curr_fps = 0
    fps = "FPS: ??"
    prev_time = time.time()
    frame_id = -1
    results = deque()
    frames = deque()
    output_info = []
    video_name = "rstp2"
    save_path = os.path.join(out_log_path, video_name + "_test")
    if os.path.exists(save_path):
        import shutil
        shutil.rmtree(save_path, ignore_errors=True)
        print("remove dir: " + save_path)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    # os.makedirs(save_path)
    logs = open(save_path + "\\" + video_name + "_log.txt", "w")
    import datetime
    logs.write("detection start time:" + datetime.datetime.now().ctime())
    logs.write("video_rstp_address:" + video_rstp_address)
    logs.write("\nvideo fps: " + str(video_fps))
    save_image = True
    detection_mode = 'video'
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
                # 防止frame_id过大
                frame_id += (frame_id % 5*3600) + 1
                if frame_id % frame_interval != 0:
                # if frame_id % 50 > 10:
                    continue

                if return_value:
                    ###################
                    # Actual detection.
                    output_dict = run_inference_for_single_image(image_np, sess, tensor_dict)
                    filtered_outputdict, result = filtered_box_stage1(detection_mode, output_dict, frame_id, video_fps,
                                                                      video_size,
                                                                      use_normalized_coordinates,
                                                                      nms_threshold,
                                                                      min_prob, min_x,
                                                                      min_y, min_area)
                    image = visualize(
                            image_np,
                            filtered_outputdict['boxes'],
                            filtered_outputdict['classes'],
                            filtered_outputdict['scores'],
                            category_index,
                            use_normalized_coordinates,
                            line_thickness=3,
                            )
                    results.append(result)
                    frames.append(image)
                    if len(results) >= frame_num_for_judge:
                        no_hat_time_, no_hat_num, id = filtered_box_stage2(results, frame_num_for_judge, min_ratio)
                        # if no_hat_time_==True, at this time, there are no_hat chefs
                        if no_hat_time_:
                            output_info.append([no_hat_time_, no_hat_num])
                            now_time = datetime.datetime.now()
                            date = str(now_time.month) + "_" + str(now_time.day) + "_" + str(now_time.hour)\
                                + "_" + str(now_time.minute) + "_" + str(now_time.second)
                            print("Found %d chefs without hats at " % no_hat_num, now_time.ctime())
                            if save_image:
                                cv2.imwrite(save_path + "\\" + video_name + "_time=" +
                                           date + "_no_hats" + '.jpg', frames[id])
                        results.popleft()
                        frames.popleft()

                    #######################
                    curr_time = time.time()
                    exec_time = curr_time - prev_time
                    prev_time = curr_time
                    accum_time = accum_time + exec_time
                    curr_fps = curr_fps + 1
                    if accum_time > 1:
                        accum_time = accum_time - 1
                        fps = "FPS: " + str(curr_fps)
                        curr_fps = 0
                    cv2.putText(image, text=fps, org=(3, 15), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                                fontScale=0.50, color=(255, 0, 0), thickness=2)
                    cv2.namedWindow("result", cv2.WINDOW_NORMAL)
                    cv2.imshow("result", image)
                    if isOutput:
                        out.write(image)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                else:
                    break

    testing_time = (time.time() - time1) / 60
    logs.write("\ntest time: " + str(round(testing_time, 2)) + "min\n")
    logs.close()
    print("cost time: %.2f min" % testing_time)





