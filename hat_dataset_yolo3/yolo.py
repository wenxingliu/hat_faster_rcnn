# -*- coding: utf-8 -*-
"""
Class definition of YOLO_v3 style detection model on image and video
"""
import configparser
import colorsys
import os
from timeit import default_timer as timer
from collections import deque
import numpy as np
from keras import backend as K
from keras.models import load_model
from keras.layers import Input
from PIL import Image, ImageFont, ImageDraw

from hat_dataset_yolo3.yolo3.model import yolo_eval, yolo_body, tiny_yolo_body
from hat_dataset_yolo3.yolo3.utils import letterbox_image, aug_bbox_range
from keras.utils import multi_gpu_model
from heuristic_filter import visualize, filtered_box_stage1, filtered_box_stage2

__author__ = 'WangZe'


class YOLO(object):
    @classmethod
    def get_defaults(cls, n):
        if n in cls._defaults:
            return cls._defaults[n]
        else:
            return "Unrecognized attribute name '" + n + "'"

    def __init__(self, config_path):
        # self.__dict__.update(self._defaults)  # set up default values
        # self.__dict__.update(kwargs)  # and update with user overrides
        # config_path = "test_configuration.config"
        cf = configparser.ConfigParser()
        cf.read(config_path)
        secton = cf.sections()
        self.model_path = cf.get("yolo_v3", "model_path")
        self.anchors_path = cf.get("yolo_v3", "anchor_path")
        self.classes_path = cf.get("yolo_v3", "classes_path")
        self.score = cf.getfloat("yolo_v3", "score")
        self.iou = cf.getfloat("yolo_v3", "iou")
        self.gpu_num = cf.getint("yolo_v3", "gpu_num")
        self.model_image_size = (416, 416)

        self.class_names = self._get_class()
        self.anchors = self._get_anchors()
        self.sess = K.get_session()
        self.boxes, self.scores, self.classes = self.generate()

    def _get_class(self):
        classes_path = os.path.expanduser(self.classes_path)
        with open(classes_path) as f:
            class_names = f.readlines()
        class_names = [c.strip() for c in class_names]
        return class_names

    def _get_anchors(self):
        anchors_path = os.path.expanduser(self.anchors_path)
        with open(anchors_path) as f:
            anchors = f.readline()
        anchors = [float(x) for x in anchors.split(',')]
        return np.array(anchors).reshape(-1, 2)

    def generate(self):
        model_path = os.path.expanduser(self.model_path)
        assert model_path.endswith('.h5'), 'Keras model or weights must be a .h5 file.'

        # Load model, or construct model and load weights.
        num_anchors = len(self.anchors)
        num_classes = len(self.class_names)
        is_tiny_version = num_anchors==6 # default setting
        try:
            self.yolo_model = load_model(model_path, compile=False)
        except:
            self.yolo_model = tiny_yolo_body(Input(shape=(None,None,3)), num_anchors//2, num_classes) \
                if is_tiny_version else yolo_body(Input(shape=(None,None,3)), num_anchors//3, num_classes)
            self.yolo_model.load_weights(self.model_path) # make sure model, anchors and classes match
        else:
            assert self.yolo_model.layers[-1].output_shape[-1] == \
                num_anchors/len(self.yolo_model.output) * (num_classes + 5), \
                'Mismatch between model and given anchor and class sizes'
        # print(self.yolo_model.summary())
        print('{} model, anchors, and classes loaded.'.format(model_path))

        # Generate colors for drawing bounding boxes.
        hsv_tuples = [(x / len(self.class_names), 1., 1.)
                      for x in range(len(self.class_names))]
        self.colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
        self.colors = list(
            map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)),
                self.colors))
        np.random.seed(10101)  # Fixed seed for consistent colors across runs.
        np.random.shuffle(self.colors)  # Shuffle colors to decorrelate adjacent classes.
        np.random.seed(None)  # Reset seed to default.

        # Generate output tensor targets for filtered bounding boxes.
        self.input_image_shape = K.placeholder(shape=(2, ))
        if self.gpu_num >= 2:
            self.yolo_model = multi_gpu_model(self.yolo_model, gpus=self.gpu_num)
        boxes, scores, classes = yolo_eval(self.yolo_model.output, self.anchors,
                                           len(self.class_names), self.input_image_shape,
                score_threshold=self.score, iou_threshold=self.iou)
        return boxes, scores, classes

    def detect_image(self, image):
        start = timer()

        if self.model_image_size != (None, None):
            assert self.model_image_size[0] % 32 == 0, 'Multiples of 32 required'
            assert self.model_image_size[1] % 32 == 0, 'Multiples of 32 required'
            boxed_image = letterbox_image(image, tuple(reversed(self.model_image_size)))
        else:
            new_image_size = (image.width - (image.width % 32),
                              image.height - (image.height % 32))
            boxed_image = letterbox_image(image, new_image_size)
        image_data = np.array(boxed_image, dtype='float32')

        print(image_data.shape)
        image_data /= 255.
        image_data = np.expand_dims(image_data, 0)  # Add batch dimension.

        out_boxes, out_scores, out_classes = self.sess.run(
            [self.boxes, self.scores, self.classes],
            feed_dict={
                self.yolo_model.input: image_data,
                self.input_image_shape: [image.size[1], image.size[0]],
                K.learning_phase(): 0
            })

        print('Found {} boxes for {}'.format(len(out_boxes), 'img'))

        font = ImageFont.truetype(font='font/FiraMono-Medium.otf',
                                  size=np.floor(3e-2 * image.size[1] + 0.5).astype('int32'))
        thickness = (image.size[0] + image.size[1]) // 300

        for i, c in reversed(list(enumerate(out_classes))):
            predicted_class = self.class_names[c]
            box = out_boxes[i]
            score = out_scores[i]

            label = '{} {:.2f}'.format(predicted_class, score)
            draw = ImageDraw.Draw(image)
            label_size = draw.textsize(label, font)

            top, left, bottom, right = box
            top = max(0, np.floor(top + 0.5).astype('int32'))
            left = max(0, np.floor(left + 0.5).astype('int32'))
            bottom = min(image.size[1], np.floor(bottom + 0.5).astype('int32'))
            right = min(image.size[0], np.floor(right + 0.5).astype('int32'))
            print(label, (left, top), (right, bottom))

            if top - label_size[1] >= 0:
                text_origin = np.array([left, top - label_size[1]])
            else:
                text_origin = np.array([left, top + 1])

            # My kingdom for a good redistributable image drawing library.
            for i in range(thickness):
                draw.rectangle(
                    [left + i, top + i, right - i, bottom - i],
                    outline=self.colors[c])
            draw.rectangle(
                [tuple(text_origin), tuple(text_origin + label_size)],
                fill=self.colors[c])
            draw.text(text_origin, label, fill=(0, 0, 0), font=font)
            del draw

        end = timer()
        print(end - start)
        return image

    def crop_image(self, image):
        start = timer()

        if self.model_image_size != (None, None):
            assert self.model_image_size[0] % 32 == 0, 'Multiples of 32 required'
            assert self.model_image_size[1] % 32 == 0, 'Multiples of 32 required'
            boxed_image = letterbox_image(image, tuple(reversed(self.model_image_size)))
        else:
            new_image_size = (image.width - (image.width % 32),
                              image.height - (image.height % 32))
            boxed_image = letterbox_image(image, new_image_size)
        image_data = np.array(boxed_image, dtype='float32')

        print(image_data.shape)
        image_data /= 255.
        image_data = np.expand_dims(image_data, 0)  # Add batch dimension.

        out_boxes, out_scores, out_classes = self.sess.run(
            [self.boxes, self.scores, self.classes],
            feed_dict={
                self.yolo_model.input: image_data,
                self.input_image_shape: [image.size[1], image.size[0]],
                K.learning_phase(): 0
            })

        print('Found {} boxes for {}'.format(len(out_boxes), 'img'))
        image_size = [image.size[1], image.size[0]]
        cropped_imgs = []
        for bbox in out_boxes:
            top, left, bottom, right = bbox
            top = max(0, np.floor(top + 0.5).astype('int32'))
            left = max(0, np.floor(left + 0.5).astype('int32'))
            bottom = min(image.size[1], np.floor(bottom + 0.5).astype('int32'))
            right = min(image.size[0], np.floor(right + 0.5).astype('int32'))
            cropped_img = np.array(image)[int(top): int(bottom), int(left): int(right)]
            # img = Image.fromarray(cropped_img)
            cropped_imgs.append(cropped_img)
        end = timer()
        print(end - start)
        return cropped_imgs

    def close_session(self):
        self.sess.close()


def detect_video(yolo, video_path, output_path=""):
    import cv2
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
    prev_time = timer()
    while True:
        return_value, frame = vid.read()
        image = Image.fromarray(frame)
        image = yolo.detect_image(image)
        result = np.asarray(image)
        curr_time = timer()
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
    yolo.close_session()


def detect_video_yolo(config_path):
    yolo = YOLO(config_path)
    cf = configparser.ConfigParser()
    cf.read(config_path)
    video_path = cf.get("yolo_test_video_path", "video_path")
    output_path = cf.get("yolo_test_video_path", "output_video_path")
    out_log_path = cf.get("yolo_test_video_path", "out_log_path")

    frame_num_for_judge = cf.getint("test_video_config", "frame_num_for_judge")
    use_normalized_coordinates = cf.getboolean("yolo_v3", "use_normalized_coordinates")
    nms_threshold = cf.getfloat("test_video_config", "nms_threshold")
    min_prob = cf.getfloat("test_video_config", "min_prob")
    min_x = cf.getfloat("test_video_config", "min_x")
    min_y = cf.getfloat("test_video_config", "min_y")
    min_area = cf.getfloat("test_video_config", "min_area")
    min_ratio = cf.getfloat("test_video_config", "min_ratio")
    frame_interval = cf.getint("test_video_config", "interval")

    import time
    import cv2
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
    logs.write("\nvideo length: " + str(int(video_length/video_fps)) + "s")
    logs.write("                " + str(int(video_length / video_fps/60)) + "min")
    save_image = True
    detection_mode = 'video'
    while True:
        return_value, image_np = vid.read()
        frame_id += 1
        if frame_id % frame_interval != 0:
            continue
        if return_value:
        ###################
        # Actual detection.
            image = Image.fromarray(image_np)
            output_dict = generate_output_dict(yolo, image)
            filtered_outputdict, result = filtered_box_stage1(detection_mode, output_dict, frame_id, video_fps,
                                                              video_size, use_normalized_coordinates,
                                                              nms_threshold, min_prob, min_x, min_y, min_area)
            image = visualize(
                    image_np,
                    filtered_outputdict['boxes'],
                    filtered_outputdict['classes'],
                    filtered_outputdict['scores'],
                    category_index=None,
                    use_normalized_coordinates=False,
                    line_thickness=3,
                    )
            results.append(result)
            frames.append(image)
            if len(results) == frame_num_for_judge:
                no_hat_time_, no_hat_nums = filtered_box_stage2(results, frame_num_for_judge, min_ratio)
                # if no_hat_time_==True, at this time, there are no_hat chefs
                if no_hat_time_:
                    output_info.append([no_hat_time_, no_hat_nums])
                    print("Found %d chefs without hats at %.2f seconds;" % (no_hat_nums, no_hat_time_))
                    logs.write("\nFound %d chefs without hats at %.2f seconds." % (no_hat_nums, no_hat_time_))
                    if save_image:
                        for j, img in enumerate(frames):
                            cv2.imwrite(save_path + "\\" + video_name + "_time=" + str(no_hat_time_) +
                                        "no_hats=" + str(no_hat_nums) + "_" + str(j) + '.jpg', img)
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


def detect_video_list_yolo(config_path):
    yolo = YOLO(config_path)
    cf = configparser.ConfigParser()
    cf.read(config_path)
    video_list_path = cf.get("yolo_test_video_path", "video_list_path")
    output_path = cf.get("yolo_test_video_path", "output_video_path")
    out_log_path = cf.get("yolo_test_video_path", "out_log_path")

    frame_num_for_judge = cf.getint("test_video_config", "frame_num_for_judge")
    use_normalized_coordinates = cf.getboolean("yolo_v3", "use_normalized_coordinates")
    nms_threshold = cf.getfloat("test_video_config", "nms_threshold")
    min_prob = cf.getfloat("test_video_config", "min_prob")
    min_x = cf.getfloat("test_video_config", "min_x")
    min_y = cf.getfloat("test_video_config", "min_y")
    min_area = cf.getfloat("test_video_config", "min_area")
    min_ratio = cf.getfloat("test_video_config", "min_ratio")
    frame_interval = cf.getint("test_video_config", "interval")

    # video_list_path = os.listdir(video_list_path)
    video_list_path = [os.path.join(video_list_path, path) for path in os.listdir(video_list_path)]
    import time
    import cv2
    for video_path in video_list_path:

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
        logs.write("                " + str(int(video_length / video_fps / 60)) + "min")
        save_image = True
        detection_mode = 'video'
        while True:
            return_value, image_np = vid.read()
            frame_id += 1
            if frame_id % 50 > 10:
            # if frame_id % frame_interval != 0:
                continue
            if return_value:
                ###################
                # Actual detection.
                image = Image.fromarray(image_np)
                output_dict = generate_output_dict(yolo, image)
                filtered_outputdict, result = filtered_box_stage1(detection_mode, output_dict, frame_id, video_fps,
                                                                  video_size, use_normalized_coordinates,
                                                                  nms_threshold, min_prob, min_x, min_y, min_area)
                image = visualize(
                    image_np,
                    filtered_outputdict['boxes'],
                    filtered_outputdict['classes'],
                    filtered_outputdict['scores'],
                    category_index=None,
                    use_normalized_coordinates=False,
                    line_thickness=3,
                )
                results.append(result)
                frames.append(image)
                if len(results) == frame_num_for_judge:
                    no_hat_time_, no_hat_nums = filtered_box_stage2(results, frame_num_for_judge, min_ratio)
                    # if no_hat_time_==True, at this time, there are no_hat chefs
                    if no_hat_time_:
                        output_info.append([no_hat_time_, no_hat_nums])
                        print("Found %d chefs without hats at %.2f seconds;" % (no_hat_nums, no_hat_time_))
                        logs.write("\nFound %d chefs without hats at %.2f seconds." % (no_hat_nums, no_hat_time_))
                        if save_image:
                            for j, img in enumerate(frames):
                                cv2.imwrite(save_path + "\\" + video_name + "_time=" + str(no_hat_time_) +
                                            "no_hats=" + str(no_hat_nums) + "_" + str(j) + '.jpg', img)
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


def detect_image_yolo(config_path):
    cf = configparser.ConfigParser()
    cf.read(config_path)
    input_image_path = cf.get("test_image_path", "image_path")
    image_output_path = cf.get("test_image_path", "output_image_path")

    use_normalized_coordinates = cf.getboolean("yolo_v3", "use_normalized_coordinates")
    nms_threshold = cf.getfloat("test_video_config", "nms_threshold")
    min_prob = cf.getfloat("test_video_config", "min_prob")
    min_x = cf.getfloat("test_video_config", "min_x")
    min_y = cf.getfloat("test_video_config", "min_y")
    min_area = cf.getfloat("test_video_config", "min_area")

    image_output_path = os.path.abspath(image_output_path) + "\\"
    detection_mode = 'image'
    import time
    time1 = time.time()
    input_image_path = os.path.abspath(input_image_path)
    image_path_list = [input_image_path + "\\" + image_name for image_name in os.listdir(input_image_path)]
    yolo = YOLO(config_path)
    frame_id = None
    video_fps = None
    video_size = None
    import cv2
    for i, img_path in enumerate(image_path_list):
        image = Image.open(img_path)
        video_size = (image.width, image.height)
        output_dict = generate_output_dict(yolo, image)
        filtered_outputdict, result = filtered_box_stage1(detection_mode, output_dict, frame_id, video_fps,
                                                          video_size, use_normalized_coordinates,
                                                          nms_threshold, min_prob, min_x, min_y, min_area)
        image = visualize(
            np.array(image),
            filtered_outputdict['boxes'],
            filtered_outputdict['classes'],
            filtered_outputdict['scores'],
            category_index=None,
            use_normalized_coordinates=False,
            line_thickness=3)
        no_hat_num = result[-2]
        image = image[..., ::-1]
        cv2.imwrite(image_output_path + str(i) + '_nohat_nums=' + str(no_hat_num) + ".jpg", image)
    process_time = round((time.time() - time1) / 60, 2)
    print("test time: ", process_time)


def save_image_to_file(dir_path, file_name, image_obj):
    if not os.path.isdir(dir_path):
        os.mkdir(dir_path)

    if 'numpy' not in str(type(image_obj)):
        image_arr = np.array(image_obj, dtype='float32')
    else:
        image_arr = image_obj

    file_path = os.path.join(dir_path, file_name + '.jpg')

    if os.path.isfile(file_path):
        print('overwrite %s\%s' % (dir_path, file_name))
        os.remove(file_path)
    import cv2
    cv2.imwrite(file_path, image_arr)


def generate_output_dict(self, image):
    if self.model_image_size != (None, None):
        assert self.model_image_size[0] % 32 == 0, 'Multiples of 32 required'
        assert self.model_image_size[1] % 32 == 0, 'Multiples of 32 required'
        boxed_image = letterbox_image(image, tuple(reversed(self.model_image_size)))
    else:
        new_image_size = (image.width - (image.width % 32),
                          image.height - (image.height % 32))
        boxed_image = letterbox_image(image, new_image_size)
    image_data = np.array(boxed_image, dtype='float32')

    # print(image_data.shape)
    image_data /= 255.
    image_data = np.expand_dims(image_data, 0)  # Add batch dimension.

    out_boxes, out_scores, out_classes = self.sess.run(
        [self.boxes, self.scores, self.classes],
        feed_dict={
            self.yolo_model.input: image_data,
            self.input_image_shape: [image.size[1], image.size[0]],
            K.learning_phase(): 0
        })

    for i in range(len(out_classes)):
        out_classes[i] += 1

    output_dict = dict(detection_boxes=out_boxes, detection_classes=out_classes, detection_scores=out_scores)
    return output_dict
