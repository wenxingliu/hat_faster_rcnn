import collections
from object_detection.utils.visualization_utils import *
import numpy as np
from collections import Counter

__author__ = 'WangZe'


def visualize(
    image,
    boxes,
    classes,
    scores,
    category_index,
    use_normalized_coordinates=False,
    agnostic_mode=False,
    line_thickness=4,
    skip_scores=False,
    skip_labels=False):

    """Overlay labeled boxes on an image with formatted scores and label names.

    This function groups boxes that correspond to the same location
    and creates a display string for each detection and overlays these
    on the image. Note that this function modifies the image in place, and returns
    that same image.

    Args:
    image: uint8 numpy array with shape (img_height, img_width, 3)
    boxes: a numpy array of shape [N, 4]
    classes: a numpy array of shape [N]. Note that class indices are 1-based,
      and match the keys in the label map.
    scores: a numpy array of shape [N] or None.  If scores=None, then
      this function assumes that the boxes to be plotted are groundtruth
      boxes and plot all boxes as black with no classes or scores.
    category_index: a dict containing category dictionaries (each holding
      category index `id` and category name `name`) keyed by category indices.
    use_normalized_coordinates: whether boxes is to be interpreted as
      normalized coordinates or not.
    agnostic_mode: boolean (default: False) controlling whether to evaluate in
      class-agnostic mode or not.  This mode will display scores but ignore
      classes.
    line_thickness: integer (default: 4) controlling line width of the boxes.
    skip_scores: whether to skip score when drawing a single detection
    skip_labels: whether to skip label when drawing a single detection
    Returns:
    uint8 numpy array with shape (img_height, img_width, 3) with overlaid boxes.
    """
    # Create a display string (and color) for every box location, group any boxes
    # that correspond to the same location.
    box_to_display_str_map = collections.defaultdict(list)
    box_to_color_map = collections.defaultdict(str)

    for i in range(len(boxes)):
      box = tuple(boxes[i])
      display_str = ''
      if not skip_labels:
          if not agnostic_mode:
              if category_index:
                if classes[i] in category_index.keys():
                  class_name = category_index[classes[i]]['name']
              elif not category_index:
                class_name = 'hat' if classes[i] == 1 else 'no_hat'
              else:
                class_name = 'N/A'
              display_str = str(class_name)
      if not skip_scores:
          if not display_str:
              display_str = '{}%'.format(int(100*scores[i]))
          else:
              display_str = '{}: {}%'.format(display_str, int(100*scores[i]))
      box_to_display_str_map[box].append(display_str)
      box_to_color_map[box] = STANDARD_COLORS[classes[i] % len(STANDARD_COLORS)]


  # Draw all boxes onto image.
    for box, color in box_to_color_map.items():
        ymin, xmin, ymax, xmax = box
        draw_bounding_box_on_image_array(image, ymin, xmin, ymax, xmax, color=color, thickness=line_thickness,
                                         display_str_list=box_to_display_str_map[box],
                                         use_normalized_coordinates=use_normalized_coordinates)

    return image


def filtered_box_stage1(detection_mode, output_dicts, frame_id, video_fps, video_size, use_normalized_coordinates,
                        overlap_threshold, min_prob, min_x, min_y, min_area):
    """
    :param output_dicts: [box, scores, class],      box;300*4; scores:300; class: 300
        box : [ymin, xmin, ymax, xmax]
    :param frame_id:
    :param video_fps:
    :param frame_num_for_judge:
    :param min_prob:
    :param min_x:
    :param min_y:
    :param min_area:
    :return: filtered_outputdicts, result
    """
    if detection_mode == "image":
        image_path = frame_id
        video_fps = None

    boxes = output_dicts['detection_boxes'].tolist()
    classes = output_dicts['detection_classes'].tolist()
    scores = output_dicts['detection_scores'].tolist()
    hat_nums = 0
    no_hat_nums = 0
    filtered_box = []
    filtered_classes = []
    filtered_scores = []
    if not use_normalized_coordinates:
        width, height = video_size
        min_area = width * height * min_area
        min_x = min_x * width
        min_y = min_y * height
        max_x = width - min_x
    else:
        max_x = 1 - min_x
    for box_id, box in enumerate(boxes):
        #  filter the box that in the frame edge, the box of small area and the box with low score
        box_area = (box[2] - box[0]) * (box[3] - box[1])
        box_score = scores[box_id]
        if box[1] > min_x and box[3] < max_x and box[0] > min_y and box_area > min_area and box_score > min_prob:
            filtered_box.append(box)
            filtered_classes.append(classes[box_id])
            filtered_scores.append(scores[box_id])
            if classes[box_id] == 1:
                hat_nums += 1
            else:
                no_hat_nums += 1

    # threshold = 0.8
    # filter the box of high iou with other boxes
    if len(filtered_box) > 1:
        for box_id, box in enumerate(filtered_box):
            # filtered_box_copy = copy.deepcopy(filtered_box)
            box_and_other_box_iou = [bboxes_iou(box, box2) for box2 in filtered_box]
            iou_sorted = sorted(box_and_other_box_iou, reverse=True)
            max_iou = iou_sorted[1]
            if max_iou > overlap_threshold:
                box2_id = box_and_other_box_iou.index(max_iou)
                box1_score = filtered_scores[box_id]  # current box score
                box2_score = filtered_scores[box2_id]  # max iou box score
                if box1_score > box2_score:
                    if filtered_classes[box2_id] == 1:
                        hat_nums -= 1
                    else:
                        no_hat_nums -= 1
                    del filtered_box[box2_id], filtered_classes[box2_id], filtered_scores[box2_id]
                else:
                    if filtered_classes[box2_id] == 1:
                        hat_nums -= 1
                    else:
                        no_hat_nums -= 1
                    del filtered_box[box_id], filtered_classes[box_id], filtered_scores[box_id]

    total_nums = hat_nums + no_hat_nums
    if detection_mode == "image":
        result = [image_path, filtered_box, filtered_scores, filtered_classes, hat_nums, no_hat_nums, total_nums]
    else:
        video_time = round(frame_id / video_fps, 2)
        result = [video_time, filtered_box, filtered_scores, filtered_classes, hat_nums, no_hat_nums, total_nums]
    filter_output_dict = dict(boxes=filtered_box, classes=filtered_classes, scores=filtered_scores)
    return filter_output_dict, result


def filtered_box_stage2(results, frame_num_for_judge, min_ratio):

    # second filtered the box of unstable no_hat nums among the continuous frame
    no_hat_nums = [result[-2] for result in results]
    frame_nums = len(results)
    assert frame_nums == frame_num_for_judge

    no_hat_num, frequents = Counter(no_hat_nums).most_common(1)[0]
    if no_hat_num > 0 and frequents > min_ratio * frame_num_for_judge:
        video_time = results[0][0]
        return video_time, no_hat_num
    else:
        return None, None


def compute_bbox_sizes(bboxes):
    box_sizes = np.product([bboxes[:, 2] - bboxes[:, 0], bboxes[:, 3] - bboxes[:, 1]], axis=0)
    return box_sizes


def bboxes_iou(bbox_0, bbox_1):
    inter_area_top = max(bbox_0[0], bbox_1[0])
    inter_area_left = max(bbox_0[1], bbox_1[1])
    inter_area_bottom = min(bbox_0[2], bbox_1[2])
    inter_area_right = min(bbox_0[3], bbox_1[3])
    inter_bbox = np.array([inter_area_top, inter_area_left, inter_area_bottom, inter_area_right])
    bbox_size_0, bbox_size_1, inter_bbox_size = compute_bbox_sizes(np.array([bbox_0, bbox_1, inter_bbox]))
    iou = inter_bbox_size / float(bbox_size_0 + bbox_size_1 - inter_bbox_size)
    return iou



