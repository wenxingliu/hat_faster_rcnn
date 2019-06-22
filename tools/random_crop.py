import cv2
import os
import random

__anthor__ = 'wangze'

'''批量随机裁剪图片产生负样本'''

cropped_img_height = 300
cropped_img_width = 300

images_path = 'D:\\project3_faster_rcnn\\models-master\\research\\hat_detection\\video_data\\negative_samples\\images'
cropped_images_path = \
    'D:\\project3_faster_rcnn\\models-master\\research\\hat_detection\\video_data\\negative_samples\\cropped_images'


image_paths = [os.path.join(images_path, path) for path in os.listdir(images_path)]
# id is the index of image, count is the cropped img nums for per image
for id, per_img_path in enumerate(image_paths):
    img = cv2.imread(per_img_path)
    count = 1
    img_height, img_width = img.shape[0], img.shape[1]
    while img_height > cropped_img_height and img_width > cropped_img_width:
        y1 = random.randint(1, img_height - cropped_img_height)
        x1 = random.randint(1, img_width - cropped_img_width)
        # 随机截图
        cropImg = img[y1: y1 + cropped_img_height, x1: x1 + cropped_img_width]
        cv2.imwrite(cropped_images_path + '\\' + str(id) + '_' + str(count) + '.jpg', cropImg)
        count += 1
        if count == 20:
            break
