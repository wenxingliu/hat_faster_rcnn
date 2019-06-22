import os
import cv2
import xml.etree.ElementTree as ET

xml_paths = 'D:\\project3_faster_rcnn\\models-master\\research\\hat_detection\\data(single_no_hat)\\train_xmls\\'

all_xml_paths = os.listdir(xml_paths)

for per_path in all_xml_paths:
    per_xml_path = os.path.join(xml_paths, per_path)
    try:
        et = ET.parse(per_xml_path)
    except:
        print(per_xml_path)
        exit(-1)
    root = et.getroot()
    size = root.find('size')
    img_path = root.find('path').text
    try:
        img_shape = cv2.imread(img_path).shape
    except:
        print(img_path)
        exit(-1)
    height, width, depth = img_shape[0], img_shape[1], img_shape[2]
    if size.find('width').text == '0':
        size.find('width').text = str(width)
        size.find('height').text = str(height)
        size.find('depth').text = str(depth)
    et.write(per_xml_path)


