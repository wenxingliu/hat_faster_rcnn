import os
import xml.etree.ElementTree as ET


__author__ = 'sliu'


def update_xml_file(fn):
    PATH = 'D:\\project3_faster_rcnn\\models-master\\research\\hat_detection\\'
    et = ET.parse(fn)
    root = et.getroot()
    root.find('folder').text = 'images'
    root.find('filename').text = root.find('filename').text.replace('.png', '.jpg')
    root.find('path').text = PATH + "data(single_no_hat)\\train_images\\" + root.find('filename').text
    et.write(fn)


if __name__ == '__main__':
    ANNOTATION_PATH = 'D:\\project3_faster_rcnn\\models-master\\research\\hat_detection\\data(single_no_hat)\\train_xmls\\'
    for xml_file in os.listdir(ANNOTATION_PATH):
        if xml_file.endswith('.xml') is True:
            update_xml_file(ANNOTATION_PATH + xml_file)