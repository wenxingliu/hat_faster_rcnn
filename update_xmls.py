import os
import xml.etree.ElementTree as ET


__author__ = 'sliu'

def update_xml_file(fn):
    PATH = 'C:\\dev\\models\\research\\hat_dataset\\'
    et = ET.parse(fn)
    root = et.getroot()
    root.find('folder').text = 'images'
    root.find('filename').text = root.find('filename').text.replace('.png', '.jpg')
    root.find('path').text = PATH + "\\images\\" + root.find('filename').text
    et.write(fn)


if __name__=='__main__':
    ANNOTATION_PATH = 'C:\\dev\\models\\research\\hat_dataset\\annotations\\xmls\\'
    for xml_file in os.listdir(ANNOTATION_PATH):
        if xml_file.endswith('.xml') is True:
            update_xml_file(ANNOTATION_PATH + xml_file)