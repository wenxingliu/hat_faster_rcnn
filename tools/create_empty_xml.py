import os
import cv2
import xml.etree.ElementTree as ET

__anthor__ = 'wangze'


def __indent(elem, level=0):
    i = "\n" + level*"\t"
    if len(elem):
        if not elem.text or not elem.text.strip():
            elem.text = i + "\t"
        if not elem.tail or not elem.tail.strip():
            elem.tail = i
        for elem in elem:
            __indent(elem, level+1)
        if not elem.tail or not elem.tail.strip():
            elem.tail = i
    else:
        if level and (not elem.tail or not elem.tail.strip()):
            elem.tail = i

def create_per_xmls(img_path, output_path):
    root = ET.Element('annotation')       # 创建节点
    tree = ET.ElementTree(root)     # 创建文档

    folder = ET.Element('folder')
    folder.text = img_path.split("\\")[-2]
    img_name = img_path.split('\\')[-1]
    filename = ET.Element('filename')
    filename.text = img_name
    path = ET.Element('path')
    path.text = img_path
    source = ET.Element('source')
    database = ET.Element('database')
    database.text = 'Unknown'
    source.append(database)
    img_size = ET.Element('size')
    shape = cv2.imread(img_path).shape
    height, width, depth = shape[0], shape[1], shape[2]
    element_h = ET.Element('height')
    element_h.text = str(height)
    element_w = ET.Element('width')
    element_w.text = str(width)
    element_d = ET.Element('depth')
    element_d.text = str(depth)
    img_size.append(element_h)
    img_size.append(element_w)
    img_size.append(element_d)
    segmented = ET.Element('segmented')
    segmented.text = '0'
    root.append(folder)
    root.append(filename)
    root.append(path)
    root.append(source)
    root.append(img_size)
    root.append(segmented)

    __indent(root)          # 增加换行符
    # tree.write('test.xml', encoding='utf-8', xml_declaration=True)
    tree.write(output_path + '\\' + img_name + '.xml')


empty_img_paths = 'D:\\project3_faster_rcnn\\models-master\\research\\hat_detection\\' \
                  'video_data\\negative_samples\\neg_samples_0617'

img_paths = os.listdir(empty_img_paths)
output_path = 'D:\\project3_faster_rcnn\\models-master\\research\\hat_detection\\' \
                  'video_data\\negative_samples\\empty_xmls_0617'

for per_path in img_paths:
    create_per_xmls(os.path.join(empty_img_paths, per_path), output_path)
