# -*- coding: utf-8 -*-
"""
Created on Fri Jan 18 20:57:14 2019

@author: admin
"""

# all_xmls_path = "C:\\Users\\admin\\Desktop\\xmls"
all_xmls_path = "D:\\project3_faster_rcnn\\models-master\\research\\hat_dataset\\data_single_no_hat\\test_xmls"

import os
import xml.etree.ElementTree as ET

#os.chdir(all_xmls_path)

xml_list = os.listdir(all_xmls_path)

for id, xml in enumerate(xml_list):
    xml_path = os.path.join(all_xmls_path, xml)

    xml_tree = ET.parse(xml_path)
    
    root = xml_tree.getroot()

    object_bboxes = root.findall("object")

    for ob in object_bboxes:
        if ob.find("name").text == "hat":
            root.remove(ob)

    xml_tree.write(xml_path)

