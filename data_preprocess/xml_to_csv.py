import os
import pandas as pd
import xml.etree.ElementTree as ET


def xml_to_csv(annotation_path, selected_list=range(401)):
    xml_list = []
    for xml_file in os.listdir(annotation_path):
        # if xml_file.endswith('.xml') is False:
        #     continue
        # if isinstance(xml_file.split(".xml"), int):
        #     file_num = int(xml_file.split('.xml')[0])
        # else:
        #     file_num = int(xml_file.split(".xml")[0].split("_")[-1])
        # if file_num not in selected_list:
        #     continue
        tree = ET.parse(annotation_path + xml_file)
        root = tree.getroot()
        for member in root.findall('object'):
            value = (root.find('filename').text.replace('png', 'jpg'),
                     int(root.find('size')[0].text),
                     int(root.find('size')[1].text),
                     member[0].text,
                     int(member[4][0].text),
                     int(member[4][1].text),
                     int(member[4][2].text),
                     int(member[4][3].text)
                     )
            xml_list.append(value)
    column_name = ['filename', 'width', 'height', 'class', 'xmin', 'ymin', 'xmax', 'ymax']
    xml_df = pd.DataFrame(xml_list, columns=column_name)
    return xml_df


def main(annotation_path, out_dir, selected_list):
    xml_df = xml_to_csv(annotation_path, selected_list)
    xml_df.to_csv(out_dir, index=None)
    print('Successfully converted xml to csv.')


if __name__ == "__main__":
    annotation_path = \
        "D:\\project3_faster_rcnn\\models-master\\research\\hat_detection\\data(single_no_hat)\\train_xmls\\"

    # use first 400 images to train, and use the rest for test
    train_out_dir = "D:\\project3_faster_rcnn\\models-master\\research\\hat_detection\\train_labels.csv"
    main(annotation_path, train_out_dir, selected_list=range(0, 1240))

    # use first 400 images to train, and use the rest for test
    # val_out_dir = "D:\\project3_faster_rcnn\\models-master\\research\\hat_dataset\\val_labels.csv"
    # main(annotation_path, val_out_dir, selected_list=range(1240, 1540))
