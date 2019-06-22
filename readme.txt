模型训练步骤：
1 数据预处理（data_preprocess目录下）：
使用labelImg软件进行标注生成xml文件 python labelImg.py，多人标注时综合到一起时需要更新xml文件中的路径update_xmls.py；
xml转为csv文件，csv文件包含每个object的位置信息和类别信息xml_to_csv.py；
生成tfrecords文件create_hat_tf_record.py；

过程中可能用到tools目录下的各种文件：产生空标注文件的脚本create_empty_xml.py，从视频中抽取帧extract_frames_from_video.py，
png图片转jpg格式png_to_jpg.py，随机裁剪负样本扩充给random_crop.py，移除hat onject（单一目标检测）remove_hat_objects_from_xmls.py，
移除xml文件中错误的文件（height or width == 0）update_png_img_shape_in_xml.py

2 模型训练：
python model_training/train.py
train_configuration.config文件中设置模型的输出路径等参数

3 测试：
设置好test_configuration.config中的输入输出各项
apps下提供detect_on_web, rstp_video_detect,  video_detect ,image_detect的方法，直接运行即可。

4 需改进的问题：
      1 对10帧results的处理只是用了简单的数量投票法，可以考虑用以IOU作为距离度量，对10帧结果进行聚类
      2 对10帧results处理后能改变是否保存结果（证据），但没有改变已经展示	在视频窗口的结果。

程序说明：
1
apps下包括四个文件：分别是detect_on_web.py用于web端展示；detect_rstp_video.py用于输入rstp或m3u8地址进行识别；detect_video.py
和detect_video_list.py用于对本地video文件进行识别
识别时，输入视频流地址或本地视频路径以及输出路径都在test_configuration.config文件中设置
data(single_no_hat)下是image和xml文件
data_preprocess目录下分别是update_xmls.py文件更新xml文件的路径；xml_to_csv.py用于生成数据的路径与标注信息的csv文件，create_hat_tf_record.py生成tfrecords格式的数据
font目录下是字体文件，如图像上‘no hat’的字体样式
model_training目录下train.py中的frcnn_train函数调用train_frcnn.py模块开始模型的训练，train_configuration为配置文件，其中设置模型的输出路径，训练集测试集划分比例，batch, RPN层和rcnn层的学习率，score，iou等
object_detection目录为API自带的库
saved_models为训练好的模型文件经转化后的pb文件
tools目录下主要是清晰数据时用到的脚本，功能如名称所示了，create_empty_xml.py
random_crop.py是产生负样本的脚本，

2
test_configuration中的参数含义:
【test_video_config】中
frame_num_for_judge 为判断结果用到的帧数，10帧中no_hat的数量>=nms_threshold*frame_num_for_judge时认为是正确的nohat数量
nms_threshold=0.7
min_prob=0.99
min_x=0.02
min_y=0.01
min_area=0.008
min_ratio = 0.5
interval = 5
min_prob为置信度，设为0.99降低误判，min_x和min_y为box的左边线和上边线的值，去掉边上的object降低误判，min_area为box面积，过滤掉比较小的目标，interval 是取帧间隔，为达到实时监测，实际视频fps=25，每隔5帧即0.2s取1帧以达到实时监测

3
frcnn-detection.py文件中的detect_rstp_video_frcnn函数详细流程：
先从config文件中读取rstp地址以及模型路径，参数等，
然后读取视频参数
建立logs.txt文件保存相关信息（可删除）
cv2隔5帧读取一帧进行分析处理，
建立长度为10的队列frames和results，每append一帧，popLeft一帧
filtered_box_stage1函数利用min_x,min_y，min_area等过滤掉不符合要求的box，然后nms过滤掉box，
 filtered_box_stage2函数对10帧frame的10个results做处理，按投票法得到最可靠的no_hat数量，满足nms_threshold的要求时再保存图片文件