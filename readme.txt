ģ��ѵ�����裺
1 ����Ԥ����data_preprocessĿ¼�£���
ʹ��labelImg������б�ע����xml�ļ� python labelImg.py�����˱�עʱ�ۺϵ�һ��ʱ��Ҫ����xml�ļ��е�·��update_xmls.py��
xmlתΪcsv�ļ���csv�ļ�����ÿ��object��λ����Ϣ�������Ϣxml_to_csv.py��
����tfrecords�ļ�create_hat_tf_record.py��

�����п����õ�toolsĿ¼�µĸ����ļ��������ձ�ע�ļ��Ľű�create_empty_xml.py������Ƶ�г�ȡ֡extract_frames_from_video.py��
pngͼƬתjpg��ʽpng_to_jpg.py������ü������������random_crop.py���Ƴ�hat onject����һĿ���⣩remove_hat_objects_from_xmls.py��
�Ƴ�xml�ļ��д�����ļ���height or width == 0��update_png_img_shape_in_xml.py

2 ģ��ѵ����
python model_training/train.py
train_configuration.config�ļ�������ģ�͵����·���Ȳ���

3 ���ԣ�
���ú�test_configuration.config�е������������
apps���ṩdetect_on_web, rstp_video_detect,  video_detect ,image_detect�ķ�����ֱ�����м��ɡ�

4 ��Ľ������⣺
      1 ��10֡results�Ĵ���ֻ�����˼򵥵�����ͶƱ�������Կ�������IOU��Ϊ�����������10֡������о���
      2 ��10֡results������ܸı��Ƿ񱣴�����֤�ݣ�����û�иı��Ѿ�չʾ	����Ƶ���ڵĽ����

����˵����
1
apps�°����ĸ��ļ����ֱ���detect_on_web.py����web��չʾ��detect_rstp_video.py��������rstp��m3u8��ַ����ʶ��detect_video.py
��detect_video_list.py���ڶԱ���video�ļ�����ʶ��
ʶ��ʱ��������Ƶ����ַ�򱾵���Ƶ·���Լ����·������test_configuration.config�ļ�������
data(single_no_hat)����image��xml�ļ�
data_preprocessĿ¼�·ֱ���update_xmls.py�ļ�����xml�ļ���·����xml_to_csv.py�����������ݵ�·�����ע��Ϣ��csv�ļ���create_hat_tf_record.py����tfrecords��ʽ������
fontĿ¼���������ļ�����ͼ���ϡ�no hat����������ʽ
model_trainingĿ¼��train.py�е�frcnn_train��������train_frcnn.pyģ�鿪ʼģ�͵�ѵ����train_configurationΪ�����ļ�����������ģ�͵����·����ѵ�������Լ����ֱ�����batch, RPN���rcnn���ѧϰ�ʣ�score��iou��
object_detectionĿ¼ΪAPI�Դ��Ŀ�
saved_modelsΪѵ���õ�ģ���ļ���ת�����pb�ļ�
toolsĿ¼����Ҫ����������ʱ�õ��Ľű���������������ʾ�ˣ�create_empty_xml.py
random_crop.py�ǲ����������Ľű���

2
test_configuration�еĲ�������:
��test_video_config����
frame_num_for_judge Ϊ�жϽ���õ���֡����10֡��no_hat������>=nms_threshold*frame_num_for_judgeʱ��Ϊ����ȷ��nohat����
nms_threshold=0.7
min_prob=0.99
min_x=0.02
min_y=0.01
min_area=0.008
min_ratio = 0.5
interval = 5
min_probΪ���Ŷȣ���Ϊ0.99�������У�min_x��min_yΪbox������ߺ��ϱ��ߵ�ֵ��ȥ�����ϵ�object�������У�min_areaΪbox��������˵��Ƚ�С��Ŀ�꣬interval ��ȡ֡�����Ϊ�ﵽʵʱ��⣬ʵ����Ƶfps=25��ÿ��5֡��0.2sȡ1֡�Դﵽʵʱ���

3
frcnn-detection.py�ļ��е�detect_rstp_video_frcnn������ϸ���̣�
�ȴ�config�ļ��ж�ȡrstp��ַ�Լ�ģ��·���������ȣ�
Ȼ���ȡ��Ƶ����
����logs.txt�ļ����������Ϣ����ɾ����
cv2��5֡��ȡһ֡���з�������
��������Ϊ10�Ķ���frames��results��ÿappendһ֡��popLeftһ֡
filtered_box_stage1��������min_x,min_y��min_area�ȹ��˵�������Ҫ���box��Ȼ��nms���˵�box��
 filtered_box_stage2������10֡frame��10��results��������ͶƱ���õ���ɿ���no_hat����������nms_threshold��Ҫ��ʱ�ٱ���ͼƬ�ļ�