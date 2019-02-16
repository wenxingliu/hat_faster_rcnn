## Train Tensorflow Object Detection Models On Customized Data

***Wenxing Liu 11/18/2018***

### Setup

Follow steps on:
https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/installation.md

You may encounter issues installing cocoapi or protobuf on Windows OS. Find details and follow steps on:
- https://github.com/maierfelix/POGOserver/wiki/Windows-Protobuf-How-To
- https://github.com/philferriere/cocoapi



### Local Dir Structure:

Create `hat_dataset` folder under `research`, then clone this repo to `hat_dataset`. Folder structure should look like the following.

- research
    - hat_dataset
        - annotations
            - xmls
        - images
        - TFRecords
        - checkpoints
        - saved_models
        

### Train Faster RCNN on customized dataset
Before you start, make sure the paths in xml files are correct. If not, you can use `update_xmls.py` as a template to update file paths or file names.
Also, you can use different models from tensorflow model zoo. Here I'll take resnet50 as an example. 

1. Run `xml_to_csv.py` to generate `train_labels.csv`.
2. Run `create_hat_tf_record.py` to generate `train.record`.
3. Download `faster_rcnn_resnet50_coco` from tensorflow model zoo to `research\object_detection\` folder.
4. Update paths in `hat_resnet50_config.config` to map your local directories.
5. Update paths and configs in `train.py`. Then cd into `hat_dataset` folder, and run `python train.py --logtostderr`. Checkpoints will be saved to `checkpoints` folder.
6. Update paths and configs in `export_inference_graph.py` to read from saved checkpoints from the previous step. Run `python export_inference_graph.py --input_type image_tensor` under `hat_dataset` folder.A trained graph will be saved to `saved_models` folder.
7. Test saved model on test data.

### Test Faster RCNN on customized dataset
First, make sure the raw trained model file in the directory model_training/checkpoints/,and run tool/export_inference_graph.py to generate the frozen_inference_graph.pb in the directory saved_models. Then, tune the parameters and test dataset paths through the file test_configuration.config
1. Run `apps/detect_rstp_video.py ` to test the rstp video stream, results will be saved in the directory test_log, the same as below.
2. Run `apps/detect_video_list.py ` to test videos in the local directories.