## Train Tensorflow Object Detection Models On Customized Data

***Wenxing Liu 11/18/2018***

### Setup

Follow steps on:
https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/installation.md

You may encounter issues installing cocoapi or protobuf on Windows OS. Find details and follow steps on:
- https://github.com/maierfelix/POGOserver/wiki/Windows-Protobuf-How-To
- https://github.com/philferriere/cocoapi
- https://github.com/cocodataset/cocoapi/issues/51

### Tips：

Before compile the protobuf libraries, copy the protoc.exe into ../research/ directory 

When compile with command "protoc object_detection/protos/*.proto --python_out=.", you can use git bash command if winows cmd or power shell don't work. 

If you're using python3 , add list() to category_index.values() in model_lib.py about line 381 as this list(category_index.values()).

reference for other errors:
- https://github.com/tensorflow/models/issues/4881

### Local Dir Structure:

Create `hat_dataset` folder under `research`, then clone this repo to `hat_dataset`. Folder structure should look like the following.

- research
    - hat_dataset
        - annotations
            - xmls
            - hat_xmls
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
