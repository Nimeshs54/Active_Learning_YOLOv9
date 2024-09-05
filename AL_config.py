# Config for active learning
max_queried = 800
unlabeled = 'dataset/TRUCK_Real/unlabeled'
labeled = 'dataset/TRUCK_Real/labeled'
num_select = 2

# Config general
project = "yolov9"
weight = 'yolov9-c.pt'
device = '0' # cpu or 0,1,...
name = 'TRUCK_Real'
exist_ok = 1

# Config for train model
config_model = 'models/detect/yolov9-custom.yaml'
config_data = 'dataset/TRUCK_Real/truck_data.yaml'
batch_size = 1
epochs = 1
optimizer = 'SGD'
patience = 25
project_train = 'runs/train'

# Config for detection
source = 'dataset/TRUCK_Real/unlabeled' # '0' for webcam
conf_thres = 0.25
iou_thres = 0.45
project_detect = 'runs/detect'
save_conf = 0

# # Config for Validation(Test)
# val_data = 'dataset/pascal_voc/VOC2007.yaml'
# weights = 'runs/train/voc2007/weights/best.pt'
# device = '0'
# task = 'test'
# val_name = 'voc2007_test'