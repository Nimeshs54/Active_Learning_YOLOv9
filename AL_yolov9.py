import torch.distributed
from AL_detect import AL_detect
from val_dual import *
import argparse
import logging
import os
import sys
import random
import time
import numpy as np
import torch.distributed as dist
import torch.utils.data
import yaml
from tqdm import tqdm
from pathlib import Path

# try:
#     import comet_ml  # must be imported before torch (if installed)
# except ImportError:
#     comet_ml = None


FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from copy import deepcopy
from datetime import datetime, timedelta
from AL_train import train
from warnings import warn
from utils.callbacks import Callbacks
from utils.loggers.comet.comet_utils import check_comet_resume
from utils.downloads import attempt_download, is_url
from torch.utils.tensorboard import SummaryWriter
from utils.general import (
    LOGGER,
    TQDM_BAR_FORMAT,
    check_amp,
    check_dataset,
    check_file,
    check_git_info,
    check_git_status,
    check_img_size,
    check_requirements,
    check_suffix,
    check_yaml,
    colorstr,
    get_latest_run,
    increment_path,
    init_seeds,
    intersect_dicts,
    labels_to_class_weights,
    labels_to_image_weights,
    methods,
    one_cycle,
    print_args,
    print_mutation,
    strip_optimizer,
    yaml_save,
    set_logging
)
from utils.torch_utils import select_device
import AL_config as config

logger = logging.getLogger(__name__)

LOCAL_RANK = int(os.getenv("LOCAL_RANK", -1))  # https://pytorch.org/docs/stable/elastic/run.html
RANK = int(os.getenv("RANK", -1))
WORLD_SIZE = int(os.getenv("WORLD_SIZE", 1))
GIT_INFO = check_git_info()

# try:
#     import wandb
# except ImportError:
#     wandb = None
#     logger.info(
#         "Install Weights & Biases for experiment logging via 'pip install wandb' (recommended)")

class Yolov9():
    model_type = 'Yolo version 9'

    def train(self, ep = 0):
        parser = argparse.ArgumentParser()
        parser.add_argument('--weights', type=str, default=config.weight, help='initial weights path')
        parser.add_argument('--cfg', type=str, default=config.config_model, help='models/yolov5s.yaml path')
        parser.add_argument('--data', type=str, default=config.config_data, help='data.yaml path')
        parser.add_argument('--hyp', type=str, default='dataset/pascal_voc/hyp.scratch.yaml', help='hyperparameters path')
        parser.add_argument('--epochs', type=int, default=config.epochs)
        parser.add_argument('--batch-size', type=int, default=config.batch_size, help='total batch size for all GPUs')
        parser.add_argument("--imgsz", "--img", "--img-size", type=int, default=640, help="train, val image size (pixels)")
        parser.add_argument('--rect', action='store_true', help='rectangular training')
        parser.add_argument('--resume', nargs='?', const=True, default=False, help='resume most recent training')
        parser.add_argument('--nosave', action='store_true', help='only save final checkpoint')
        parser.add_argument("--noval", action="store_true", help="only validate final epoch")
        parser.add_argument('--noautoanchor', action='store_true', help='disable autoanchor check')
        parser.add_argument("--noplots", action="store_true", help="save no plot files")
        parser.add_argument("--evolve", type=int, nargs="?", const=300, help="evolve hyperparameters for x generations")
        parser.add_argument('--bucket', type=str, default='', help='gsutil bucket')
        parser.add_argument("--cache", type=str, nargs="?", const="ram", help="image --cache ram/disk")
        parser.add_argument('--image-weights', action='store_true', help='use weighted image selection for training')
        parser.add_argument('--device', default=config.device, help='cuda device, i.e. 0 or 0,1,2,3 or cpu') # using GPU 0
        parser.add_argument('--multi-scale', action='store_true', help='vary img-size +/- 50%%')
        parser.add_argument('--single-cls', action='store_true', help='train multi-class data as single-class')
        parser.add_argument("--optimizer", type=str, choices=["SGD", "Adam", "AdamW"], default=config.optimizer, help="optimizer")
        parser.add_argument('--sync-bn', action='store_true', help='use SyncBatchNorm, only available in DDP mode')
        parser.add_argument('--log-imgs', type=int, default=16, help='number of images for W&B logging, max 100')
        parser.add_argument('--log-artifacts', action='store_true', help='log artifacts, i.e. final trained model')
        parser.add_argument('--workers', type=int, default=8, help='maximum number of dataloader workers')
        parser.add_argument('--project', default=config.project_train, help='save to project/name')
        parser.add_argument('--name', default=config.name, help='save to project/name')
        parser.add_argument("--patience", type=int, default=25, help="EarlyStopping patience (epochs without improvement)")
        parser.add_argument('--exist-ok', type=int, default=config.exist_ok, help='existing project/name ok, do not increment')
        parser.add_argument("--cos-lr", action="store_true", help="cosine LR scheduler")
        parser.add_argument('--flat-cos-lr', action='store_true', help='flat cosine LR scheduler')
        parser.add_argument('--fixed-lr', action='store_true', help='fixed LR scheduler')
        parser.add_argument("--quad", action="store_true", help="quad dataloader")
        parser.add_argument("--label-smoothing", type=float, default=0.0, help="Label smoothing epsilon")
        parser.add_argument("--freeze", nargs="+", type=int, default=[0], help="Freeze layers: backbone=10, first3=0 1 2")
        parser.add_argument("--save-period", type=int, default=-1, help="Save checkpoint every x epochs (disabled if < 1)")
        parser.add_argument("--seed", type=int, default=0, help="Global training seed")
        parser.add_argument("--local_rank", type=int, default=-1, help="Automatic DDP Multi-GPU argument, do not modify")
        parser.add_argument('--min-items', type=int, default=0, help='Experimental')
        parser.add_argument('--close-mosaic', type=int, default=0, help='Experimental')

        # Logger arguments
        parser.add_argument("--entity", default=None, help="Entity")
        parser.add_argument("--upload_dataset", nargs="?", const=True, default=False, help='Upload data, "val" option')
        parser.add_argument("--bbox_interval", type=int, default=-1, help="Set bounding-box image logging interval")
        parser.add_argument("--artifact_alias", type=str, default="latest", help="Version of dataset artifact to use")

        # NDJSON logging
        parser.add_argument("--ndjson-console", action="store_true", help="Log ndjson to console")
        parser.add_argument("--ndjson-file", action="store_true", help="Log ndjson to file")
        opt = parser.parse_args()

        callbacks = Callbacks()
        if RANK in {-1, 0}:
            print_args(vars(opt))
            check_git_status()
            check_requirements(ROOT / "requirements.txt")


        # # Set DDP variables
        # opt.total_batch_size = opt.batch_size
        # opt.world_size = int(os.environ['WORLD_SIZE']) if 'WORLD_SIZE' in os.environ else 1
        # opt.global_rank = int(os.environ['RANK']) if 'RANK' in os.environ else -1
        # set_logging(opt.global_rank)
        # if opt.global_rank in [-1, 0]:
        #     check_git_status()

        # Resume (from specified or most recent last.pt)
        if opt.resume and not check_comet_resume(opt) and not opt.evolve:
            last = Path(check_file(opt.resume) if isinstance(opt.resume, str) else get_latest_run())
            opt_yaml = last.parent.parent / 'opt.yaml'  # train options yaml
            opt_data = opt.data  # original dataset
            if opt_yaml.is_file():
                with open(opt_yaml, errors='ignore') as f:
                    d = yaml.safe_load(f)
            else:
                d = torch.load(last, map_location='cpu')['opt']
            opt = argparse.Namespace(**d)  # replace
            opt.cfg, opt.weights, opt.resume = '', str(last), True  # reinstate
            if is_url(opt_data):
                opt.data = check_file(opt_data)  # avoid HUB resume auth timeout
        else:
            opt.data, opt.cfg, opt.hyp, opt.weights, opt.project = \
                check_file(opt.data), check_yaml(opt.cfg), check_yaml(opt.hyp), str(opt.weights), str(opt.project)  # checks
            assert len(opt.cfg) or len(opt.weights), 'either --cfg or --weights must be specified'
            if opt.evolve:
                if opt.project == str(ROOT / 'runs/train'):  # if default project name, rename to runs/evolve
                    opt.project = str(ROOT / 'runs/evolve')
                opt.exist_ok, opt.resume = opt.resume, False  # pass resume to exist_ok and disable resume
            if opt.name == 'cfg':
                opt.name = Path(opt.cfg).stem  # use model.yaml as name
            opt.save_dir = str(increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok))

        # DDP mode
        device = select_device(opt.device, batch_size=opt.batch_size)
        if LOCAL_RANK != -1:
            msg = 'is not compatible with YOLO Multi-GPU DDP training'
            assert not opt.image_weights, f'--image-weights {msg}'
            assert not opt.evolve, f'--evolve {msg}'
            assert opt.batch_size != -1, f'AutoBatch with --batch-size -1 {msg}, please pass a valid --batch-size'
            assert opt.batch_size % WORLD_SIZE == 0, f'--batch-size {opt.batch_size} must be multiple of WORLD_SIZE'
            assert torch.cuda.device_count() > LOCAL_RANK, 'insufficient CUDA devices for DDP command'
            torch.cuda.set_device(LOCAL_RANK)
            device = torch.device('cuda', LOCAL_RANK)
            if not dist.is_initialized():
                dist.init_process_group(backend="nccl" if dist.is_nccl_available() else "gloo")
            # dist.init_process_group(backend="nccl" if dist.is_nccl_available() else "gloo")

        # Hyperparameters
        with open(opt.hyp) as f:
            hyp = yaml.load(f, Loader=yaml.FullLoader)  # load hyps
            if 'box' not in hyp:
                warn('Compatibility: %s missing "box" which was renamed from "giou" in %s' %
                    (opt.hyp, 'https://github.com/ultralytics/yolov5/pull/1120'))
                hyp['box'] = hyp.pop('giou')

        # Train
        # logger.info(opt)
        # if not opt.evolve:
        #     tb_writer = None  # init loggers
        #     if opt.global_rank in [-1, 0]:
        #         logger.info(f'Start Tensorboard with "tensorboard --logdir {opt.project}", view at http://localhost:6006/')
        #         tb_writer = SummaryWriter(opt.save_dir)  # Tensorboard
        #     train(hyp, opt, device, tb_writer, wandb, ep)
        if not opt.evolve:
            train(opt.hyp, opt, device, callbacks, ep)
            

    def detect(self):
        parser = argparse.ArgumentParser()
        parser.add_argument('--weights', nargs='+', type=str, default=config.weight, help='model.pt path(s)')
        parser.add_argument('--source', type=str, default=config.source, help='source')  # file/folder, 0 for webcam
        parser.add_argument('--data', type=str, default=config.config_data, help='data.yaml path')
        parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[640, 640], help='inference size h,w')
        parser.add_argument('--conf-thres', type=float, default=config.conf_thres, help='object confidence threshold')
        parser.add_argument('--iou-thres', type=float, default=config.iou_thres, help='IOU threshold for NMS')
        parser.add_argument('--max-det', type=int, default=1000, help='maximum detections per image')
        parser.add_argument('--device', default=config.device, help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
        parser.add_argument('--save-conf', type=int, default=config.save_conf, help='save confidences in --save-txt labels')
        parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
        parser.add_argument('--save-crop', action='store_true', help='save cropped prediction boxes')
        parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
        parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
        parser.add_argument('--augment', action='store_true', help='augmented inference')
        parser.add_argument('--visualize', action='store_true', help='visualize features')
        parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
        parser.add_argument('--project', default=ROOT / 'runs/detect', help='save results to project/name')
        parser.add_argument('--name', default='exp', help='save results to project/name')
        parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
        parser.add_argument('--line-thickness', default=3, type=int, help='bounding box thickness (pixels)')
        parser.add_argument('--hide-labels', default=False, action='store_true', help='hide labels')
        parser.add_argument('--hide-conf', default=False, action='store_true', help='hide confidences')
        parser.add_argument('--half', action='store_true', help='use FP16 half-precision inference')
        parser.add_argument('--dnn', action='store_true', help='use OpenCV DNN for ONNX inference')
        parser.add_argument('--vid-stride', type=int, default=1, help='video frame-rate stride')
        opt = parser.parse_args()

        with torch.no_grad():
            result = AL_detect(opt)
        return result
    
    
    def val(self):
        parser = argparse.ArgumentParser()
        parser.add_argument('--data', type=str, default='dataset/pascal_voc/VOC2007.yaml', help='dataset.yaml path')
        parser.add_argument('--weights', nargs='+', type=str, default='runs/train/voc2007/weights/best.pt', help='model path(s)')
        parser.add_argument('--batch-size', type=int, default=32, help='batch size')
        parser.add_argument('--imgsz', '--img', '--img-size', type=int, default=640, help='inference size (pixels)')
        parser.add_argument('--conf-thres', type=float, default=0.001, help='confidence threshold')
        parser.add_argument('--iou-thres', type=float, default=0.7, help='NMS IoU threshold')
        parser.add_argument('--max-det', type=int, default=300, help='maximum detections per image')
        parser.add_argument('--task', default=config.task, help='train, val, test, speed or study')
        parser.add_argument('--device', default=config.device, help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
        parser.add_argument('--workers', type=int, default=8, help='max dataloader workers (per RANK in DDP mode)')
        parser.add_argument('--single-cls', action='store_true', help='treat as single-class dataset')
        parser.add_argument('--augment', action='store_true', help='augmented inference')
        parser.add_argument('--verbose', action='store_true', help='report mAP by class')
        parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
        parser.add_argument('--save-hybrid', action='store_true', help='save label+prediction hybrid results to *.txt')
        parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
        parser.add_argument('--save-json', action='store_true', help='save a COCO-JSON results file')
        parser.add_argument('--project', default=ROOT / 'runs/val', help='save to project/name')
        parser.add_argument('--name', default=config.val_name, help='save to project/name')
        parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
        parser.add_argument('--half', action='store_true', help='use FP16 half-precision inference')
        parser.add_argument('--dnn', action='store_true', help='use OpenCV DNN for ONNX inference')
        parser.add_argument('--min-items', type=int, default=0, help='Experimental')
        opt = parser.parse_args()

        opt.data = check_yaml(opt.data)  # check YAML
        opt.save_json |= opt.data.endswith('coco.yaml')
        opt.save_txt |= opt.save_hybrid
        print_args(vars(opt))

        with torch.no_grad():
            result = main(opt)
        return result
        # opt.data = check_yaml(opt.data)  # check YAML
        # opt.save_json |= opt.data.endswith('coco.yaml')
        # opt.save_txt |= opt.save_hybrid
        # print_args(vars(opt))
        # return opt


if __name__ == '__main__':
    model = Yolov9()
    model.detect()