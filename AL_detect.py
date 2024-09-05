import argparse
import os
import platform
import sys
from pathlib import Path
import time
import torch

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLO root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from models.common import DetectMultiBackend
from utils.dataloaders import IMG_FORMATS, VID_FORMATS, LoadImages, LoadScreenshots, LoadStreams
from utils.general import (LOGGER, Profile, check_file, check_img_size, check_imshow, check_requirements, colorstr, cv2,
                           increment_path, non_max_suppression, print_args, scale_boxes, strip_optimizer, xyxy2xywh)
from utils.plots import Annotator, colors, save_one_box
from utils.torch_utils import select_device, smart_inference_mode

from models.experimental import attempt_load
from utils.dataloaders import LoadImages
from utils.general import check_img_size, non_max_suppression, xyxy2xywh, scale_coords
from utils.torch_utils import select_device, time_sync
import AL_config as config
import time

def AL_detect(opt):
    # Detect images
    # File weight model, detect data source, image size used
    weights, source, imgsz = opt.weights, opt.source, opt.imgsz

    nosave = opt.nosave
    source = str(source)
    save_img = not nosave and not source.endswith('.txt')  # save inference images
    is_file = Path(source).suffix[1:] in (IMG_FORMATS + VID_FORMATS)
    is_url = source.lower().startswith(('rtsp://', 'rtmp://', 'http://', 'https://'))
    webcam = source.isnumeric() or source.endswith('.txt') or (is_url and not is_file)
    screenshot = source.lower().startswith('screen')
    if is_url and is_file:
        source = check_file(source)  # download

    # Directories
    save_dir = increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok)  # increment run
    (save_dir / 'labels' if opt.save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    # Initialization
    device = select_device(opt.device)
    half = device.type != 'cpu'

    # Load model
    # model = attempt_load(weights=weights, device=device)
    model = DetectMultiBackend(weights, device=device, dnn=opt.dnn, data=opt.data, fp16=half)
    stride, names, pt = model.stride, model.names, model.pt
    imgsz = check_img_size(imgsz, s=stride)  # check image size

    # model = attempt_load(weights=weights, device=device)
    # imgsz = check_img_size(imgsz, s=model.stride.max())  # check img_size
    # if half:
    #     model.half()  # to FP16
    # dataset = LoadImages(source, img_size=imgsz)

    # Get names and colors
    names = model.module.names if hasattr(model, 'module') else model.names

    # Dataloader
    vid_stride=1
    bs = 1  # batch_size
    if webcam:
        view_img = check_imshow(warn=True)
        dataset = LoadStreams(source, img_size=imgsz, stride=stride, auto=pt, vid_stride=vid_stride)
        bs = len(dataset)
    elif screenshot:
        dataset = LoadScreenshots(source, img_size=imgsz, stride=stride, auto=pt)
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt, vid_stride=vid_stride)
    vid_path, vid_writer = [None] * bs, [None] * bs


    # Run inference
    model.warmup(imgsz=(1 if pt or model.triton else bs, 3, *imgsz))  # warmup
    seen, windows, dt = 0, [], (Profile(), Profile(), Profile())
    # t0 = time.time()
    # img = torch.zeros((1, 3, imgsz, imgsz), device=device)  # init img
    # _ = model(img.half() if half else img) if device.type != 'cpu' else None  # run once

    result = {}
    # Browse all photos
    for path, im, im0s, vid_cap, s in dataset:
        with dt[0]:
            im = torch.from_numpy(im).to(model.device)
            im = im.half() if model.fp16 else im.float()  # uint8 to fp16/32
            im /= 255  # 0 - 255 to 0.0 - 1.0
            if len(im.shape) == 3:
                im = im[None]  # expand for batch dim
    # for path, img, im0s, vid_cap, s in dataset:
    #     print(s)
    #     img = torch.from_numpy(img).to(device)
    #     img = img.half() if half else img.float()  # uint8 to fp16/32
    #     # image normalization
    #     img /= 255.0  # 0 - 255 to 0.0 - 1.0
    #     if img.ndimension() == 3:
    #         img = img.unsqueeze(0)

        # Inference
        visualize = opt.visualize
        with dt[1]:
            visualize = increment_path(save_dir / Path(path).stem, mkdir=True) if visualize else False
            pred = model(im, augment=opt.augment, visualize=visualize)
            pred = pred[0][1]
        # t1 = time_sync()
        # pred = model(img, augment=opt.augment)[0]

        # Apply NMS
        with dt[2]:
            pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, opt.classes, opt.agnostic_nms, max_det=opt.max_det)
        # pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)
        # t2 = time_sync()


        # Process predictions
        result[path] = []

        for i, det in enumerate(pred):  # per image
            seen += 1
            if webcam:  # batch_size >= 1
                p, im0, frame = path[i], im0s[i].copy(), dataset.count
                s += f'{i}: '
            else:
                p, im0, frame = path, im0s.copy(), getattr(dataset, 'frame', 0)

            p = Path(p)  # to Path
            save_path = str(save_dir / p.name)  # im.jpg
            txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # im.txt
            s += '%gx%g ' % im.shape[2:]  # print string
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            imc = im0.copy() if opt.save_crop else im0  # for save_crop
            annotator = Annotator(im0, line_width=opt.line_thickness, example=str(names))
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, 5].unique():
                    n = (det[:, 5] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                # Write results
                for *xyxy, conf, cls in reversed(det):
                    xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                    x,y,w,h = xywh
                    data = {"class": cls.item(), "box": [x,y,w,h], "conf": conf.item()}
                    result[path].append(data)

            # Print time (inference-only)
            LOGGER.info(f"{s}{'' if len(det) else '(no detections), '}{dt[1].dt * 1E3:.1f}ms")
    print(f'Done.')
    return result




# def AL_detect(opt):
#     # Detect images
#     # File weight model, detect data source, image size used
#     weights, source, imgsz = opt.weights, opt.source, opt.imgsz

#     # Initialization
#     device = select_device(opt.device)
#     half = device.type != 'cpu'

#     # Load model
#     model = attempt_load(weights=weights, device=device)
#     imgsz = check_img_size(imgsz, s=model.stride.max())  # check img_size
#     if half:
#         model.half()  # to FP16
#     dataset = LoadImages(source, img_size=imgsz)

#     # Get names and colors
#     names = model.module.names if hasattr(model, 'module') else model.names

#     # Run inference
#     t0 = time.time()
#     img = torch.zeros((1, 3, imgsz, imgsz), device=device)  # init img
#     _ = model(img.half() if half else img) if device.type != 'cpu' else None  # run once

#     result = {}
#     # Browse all photos
#     for path, img, im0s, vid_cap, s in dataset:
#         print(s)
#         img = torch.from_numpy(img).to(device)
#         img = img.half() if half else img.float()  # uint8 to fp16/32
#         # image normalization
#         img /= 255.0  # 0 - 255 to 0.0 - 1.0
#         if img.ndimension() == 3:
#             img = img.unsqueeze(0)

#         # Inference
#         t1 = time_sync()
#         pred = model(img, augment=opt.augment)[0]

#         # Apply NMS
#         pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)
#         t2 = time_sync()

#         # Process detections
#         result[path] = []

#         for i, det in enumerate(pred):  # detections per image
#             p, s, im0, frame = path, '', im0s, getattr(dataset, 'frame', 0)
#             s += '%gx%g ' % img.shape[2:]  # print string
#             gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            
#             if len(det):
#                 # Rescale boxes from img_size to im0 size
#                 det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()
                
#                 # Print results
#                 for c in det[:, -1].unique():
#                     n = (det[:, -1] == c).sum()  # detections per class
#                     s += f'{n} {names[int(c)]}s, '  # add to string
                
#                 # Save information about the box 1 file
#                 for *xyxy, conf, cls in reversed(det):
#                     xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh)
#                     x,y,w,h = xywh
#                     data = {"class": cls.item(), "box": [x,y,w,h], "conf": conf.item()}
#                     result[path].append(data)
#             # Print time (inference + NMS)
#             print(f'{s}Done. ({t2 - t1:.3f}s)')
#     print(f'Done. ({time.time() - t0:.3f}s)')
#     return result