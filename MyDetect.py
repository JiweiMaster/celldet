import argparse
import time
from pathlib import Path

import cv2
from numpy.lib.arraysetops import isin
import torch
import torch.backends.cudnn as cudnn
from numpy import random

from models.experimental_analysis import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general_analysis import check_img_size, check_requirements, check_imshow, non_max_suppression, apply_classifier, \
    scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path
from utils.plots import plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized
import json
import numpy as np
from tqdm import tqdm

def detect(save_img=False):
    #
    isShowEachClass = False
    source, weights, view_img, save_txt, imgsz = opt.source, opt.weights, opt.view_img, opt.save_txt, opt.img_size
    # Directories
    save_dir = Path(increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok))  # increment run
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    # Initialize
    set_logging()
    device = select_device(opt.device)
    half = device.type != 'cpu'  # half precision only supported on CUDA

    # Load model
    model = attempt_load(weights, map_location=device)  # load FP32 model
    stride = int(model.stride.max())  # model stride
    imgsz = check_img_size(imgsz, s=stride)  # check img_size
    if half:
        model.half()  # to FP16
    # Set Dataloader
    save_img = True
    dataset = LoadImages(source, img_size=imgsz, stride=stride)
    # Get names and colors
    names = model.module.names if hasattr(model, 'module') else model.names
    # colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]
    # 'fake','negative','others','positive'
    colors = [[200,0,0],[0,200,200],[0,200,0],[0,0,200]]
    # Run inference
    if device.type != 'cpu':
        model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once
    t0 = time.time()
    for path, img, im0s, vid_cap in dataset:
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)
        # Inference
        t1 = time_synchronized()
        # 不输出密度图
        # pred = model(img, augment=opt.augment)[0]
        # 输出密度图
        pred, density_out = model(img, augment=opt.augment)[0]
        # Apply NMS
        pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)
        t2 = time_synchronized()
        # Process detections
        for i, det in enumerate(pred):  # detections per image
            p, s, im0 = path, '', im0s
            # more image
            im1 = np.copy(im0)
            im2 = np.copy(im0)
            imAll = np.copy(im0)
            # more image
            p = Path(p)  # to Path
            save_path = str(save_dir / p.name)  # img.jpg
            s += '%gx%g ' % img.shape[2:]  # print string
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()
                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string
                # 画一下图像
                for *xyxy, conf, cls, source_layer in reversed(det):
                    label = ''
                    plot_one_box(xyxy, imAll, label=label, color=colors[int(cls)], line_thickness=3)
                    if source_layer == 0:
                        if save_img or view_img:  # Add bbox to image
                            label = ''
                            plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=3)
                    if source_layer == 1:
                        if save_img or view_img:  # Add bbox to image
                            label = ''
                            plot_one_box(xyxy, im1, label=label, color=colors[int(cls)], line_thickness=3)
                    if source_layer == 2:
                        if save_img or view_img:  # Add bbox to image
                            label = ''
                            plot_one_box(xyxy, im2, label=label, color=colors[int(cls)], line_thickness=3)
                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string
                # Write results
                for *xyxy, conf, cls in reversed(det):
                    label = f'{names[int(cls)]} {conf:.2f}'
                    plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=3)
            # Print time (inference + NMS)
            print(f'{s}Done. ({t2 - t1:.3f}s)')
            # Save results (image with detections)
            if save_img:
                paths = save_path.split('.')
                save_path0 = paths[0] + '_f0.' + paths[1]
                save_path1 = paths[0] + '_f1.' + paths[1]
                save_path2 = paths[0] + '_f2.' + paths[1]
                if isShowEachClass:
                    cv2.imwrite(save_path0, im0)
                    cv2.imwrite(save_path1, im1)
                    cv2.imwrite(save_path2, im2)
                cv2.imwrite(save_path, imAll)

    if save_txt or save_img:
        s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
        print(f"Results saved to {save_dir}{s}")
    print(f'Done. ({time.time() - t0:.3f}s)')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # weight = 'runs/train/localunet_density_single/weights/best.pt' 
    # weight = 'runs/train/yolo-baseline/weights/best.pt'
    # weight = 'runs/train_old/exp3_better/weights/best.pt'
    weight = 'runs/train/localunet_density_single/weights/best.pt'
    parser.add_argument('--weights', nargs='+', type=str, default=weight, help='model.pt path(s)')
    # parser.add_argument('--source', type=str, default='../ki67Dataset/AllImage/val/images', help='source')
    parser.add_argument('--source', type=str, default='analysisData', help='source')
    parser.add_argument('--img-size', type=int, default= 640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default= 0.4, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default= 0.45, help='IOU threshold for NMS')
    parser.add_argument('--device', default='0,1', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='display results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-json', action='store_true', help='save results to *.json')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default='runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--save-label-json', default=True, action='store_true', help='save label results to *.json')
    # save_label_json
    opt = parser.parse_args()
    print(opt)
    check_requirements()
    opt.save_txt = True
    with torch.no_grad():
        detect()




