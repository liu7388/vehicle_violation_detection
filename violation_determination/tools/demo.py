import argparse
import os, sys
import shutil
import time
from pathlib import Path
import imageio

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)
sys.path.append(str(Path(os.path.abspath(__file__)).parents[2]))

import cv2
import torch
import torch.backends.cudnn as cudnn
from numpy import random
import numpy as np
import torchvision.transforms as transforms

from violation_determination.lib.config import cfg
from violation_determination.lib.utils.utils import create_logger, select_device, time_synchronized
from violation_determination.lib.models import get_net
from violation_determination.lib.dataset import LoadImages, LoadStreams
from violation_determination.lib.core.general import non_max_suppression, scale_coords
from violation_determination.lib.utils import show_seg_result
from violation_determination.lib.core.function import AverageMeter
from tqdm import tqdm

from violation_determination.my_scripts.houghlines_merge import houghlines_merge
import json
from violation_determination.my_scripts.car_in_which_lane import car_in_which_lane
from violation_determination.my_scripts.determine_blinkers_violation import determine_blinkers_violation

normalize = transforms.Normalize(
    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
)

transform = transforms.Compose([
    transforms.ToTensor(),
    normalize,
])


def detect(cfg, opt):
    """
    Perform detection and violation determination on images or videos using a pre-trained model.

    Args:
        cfg (module): Configuration module.
        opt (argparse.Namespace): Command-line arguments.

    """
    logger, _, _ = create_logger(
        cfg, cfg.LOG_DIR, 'demo')

    device = select_device(logger, opt.device)
    if os.path.exists(opt.save_dir):
        shutil.rmtree(opt.save_dir)
    os.makedirs(opt.save_dir)
    half = device.type != 'cpu'

    # Load model
    model = get_net(cfg)
    checkpoint = torch.load(opt.weights, map_location=device)
    model.load_state_dict(checkpoint['state_dict'])
    model = model.to(device)

    if half:
        model.half()

    # Set Dataloader
    if opt.source.isnumeric():
        cudnn.benchmark = True
        dataset = LoadStreams(opt.source, img_size=opt.img_size)
        bs = len(dataset)
    else:
        dataset = LoadImages(opt.source, img_size=opt.img_size)
        bs = 1

    # Get names and colors
    names = model.module.names if hasattr(model, 'module') else model.names
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(names))]

    # Run inference
    t0 = time.time()

    vid_path, vid_writer = None, None
    img = torch.zeros((1, 3, opt.img_size, opt.img_size), device=device)
    _ = model(img.half() if half else img) if device.type != 'cpu' else None
    model.eval()

    inf_time = AverageMeter()
    nms_time = AverageMeter()

    blinkers_history_length = 30
    blinkers_history_timeline = []
    lane_history_timeline_margin = 10
    lane_history_timeline = []
    previous_lane = None
    current_lane = None
    lane_history_timeline_counter = 0

    detect_violation_result = None

    for i, (path, img, img_det, vid_cap, shapes) in tqdm(enumerate(dataset), total=len(dataset)):
        img = transform(img).to(device)
        img = img.half() if half else img.float()
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Inference
        t1 = time_synchronized()
        det_out, da_seg_out, ll_seg_out = model(img)
        t2 = time_synchronized()

        inf_out, _ = det_out
        inf_time.update(t2 - t1, img.size(0))

        # Apply NMS
        t3 = time_synchronized()
        det_pred = non_max_suppression(inf_out, conf_thres=opt.conf_thres, iou_thres=opt.iou_thres, classes=None,
                                       agnostic=False)
        t4 = time_synchronized()

        nms_time.update(t4 - t3, img.size(0))
        det = det_pred[0]

        save_path = str(opt.save_dir + '/' + Path(path).name) if dataset.mode != 'stream' else str(
            opt.save_dir + '/' + "web.mp4")

        _, _, height, width = img.shape
        h, w, _ = img_det.shape
        pad_w, pad_h = shapes[1][1]
        pad_w = int(pad_w)
        pad_h = int(pad_h)
        ratio = shapes[1][0][1]

        da_predict = da_seg_out[:, :, pad_h:(height - pad_h), pad_w:(width - pad_w)]
        da_seg_mask = torch.nn.functional.interpolate(da_predict, scale_factor=int(1 / ratio), mode='bilinear')
        _, da_seg_mask = torch.max(da_seg_mask, 1)
        da_seg_mask = da_seg_mask.int().squeeze().cpu().numpy()

        ll_predict = ll_seg_out[:, :, pad_h:(height - pad_h), pad_w:(width - pad_w)]
        ll_seg_mask = torch.nn.functional.interpolate(ll_predict, scale_factor=int(1 / ratio), mode='bilinear')
        _, ll_seg_mask = torch.max(ll_seg_mask, 1)
        ll_seg_mask = ll_seg_mask.int().squeeze().cpu().numpy()

        lanes_mask = ll_seg_mask.copy()
        min_val = np.min(lanes_mask)
        max_val = np.max(lanes_mask)
        lanes_mask = (lanes_mask - min_val) * (255 / (max_val - min_val))
        lanes_mask = lanes_mask.astype(np.uint8)

        img_det = show_seg_result(img_det, (da_seg_mask, ll_seg_mask), _, _, is_demo=True)

        houghlines = houghlines_merge(lanes_mask)
        if houghlines is not None:
            for line in houghlines:
                x1, y1, x2, y2 = line[0]
                x1 = int(x1 * 2 / 3)
                y1 = int(y1 * 2 / 3)
                x2 = int(x2 * 2 / 3)
                y2 = int(y2 * 2 / 3)
                cv2.line(img_det, (x1, y1), (x2, y2), (0, 0, 255), 2)

        parser = argparse.ArgumentParser(description='Annotate video with detections')
        parser.add_argument('--source', type=str)
        parser.add_argument('--save-dir', type=str)
        parser.add_argument('--target_id', type=int, default=5, help='Target ID for annotation (default: 5)')
        args = parser.parse_args()

        video = os.path.basename(args.source[:-14])
        car_ID = args.target_id

        with open(
                "../vehicle/data/videos/train/" + video + '_merged.json',
                'r') as file:
            data = json.load(file)

        car_blinker_detection_width = 1920
        scale_ratio = img_det.shape[1] / car_blinker_detection_width

        car_blinker_detection_list = [item for item in data if item['frame_number'] == i + 1]
        car_blinker_detection_target = next((item for item in car_blinker_detection_list if item['id'] == car_ID), None)
        if car_blinker_detection_target is not None:
            current_lane = car_in_which_lane(car_blinker_detection_target, houghlines)
            if current_lane == "Unavailable yet":
                print("Current lane:", current_lane,
                      "(Number of lanes detected either on the left or right is less than 2!)", end="\n\n")
            else:
                print("Current lane:", current_lane, end="\n\n")
            x1, y1, x2, y2 = car_blinker_detection_target["bbox"]
            x1 = int(round(x1 * scale_ratio))
            y1 = int(round(y1 * scale_ratio))
            x2 = int(round(x2 * scale_ratio))
            y2 = int(round(y2 * scale_ratio))
            cv2.putText(img_det, f'Lane: {current_lane}', (min(list([x1, x2])), max(list([y1, y2])) + 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2, cv2.LINE_AA)
            if len(blinkers_history_timeline) == blinkers_history_length:
                blinkers_history_timeline.pop(0)
                blinkers_history_timeline.append(car_blinker_detection_target["left_or_right"])
            elif 0 <= len(blinkers_history_timeline) < 10:
                blinkers_history_timeline.append(car_blinker_detection_target["left_or_right"])
        else:
            print('No car_target')

        if (current_lane != "Unavailable yet") and (previous_lane != "Unavailable yet"):
            if current_lane == previous_lane:
                if len(lane_history_timeline) <= lane_history_timeline_margin:
                    lane_history_timeline.append(current_lane)
                else:
                    lane_history_timeline.pop(0)
                    lane_history_timeline.append(current_lane)
            elif current_lane != previous_lane:
                lane_history_timeline.pop(0)
                lane_history_timeline.append(current_lane)
                detect_violation_result = determine_blinkers_violation(blinkers_history_timeline, lane_history_timeline)
                print(f"{lane_history_timeline[0]} => {lane_history_timeline[-1]}\n{detect_violation_result}")
                lane_history_timeline = []

        previous_lane = current_lane

        if detect_violation_result:
            x1, y1, x2, y2 = car_blinker_detection_target["bbox"]
            x1 = int(round(x1 * scale_ratio))
            y1 = int(round(y1 * scale_ratio))
            x2 = int(round(x2 * scale_ratio))
            y2 = int(round(y2 * scale_ratio))
            cv2.putText(img_det, f'{detect_violation_result}', (min(list([x1, x2])), max(list([y1, y2])) + 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2, cv2.LINE_AA)

        if dataset.mode == 'images':
            cv2.imwrite(save_path, img_det)

        elif dataset.mode == 'video':
            if vid_path != save_path:
                vid_path = save_path
                if isinstance(vid_writer, cv2.VideoWriter):
                    vid_writer.release()

                fourcc = 'mp4v'
                fps = vid_cap.get(cv2.CAP_PROP_FPS)
                h, w, _ = img_det.shape
                vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*fourcc), fps, (w, h))
            vid_writer.write(img_det)

        else:
            cv2.imshow('image', img_det)
            cv2.waitKey(1)

    print('Results saved to %s' % Path(opt.save_dir))
    print('Done. (%.3fs)' % (time.time() - t0))
    print('inf : (%.4fs/frame)   nms : (%.4fs/frame)' % (inf_time.avg, nms_time.avg))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str,
                        default='../weights/End-to-end.pth',
                        help='model.pth path(s)')
    parser.add_argument('--source', type=str, default='inference/videos', help='source')
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='IOU threshold for NMS')
    parser.add_argument('--device', default='cpu', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--save-dir', type=str, default='inference/output', help='directory to save results')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--target_id', type=int, default=5, help='Target ID for annotation (default: 5)')

    opt = parser.parse_args()
    with torch.no_grad():
        detect(cfg, opt)
