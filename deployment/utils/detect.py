import argparse

import cv2
import torch
import torch.backends.cudnn as cudnn
from numpy import random
import numpy as np

from models.experimental import attempt_load
from utils.datasets import letterbox
from utils.general import check_img_size, check_inside, check_requirements, non_max_suppression, \
    scale_coords, set_logging
from utils.plots import get_result, plot_one_box
from utils.torch_utils import time_synchronized

def detect(opt):
    source, weights, imgsz, frame_rate = opt.source, opt.weights, opt.img_size, opt.frame_rate
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    set_logging()
    model = attempt_load(weights, map_location=device)  # load FP32 model
    stride = int(model.stride.max())  # model stride
    imgsz = check_img_size(imgsz, s=stride)  # check img_size

    cap = cv2.VideoCapture(0)

    # Get names and colors
    names = model.module.names if hasattr(model, 'module') else model.names
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]

    # Run inference
    if device.type != 'cpu':
        model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once
    
    print(f"Video capturing at {cap.get(cv2.CAP_PROP_FPS)} FPS")
    
    is_predicting = False
    count_frame = 1
    while(True):
        res, img0 = cap.read()
        cudnn.benchmark = True  # set True to speed up constant image size inference

        # Pad size
        img = letterbox(img0, imgsz, 32)[0]
        # BGR to RGB, to 3x416x416
        img = img[:, :, ::-1].transpose(2, 0, 1)  
        img = np.ascontiguousarray(img)

        img = torch.from_numpy(img).to(device)
        img = img.float() # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        count_frame += 1
        if (count_frame > frame_rate):
            is_predicting = True
            count_frame = 0

        # string result
        s = f'Scanning | '+'%gx%g ' % img.shape[2:]

        if is_predicting:    
            is_predicting = False
            t1 = time_synchronized()
            pred = model(img, augment=opt.augment)[0]
            # Apply NMS
            pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)
            t2 = time_synchronized()

            for i, det in enumerate(pred):  # detections per image
                if len(det):
                    s = 'Found | '
                    # Rescale boxes from img_size to im0 size
                    det[:, :4] = scale_coords(img.shape[2:], det[:, :4], img0.shape).round()

                    # Check the element in number plate
                    # det = check_inside(det, device)

                    # Print results
                    s += get_result(det, names)

                    # Print time (inference + NMS)
                    s += f"| Done. ({t2 - t1:.3f}s)"

                    # Write results
                    for *xyxy, conf, cls in reversed(det):
                        label = f'{names[int(cls)]}'
                        plot_one_box(xyxy, img0, label=label, color=colors[int(cls)], line_thickness=3)
        # show image
        cv2.imshow('Number plate detection', img0)
        
        # String results
        print(s)
        # wait key to break
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        # Release capture
    cap.release()
    cv2.destroyAllWindows()