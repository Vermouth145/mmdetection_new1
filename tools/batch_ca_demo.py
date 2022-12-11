# Copyright (c) OpenMMLab. All rights reserved.
import asyncio
import matplotlib.pyplot as plt
import numpy as np

import torch
from tools.feature_visualization import draw_feature_map1, feature_map_channel

from argparse import ArgumentParser

from mmdet.apis import (inference_detector,
                        init_detector, show_result_pyplot)
import tqdm
import os
import cv2

def parse_args():
    parser = ArgumentParser()
    parser.add_argument('--img', default='../demo/test_data-copy', help='Image file')
    parser.add_argument('--config', default='../configs/yolo/yolov3_d53_mstrain-608_273e_coco.py', help='Config file')
    parser.add_argument('--checkpoint', default='../ckpt/yolov3_cbam/epoch_200.pth', help='Checkpoint file')
    parser.add_argument('--out-file', default=None, help='Path to output file')
    parser.add_argument(
        '--device', default='cuda:0', help='Device used for inference')
    parser.add_argument(
        '--palette',
        default='coco',
        choices=['coco', 'voc', 'citys', 'random'],
        help='Color palette used for visualization')
    parser.add_argument(
        '--score-thr', type=float, default=0.3, help='bbox score threshold')
    parser.add_argument(
        '--async-test',
        action='store_true',
        help='whether to set async options for async inference.')
    args = parser.parse_args()
    return args


def main(args):
    # build the model from a config file and a checkpoint file
    model = init_detector(args.config, args.checkpoint, device=args.device)
    # test a single image
    for filename in tqdm.tqdm(os.listdir(args.img)):
        img = os.path.join(args.img, filename)
        inimg = cv2.imread(args.img + '/' + filename)
        result = inference_detector(model, img)
        # out_file = os.path.join(args.out_file, filename)
        # show the results
        show_result_pyplot(
            model,
            img,
            result,
            palette=args.palette,
            score_thr=args.score_thr,
            out_file=None)


if __name__ == '__main__':
    args = parse_args()
    main(args)
