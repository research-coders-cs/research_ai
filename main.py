import os
import gc
import math
import time
import random
import requests
import itertools

import logging
logging.basicConfig(level=logging.INFO)

# import wandb
from datetime import datetime

import torch
# from torch import nn
# from torch.nn import functional
# from torch.utils.tensorboard import SummaryWriter

from torch.utils.data import DataLoader

import torchvision
from torchvision import transforms

# from torchsummary import summary
import digitake


import matplotlib.pyplot as plt
# import cv2

import numpy as np

from src.wsdan import WSDAN
from src.transform import ThyroidDataset, get_transform##, get_transform_center_crop, transform_fn


ARTIFACTS_OUTPUT = './output'

def mk_artifact_dir(dirname):
    path = f'{ARTIFACTS_OUTPUT}/{dirname}'
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)
    return path

def get_device():
    USE_GPU = False#True
    digitake.model.set_reproducible(2565)

    if USE_GPU:
        # GPU settings
        assert torch.cuda.is_available(), "Don't forget to turn on gpu runtime!"
        os.environ['CUDA_VISIBLE_DEVICES'] = '0'
        device = torch.device("cuda")
        torch.backends.cudnn.benchmark = True
    else:
        device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

    return device

def demo_thyroid_test():
    print('\n\n\n\n@@ demo_thyroid_test(): ^^')
    from src.thyroid_test import test  # "Prediction"

    device = get_device()
    print("@@ device:", device)

    #

    test_ds_path_no = digitake.preprocess.build_dataset({
      'malignant': ['Test/Malignant'],
      'benign': ['Test/Benign'],
    }, root='Dataset_train_test_val')

    print('@@ test_ds_path_no:', test_ds_path_no)
    print("@@ len(test_ds_path_no['malignant']):", len(test_ds_path_no['malignant']))
    print("@@ len(test_ds_path_no['benign']):", len(test_ds_path_no['benign']))

    #

    # pretrain = 'resnet'
    pretrain = 'densenet'

    target_resize = 250
    batch_size = 8 #@param ["8", "16", "4", "1"] {type:"raw"}

    num_classes = 2
    num_attention_maps = 32

    #@@workers = 2
    workers = 0  # @@

    #

    # No Markers
    test_dataset_no = ThyroidDataset(
        phase='test',
        dataset=test_ds_path_no,
        transform=get_transform(target_resize, phase='basic'),
        with_alpha_channel=False)

    test_loader_no = DataLoader(
        test_dataset_no,
        batch_size=batch_size * 4,
        shuffle=False,
        num_workers=workers,
        pin_memory=True)

    #

    print('\n\n@@ ======== Calling `net = WSDAN(...)`')
    net = WSDAN(num_classes=num_classes, M=num_attention_maps, net=pretrain, pretrained=True)

    net.to(device)

    #

    print('\n\n@@ ======== Calling `test()`')

    ckpt = "WSDAN_densenet_224_16_lr-1e5_n1-remove_220828-0837_85.714.ckpt"
    #ckpt = "WSDAN_doppler_densenet_224_16_lr-1e5_n5_220905-1309_78.571.ckpt"
    #ckpt = "densenet_250_8_lr-1e5_n4_60.000"

    results = test(device, net, batch_size, test_loader_no, ckpt,
                   savepath=mk_artifact_dir('demo_thyroid_test'))
    # print('@@ results:', results)

    #

    if 1:  #  legacy
        from src.legacy import print_scores, print_auc

        print('\n\n@@ ======== print_scores(results)')
        print_scores(results)

        _enable_plot = 0  # @@
        print(f'\n\n@@ ======== print_auc(results, enable_plot={_enable_plot})')
        print_auc(results, len(test_dataset_no), enable_plot=_enable_plot)

    #

    print('@@ demo_thyroid_test(): vv')


def demo_thyroid_train():
    print('\n\n\n\n@@ demo_thyroid_train(): ^^')
    from src.thyroid_train import training

    device = get_device()
    print("@@ device:", device)

    #

    train_ds_path = digitake.preprocess.build_dataset({
      'malignant': ['Train/Malignant'],
      'benign': ['Train/Benign'],
    }, root='Dataset_train_test_val')

    print(train_ds_path)
    print(len(train_ds_path['malignant']), len(train_ds_path['benign']))  # @@ 20 21

    #

    # pretrain = 'resnet'
    pretrain = 'densenet'

    target_resize = 250
    batch_size = 8 #@param ["8", "16", "4", "1"] {type:"raw"}

    #@@workers = 2
    workers = 0  # @@

    lr = 0.001 #@param ["0.001", "0.00001"] {type:"raw"}
    lr_ = "lr-1e5" #@param ["lr-1e3", "lr-1e5"]

    start_epoch = 0
    total_epochs = 5         ################### 60

    #

    train_dataset = ThyroidDataset(
        phase='train',
        dataset=train_ds_path,
        transform=get_transform(target_resize, phase='basic'),
        with_alpha_channel=False  # if False, it will load image as RGB(3-channel)
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=workers,
        pin_memory=True
    )


    #

    net = None  # !!!!
    #training(device, net, batch_size, train_loader, validate_loader)

    #

    print('@@ demo_thyroid_train(): vv')


def demo_doppler_comp():
    print('\n\n\n\n@@ demo_doppler_comp(): ^^')

    from src.doppler import doppler_comp, get_iou, plot_comp, get_sample_paths
    savepath = mk_artifact_dir('demo_doppler_comp')

    for path_doppler, path_markers, path_markers_label in get_sample_paths():
        print('\n@@ -------- calling doppler_comp() for')
        print(f'  {os.path.basename(path_doppler)} vs')
        print(f'  {os.path.basename(path_markers)}')

        bbox_doppler, bbox_markers, border_img_doppler, border_img_markers = doppler_comp(
            path_doppler, path_markers, path_markers_label)
        print('@@ bbox_doppler:', bbox_doppler)
        print('@@ bbox_markers:', bbox_markers)

        iou = get_iou(bbox_doppler, bbox_markers)
        print('@@ iou:', iou)

        plt = plot_comp(border_img_doppler, border_img_markers, path_doppler, path_markers)
        stem = os.path.splitext(os.path.basename(path_doppler))[0]
        fname = f'{savepath}/comp-doppler-{stem}.jpg'
        plt.savefig(fname, bbox_inches='tight')
        print('@@ saved -', fname)

    print('@@ demo_doppler_comp(): vv')


if __name__ == '__main__':
    print("@@ torch.__version__:", torch.__version__)

    if 0:  # adaptation of 'compare.{ipynb,py}' exported from https://colab.research.google.com/drive/1kxMFgo1LyVqPYqhS6_UJKUsVvA2-l9wk
        demo_doppler_comp()  # TODO - renaming

    if 1:  # the "Traning/Validation" flow of 'WSDAN_Pytorch_Revised_v1_01_a.ipynb'
        demo_thyroid_train()

    if 0:  # the "Prediction" flow of 'WSDAN_Pytorch_Revised_v1_01_a.ipynb' - https://colab.research.google.com/drive/1LN4KjBwtq6hUG42LtSLCmIVPasehKeKq
        demo_thyroid_test()  # TODO - generate 'confusion_matrix_test-*.png', 'test-*.png'
