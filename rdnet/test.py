# Copyright (C) 2019 Jin Han Lee
#
# This file is a part of BTS.
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program. If not, see <http://www.gnu.org/licenses/>

from __future__ import absolute_import, division, print_function

import os
import argparse
import time
import numpy as np
import cv2
import sys

import torch
import torch.nn as nn
from torch.autograd import Variable
from bts_dataloader import *

import errno
import matplotlib.pyplot as plt
from tqdm import tqdm

from dataloader import *
from arg_handler import Arg_train

args = Arg_train()
DEVICE = torch.device('cuda')

model_dir = os.path.dirname(args.checkpoint_path)
sys.path.append(model_dir)

for key, val in vars(__import__(args.model_name)).items():
    if key.startswith('__') and key.endswith('__'):
        continue
    vars()[key] = val


def get_num_lines(file_path):
    f = open(file_path, 'r')
    lines = f.readlines()
    f.close()
    return len(lines)


def test(params):
    """Test function."""
    args.mode = 'test'
    dataloader = BtsDataLoader(args, 'test')

    model = BtsModel(params=args)
    model = torch.nn.DataParallel(model)

    checkpoint = torch.load(args.checkpoint_path)
    model.load_state_dict(checkpoint['model'])
    model.eval()
    model.cuda()

    num_params = sum([np.prod(p.size()) for p in model.parameters()])
    print("Total number of parameters: {}".format(num_params))

    if args.data_path[-1] != '/':
        args.data_path += '/'
    path = args.data_path + 'nyu_test/'
    num_test_samples = len([f for f in os.listdir(path)
                            if os.path.isfile(os.path.join(path, f))])

    pred_depths = []

    start_time = time.time()
    with torch.no_grad():
        for _, sample in enumerate(tqdm(dataloader.data)):
            image = sample['image'].to(DEVICE)
            embedding = sample['embedding'].to(DEVICE)
            location = sample['bbox'].to(DEVICE)
            # Predict
            depth_est = model(image, embedding, location)
            pred_depths.append(depth_est.cpu().numpy().squeeze())

    elapsed_time = time.time() - start_time
    print('Elapesed time: %s' % str(elapsed_time))
    print('Done.')

    save_name = 'result_' + args.model_name

    print('Saving result pngs..')
    if not os.path.exists(os.path.dirname(save_name)):
        try:
            os.mkdir(save_name)
            os.mkdir(save_name + '/raw')
            os.mkdir(save_name + '/cmap')
            os.mkdir(save_name + '/rgb')
            os.mkdir(save_name + '/gt')
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise

    filenames = dataloader.training_samples.filenames
    for s in tqdm(range(num_test_samples)):
        scene_name = filenames[s].split()[0].split('/')[0]
        filename_pred_png = save_name + '/raw/' + scene_name + '_' + filenames[s].split()[0].split('/')[1].replace(
            '.jpg', '.png')
        filename_cmap_png = save_name + '/cmap/' + scene_name + '_' + filenames[s].split()[0].split('/')[1].replace(
            '.jpg', '.png')
        filename_gt_png = save_name + '/gt/' + scene_name + '_' + filenames[s].split()[0].split('/')[1].replace(
            '.jpg', '.png')
        filename_image_png = save_name + '/rgb/' + \
            scene_name + '_' + filenames[s].split()[0].split('/')[1]
        rgb_path = os.path.join(args.data_path, './' + filenames[s].split()[0])
        image = cv2.imread(rgb_path)
        if args.dataset == 'nyu':
            gt_path = os.path.join(
                args.data_path, './' + filenames[s].split()[1])
            gt = cv2.imread(gt_path, -1).astype(np.float32) / \
                1000.0  # Visualization purpose only
            gt[gt == 0] = np.amax(gt)

        pred_depth = pred_depths[s]

        pred_depth_scaled = pred_depth * 1000.0

        pred_depth_scaled = pred_depth_scaled.astype(np.uint16)
        cv2.imwrite(filename_pred_png, pred_depth_scaled,
                    [cv2.IMWRITE_PNG_COMPRESSION, 0])

        if args.save_lpg:
            cv2.imwrite(filename_image_png, image[10:-1 - 9, 10:-1 - 9, :])
            plt.imsave(filename_gt_png, np.log10(
                gt[10:-1 - 9, 10:-1 - 9]), cmap='Greys')
            pred_depth_cropped = pred_depth[10:-1 - 9, 10:-1 - 9]
            plt.imsave(filename_cmap_png, np.log10(
                pred_depth_cropped), cmap='Greys')

    return


if __name__ == '__main__':
    test(args)
