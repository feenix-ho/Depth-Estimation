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

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torch.utils.data.distributed
import torch.distributed as dist
from torchvision import transforms
from PIL import Image, ImageDraw
import os
import random
import json
import argparse
import sys


def _is_pil_image(img):
    return isinstance(img, Image.Image)


def _is_numpy_image(img):
    return isinstance(img, np.ndarray) and (img.ndim in {2, 3})


def preprocessing_transforms(mode):
    return transforms.Compose([
        ToTensor(mode=mode)
    ])


class BtsDataLoader(object):
    def __init__(self, args, mode):
        if mode == 'train':
            self.training_samples = DataLoadPreprocess(
                args, mode, transform=preprocessing_transforms(mode))
            self.train_sampler = None
            self.data = DataLoader(self.training_samples, args.batch_size,
                                   shuffle=(self.train_sampler is None),
                                   num_workers=args.num_threads,
                                   pin_memory=True,
                                   sampler=self.train_sampler)

        elif mode == 'online_eval':
            self.testing_samples = DataLoadPreprocess(
                args, mode, transform=preprocessing_transforms(mode))
            self.eval_sampler = None
            self.data = DataLoader(self.testing_samples, 1,
                                   shuffle=False,
                                   num_workers=1,
                                   pin_memory=True,
                                   sampler=self.eval_sampler)

        elif mode == 'test':
            self.testing_samples = DataLoadPreprocess(
                args, mode, transform=preprocessing_transforms(mode))
            self.data = DataLoader(self.testing_samples,
                                   1, shuffle=False, num_workers=1)

        else:
            print(
                'mode should be one of \'train, test, online_eval\'. Got {}'.format(mode))


def bbox_resize(location):
    x_scale = 384 / 640
    y_scale = 256 / 480
    x = int(np.round(location[0] * x_scale))
    y = int(np.round(location[1] * y_scale))
    xmax = int(np.round(location[2] * x_scale))
    ymax = int(np.round(location[3] * y_scale))
    return [x, y, xmax, ymax]


class DataLoadPreprocess(Dataset):
    def __init__(self, args, mode, transform=None, is_for_online_eval=False):
        self.args = args
        if args.data_path[-1] != '/':
            args.data_path += '/'
        if mode == 'online_eval':
            mode = 'test'

        # load file names in 'custom_data_info.json'
        open_file = open(args.data_path + 'data.json')
        d = json.load(open_file)
        self.filenames = d['idx_to_' + mode + '_files']
        self.idx_to_bbox_embed = d['idx_to_' + mode + '_bbox_embed']
        open_file.close()

        self.images_path = args.data_path + 'nyu_' + mode + '/'

        self.bbox_embed_path = args.data_path + 'bbox_embed/'

        self.depths_path = args.data_path + 'nyu_depth_' + mode + '/'

        self.mode = mode
        self.transform = transform
        self.to_tensor = ToTensor
        self.is_for_online_eval = is_for_online_eval

    def __getitem__(self, idx):
        sample_path = self.filenames[idx]
        idx_bbox_embed = self.idx_to_bbox_embed[idx]
        filename = int(sample_path[:-4])
        size = (384, 256)

        if self.mode == 'train':
            # set path
            image_path = self.images_path + sample_path
            depth_path = self.depths_path + str(filename) + '.npz'
            bbox_embed_path = self.bbox_embed_path + \
                str(idx_bbox_embed) + '.npz'
            # load & resize image
            image = Image.open(image_path).resize(size, Image.BICUBIC)
            # load depth
            f = np.load(depth_path)
            depth_gt = f['depth'].T
            f.close()
            # resize depth
            img_depth = depth_gt * 1000.0
            img_depth_uint16 = img_depth.astype(np.uint16)
            depth_gt = Image.fromarray(
                img_depth_uint16).resize(size, Image.NEAREST)
            # load bbox & embed
            f = np.load(bbox_embed_path)
            bbox = f['bbox']
            # resize bbox
            bbox = np.apply_along_axis(bbox_resize, 1, bbox)
            embedding = f['embed']
            f.close()

            image = np.asarray(image, dtype=np.float32) / 255.0
            depth_gt = np.asarray(depth_gt, dtype=np.float32)
            depth_gt = np.expand_dims(depth_gt, axis=2)
            bbox = np.asarray(bbox, dtype=np.float32)
            embedding = np.asarray(embedding, dtype=np.float32)

            mask = np.zeros(depth_gt.shape, dtype=bool)
            mask[45:471, 41:601] = 1
            depth_gt = depth_gt / 1000.0
            mask &= depth_gt > .1

            image, depth_gt = self.train_preprocess(image, depth_gt)
            sample = {'image': image, 'depth': depth_gt,
                      'embedding': embedding, 'bbox': bbox, 'mask': mask
                      }

        else:
            # set path
            image_path = self.images_path + sample_path
            bbox_embed_path = self.bbox_embed_path + \
                str(idx_bbox_embed) + '.npz'
            # load & resize image
            image = Image.open(image_path).resize(size, Image.BICUBIC)
            # resize depth
            img_depth = depth_gt * 1000.0
            img_depth_uint16 = img_depth.astype(np.uint16)
            depth_gt = Image.fromarray(
                img_depth_uint16).resize(size, Image.NEAREST)
            # load bbox & embed
            f = np.load(bbox_embed_path)
            bbox = f['bbox']
            # resize bbox
            bbox = np.apply_along_axis(bbox_resize, 1, bbox)
            embedding = f['embed']
            f.close()

            image = np.asarray(image, dtype=np.float32) / 255.0
            bbox = np.asarray(bbox, dtype=np.float32)
            embedding = np.asarray(embedding, dtype=np.float32)

            if self.mode == 'online_eval':
                depth_path = self.depths_path + str(x) + '.npz'
                has_valid_depth = False
                try:
                    # load depth
                    f = np.load(depth_path)
                    depth_gt = np.load(depth_path)['depth'].T
                    f.close()
                    i  # resize depth
                    img_depth = depth_gt * 1000.0
                    img_depth_uint16 = img_depth.astype(np.uint16)
                    depth_gt = Image.fromarray(
                        img_depth_uint16).resize(size, Image.NEAREST)
                    has_valid_depth = True
                except IOError:
                    depth_gt = False
                    # print('Missing gt for {}'.format(image_path))

                if has_valid_depth:
                    depth_gt = np.asarray(depth_gt, dtype=np.float32)
                    depth_gt = np.expand_dims(depth_gt, axis=2)

                    mask = np.zeros(depth_gt.shape, dtype=bool)
                    mask[45:471, 41:601] = 1
                    depth_gt = depth_gt / 1000.0
                    mask &= depth_gt > .1

                    image, depth_gt = self.train_preprocess(image, depth_gt)
                    depth_gt = np.asarray(depth_gt, dtype=np.float32)
                    depth_gt = np.expand_dims(depth_gt, axis=2)
                    depth_gt = depth_gt / 1000.0

                sample = {'image': image, 'depth': depth_gt, 'mask': mask,
                          'embedding': embedding, 'bbox': bbox, 'valid': has_valid_depth
                          }
            else:
                sample = {'image': image, 'embedding': embedding, 'bbox': bbox}

        if self.transform:
            sample = self.transform(sample)

        return sample

    def rotate_image(self, image, angle, flag=Image.BILINEAR):
        result = image.rotate(angle, resample=flag)
        return result

    def random_crop(self, img, depth, height, width):
        assert img.shape[0] >= height
        assert img.shape[1] >= width
        assert img.shape[0] == depth.shape[0]
        assert img.shape[1] == depth.shape[1]
        x = random.randint(0, img.shape[1] - width)
        y = random.randint(0, img.shape[0] - height)
        img = img[y:y + height, x:x + width, :]
        depth = depth[y:y + height, x:x + width, :]
        return img, depth

    def train_preprocess(self, image, depth_gt):
        # Random flipping
        '''
        do_flip = random.random()
        if do_flip > 0.5:
            image = (image[:, ::-1, :]).copy()
            depth_gt = (depth_gt[:, ::-1, :]).copy()
        '''
        # Random gamma, brightness, color augmentation
        do_augment = random.random()
        if do_augment > 0.5:
            image = self.augment_image(image)

        return image, depth_gt

    def augment_image(self, image):
        # gamma augmentation
        gamma = random.uniform(0.9, 1.1)
        image_aug = image ** gamma

        # brightness augmentation
        brightness = random.uniform(0.75, 1.25)

        image_aug = image_aug * brightness

        # color augmentation
        colors = np.random.uniform(0.9, 1.1, size=3)
        white = np.ones((image.shape[0], image.shape[1]))
        color_image = np.stack([white * colors[i] for i in range(3)], axis=2)
        image_aug *= color_image
        image_aug = np.clip(image_aug, 0, 1)

        return image_aug

    def __len__(self):
        return len(self.filenames)


class ToTensor(object):
    def __init__(self, mode):
        self.mode = mode
        self.normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    def __call__(self, sample):
        image = self.to_tensor(sample['image'])
        image = self.normalize(image)
        embedding = torch.Tensor(sample['embedding'])
        bbox = torch.LongTensor(sample['bbox'])

        if self.mode == 'test':
            return {'image': image, 'embedding': embedding, 'bbox': bbox}

        mask = torch.BoolTensor(sample['mask'])
        depth = self.to_tensor(sample['depth'])

        if self.mode != 'train':
            valid_depth = sample['valid']
            return {'image': image, 'mask': mask, 'embedding': embedding,
                    'bbox': bbox, 'depth': depth, 'valid': valid_depth}

        return {'image': image, 'mask': mask, 'embedding': embedding,
                'bbox': bbox, 'depth': depth}

    def to_tensor(self, pic):
        if not (_is_pil_image(pic) or _is_numpy_image(pic)):
            raise TypeError(
                'pic should be PIL Image or ndarray. Got {}'.format(type(pic)))

        if isinstance(pic, np.ndarray):
            img = torch.from_numpy(pic.transpose((2, 0, 1)))
            return img

        # handle PIL Image
        if pic.mode == 'I':
            img = torch.from_numpy(np.array(pic, np.int32, copy=False))
        elif pic.mode == 'I;16':
            img = torch.from_numpy(np.array(pic, np.int16, copy=False))
        else:
            img = torch.ByteTensor(
                torch.ByteStorage.from_buffer(pic.tobytes()))
        # PIL image mode: 1, L, P, I, F, RGB, YCbCr, RGBA, CMYK
        if pic.mode == 'YCbCr':
            nchannel = 3
        elif pic.mode == 'I;16':
            nchannel = 1
        else:
            nchannel = len(pic.mode)
        img = img.view(pic.size[1], pic.size[0], nchannel)

        img = img.transpose(0, 1).transpose(0, 2).contiguous()
        if isinstance(img, torch.ByteTensor):
            return img.float()
        else:
            return img


def main():
    args = Arg_train()
    obj = BtsDataLoader(args, 'train')
    a = obj.training_samples.__getitem__(1)
    print(obj)
    del obj.training_samples.depth


if __name__ == '__main__':
    main()
