import numpy as np
import os
import os.path as osp
import matplotlib.pyplot as plt
import json
import random
import torch.utils.data as data
import torchvision.transforms as transforms
from Common.utils import ImgAugTransformRefine, ImgAugTransform

from skimage.filters import threshold_otsu
from PIL import Image
import torchvision.transforms.functional as TF
from PIL import ImageDraw

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")


class refineDataSetExtract():
    def __init__(self=None, height=0):
        super(refineDataSetExtract, self).__init__()
        # base setting
        path_ = os.getcwd()
        self.root = path_ + '/data/'
        self.datamode = 'train'  # train or test or self-define
        self.data_list = "train_pairs.txt"
        self.fine_height = height
        self.fine_width = 128
        self.radius = 3
        self.data_path = osp.join(self.root, self.datamode)
        self.transform = transforms.Compose(
            (transforms.Scale(self.fine_height), transforms.ToTensor(), transforms.Normalize(0.5, 0.5)))

        self.transform_input = transforms.Compose(
            [ImgAugTransform(), transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        # load data list
        im_names = []
        c_names = []
        with open(osp.join(self.root, self.data_list), 'r') as f:
            for line in f.readlines():
                im_name, c_name = line.strip().split()
                im_names.append(im_name)
                c_names.append(c_name)

        self.im_names = im_names
        self.c_names = c_names
        self.rotate = ImgAugTransformRefine()

    def name(self):
        return "refineDataSetExtract"

    def transformData(self, src, mask, target, cloth, wrap, diff, head):
        # Resize
        resize = transforms.Resize(size=(128, 128))
        src = resize(src)  # Source with missing cloth
        mask = resize(mask)  # mask of the missing cloth
        target = resize(target)  # target/ Ground truth
        cloth = resize(cloth)  # Cloth ground truth, how it should look before applying
        wrap = resize(wrap)  # skeleton
        diff = resize(diff)
        head = resize(head)

        src = TF.to_tensor(src)
        mask = TF.to_tensor(mask)
        target = TF.to_tensor(target)
        cloth = TF.to_tensor(cloth)
        wrap = TF.to_tensor(wrap)
        diff = TF.to_tensor(diff)
        #head = TF.to_tensor(head)

        src = TF.normalize(src, (0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        mask = TF.normalize(mask, (0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        target = TF.normalize(target, (0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        cloth = TF.normalize(cloth, (0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        wrap = TF.normalize(wrap, (0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        diff = TF.normalize(diff, (0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        return src, mask, target, cloth, wrap, diff, head

    def get_binary_from_img(self, image_name):
        loader2 = transforms.Compose([transforms.Resize((256, 192)), transforms.ToTensor()])
        """load image, returns cuda tensor"""
        image = Image.fromarray(np.uint8(image_name))
        image = loader2(image).float()
        better_contrast = image.permute(1, 2, 0).detach().cpu().numpy()
        better_contrast[better_contrast > 1] = 1
        #     print(lol.shape)

        thresh = threshold_otsu(better_contrast)
        binary = better_contrast > thresh
        return binary  # assumes that you're using GPU

    def get_binary(self, image_name):
        loader2 = transforms.Compose([transforms.Resize((256, 192)), transforms.ToTensor()])
        """load image, returns cuda tensor"""
        image = Image.open(image_name)
        image = loader2(image).float()
        better_contrast = image.permute(1, 2, 0).detach().cpu().numpy()
        better_contrast[better_contrast > 1] = 1
        #     print(lol.shape)

        thresh = threshold_otsu(better_contrast)
        binary = better_contrast > thresh
        return binary  # assumes that you're using GPU

    def __getitem__(self, index):
        c_name = self.c_names[index]
        im_name = self.im_names[index]

        # person image
        im = plt.imread(osp.join(self.data_path, 'image', im_name))
        cm = plt.imread(osp.join(self.data_path, 'cloth', c_name))
        wrap = plt.imread(osp.join(self.data_path, 'image_shaped_cloth', im_name))
        diff = plt.imread(osp.join(self.data_path, 'changed_diff', im_name))
        #         im = self.transform(im) # [-1,1]

        # load parsing image

        parse_name = im_name.replace('.jpg', '.png')
        im_parse = Image.open(osp.join(self.data_path, 'image-parse', parse_name))
        parse_array = np.array(im_parse)
        #         parse_shape = (parse_array > 0).astype(np.float32)

        parse_head = (parse_array == 1).astype(np.float32) + \
                     (parse_array == 2).astype(np.float32) + \
                     (parse_array == 4).astype(np.float32) + \
                     (parse_array == 13).astype(np.float32)

        parse_cloth = (parse_array == 5).astype(np.float32) + (parse_array == 6).astype(np.float32) + (
                    parse_array == 7).astype(np.float32) + (parse_array == 9).astype(np.float32) + (
                                  parse_array == 15).astype(np.float32) + (parse_array == 3).astype(np.float32) + (
                                  parse_array == 14).astype(np.float32)

        pcm = self.get_binary_from_img(parse_cloth)
        phead = self.get_binary_from_img(parse_head)  # [0,1]
        im_h = im * phead - (1 - phead)  # [-1,1], fill 0 for other parts

        source = im * pcm
        source[source == 0] = 255
        mask = plt.imread(osp.join(self.data_path, 'nested_unet_msk', im_name))

        lol = self.get_binary(osp.join(self.data_path, 'nested_unet_msk', im_name))
        lol2 = source * (1 - lol)
        lol2[lol2 == 0] = 255

        lol3 = source * (lol)
        lol3[lol3 == 0] = 255

        input = Image.fromarray(np.uint8(lol2))
        mask = Image.fromarray(np.uint8(mask))
        style = Image.fromarray(np.uint8(lol3))
        target = Image.fromarray(np.uint8(source))
        cloth = Image.fromarray(np.uint8(cm))
        wrap = Image.fromarray(np.uint8(wrap))
        diff = Image.fromarray(np.uint8(diff))
        head = Image.fromarray(np.uint8(im_h))

        #         source = self.transform_input(input)  # [-1,1]
        #         mask = self.transform_input(mask)  # [-1,1]
        style_ = self.transform(style)
        cloth = self.transform(cloth)
        head = self.transform(head)
        #         targ = self.transform_input(style)
        #         skel = self.transform_input(one_map)

        resize = transforms.Resize(size=(128, 128))
        cloth = resize(cloth)  # Cloth ground truth, how it should look before applying
        style_ = resize(style_)

        source, mask, target, targ, wrap, diff, head = self.transformData(input, mask, target, style, wrap, diff, head)
        del lol3, lol2, pcm, im, parse_cloth, im_parse, lol
        return source, mask, style_, target, targ, wrap, diff, cloth, head  # , skel

    #

    def __len__(self):
        return len(self.im_names)