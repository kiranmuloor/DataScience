import os
# Ignore warnings
import warnings

warnings.filterwarnings("ignore")
import os.path as osp
import numpy as np
from Common.utils import ImgAugTransform
from torchvision import datasets, transforms

from PIL import Image
from PIL import ImageDraw

from matplotlib import pyplot as plt
import warnings

warnings.filterwarnings("ignore")
import os.path as osp
import json
import torchvision.transforms.functional as TF
from skimage.filters import threshold_otsu



class data_loader():
    def __init__(self, mode="train"):
        path_ = os.getcwd()
        self.root = path_ + '/data/'
        self.datamode = mode  # train or test or self-define
        self.fine_height = 128
        self.fine_width = 128
        self.radius = 3
        self.data_path = osp.join(self.root, self.datamode)
        self.transform = transforms.Compose([transforms.Resize((self.fine_height, self.fine_height)), \
 \
                                             transforms.ToTensor(), \
                                             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        self.transform_input = transforms.Compose([ImgAugTransform(), \
                                                   transforms.ToTensor(),
                                                   transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

        self.rotate = ImgAugTransform()

    def transformData(self, src, mask, target, cloth, skel, head):
        # Resize
        resize = transforms.Resize(size=(128, 128))
        src = resize(src)  # Source with missing cloth
        mask = resize(mask)  # mask of the missing cloth
        target = resize(target)  # target/ Ground truth
        cloth = resize(cloth)  # Cloth ground truth, how it should look before applying
        skel = resize(skel)  # skeleton
        head = resize(head)  # head

        # src = self.rotate(src)
        # mask = self.rotate(mask)
        # target = self.rotate(target)
        # cloth = self.rotate(cloth)
        # skel = self.rotate(skel)

        src = TF.to_tensor(src)
        mask = TF.to_tensor(mask)
        target = TF.to_tensor(target)
        cloth = TF.to_tensor(cloth)
        skel = TF.to_tensor(skel)

        src = TF.normalize(src, (0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        mask = TF.normalize(mask, (0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        target = TF.normalize(target, (0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        cloth = TF.normalize(cloth, (0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        skel = TF.normalize(skel, (0.5, 0.5, 0.5), (0.5, 0.5, 0.5))

        return src, mask, target, cloth, skel, head

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

    def get_img(self, im_name, c_name):

        # person image
        im = plt.imread(osp.join(self.data_path, 'image', im_name))
        cm = plt.imread(osp.join(self.data_path, 'cloth', c_name))

        # print(im)
        # img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        plt.imshow(im)
        plt.imshow(cm)

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

        parse_cloth = (parse_array == 5).astype(np.float32) + \
                      (parse_array == 6).astype(np.float32) + \
                      (parse_array == 7).astype(np.float32) + \
                      (parse_array == 9).astype(np.float32) + \
                      (parse_array == 15).astype(np.float32) + (parse_array == 3).astype(np.float32) + (
                              parse_array == 14).astype(np.float32)

        pcm = self.get_binary_from_img(parse_cloth)
        phead = self.get_binary_from_img(parse_head)  # [0,1]
        im_h = im * phead - (1 - phead)  # [-1,1], fill 0 for other parts

        pose_name = im_name.replace('.jpg', '_keypoints.json')
        with open(osp.join(self.data_path, 'pose', pose_name), 'r') as f:
            pose_label = json.load(f)
            pose_data = pose_label['people'][0]['pose_keypoints']
            pose_data = np.array(pose_data)
            pose_data = pose_data.reshape((-1, 3))

        point_num = pose_data.shape[0]
        r = 7  # self.radius
        #         pdb.set_trace()
        coop = {}
        coop2 = {}
        ai = 0
        for lol, i in enumerate(
                [1, 2, 3, 4, 5, 6, 7, 8, 11]):  # leaving out head and legs joints, heap and hands are kept
            pointx = pose_data[i, 0]
            pointy = pose_data[i, 1]
            if pointx > 1 and pointy > 1:
                coop[ai] = (pointx, pointy)
                coop2[ai] = (pointx, pointy)
                ai = ai + 1
            else:
                coop2[ai] = (pointx, pointy)
                ai = ai + 1

        # creating skeleton
        bone_list = [[x[0], x[1]] for key, x in coop2.items()]
        #         bone_list = bone_list.numpy()
        bone_list = np.array(bone_list) - 1
        itemindex = np.where(bone_list == -1)
        if len(itemindex[0]) == 0:
            it = 100
        else:
            it = np.unique(itemindex[0])

        one_map = Image.new('RGB', (192, 256))
        draw = ImageDraw.Draw(one_map)
        if np.logical_not(np.isin(it, 0)).all() and np.logical_not(np.isin(it, 1)).all():
            draw.line((bone_list[0][0], bone_list[0][1], bone_list[1][0], bone_list[1][1]), fill='red', width=14)
        if np.logical_not(np.isin(it, 1)).all() and np.logical_not(np.isin(it, 2)).all():
            draw.line((bone_list[1][0], bone_list[1][1], bone_list[2][0], bone_list[2][1]), fill='blue', width=14)
        if np.logical_not(np.isin(it, 3)).all() and np.logical_not(np.isin(it, 2)).all():
            draw.line((bone_list[2][0], bone_list[2][1], bone_list[3][0], bone_list[3][1]), fill='white', width=14)
        if np.logical_not(np.isin(it, 0)).all() and np.logical_not(np.isin(it, 4)).all():
            draw.line((bone_list[0][0], bone_list[0][1], bone_list[4][0], bone_list[4][1]), fill='orange', width=14)
        if np.logical_not(np.isin(it, 4)).all() and np.logical_not(np.isin(it, 5)).all():
            draw.line((bone_list[4][0], bone_list[4][1], bone_list[5][0], bone_list[5][1]), fill='orchid', width=14)
        if np.logical_not(np.isin(it, 6)).all() and np.logical_not(np.isin(it, 5)).all():
            draw.line((bone_list[5][0], bone_list[5][1], bone_list[6][0], bone_list[6][1]), fill='yellow', width=14)

        if np.logical_not(np.isin(it, 0)).all() and np.logical_not(np.isin(it, 1)).all():
            draw.line((bone_list[1][0], bone_list[1][1], bone_list[7][0], bone_list[7][1]), fill='gold', width=14)
        if np.logical_not(np.isin(it, 4)).all() and np.logical_not(np.isin(it, 8)).all():
            draw.line((bone_list[4][0], bone_list[4][1], bone_list[8][0], bone_list[8][1]), fill='pink', width=14)
        if np.logical_not(np.isin(it, 7)).all() and np.logical_not(np.isin(it, 8)).all():
            draw.line((bone_list[7][0], bone_list[7][1], bone_list[8][0], bone_list[8][1]), fill='brown', width=14)

        source = im * pcm
        #         face = im*(1-pcm)
        source[source == 0] = 255
        #         face[face==0]=255
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
        style_ = self.transform(style)
        cloth = self.transform(cloth)
        head = Image.fromarray(np.uint8(im_h))
        head = self.transform(head)

        source, mask, target, targ, skel, head = self.transformData(input, mask, target, style, one_map, head)
        del lol3, lol2, pcm, im, parse_cloth, im_parse, lol
        return source, mask, style_, target, targ, skel, cloth, head
