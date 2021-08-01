import os
import random
import matplotlib.pyplot as plt
import numpy as np
import torch
from imgaug import augmenters as iaa
from torch.autograd import Variable

import base64

def decodeImage(imgstring, fileName):
    imgdata = base64.b64decode(imgstring)
    print(fileName)
    with open(fileName) as f:
        #commonFunctions.createDir(fileName)
        f.write(imgdata)
        f.close()


def encodeImageIntoBase64(croppedImagePath):
    with open(croppedImagePath) as f:
        return base64.b64encode(f.read())


# Initialize kernel weights to uniform. We are not using BatchNorm in final code.
def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find('Conv2d') != -1:
        torch.nn.init.normal(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm2d') != -1:
        torch.nn.init.normal(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant(m.bias.data, 0.0)


# LambdaLR is use for Learning rate scheduling (Not used in main code).
class LambdaLR():
    def __init__(self, n_epochs, offset, decay_start_epoch):
        assert ((n_epochs - decay_start_epoch) > 0), "Decay must start before the training session ends!"
        self.n_epochs = n_epochs
        self.offset = offset
        self.decay_start_epoch = decay_start_epoch

    def step(self, epoch):
        return 1.0 - max(0, epoch + self.offset - self.decay_start_epoch) / (self.n_epochs - self.decay_start_epoch)


class commonFunctions:
    def name(self):
        return 'commonFunctions'

    def __init__(self):
        super(commonFunctions, self).__init__()

    def createDir(self, path):
        if not os.path.exists(path):
            os.makedirs(path)

    def display_img(self, img, cmap=None):
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111)
        ax.imshow(img, cmap)


class ImgAugTransform:
    def __init__(self):
        self.aug = iaa.Sequential([
            #         iaa.Scale((128, 128)),
            #         iaa.Sometimes(0.25, iaa.GaussianBlur(sigma=(0, 3.0))),
            #         iaa.Fliplr(0.5),
            #         iaa.Affine(rotate=(-40, 40), mode='symmetric'),
            iaa.Affine(rotate=40, mode='symmetric')
            #         iaa.Sometimes(0.25,
            #                       iaa.OneOf([iaa.Dropout(p=(0, 0.1)),
            #                                  iaa.CoarseDropout(0.1, size_percent=0.5)])),
            #         iaa.AddToHueAndSaturation(value=(-10, 10), per_channel=True)
        ])

    def __call__(self, img):
        img = np.array(img)
        return self.aug.augment_image(img)


class ImgAugTransformStitching:
    def __init__(self):
        sometimes = lambda aug: iaa.Sometimes(0.5, aug)

        self.aug = iaa.Sequential([
            #         iaa.Scale((128, 128)),
            #         iaa.Sometimes(0.25, iaa.GaussianBlur(sigma=(0, 3.0))),
            #         iaa.Fliplr(0.5),
            iaa.Affine(rotate=40, mode='symmetric'),
            #             iaa.Affine( rotate = 20 , mode='symmetric')
            #         iaa.Sometimes(0.25,
            #                       iaa.OneOf([iaa.Dropout(p=(0, 0.1)),
            #                                  iaa.CoarseDropout(0.1, size_percent=0.5)])),
            #         iaa.AddToHueAndSaturation(value=(-10, 10), per_channel=True)
            iaa.Affine(
                translate_percent={"x": 0.2, "y": 0.1},
                #             rotate=(-45, 45),
                #             shear=(-16, 16),
                #             order=[0, 1],
                #             cval=(0, 255),
                mode='symmetric'
            )
        ])

    def __call__(self, img, img1, img2):
        img = np.array(img)
        img1 = np.array(img1)
        img2 = np.array(img2)

        return self.aug.augment_image(img), self.aug.augment_image(img1), self.aug.augment_image(img2)


class ImgAugTransformRefine:
    def __init__(self):
        sometimes = lambda aug: iaa.Sometimes(0.5, aug)

        self.aug = iaa.Sequential([
            iaa.Affine(
                translate_percent={"x": 0.2, "y": 0.1},
                mode='symmetric'
            )
        ])

    def __call__(self, img, img1, img2):
        img = np.array(img)
        img1 = np.array(img1)
        img2 = np.array(img2)

        return self.aug.augment_image(img), self.aug.augment_image(img1), self.aug.augment_image(img2)


# Photo-Realistic Single Image Super-Resolution Using a Generative Adversarial Network
# ReplayBuffer was first introduced in the above mentioned paper, It's effect mathematically has been supported in
# latest ICLR paper ProbGAN. Replay buffer uses previous data as prior for the Discriminator which it has seen already.
# Page 5 of the paper, just over Theory section.
# Hence we propose to maintain a subset of discriminators by subsampling the whole sequence of discriminators.

class ReplayBuffer():
    def __init__(self, max_size=50):
        assert (max_size > 0), 'Empty buffer or trying to create a black hole. Be careful.'
        self.max_size = max_size
        self.data = []

    def push_and_pop(self, data):
        to_return = []
        for element in data.data:
            element = torch.unsqueeze(element, 0)
            if len(self.data) < self.max_size:
                self.data.append(element)
                to_return.append(element)
            else:
                if random.uniform(0, 1) > 0.5:
                    i = random.randint(0, self.max_size - 1)
                    to_return.append(self.data[i].clone())
                    self.data[i] = element
                else:
                    to_return.append(element)
        return Variable(torch.cat(to_return))
