import os, sys, gc, argparse, numpy as np
import torch
from torchvision.utils import save_image
from torch.utils.data import DataLoader
from torch.autograd import Variable
from skimage.transform import rescale, resize, downscale_local_mean
from skimage.filters import threshold_otsu, threshold_local  # threshold_adaptive
from skimage.morphology import binary_closing, binary_opening, binary_erosion, binary_dilation
from skimage.exposure import rescale_intensity
import torchvision.transforms.functional as TF
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
from PIL import Image


def diffMask(img1=None, img2=None, opt=None, dataset=None, args=None):
    netG = args[0]
    netB = args[1]
    netD = args[2]
    f = args[3]
    res_path = opt.results_Stage3
    res_folders = ['temp_masks',
                   'temp_Stage2',
                   'temp_ref',
                   'temp_diff',
                   'temp_Stage3',
                   'temp_skel',
                   'temp_res',
                   'temp_Stage1',
                   'temp_src']
    for x in res_folders:
        if os.path.isdir("{}{}".format(res_path, x)) == False:
            os.mkdir("{}{}".format(res_path, x))
    save_masks = "{}{}".format(res_path, "temp_masks")
    save_Stage2 = "{}{}".format(res_path, "temp_Stage2")
    save_ref = "{}{}".format(res_path, "temp_ref")
    save_diff = "{}{}".format(res_path, "temp_diff")
    save_Stage3 = "{}{}".format(res_path, "temp_Stage3")
    save_skel = "{}{}".format(res_path, "temp_skel")
    save_res = "{}{}".format(res_path, "temp_res")
    save_Stage1 = "{}{}".format(res_path, "temp_Stage1")
    save_src = "{}{}".format(res_path, "temp_src")

    resize2 = transforms.Resize(size=(128, 128))
    src, mask, style_img, target, gt_cloth, skel, cloth, face = dataset.get_img("{}_0.jpg".format(img1[:-6]),
                                                                                "{}_1.jpg".format(img1[:-6]))
    src, mask, style_img, target, gt_cloth, skel, cloth, face = src.unsqueeze(0), mask.unsqueeze(
        0), style_img.unsqueeze(
        0), target.unsqueeze(0), gt_cloth.unsqueeze(0), skel.unsqueeze(0), cloth.unsqueeze(0), face.unsqueeze(0)
    src1, mask1, style_img1, target1, gt_cloth1, skel1, cloth1, face1 = Variable(src), Variable(mask), Variable(
        style_img), Variable(target), Variable(gt_cloth), Variable(skel), Variable(
        cloth), Variable(face)
    src, mask, style_img, target, gt_cloth, skel, cloth, face = dataset.get_img("{}_0.jpg".format(img2[:-6]),
                                                                                "{}_1.jpg".format(img2[:-6]))
    src, mask, style_img, target, gt_cloth, skel, cloth, face = src.unsqueeze(0), mask.unsqueeze(0), style_img.unsqueeze(
        0), target.unsqueeze(0), gt_cloth.unsqueeze(0), skel.unsqueeze(0), cloth.unsqueeze(0), face.unsqueeze(0)
    src2, mask2, style_img2, target2, gt_cloth2, skel2, cloth2, face2 = Variable(src), Variable(mask), Variable(
        style_img), Variable(target), Variable(gt_cloth), Variable(skel), Variable(
        cloth), Variable(face)
    gen_targ_Stage1, s_128, s_64, s_32, s_16, s_8, s_4 = netG(skel1, cloth2)  # gen_targ11 is structural change cloth
    gen_targ_Stage2, s_128, s_64, s_32, s_16, s_8, s_4 = netB(src1, gen_targ_Stage1,
                                                              skel1)  # gen_targ12 is Stage2 image

    # saving structural
    pic_Stage2 = (torch.cat([gen_targ_Stage2], dim=0).data + 1) / 2.0
    plt.imsave("./pic_Stage2.jpg", pic_Stage2)
    plt.imsave("./pic_Stage1.jpg", (torch.cat([gen_targ_Stage1], dim=0).data + 1) / 2.0)
    plt.imsave("./pic_face.jpg", face2)
    #     save_dir = "/home/np9207/PolyGan_res/temp_Stage2/"
    save_image(pic_Stage2, '%s/%d_%s_%d.jpg' % (save_Stage2, f, img1[:-6], 0), nrow=1)
    print(save_Stage2)
    msk1 = torch.as_tensor(mask1[0, :, :, :].detach().permute(1, 2, 0))

    # print(save_masks)

    #plt.imsave("./res_msk1.jpg", msk1)

    plt.imsave("{}/{}_{}_mask.jpg".format(save_masks, f, img1[:-6]), np.array(msk1, dtype='uint8'), cmap="gray")
    plt.imsave("{}/{}_{}_ref.jpg".format(save_ref, f, img1[:-6]),
               resize(plt.imread("data/{}/image/{}_0.jpg".format(opt.datamode, img1[:-6])),
                      (128, 128)))
    Stage2 = rescale_intensity(plt.imread("{}/{}_{}_0.jpg".format(save_Stage2, f, img1[:-6])) / 255)
    mask = rescale_intensity(plt.imread("{}/{}_{}_mask.jpg".format(save_masks, f, img1[:-6])) / 255)
    ref = rescale_intensity(plt.imread("{}/{}_{}_ref.jpg".format(save_ref, f, img1[:-6])) / 255)

    temp_im = ref * (1 - mask)
    temp1 = ref * mask  # Gives original image without cloth
    temp2 = Stage2 * mask  # Gives
    temp2[:, :, 0][temp2[:, :, 0] < 0.95] = 0
    #     print(lol.Stage1)

    block_size = 13
    binary = threshold_local(temp2[:, :, 0], block_size, offset=0)

    plt.imshow(binary * 1, cmap="gray")
    plt.imsave("{}/{}_{}_diff.jpg".format(save_diff, f, img1[:-6]), binary * 1, cmap="gray")
    diff = plt.imread("{}/{}_{}_diff.jpg".format(save_diff, f, img1[:-6]))
    diff = Image.fromarray(np.uint8(diff))
    diff = resize2(diff)
    diff = TF.to_tensor(diff)
    diff = TF.normalize(diff, (0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    diff = diff.unsqueeze(0)
    diff = Variable(diff)

    gen_targ_Stage3, s_128, s_64, s_32, s_16, s_8, s_4 = netD(diff, gen_targ_Stage2)

    pic = (torch.cat([gen_targ_Stage3], dim=0).data + 1) / 2.0

    save_image(pic, '{}/{}_{}_{}.jpg'.format(save_Stage3, f, img1[:-6], 54), nrow=1)

    pic2 = (torch.cat([gen_targ_Stage1], dim=0).data + 1) / 2.0
    pic3 = (torch.cat([skel1, face1], dim=0).data + 1) / 2.0

    pic00 = (torch.cat([src1], dim=0).data + 1) / 2.0
    save_image(pic00, '{}/{}_{}_src.jpg'.format(save_src, f, img1[:-6]), nrow=3)
    save_image(pic2, '{}/{}_{}_{}_Stage1.jpg'.format(save_Stage1, f, img1[:-6], img2[:-6]), nrow=1)
    save_image(pic3, '{}/{}_{}_skel.jpg'.format(save_skel, f, img1[:-6]), nrow=1)


def saveFullTranslation(image1=None, image2=None, opt=None, f=0):
    res_path = opt.results_Stage3
    save_masks = "{}{}".format(res_path, "temp_masks")
    save_Stage2 = "{}{}".format(res_path, "temp_Stage2")
    save_ref = "{}{}".format(res_path, "temp_ref")
    save_diff = "{}{}".format(res_path, "temp_diff")
    save_Stage3 = "{}{}".format(res_path, "temp_Stage3")
    save_skel = "{}{}".format(res_path, "temp_skel")
    save_res = "{}{}".format(res_path, "temp_res")
    save_Stage1 = "{}{}".format(res_path, "temp_Stage1")
    save_src = "{}{}".format(res_path, "temp_src")

    Stage3_img = rescale_intensity(plt.imread('{}/{}_{}_{}.jpg'.format(save_Stage3, f, image1[:-6], 54)) / 255)
    diff_img = rescale_intensity(plt.imread("{}/{}_{}_diff.jpg".format(save_diff, f, image1[:-6])) / 255)
    Stage2_img = rescale_intensity(plt.imread("{}/{}_{}_0.jpg".format(save_Stage2, f, image1[:-6])) / 255)
    img4 = Stage2_img * (1 - diff_img)

    plt.imsave("./res_Stage3.jpg", Stage3_img)
    plt.imsave("./res_Stage2.jpg", Stage2_img)

    img5 = binary_closing(diff_img[:, :, 0], )
    ms2 = img5 * 1
    ms2 = np.expand_dims(ms2, axis=2)
    ms2 = np.repeat(ms2, repeats=3, axis=2)
    img7 = Stage2_img * (1 - ms2)
    img8 = Stage3_img * ms2
    im88 = ((img7 + img8) - img8.min()) / (img8.max() - img8.min())
    #     pdb.set_trace()
    ms = rescale_intensity(plt.imread("{}/{}_{}_mask.jpg".format(save_masks, f, image1[:-6])) / 255)
    thresh = threshold_otsu(ms[:, :, 0])
    #plt.imsave(img4)
    #plt.imsave(img5)
    #plt.imsave(diff_img)

    binary1 = ms[:, :, 0] > thresh

    ms = binary1 * 1
    ms = np.expand_dims(ms, axis=2)
    ms = np.repeat(ms, repeats=3, axis=2)

    im2 = rescale_intensity(plt.imread("{}/{}_{}_ref.jpg".format(save_ref, f, image1[:-6])) / 255)

    im3 = im2 * (1 - ms)
    im3[im3 == 0] = 1

    plt.imsave("./res_1.jpg", im3 * im88)
    res = rescale_intensity(plt.imread("./res_1.jpg") / 255)
    plt.imsave("{}/{}_{}a.jpg".format(save_res, f, image1[:-6]), img4)
    plt.imsave("{}/{}_{}b.jpg".format(save_res, f, image1[:-6]), img5)
    plt.imsave("{}/{}_{}c.jpg".format(save_res, f, image1[:-6]), diff_img)
    plt.imsave("{}/{}_{}.jpg".format(save_res, f, image1[:-6]), res)
