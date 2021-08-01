import os, sys, gc, argparse, numpy as np
from Datasets.DLoaderDatasets import data_loader
from Models.models import GeneratorCoarse, Discriminator
from Common.testModel import diffMask, saveFullTranslation
import torch
from torchvision.utils import save_image
from torch.utils.data import DataLoader
from torch.autograd import Variable


def get_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument("dataroot", default="data")
    parser.add_argument("datamode", default="train")
    parser.add_argument("stage", default="Stage1", help='Stage1, Stage2, Stage3')
    parser.add_argument('Stage1', type=str, default='pre_trained_models/Stage_1/Gan_44.pth', help='load_Stage_1_model')
    parser.add_argument('Stage2', type=str, default='pre_trained_models/Stage_2/Gan_42.pth', help='load_Stage_2_model')
    parser.add_argument('Stage3', type=str, default='pre_trained_models/Stage_3/Gan_48.pth', help='load_Stage_3_model')
    parser.add_argument('results_Stage1', type=str, default='results/test/Stage1', help='save results')
    parser.add_argument('results_Stage2', type=str, default='results/test/Stage2', help='save results')
    parser.add_argument('results_Stage3', type=str, default='results/test/Stage3', help='save results')
    parser.add_argument('model_image', default="006852_0.jpg", type=str, help='Model of the person wearing cloth')
    parser.add_argument('reference_image', default="006852_1.jpg", type=str, help='Reference cloth to swap')

    argv = ["", "data", "train", "Stage3",
            "pre_trained_models/Stage_1/Gan_1.pth", "pre_trained_models/Stage_2/Gan_5.pth",
            "pre_trained_models/Stage_3/Gan_5.pth",
            "results/test/Stage1", "results/test/Stage2", "results/test/Stage3",
            "000046_0.jpg", "000189_1.jpg"]
    opt = parser.parse_args(argv[1:])
    return opt


def test(opt, test_loader, image1, image2, *args):
    print("Stage1 execute Start")
    print(image1)
    print(image2)
    src, mask, style_img, target, gt_cloth, skel, cloth, head = test_loader.get_img(image1, image2)
    src, mask, style_img, target, gt_cloth, skel, cloth, head = src.unsqueeze(0), mask.unsqueeze(
        0), style_img.unsqueeze(0), target.unsqueeze(0), gt_cloth.unsqueeze(0), skel.unsqueeze(0), cloth.unsqueeze(0), head.unsqueeze(0)
    src, mask, style_img, target, gt_cloth, skel, cloth, head = Variable(src), Variable(mask), Variable(
        style_img), Variable(target), Variable(gt_cloth), Variable(skel), Variable(cloth), Variable(head)

    if opt.stage == "Stage1":
        netG = args[0]
        gen_targ, _, _, _, _, _, _ = netG(skel, cloth)  # src,conditions
        pic = (torch.cat([gen_targ], dim=0).data + 1) / 2.0
        save_dir = "{}/{}".format(os.getcwd(), opt.results_Stage1)
        print(save_dir)
        save_image(pic, '{}/{}_{}'.format(save_dir, args[1], opt.model_image), nrow=1)
    elif opt.stage == "Stage2":
        netG1 = args[0]
        netG2 = args[1]
        gen_targ_Stage1, _, _, _, _, _, _ = netG1(skel, cloth)
        gen_targ_Stage2, _, _, _, _, _, _ = netG2(src, gen_targ_Stage1, skel)
        pic1 = (torch.cat([gen_targ_Stage1], dim=0).data + 1) / 2.0
        pic2 = (torch.cat([gen_targ_Stage2], dim=0).data + 1) / 2.0
        save_dir1 = "{}/{}".format(os.getcwd(), opt.results_Stage1)
        save_image(pic1, '{}/{}_{}'.format(save_dir1, args[2], opt.model_image), nrow=1)
        save_dir2 = "{}/{}".format(os.getcwd(), opt.results_Stage2)
        save_image(pic2, '{}/{}_{}'.format(save_dir2, args[2], opt.model_image), nrow=1)
    elif opt.stage == "Stage3":
        diffMask(image1, image2, opt, test_loader, args)
        saveFullTranslation(image1, image2, opt, args[3])

    print("Stage1 execute Completed")


def main():
    print('test1')
    opt = get_opt()
    print('test2')
    print(opt)
    print("Start to test stage: %s" % (opt.stage))
    # create dataset
    test_loader = data_loader(opt.datamode)

    if not os.path.exists(opt.results_Stage2):
        os.makedirs(opt.results_Stage2)
    if not os.path.exists(opt.results_Stage1):
        os.makedirs(opt.results_Stage1)
    if not os.path.exists(opt.results_Stage3):
        os.makedirs(opt.results_Stage3)

    if opt.stage == "Stage1":
        print(opt.Stage1)
        netG_Stage1 = GeneratorCoarse(6, 3)
        netG_Stage1.load_state_dict(torch.load("./{}".format(opt.Stage1)))
        # print(netG_Stage1)
        test(opt, test_loader, opt.model_image, opt.reference_image, netG_Stage1, 1)
    elif opt.stage == "Stage2":
        print(opt.stage)
        netG_Stage1 = GeneratorCoarse(6, 3)
        netG_Stage2 = GeneratorCoarse(9, 3)

        netG_Stage1.load_state_dict(torch.load("{}".format(opt.Stage1)))
        netG_Stage2.load_state_dict(torch.load("{}".format(opt.Stage2)))
        test(opt, test_loader, opt.model_image, opt.reference_image, netG_Stage1, netG_Stage2, 1)

    elif opt.stage == "Stage3":
        netG_Stage1 = GeneratorCoarse(6, 3)
        netG_Stage2 = GeneratorCoarse(9, 3)
        netG_Stage3 = GeneratorCoarse(6, 3)

        netG_Stage1.load_state_dict(torch.load("{}".format(opt.Stage1)))
        netG_Stage2.load_state_dict(torch.load("{}".format(opt.Stage2)))
        netG_Stage3.load_state_dict(torch.load("{}".format(opt.Stage3)))
        test(opt, test_loader, opt.model_image, opt.reference_image, netG_Stage1, netG_Stage2, netG_Stage3, 1)

    print('Finished testing %s!' % (opt.stage))


if __name__ == "__main__":
    main()
