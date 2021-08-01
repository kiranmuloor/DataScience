from __future__ import print_function
import os, sys, gc, argparse, numpy as np

import torch
from torchvision.utils import save_image
from torch.utils.data import DataLoader
from torch.autograd import Variable
import torch.optim as optim
import torch.utils.data

from Common.utils import weights_init_normal, LambdaLR, ReplayBuffer, commonFunctions

from Datasets.ShapeDatasets import shapeDataSetExtract
from Datasets.StitchDataset import stitchDataSetExtract
from Datasets.RefineDatasets import refineDataSetExtract
from Models.models import GeneratorCoarse, Discriminator
# Imports PIL module
from PIL import Image

# set the default values to run the training model
def get_options():
    parser = argparse.ArgumentParser()
    parser.add_argument("dataroot", type=str, default="data")
    parser.add_argument("datamode", default="train")
    parser.add_argument("stage", default="Shape", help='Shape, Stitch, Refine')
    parser.add_argument("data_list", default="train_pairs.txt")
    parser.add_argument("thread", default="0")  # number of workers/thread to use for loading data
    parser.add_argument('batch', type=str, default="1")  # batch size
    parser.add_argument('results', type=str, default='results/Shape', help='save results')
    parser.add_argument("epochs", type=str, default="45")
    parser.add_argument("input_channel", type=str, default="6")
    parser.add_argument("decay_epoch", type=str, default="10")
    parser.add_argument('learn_rate', type=str, default="0.0002", help='initial learning rate for adam')
    parser.add_argument("critic", type=str, default="10")  # Number of times after which to update Discriminator.
    parser.add_argument("display_count", type=str, default="1000")
    parser.add_argument("save_model", type=str, default="2")
    # set default values
    argv = ["", "Data", "train", "Refine", "train_pairs.txt", "0", "1", "results/"
        , "11", "6", "10", "0.0002", "10", "1000", "2"]
    opt = parser.parse_args(argv[1:])
    print("arguments are set for training the model")
    return opt


def trainRefineModel(opt, netG, netD):
    print("training the model for different model types: %s" % (opt.stage))
    dataset = shapeDataSetExtract(128)
    train_loader = DataLoader(dataset,
                              batch_size=int(opt.batch),
                              shuffle=False,
                              num_workers=int(opt.thread),
                              drop_last=True, pin_memory=True)

    epoch = 0
    n_epochs = int(opt.epochs)
    decay_epoch = int(opt.decay_epoch)
    batchSize = int(opt.batch)
    size = 128
    input_nc = int(opt.input_channel)
    output_nc = 3
    lr = float(opt.learn_rate)
    nRow = 3

    criterion_GAN = torch.nn.MSELoss()
    criterion_identity = torch.nn.L1Loss()

    optimizer_G = torch.optim.Adam(netG.parameters(), lr=lr, betas=(0.5, 0.999))
    optimizer_D = torch.optim.Adam(netD.parameters(), lr=lr, betas=(0.5, 0.999))

    lr_scheduler_G = torch.optim.lr_scheduler.LambdaLR(optimizer_G,
                                                       lr_lambda=LambdaLR(n_epochs, epoch, decay_epoch).step)
    lr_scheduler_D = torch.optim.lr_scheduler.LambdaLR(optimizer_D,
                                                       lr_lambda=LambdaLR(n_epochs, epoch, decay_epoch).step)

    # Inputs & targets memory allocation
    Tensor = torch.FloatTensor
    input_A = Tensor(batchSize, input_nc, size, size)

    target_real = Variable(Tensor(batchSize).fill_(1.0), requires_grad=False)
    target_fake = Variable(Tensor(batchSize).fill_(0.0), requires_grad=False)

    fake_buffer = ReplayBuffer()
    print(n_epochs)

    for epoch in range(0, n_epochs):
        gc.collect()
        print(epoch)
        Source = iter(train_loader)
        avg_loss_g = 0
        avg_loss_d = 0
        for i in range(0, len(train_loader)):
            netG.train()
            target_real = Variable(torch.ones(1, 1), requires_grad=False)
            target_fake = Variable(torch.zeros(1, 1), requires_grad=False)
            optimizer_G.zero_grad()

            src, mask, style_img, target, gt_cloth, skel, cloth, head = Source.next()
            src, mask, style_img, target, gt_cloth, skel, cloth, head = Variable(src), Variable(mask), Variable(
                style_img), \
                                                                        Variable(target), Variable(gt_cloth), Variable(
                skel), Variable(cloth), Variable(head)

            # inverse identity
            gen_targ, _, _, _, _, _, _ = netG(skel, cloth)  # src,conditions

            pred_fake = netD(gen_targ)
            loss_GAN = 10 * criterion_GAN(pred_fake, target_real) + 10 * criterion_identity(gen_targ, gt_cloth)

            loss_G = loss_GAN
            loss_G.backward()

            optimizer_G.step()

            optimizer_D.zero_grad()

            pred_real = netD(gt_cloth)

            loss_D_real = criterion_GAN(pred_real, target_real)

            # Fake loss
            gen_targ = fake_buffer.push_and_pop(gen_targ)
            pred_fake = netD(gen_targ.detach())
            loss_D_fake = criterion_GAN(pred_fake, target_fake)

            # Total loss
            loss_D = (loss_D_real + loss_D_fake) * 0.5
            loss_D.backward()
            if (i + 1) % int(opt.critic) == 0:
                optimizer_D.step()

            avg_loss_g = (avg_loss_g + loss_G) / (i + 1)
            avg_loss_d = (avg_loss_d + loss_D) / (i + 1)

            if (i + 1) % 10 == 0:
                print("Epoch: (%3d) (%5d/%5d) Loss: (%0.0003f) (%0.0003f)" % (
                    epoch, i + 1, len(train_loader), avg_loss_g * 1000, avg_loss_d * 1000))

                # if (i + 1) % int(opt.display_count) == 0:
                pic = (torch.cat([src, style_img, gen_targ, cloth, skel, target, gt_cloth, head], dim=0).data + 1) / 2.0

                save_dir = "{}/{}{}".format(os.getcwd(), opt.results, opt.stage)
                #             os.mkdir(save_dir)
                save_image(pic, '%s/Epoch_(%d)_(%dof%d).jpg' % (save_dir, epoch, i + 1, len(train_loader)), nrow=nRow)
            if (epoch + 1) % int(opt.save_model) == 0:
                save_dir = "{}/{}{}".format(os.getcwd(), opt.results, opt.stage)
                torch.save(netG.state_dict(), '{}/Gan_{}.pth'.format(save_dir, epoch))
                # Update learning rates
            lr_scheduler_G.step()
            lr_scheduler_D.step()
    print('Stitch process is end')


def trainStitchModel(opt, netG, netD):
    print("training the model for different model types: %s" % (opt.stage))
    dataset = stitchDataSetExtract(128)

    train_loader = DataLoader(dataset,
                              batch_size=int(opt.batch),
                              shuffle=False,
                              num_workers=int(opt.thread),
                              drop_last=True, pin_memory=True)
    epoch = 0
    n_epochs = int(opt.epochs)
    decay_epoch = int(opt.decay_epoch)
    batchSize = int(opt.batch)
    size = 128
    input_nc = int(opt.input_channel)
    output_nc = 3
    lr = float(opt.learn_rate)
    nRow = 3

    criterion_GAN = torch.nn.MSELoss()
    criterion_identity = torch.nn.L1Loss()

    optimizer_G = torch.optim.Adam(netG.parameters(), lr=lr, betas=(0.5, 0.999))
    optimizer_D = torch.optim.Adam(netD.parameters(), lr=lr, betas=(0.5, 0.999))

    lr_scheduler_G = torch.optim.lr_scheduler.LambdaLR(optimizer_G,
                                                       lr_lambda=LambdaLR(n_epochs, epoch, decay_epoch).step)
    lr_scheduler_D = torch.optim.lr_scheduler.LambdaLR(optimizer_D,
                                                       lr_lambda=LambdaLR(n_epochs, epoch, decay_epoch).step)

    # Inputs & targets memory allocation
    Tensor = torch.FloatTensor
    input_A = Tensor(batchSize, input_nc, size, size)

    target_real = Variable(Tensor(batchSize).fill_(1.0), requires_grad=False)
    target_fake = Variable(Tensor(batchSize).fill_(0.0), requires_grad=False)

    fake_buffer = ReplayBuffer()
    print(n_epochs)
    # print(len(train_loader))

    for epoch in range(0, n_epochs):
        gc.collect()
        print(epoch)
        Source = iter(train_loader)
        avg_loss_g = 0
        avg_loss_d = 0
        for i in range(0, len(train_loader)):
            netG.train()
            target_real = Variable(torch.ones(1, 1), requires_grad=False)
            target_fake = Variable(torch.zeros(1, 1), requires_grad=False)
            optimizer_G.zero_grad()

            src, mask, style_img, target, gt_cloth, skel, cloth = Source.next()
            src, mask, style_img, target, gt_cloth, skel, cloth = Variable(src), Variable(mask), Variable(style_img), \
                                                                  Variable(target), Variable(gt_cloth), Variable(
                skel), Variable(cloth)

            # print(src.shape)

            gen_targ, _, _, _, _, _, _ = netG(src, style_img, skel)
            pred_fake = netD(gen_targ)

            loss_GAN = 10 * criterion_GAN(pred_fake, target_real) + 10 * criterion_identity(gen_targ, target)

            loss_G = loss_GAN
            loss_G.backward()

            optimizer_G.step()
            #############################################

            optimizer_D.zero_grad()
            pred_real = netD(target)
            loss_D_real = criterion_GAN(pred_real, target_real)

            # Fake loss
            gen_targ = fake_buffer.push_and_pop(gen_targ)
            pred_fake = netD(gen_targ.detach())
            loss_D_fake = criterion_GAN(pred_fake, target_fake)

            # Total loss
            loss_D = (loss_D_real + loss_D_fake) * 0.5
            loss_D.backward()
            if (i + 1) % int(opt.critic) == 0:
                optimizer_D.step()

            avg_loss_g = (avg_loss_g + loss_G) / (i + 1)
            avg_loss_d = (avg_loss_d + loss_D) / (i + 1)

            if (i + 1) % 10 == 0:
                print("Epoch: (%3d) (%5d/%5d) Loss: (%0.0003f) (%0.0003f)" % (
                    epoch, i + 1, len(train_loader), avg_loss_g * 1000, avg_loss_d * 1000))

                # if (i + 1) % int(opt.display_count) == 0:
                pic = (torch.cat([src, gen_targ, cloth, skel, target, gt_cloth], dim=0).data + 1) / 2.0

                save_dir = "{}/{}{}".format(os.getcwd(), opt.results, opt.stage)
                #             os.mkdir(save_dir)
                save_image(pic, '%s/Epoch_(%d)_(%dof%d).jpg' % (save_dir, epoch, i + 1, len(train_loader)), nrow=nRow)

            if (epoch + 1) % int(opt.save_model) == 0:
                save_dir = "{}/{}{}".format(os.getcwd(), opt.results, opt.stage)
                torch.save(netG.state_dict(), '{}/Gan_{}.h5'.format(save_dir, epoch))
                # Update learning rates
            lr_scheduler_G.step()
            lr_scheduler_D.step()


def trainRefineModel(opt, netG, netD):
    print("training the model for different model types: %s" % (opt.stage))
    dataset = refineDataSetExtract(128)
    train_loader = DataLoader(dataset,
                              batch_size=int(opt.batch),
                              shuffle=False,
                              num_workers=int(opt.thread),
                              drop_last=True, pin_memory=True)

    epoch = 0
    n_epochs = int(opt.epochs)
    decay_epoch = int(opt.decay_epoch)
    batchSize = int(opt.batch)
    size = 128
    input_nc = int(opt.input_channel)
    output_nc = 3
    lr = float(opt.learn_rate)
    nRow = 4

    criterion_GAN = torch.nn.MSELoss()
    criterion_identity = torch.nn.L1Loss()

    optimizer_G = torch.optim.Adam(netG.parameters(), lr=lr, betas=(0.5, 0.999))
    optimizer_D = torch.optim.Adam(netD.parameters(), lr=lr, betas=(0.5, 0.999))

    lr_scheduler_G = torch.optim.lr_scheduler.LambdaLR(optimizer_G,
                                                       lr_lambda=LambdaLR(n_epochs, epoch, decay_epoch).step)
    lr_scheduler_D = torch.optim.lr_scheduler.LambdaLR(optimizer_D,
                                                       lr_lambda=LambdaLR(n_epochs, epoch, decay_epoch).step)

    # Inputs & targets memory allocation
    Tensor = torch.FloatTensor
    input_A = Tensor(batchSize, input_nc, size, size)

    target_real = Variable(Tensor(batchSize).fill_(1.0), requires_grad=False)
    target_fake = Variable(Tensor(batchSize).fill_(0.0), requires_grad=False)

    fake_buffer = ReplayBuffer()
    print(n_epochs)
    for epoch in range(0, n_epochs):
        gc.collect()
        print(epoch)
        Source = iter(train_loader)
        avg_loss_g = 0
        avg_loss_d = 0
        for i in range(0, len(train_loader)):
            netG.train()
            target_real = Variable(torch.ones(1, 1), requires_grad=False)
            target_fake = Variable(torch.zeros(1, 1), requires_grad=False)
            optimizer_G.zero_grad()

            # src, mask, style_img, target, gt_cloth, skel, cloth = Source.next()
            # src, mask, style_img, target, gt_cloth, skel, cloth = Variable(src), Variable(mask), Variable(style_img), Variable(target)\
            #   , Variable(gt_cloth), Variable(skel), Variable(cloth)

            src, mask, style_img, target, gt_cloth, wrap, diff, cloth, head = Source.next()
            src, mask, style_img, target, gt_cloth, wrap, diff, cloth, head = Variable(src), Variable(mask), Variable(
                style_img), Variable(target), Variable(gt_cloth), Variable(wrap), Variable(diff), Variable(
                cloth), Variable(head)

            # print(src.shape)
            # print(mask.shape)
            # print(style_img.shape)
            # print(target.shape)
            # print(gt_cloth.shape)
            # print(wrap.shape)
            # print(diff.shape)
            # print(cloth.shape)

            gen_targ, _, _, _, _, _, _ = netG(diff, wrap)
            pred_fake = netD(gen_targ)

            loss_GAN = 10 * criterion_GAN(pred_fake, target_real) + 10 * criterion_identity(gen_targ, target)

            loss_G = loss_GAN
            loss_G.backward()

            optimizer_G.step()
            #############################################
            optimizer_D.zero_grad()

            pred_real = netD(target)

            loss_D_real = criterion_GAN(pred_real, target_real)

            # Fake loss
            gen_targ = fake_buffer.push_and_pop(gen_targ)
            pred_fake = netD(gen_targ.detach())
            loss_D_fake = criterion_GAN(pred_fake, target_fake)

            # Total loss
            loss_D = (loss_D_real + loss_D_fake) * 0.5
            loss_D.backward()

            if (i + 1) % int(opt.critic) == 0:
                optimizer_D.step()

            avg_loss_g = (avg_loss_g + loss_G) / (i + 1)
            avg_loss_d = (avg_loss_d + loss_D) / (i + 1)

            if (i + 1) % 10 == 0:
                print("Epoch: (%3d) (%5d/%5d) Loss: (%0.0003f) (%0.0003f)" % (
                    epoch, i + 1, len(train_loader), avg_loss_g * 1000, avg_loss_d * 1000))

            # if (i + 1) % int(opt.display_count) == 0:
            pic = (torch.cat([wrap, diff, gen_targ, target, head], dim=0).data + 1) / 2.0
            pic1 = torch.stack([target, head], dim=0)
            #resize = transforms.Resize(size=(128, 128))

            pic1 = torch.cat([target, head], 0)

            pic2 = torch.cat([target, head], dim=0)
            print(target.shape)
            print(head.shape)
            #pic2 = resize(pic2)

            #pic2 = TF.to_tensor(pic2)
            save_dir = "{}/{}{}".format(os.getcwd(), opt.results, opt.stage)

            # commonFunctions.createDir(save_dir)
            # os.mkdir(save_dir)
            save_image(pic, '%s/Epoch_(%d)_(%dof%d).jpg' % (save_dir, epoch, i + 1, len(train_loader)), nrow=nRow)
            save_image(pic1, '%s/Epoch_Final_(%d)_(%dof%d).jpg' % (save_dir, epoch, i + 1, len(train_loader)))

            if (epoch + 1) % int(opt.save_model) == 0:
                save_dir = "{}/{}{}".format(os.getcwd(), opt.results, opt.stage)
                torch.save(netG.state_dict(), '{}/Gan_{}.pth'.format(save_dir, epoch))

            # Update learning rates
            lr_scheduler_G.step()
            lr_scheduler_D.step()

            # if i == 2 :
            #   break;

    print('traing refine complete')


# define main function to start executing the training model
def main():
    trn_options = get_options()
    print(trn_options)
    print("Model training started")

    if not os.path.exists(trn_options.results):
        os.makedirs(trn_options.results)
    if trn_options.stage == "Stitch":
        netG = GeneratorCoarse(9, 3)
    else:
        netG = GeneratorCoarse(int(trn_options.input_channel), 3)

    netD = Discriminator()

    # intialize the weight for the model
    netG.apply(weights_init_normal)
    netD.apply(weights_init_normal)

    if trn_options.stage == "Shape":
        print("Training started for %s" % (trn_options.stage))
        trainShapeModel(trn_options, netG, netD)

        print("Training completed for %s " % (trn_options.stage))
    if trn_options.stage == "Stitch":
        print("Training started for %s" % (trn_options.stage))
        trainStitchModel(trn_options, netG, netD)

        print("Training completed for %s " % (trn_options.stage))
    if trn_options.stage == "Refine":
        print("Training started for %s" % (trn_options.stage))
        trainRefineModel(trn_options, netG, netD)

        print("Training completed for %s " % (trn_options.stage))
    else:
        sys.exit("Please mention the Stage from [Shape, Stitch, Refine]")

    print('Finished training ')


if __name__ == "__main__":
    main()
