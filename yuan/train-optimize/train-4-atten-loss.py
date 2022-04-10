import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.autograd as autograd
from torch.autograd import Variable
from torch.utils.data import DataLoader, Dataset
import torchvision
from torchvision import transforms

import glob
import numpy as np
from models import *
from PIL import Image
import matplotlib.pyplot as plt


class MyDataset(Dataset):
    def __init__(self, is_val=False, is_test=False, transformer=None):
        super().__init__()
        self.is_test = is_test
        self.transformer = transformer

        if is_val:
            self.path = 'test_b'
        elif is_test:
            self.path = 'test_a'
        else:
            self.path = 'train'

        self.path = 'data/' + self.path
        self.data = sorted(glob.glob(self.path + '/data/*'))
        self.gt = sorted(glob.glob(self.path + '/gt/*'))
        # print('self.data', self.data)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        img = Image.open(self.data[index])
        gt = Image.open(self.gt[index])

        if self.transformer:
            img = self.transformer(img)
            gt = self.transformer(gt)

        # img = img.permute(2, 0, 1).cuda()
        # gt = gt.permute(2, 0, 1).cuda()
        img = img.cuda()
        gt = gt.cuda()
        return img, gt


class Trainer(object):
    def __init__(self):

        # config
        self.epoches = 10
        self.batchsize = 4
        self.lr = 0.0005
        self.betas = (0.5, 0.99)
        self.figure, (self.ax_img, self.ax_loss) = plt.subplots(
            1, 2, figsize=(12, 6)
        )

        # model
        self.G = Generator().cuda()
        self.D = Discriminator().cuda()

        # label
        self.label_gen = torch.FloatTensor(self.batchsize, 1).fill_(0).cuda()
        self.label_gt = torch.FloatTensor(self.batchsize, 1).fill_(1).cuda()

        # optimizor
        self.optimG = optim.Adam(
            self.G.parameters(), lr=self.lr, betas=self.betas
        )
        self.optimD = optim.Adam(
            self.D.parameters(), lr=self.lr, betas=self.betas
        )

        # loss
        self.bce_loss = nn.BCELoss().cuda()
        self.mse_loss = nn.MSELoss().cuda()

        # transformer
        my_transformer = transforms.Compose(
            [
                # transforms.Resize((480, 720)),
                transforms.Resize((224, 224)),
                # transforms.RandomHorizontalFlip(p=0.5),
                # transforms.RandomVerticalFlip(p=0.5),
                transforms.ToTensor(),
            ]
        )
        # dataset
        self.trainset = MyDataset(transformer=my_transformer)
        self.valset = MyDataset(is_val=True, transformer=my_transformer)

        self.trainloader = DataLoader(
            self.trainset, batch_size=self.batchsize, shuffle=True
        )
        self.valloader = DataLoader(self.valset, batch_size=self.batchsize)
        print('trainset:', len(self.trainset))
        print('valset:', len(self.valset))

    def get_map_loss(self, mask_gen, mask_gt, mask):
        z = Variable(torch.zeros(mask_gt.shape)).cuda()
        loss_gen = self.mse_loss(mask_gen, mask)
        loss_gt = self.mse_loss(mask_gt, z)
        return 0.05 * (loss_gen + loss_gt)

    def get_bce_loss(self, prob, is_gt=True):

        if is_gt:
            return self.bce_loss(prob, self.label_gt)
        else:
            return self.bce_loss(prob, self.label_gen)

    def get_atten_loss(self, masks_gen, masks_gt):
        loss = None
        for i in range(1, 5):
            # print('get_atten_loss masks_gen shape', masks_gen[i - 1].shape)
            # print('get_atten_loss masks_gt shape', masks_gt.shape)

            # min = torch.min(masks_gen[i-1]).item()
            # max = torch.max(masks_gen[i-1]).item()
            # print('get_atten_loss masks_gen min/max:\t {:.3f}/{:.3f}'.format(min, max))
            # min = torch.min(masks_gt).item()
            # max = torch.max(masks_gt).item()
            # print('get_atten_loss masks_gt min/max:\t {:.3f}/{:.3f}'.format(min, max))

            # print('get_atten_loss masks_gen i', masks_gen[i - 1].type)
            # torch.clamp(masks_gen, 0,1)
            # torch.clip(masks_gt,  0,1)

            if i == 1:
                loss = pow(0.8, float(4 - i)) * self.mse_loss(
                    masks_gen[i - 1], masks_gt
                )
            else:
                loss += pow(0.8, float(4 - i)) * self.mse_loss(
                    masks_gen[i - 1], masks_gt
                )
        return loss

    def get_masks(self, img, gt):
        mask = torch.abs(img - gt)

        # threshold under 30
        mask[mask < (30.0 / 255.0)] = 0.0
        mask[mask > 0.0] = 1.0
        # print('get_masks mask:', mask.shape)

        # avg? max?
        # mask = np.average(mask, axis=2)
        mask = torch.max(mask, axis=1).values
        # print('get_masks np.max mask.shape', mask.shape)
        mask = torch.unsqueeze(mask, axis=1)
        # print('get_masks expand_dims mask.shape', mask.shape)

        return mask

    def forward(self, batch, img, gt):

        # Generator
        mask_list, frame1, frame2, img_gen = self.G(img)
        mask, prob = self.D(img_gen)
        loss_g_bce = self.get_bce_loss(prob, is_gt=False)
        # loss_g_mse = self.mse_loss(img_gen, gt)
        # loss_g = loss_g_bce + loss_g_mse

        # mask_list = torch.Tensor(mask_list)

        # print('--> mask_list len', len(mask_list))
        loss_g_atten = self.get_atten_loss(mask_list, self.get_masks(img, gt))
        loss_g = 0.01 * (-loss_g_bce) + loss_g_atten

        # Discriminator
        mask_gt, prob_gt = self.D(gt)
        mask_gen, prob_gen = self.D(img_gen.detach())
        loss_gt = self.get_bce_loss(prob_gt, is_gt=True)
        loss_gen = self.get_bce_loss(prob_gen, is_gt=False)
        loss_d_map = self.get_map_loss(
            mask_gen, mask_gt, mask_list[-1].detach()
        )

        loss_d = loss_gt + loss_gen + loss_d_map

        grid = torchvision.utils.make_grid(
            img_gen,
            nrow=2,
            padding=0,
        )
        self.ax_img.imshow(
            grid.cpu().detach().permute(1, 2, 0), label='img_gen'
        )

        return loss_g, loss_d, loss_g_atten, loss_d_map

    def train(self):

        loss_history_d = []
        loss_history_g = []

        print('Start to train')
        plt.ion()

        for epoch in range(self.epoches):
            for i, (img, gt) in enumerate(self.trainloader):

                plt.cla()
                torch.cuda.memory_summary(device="cuda", abbreviated=False)
                loss_g, loss_d, loss_g_atten, loss_d_map = self.forward(
                    i, img, gt
                )

                # Generator
                self.optimG.zero_grad()
                loss_g.backward()
                self.optimG.step()

                loss_g = loss_g.item()
                loss_history_g.append(loss_g)

                # Discriminator
                self.optimD.zero_grad()
                loss_d.backward()
                self.optimD.step()

                loss_d = loss_d.item()
                loss_history_d.append(loss_d)

                self.ax_loss.plot(loss_history_g, label='loss_g')
                self.ax_loss.plot(loss_history_d, label='loss_d')
                plt.text(i, loss_g, '{:.3f}'.format(loss_g))
                plt.text(i, loss_d, '{:.3f}'.format(loss_d))
                plt.legend()
                plt.show()
                plt.pause(0.02)

                print(
                    '[{}/{}] loss_g:loss_d, loss_g_atten:loss_d_map {:.3f}:{:.3f}, {:.3f}:{:.3f}'.format(
                        i, epoch, loss_g, loss_d, loss_g_atten, loss_d_map
                    )
                )


if __name__ == '__main__':
    # with torch.autograd.set_detect_anomaly(True):
    trainer = Trainer()
    trainer.train()
