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
        self.bce_loss = nn.BCELoss()
        self.mse_loss = nn.MSELoss()

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

    def get_loss(self, prob, is_gt=True):

        if is_gt:
            return self.bce_loss(prob, self.label_gt)
        else:
            return self.bce_loss(prob, self.label_gen)

    def forward(self, img, gt):

        # Generator
        mask_list, frame1, frame2, img_gen = self.G(img)
        mask, prob = self.D(img_gen)
        loss_g = self.get_loss(prob, is_gt=False)

        # Discriminator
        mask, prob_gt = self.D(gt)
        mask, prob_gen = self.D(img_gen.detach())
        loss_gt = self.get_loss(prob_gt, is_gt=True)
        loss_gen = self.get_loss(prob_gen, is_gt=False)

        loss_d = loss_gt + loss_gen

        return loss_g, loss_d

    def train(self):

        loss_history_d = []
        loss_history_g = []

        print('Start to train')
        figure, ax = plt.subplots(1)
        plt.ion()

        for epoch in range(self.epoches):
            for i, (img, gt) in enumerate(self.trainloader):

                plt.cla()
                torch.cuda.memory_summary(device="cuda", abbreviated=False)
                loss_g, loss_d = self.forward(img, gt)

                # Generator
                self.optimG.zero_grad()
                loss_g.backward()
                self.optimG.step()
                loss_history_g.append(loss_g.item())

                # Discriminator
                self.optimD.zero_grad()
                loss_d.backward()
                self.optimD.step()
                loss_history_d.append(loss_d.item())

                ax.plot(loss_history_g, label='loss_g')
                ax.plot(loss_history_d, label='loss_d')
                plt.show()
                plt.pause(0.02)

                print(
                    '[{}/{}] loss_g: {}, loss_d: {}'.format(
                        i, epoch, loss_g, loss_d
                    )
                )


if __name__ == '__main__':
    # with torch.autograd.set_detect_anomaly(True):
    trainer = Trainer()
    trainer.train()
