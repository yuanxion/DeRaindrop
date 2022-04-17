import argparse
import glob
import math
import matplotlib.pyplot as plt
import models
import os
import gc

from GPUtil import showUtilization as gpu_usage
import torch
from torch.nn import BCELoss, MSELoss
from torch.optim import Adam
from models.discriminator import Discriminator
from models.generator import Generator
from PIL import Image
from torch.utils.data import Dataset, DataLoader, RandomSampler, random_split
from torchvision import transforms
from torchvision.utils import make_grid


# parser
def get_parser_config():
    parser = argparse.ArgumentParser()

    parser.add_argument('--type', type=str, default='test')
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--num_workers', type=int, default=1)

    config = parser.parse_args()
    print('get_parser_config config:', config)
    return config


# data folder
### data
### ├── defoggy
### │   ├── data
### │   └── gt
### └── derain
###     ├── test
###     │   ├── data
###     │   └── gt
###     ├── train
###     │   ├── data
###     │   └── gt
###     └── val
###         ├── data
###         └── gt
#
class MyComplexDataset(Dataset):
    def __init__(
        self, is_test=False, is_val=False, is_train=True, transformer=None
    ):

        self.defoggy_raw_folder = 'data/defoggy/data/'
        self.defoggy_gt_folder = 'data/defoggy/gt/'
        self.transformer = transformer

        self.defoggy_raw_files = glob.glob(self.defoggy_raw_folder + '*jpg')
        self.defoggy_gt_files = glob.glob(self.defoggy_gt_folder + '*jpg')
        # print('self.defoggy_raw_files:', len(self.defoggy_raw_files))
        # print('self.defoggy_gt_files:', len(self.defoggy_gt_files))

        images_map = {}
        # map raw to gt
        for image_path in self.defoggy_raw_files:
            raw_name = image_path.split('/')[-1]
            splits = raw_name.split('_')
            gt_name = splits[0] + '_' + splits[1] + '.jpg'

            # NYU2_1085.jpg: [NYU2_1085_1_2.jpg, NYU2_1085_1_3.jpg]
            if gt_name not in images_map:
                images_map[gt_name] = []
            images_map[gt_name].append(raw_name)
            # print('pair {}:{}'.format(gt_name, raw_name))
            # print('pair {}:{}'.format(gt_name, images_map[gt_name]))

        self.image_pairs = []
        # pair raw & gt images
        for key in images_map:
            for value in images_map[key]:
                self.image_pairs.append(
                    [
                        self.defoggy_raw_folder + value,
                        self.defoggy_gt_folder + key,
                    ]
                )

    def __getitem__(self, index):
        defoggy_raw_file, defoggy_gt_file = self.image_pairs[index]
        defoggy_raw_img = Image.open(defoggy_raw_file)
        defoggy_gt_img = Image.open(defoggy_gt_file)

        if self.transformer:
            defoggy_raw_img = self.transformer(defoggy_raw_img)
            # defoggy_raw_img = defoggy_raw_img.cuda()
            defoggy_gt_img = self.transformer(defoggy_gt_img)
            # defoggy_gt_img = defoggy_gt_img.cuda()

        return defoggy_raw_img, defoggy_gt_img

    def __len__(self):
        return len(self.image_pairs)


def forward(raw_bs, gt_bs):
    # use GPU
    raw_bs = raw_bs.cuda()
    gt_bs = gt_bs.cuda()
    # gpu_usage()

    # train D
    mask_list, frame1, frame2, gen_bs = G(raw_bs)
    # print('gen_bs.shape:', gen_bs.shape)
    print('gen_bs std_mean: {}'.format(torch.std_mean(gen_bs)))
    # print('gen_bs: {}'.format(gen_bs))

    mask_raw, pred_raw = D(gen_bs.detach())
    mask_gt, pred_gt = D(gt_bs)
    mean_pred_raw = torch.mean(pred_raw)
    mean_pred_gt = torch.mean(pred_gt)
    print(
        'mean pred_raw: {:.2f}, mean pred_gt: {:.2f}'.format(
            mean_pred_raw, mean_pred_gt
        )
    )

    loss_d_raw = bce_loss(pred_raw, label_raw)
    loss_d_gt = bce_loss(pred_gt, label_gt)
    loss_d = loss_d_raw + loss_d_gt
    # print(
    #    'loss_d_raw: {:.2f}, loss_d_gt: {:.2f}'.format(
    #        loss_d_raw.item(), loss_d_gt.item()
    #    )
    # )

    optim_d.zero_grad()
    loss_d.backward(retain_graph=True)
    optim_d.step()

    # train G
    mask_raw, pred_raw = D(gen_bs)
    loss_g_raw = bce_loss(pred_raw, label_raw)
    loss_g = loss_g_raw
    # print('loss_g_raw: {:.2f}'.format(loss_g_raw.item()))

    optim_g.zero_grad()
    loss_g.backward()
    optim_g.step()

    # show gen image
    grid_gen = make_grid(gen_bs, nrow=grid_columns, padding=5)
    ax_gen.imshow(grid_gen.cpu().detach().permute(1, 2, 0))
    # plt.show()
    # plt.pause(0.02)

    # del grid_columns,grid_raw,grid_gt, mask_list,frame1,frame2,gen_bs, mask_raw,mask_gt, loss_d_raw,loss_d_gt
    gc.collect()
    torch.cuda.empty_cache()
    # gpu_usage()

    return mean_pred_raw, mean_pred_gt


print('[+] Train')
# parser config
config = get_parser_config()

# dataset
my_transformer = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ]
)

dataset = MyComplexDataset(is_train=True, transformer=my_transformer)
total_size = len(dataset)
print('total_size:', total_size)

# train test split
epoches = 100
split_rate = 0.8
grid_columns = int(math.sqrt(config.batch_size))

train_size = int(split_rate * total_size)
val_size = total_size - train_size
# print('train_size:', train_size)
# print('val_size:', val_size)

train_ds, val_ds = random_split(dataset, [train_size, val_size])
print('train_ds size:', len(train_ds))
print('val_ds size:', len(val_ds))

# dataloader
train_loader = DataLoader(
    train_ds, batch_size=config.batch_size, shuffle=True, drop_last=True
)

# model
D = models.Discriminator().cuda()
G = models.Generator().cuda()

# optimize
lr_d = lr_g = 0.0001
beta_d = beta_g = (0.5, 0.99)
optim_d = Adam(D.parameters(), lr=lr_d, betas=beta_d)
optim_g = Adam(G.parameters(), lr=lr_g, betas=beta_g)

# label
label_raw = torch.zeros(config.batch_size, 1).cuda()
label_gt = torch.ones_like(label_raw).cuda()
# print('label_raw.shape:', label_raw.shape)
# print('label_gt.shape:', label_gt.shape)

# loss & history
bce_loss = torch.nn.BCELoss()
mse_loss = torch.nn.MSELoss()

pred_history_raw = []
pred_history_gt = []
loss_history_d = []
loss_history_g = []

# plot
figure, ([ax_raw, ax_gt], [ax_gen, ax_pred]) = plt.subplots(
    2, 2, figsize=(10, 8)
)
plt.ion()

# train
for epoch in range(epoches):
    for i, (raw_bs, gt_bs) in enumerate(train_loader):
        plt.cla()
        torch.cuda.memory_summary(device="cuda", abbreviated=False)

        # print('batch_{}: {}, {}'.format(i, raw_bs.shape, gt_bs.shape))

        # display batch images

        grid_raw = make_grid(raw_bs, nrow=grid_columns, padding=5)
        grid_gt = make_grid(gt_bs, nrow=grid_columns, padding=5)
        # print('grid_raw.shape:', grid_raw.shape)
        # print('grid_gt.shape:', grid_gt.shape)

        ax_raw.imshow(grid_raw.cpu().detach().permute(1, 2, 0))
        ax_gt.imshow(grid_gt.cpu().detach().permute(1, 2, 0))

        mean_pred_raw, mean_pred_gt = forward(raw_bs, gt_bs)

        pred_history_raw.append(mean_pred_raw.item())
        pred_history_gt.append(mean_pred_gt.item())
        ax_pred.plot(pred_history_raw, label='raw')
        ax_pred.plot(pred_history_gt, label='gt')
        ax_pred.text(
            len(pred_history_raw),
            mean_pred_raw,
            '{:.2f}'.format(mean_pred_raw),
        )
        ax_pred.text(
            len(pred_history_gt), mean_pred_gt, '{:.2f}'.format(mean_pred_gt)
        )

        plt.legend()
        plt.show()
        plt.pause(0.02)

    # plt.pause(0.02)
    # plt.ioff()
