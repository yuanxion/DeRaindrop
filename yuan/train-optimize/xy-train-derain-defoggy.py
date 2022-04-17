import argparse
import glob
import math
import matplotlib.pyplot as plt
import models
import os
import gc

from GPUtil import showUtilization as gpu_usage
import torch
from torch import nn
from torch.nn import BCELoss, MSELoss
from torch.optim import Adam
from models.discriminator import Discriminator
from models.generator import Generator
from PIL import Image
from torch.utils.data import Dataset, DataLoader, RandomSampler, random_split
from torchvision import transforms
from torchvision.utils import make_grid
from torchvision.models.vgg import vgg16

# import torch.autograd as autograd
from torch.autograd import Variable

# parser
def get_parser_config():
    parser = argparse.ArgumentParser()

    parser.add_argument('--type', type=str, default='test')
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--dataset', type=str, default='defoggy')
    parser.add_argument('--num_workers', type=int, default=1)

    config = parser.parse_args()
    print('get_parser_config config:', config)
    return config


def get_masks_diff(raw_bs, gt_bs):
    mask = torch.abs(raw_bs - gt_bs)
    std, mean = torch.std_mean(mask)
    print('get_masks_diff mean:{}, std:{}'.format(mean, std))

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


def get_atten_loss(masks_gen, masks_diff):
    loss_atten = None
    for i in range(1, 5):
        if i == 1:
            loss_atten = pow(0.8, float(4 - i)) * mse_loss(
                masks_gen[i - 1], masks_diff
            )
        else:
            loss_atten += pow(0.8, float(4 - i)) * mse_loss(
                masks_gen[i - 1], masks_diff
            )
    return loss_atten


def get_ms_loss(img_gens, gts):

    T_ = []
    ld = [0.6, 0.8, 1.0]
    width = img_gens[2].size(2)
    # print('base width', width)
    width_4 = int(width / 4)
    width_2 = int(width / 2)
    transformer1 = transforms.Compose([transforms.Resize((width_4, width_4))])
    transformer2 = transforms.Compose([transforms.Resize((width_2, width_2))])

    for i in range(config.batch_size):
        temp = []
        pyramid = transformer1(gts[i])
        temp.append(torch.unsqueeze(pyramid, axis=0))
        pyramid = transformer2(gts[i])
        temp.append(torch.unsqueeze(pyramid, axis=0))
        temp.append(torch.unsqueeze(gts[i], axis=0))
        T_.append(temp)
    temp_T = []
    for i in range(len(ld)):
        for j in range(config.batch_size):
            if j == 0:
                x = T_[j][i]
            else:
                x = torch.cat((x, T_[j][i]), axis=0)
        temp_T.append(x)
    T_ = temp_T
    loss_ML = None
    for i in range(len(ld)):
        # print('{}:{} img_gens:{} gts:{}'.format(i,len(ld), img_gens[i].shape, T_[i].shape))
        if i == 0:
            loss_ML = ld[i] * mse_loss(img_gens[i], T_[i])
        else:
            loss_ML += ld[i] * mse_loss(img_gens[i], T_[i])

    return loss_ML / float(config.batch_size)


def get_map_loss(mask_d_gen, mask_d_gt, mask_g_gen):
    z = Variable(torch.zeros(mask_d_gt.shape)).cuda()
    loss_gen = mse_loss(mask_d_gen, mask_g_gen)
    loss_gt = mse_loss(mask_d_gt, z)
    return 0.05 * (loss_gen + loss_gt)


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
        self,
        dataset='defoggy',
        is_test=False,
        is_val=False,
        is_train=True,
        transformer=None,
    ):

        self.transformer = transformer
        self.image_pairs = []

        if dataset == 'derain':
            # 9_rain.png, 9_clean.png

            if is_train:
                self.derain_raw_folder = 'data/derain/train/data/'
                self.derain_gt_folder = 'data/derain/train/gt/'
            elif is_val:
                self.derain_raw_folder = 'data/derain/val/data/'
                self.derain_gt_folder = 'data/derain/val/gt/'
            elif is_test:
                self.derain_raw_folder = 'data/derain/test/data/'
                self.derain_gt_folder = 'data/derain/test/gt/'

            self.derain_raw_files = glob.glob(self.derain_raw_folder + '*png')
            self.derain_gt_files = glob.glob(self.derain_gt_folder + '*png')

            self.raw_files = self.derain_raw_files

        elif dataset == 'defoggy':
            # defoggy: NYU2_1085.jpg: [NYU2_1085_1_2.jpg, NYU2_1085_1_3.jpg]

            self.defoggy_raw_folder = 'data/defoggy/data/'
            self.defoggy_gt_folder = 'data/defoggy/gt/'
            self.defoggy_raw_files = glob.glob(
                self.defoggy_raw_folder + '*jpg'
            )
            self.defoggy_gt_files = glob.glob(self.defoggy_gt_folder + '*jpg')
            # print('self.defoggy_raw_files:', len(self.defoggy_raw_files))
            # print('self.defoggy_gt_files:', len(self.defoggy_gt_files))

            self.raw_files = self.defoggy_raw_files
        else:
            self.raw_files = None
            print('unknown dataset')

        # map raw to gt
        images_map = {}
        for image_path in self.raw_files:
            raw_name = image_path.split('/')[-1]
            splits = raw_name.split('_')
            if dataset == 'derain':
                gt_name = splits[0] + '_clean.png'
            elif dataset == 'defoggy':
                gt_name = splits[0] + '_' + splits[1] + '.jpg'

            if gt_name not in images_map:
                images_map[gt_name] = []
            images_map[gt_name].append(raw_name)
            # print('pair {}:{}'.format(gt_name, raw_name))
            # print('pair {}:{}'.format(gt_name, images_map[gt_name]))

        # pair raw & gt images
        if dataset == 'derain':
            for key, value in images_map.items():
                # print('pair key:{}, value:{}'.format(key, value[0]))
                self.image_pairs.append(
                    [
                        self.derain_raw_folder + value[0],
                        self.derain_gt_folder + key,
                    ]
                )
        elif dataset == 'defoggy':
            for key in images_map:
                for value in images_map[key]:
                    self.image_pairs.append(
                        [
                            self.defoggy_raw_folder + value,
                            self.defoggy_gt_folder + key,
                        ]
                    )

    def __getitem__(self, index):
        raw_file, gt_file = self.image_pairs[index]
        raw_img = Image.open(raw_file)
        gt_img = Image.open(gt_file)

        if self.transformer:
            raw_img = self.transformer(raw_img)
            gt_img = self.transformer(gt_img)

        return raw_img, gt_img

    def __len__(self):
        return len(self.image_pairs)


class PerceptualLoss(nn.Module):
    def __init__(self):
        super(PerceptualLoss, self).__init__()
        self.model = vgg16(pretrained=True).cuda()
        # print('PerceptualLoss model:', self.model)

        trainable = False
        for para in self.model.parameters():
            para.requires_grad = trainable

        self.loss = MSELoss().cuda()
        self.vgg_layers = self.model.features
        self.layer_name_mapping = {
            '1': "relu1_1",
            '3': "relu1_2",
            '6': "relu2_1",
            '8': "relu2_2",
        }

    def get_layer_output(self, x):
        output = []
        for name, module in self.vgg_layers._modules.items():
            # print('get_layer_output name:{}, module:{}'.format(name, module))
            x = module(x)
            if name in self.layer_name_mapping:
                output.append(x)
        return output

    def __call__(self, raw_bs, gt_bs):
        raw_bs = self.get_layer_output(raw_bs)
        gt_bs = self.get_layer_output(gt_bs)
        loss_PL = None
        for i in range(len(gt_bs)):
            if i == 0:
                loss_PL = self.loss(raw_bs[i], gt_bs[i]) / float(len(gt_bs))
            else:
                loss_PL += self.loss(raw_bs[i], gt_bs[i]) / float(len(gt_bs))
        return loss_PL


def forward(raw_bs, gt_bs):
    # use GPU
    gc.collect()
    torch.cuda.empty_cache()
    raw_bs = raw_bs.cuda()
    gt_bs = gt_bs.cuda()
    # gpu_usage()

    # train D
    mask_list, frame1, frame2, gen_bs = G(raw_bs)
    # print('gen_bs.shape:', gen_bs.shape)
    print('gen_bs std_mean: {}'.format(torch.std_mean(gen_bs)))
    # print('gen_bs: {}'.format(gen_bs))

    mask_gen, pred_raw = D(gen_bs.detach())
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
    loss_d_map = get_map_loss(mask_gen, mask_gt, mask_list[-1].detach())
    loss_d = loss_d_raw + loss_d_gt + loss_d_map
    # print(
    #    'loss_d_raw: {:.2f}, loss_d_gt: {:.2f}'.format(
    #        loss_d_raw.item(), loss_d_gt.item()
    #    )
    # )

    optim_d.zero_grad()
    loss_d.backward(retain_graph=True)
    optim_d.step()

    # train G
    mask_gen, pred_raw = D(gen_bs)
    loss_g_raw = bce_loss(pred_raw, label_raw)
    masks_diff = get_masks_diff(raw_bs, gt_bs)
    loss_g_att = get_atten_loss(mask_list, masks_diff)
    loss_g_ms = get_ms_loss([frame1, frame2, gen_bs], gt_bs)
    loss_g_per = per_loss(gen_bs, gt_bs)
    # loss_g = loss_g_raw
    loss_g = 0.01 * loss_g_raw + loss_g_ms + loss_g_per
    # print('loss_g_raw: {:.2f}'.format(loss_g_raw.item()))

    optim_g.zero_grad()
    loss_g.backward()
    optim_g.step()

    # show gen image
    grid_gen = make_grid(gen_bs, nrow=grid_columns, padding=5)
    ax_gen.imshow(grid_gen.cpu().detach().permute(1, 2, 0))
    # plt.show()
    # plt.pause(0.02)

    # del grid_columns,grid_raw,grid_gt, mask_list,frame1,frame2,gen_bs, mask_gen,mask_gt, loss_d_raw,loss_d_gt
    # gpu_usage()

    return mean_pred_raw, mean_pred_gt, loss_d, loss_g


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

epoches = 100
split_rate_defoggy = 0.8
grid_columns = int(math.sqrt(config.batch_size))

if config.dataset == 'derain':
    train_ds = MyComplexDataset(
        dataset='derain', is_train=True, transformer=my_transformer
    )
    val_ds = MyComplexDataset(
        dataset='derain',
        is_val=True,
        is_train=False,
        transformer=my_transformer,
    )
elif config.dataset == 'defoggy':
    dataset = MyComplexDataset(
        dataset='defoggy', is_train=True, transformer=my_transformer
    )

    # train test split
    total_size = len(dataset)
    train_size = int(split_rate_defoggy * total_size)
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
per_loss = PerceptualLoss()

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

        mean_pred_raw, mean_pred_gt, loss_d, loss_g = forward(raw_bs, gt_bs)

        # plot prediction history
        pred_history_raw.append(mean_pred_raw.item())
        pred_history_gt.append(mean_pred_gt.item())
        ax_pred.plot(pred_history_raw, label='pred_raw', linestyle='-.')
        ax_pred.plot(pred_history_gt, label='pred_gt', linestyle='-.')
        ax_pred.text(
            len(pred_history_raw),
            mean_pred_raw,
            '{:.2f}'.format(mean_pred_raw),
        )
        ax_pred.text(
            len(pred_history_gt), mean_pred_gt, '{:.2f}'.format(mean_pred_gt)
        )

        # plot loss history
        loss_history_d.append(torch.clamp(loss_d, 0, 2).item())
        loss_history_g.append(torch.clamp(loss_g, 0, 2).item())
        ax_pred.plot(loss_history_d, label='loss_d')
        ax_pred.plot(loss_history_g, label='loss_g')
        ax_pred.text(len(loss_history_d), loss_d, '{:.2f}'.format(loss_d))
        ax_pred.text(len(loss_history_g), loss_g, '{:.2f}'.format(loss_g))

        plt.legend()
        plt.show()
        plt.pause(0.02)

    # plt.pause(0.02)
    # plt.ioff()
