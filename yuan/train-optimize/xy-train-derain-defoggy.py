import argparse
import glob
import math
import os
import torch
import matplotlib.pyplot as plt
from PIL import Image
from torch.utils.data import Dataset, DataLoader, RandomSampler, random_split
from torchvision import transforms
from torchvision.utils import make_grid

# parser
def get_parser_config():
    parser = argparse.ArgumentParser()

    parser.add_argument('--type', type=str, default='test')
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--num_workers', type=int, default=4)

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
        print('self.defoggy_raw_files:', len(self.defoggy_raw_files))
        print('self.defoggy_gt_files:', len(self.defoggy_gt_files))

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
            defoggy_gt_img = self.transformer(defoggy_gt_img)

        return defoggy_raw_img, defoggy_gt_img

    def __len__(self):
        return len(self.image_pairs)


if __name__ == '__main__':
    print('[+] Train')

    # parser config
    config = get_parser_config()

    # dataset
    my_transformer = transforms.Compose([
        #transforms.Resize((224,224)),
        transforms.ToTensor(),
    ])

    dataset = MyComplexDataset(is_train=True, transformer=my_transformer)
    total_size = len(dataset)
    print('total_size:', total_size)

    # train test split
    split_rate = 0.8
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

    # plot
    figure, (ax_raw,ax_gt) = plt.subplots(1,2, figsize=(10,8))
    plt.ion()

    # train
    for i, (raw_bs, gt_bs) in enumerate(train_loader):
        #print('batch_{}: {}, {}'.format(i, raw_bs.shape, gt_bs.shape))

        # display batch images
        grid_columns = int(math.sqrt(config.batch_size))
        #print('grid_columns:', grid_columns)

        grid_raw = make_grid(raw_bs, nrow=grid_columns, padding=0)
        grid_gt = make_grid(gt_bs, nrow=grid_columns, padding=0)
        #print('grid_raw.shape:', grid_raw.shape)
        #print('grid_gt.shape:', grid_gt.shape)

        ax_raw.imshow(grid_raw.permute(1,2,0))
        ax_gt.imshow(grid_gt.permute(1,2,0))

        plt.show()
        plt.pause(0.2)

    plt.pause()
    plt.ioff()
