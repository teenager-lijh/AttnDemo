import os

import torchvision.io
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from data.transform import transform as trans
from torchvision import transforms


class ImageNetDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.data_dir = os.path.join(root_dir, 'train')
        self.transform = transform
        self.images_list = []
        self.mapper = {}

        with open(os.path.join(root_dir, 'mapper.txt'), mode='r') as file:
            mapping = file.readlines()
            for line in mapping:
                file_dir, label, _ = line.strip(' \n').split(' ')
                self.mapper[file_dir] = label

        self.data_dirs = os.listdir(self.data_dir)
        for all_dir in self.data_dirs:
            files_dir = os.path.join(self.data_dir, all_dir)
            files = os.listdir(files_dir)
            files = [os.path.join(files_dir, file) for file in files]
            self.images_list.extend(files)


        self.data_nums = len(self.images_list)  # 数据总量

    def __len__(self):
        return len(self.images_list)

    def __getitem__(self, idx):
        img_path = os.path.join(self.images_list[idx])
        image = Image.open(img_path).convert('RGB')
        img_dir = img_path.split('/')[-2]

        if self.transform:
            image = self.transform(image)

        label = int(self.mapper[img_dir])

        return image, torch.tensor(label, dtype=torch.long)




if __name__ == '__main__':
    # 数据集所在的目录
    data_dir = '/home/blueberry/cache/data/image_net_30'

    # 创建数据集
    custom_dataset = ImageNetDataset(data_dir, transform=trans)

    # 创建数据加载器
    batch_size = 4
    data_loader = DataLoader(custom_dataset, batch_size=batch_size, shuffle=True)

    # 现在你可以迭代数据加载器来访问图像数据
    for images, labels in data_loader:
        # 这里可以对图像数据进行你需要的操作，例如输入到模型中进行训练
        print(images.shape)  # 这里假设你在构建图像数据的处理管道时，将图像转换为了张量
        print(labels)
        break
