from os import path
import os
from PIL import Image
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
import torch


def read_domainnet_data(dataset_path, domain_name, split="train"):
    data_paths = []
    data_labels = []
    split_file = path.join(dataset_path, "{}_{}.txt".format('labeled_source_images', domain_name))
    with open(split_file, "r") as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip()
            data_path, label = line.split(' ')
            data_path = path.join(dataset_path, data_path)
            label = int(label)
            data_paths.append(data_path)
            data_labels.append(label)
    return data_paths, data_labels


class DomainNet(Dataset):
    def __init__(self, data_paths, data_labels, transforms, domain_name):
        super(DomainNet, self).__init__()
        self.data_paths = data_paths
        self.data_labels = data_labels
        self.transforms = transforms
        self.domain_name = domain_name

    def __getitem__(self, index):
        img = Image.open(self.data_paths[index])
        if not img.mode == "RGB":
            img = img.convert("RGB")
        label = self.data_labels[index]
        img = self.transforms(img)

        return img, label

    def __len__(self):
        return len(self.data_paths)

class DomainNet_twoTrans(Dataset):
    def __init__(self, data_paths, data_labels, transforms, local_transforms, domain_name):
        super(DomainNet_twoTrans, self).__init__()
        self.data_paths = data_paths
        self.data_labels = data_labels
        self.transforms = transforms
        self.local_transforms = local_transforms
        self.domain_name = domain_name

    def __getitem__(self, index):
        img = Image.open(self.data_paths[index])
        if not img.mode == "RGB":
            img = img.convert("RGB")
        label = self.data_labels[index]
        if self.transforms is not None:
            global_img = self.transforms(img)
            local_img = self.local_transforms(img)

        return [global_img, local_img], label, index

    def __len__(self):
        return len(self.data_paths)


def global_local(resize_size=256, crop_size=224, alexnet=False, global_=True):
    if not alexnet:
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                   std=[0.229, 0.224, 0.225])
    else:
        normalize = Normalize(meanfile='./ilsvrc_2012_mean.npy')

    if global_ == True:
        # global view
        return transforms.Compose([
                transforms.Resize((resize_size, resize_size)),
                transforms.RandomCrop(crop_size),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                # normalize
            ])
            
    else:
        # local view
        return transforms.Compose([
                transforms.RandomResizedCrop(size=[crop_size, crop_size], scale=[0.25, 1.0]),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                # normalize
            ])


def get_target_fourdomainnet_dloader(base_path, domain_name, batch_size, num_workers):
    dataset_path = path.join(base_path, 'FourDomainNet')
    train_data_paths, train_data_labels = read_domainnet_data(dataset_path, domain_name, split="train")
    test_data_paths, test_data_labels = read_domainnet_data(dataset_path, domain_name, split="test")
    transforms_train = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.75, 1)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor()
    ])
    transforms_test = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor()
    ])

    train_dataset = DomainNet_twoTrans(train_data_paths, train_data_labels, transforms=global_local(global_=True), local_transforms=global_local(global_=False), domain_name=domain_name)
    val_dataset = DomainNet(train_data_paths, train_data_labels, transforms=transforms_train, domain_name=domain_name)
    test_dataset = DomainNet(test_data_paths, test_data_labels, transforms_test, domain_name)
    print('length of training data:'+str(len(train_dataset)), ', test data:'+str( len(test_dataset)))
    train_dloader = DataLoader(train_dataset, batch_size=batch_size, num_workers=num_workers, pin_memory=False, shuffle=True)
    val_dloader = DataLoader(val_dataset, batch_size=batch_size, num_workers=num_workers, pin_memory=False, shuffle=False)
    test_dloader = DataLoader(test_dataset, batch_size=batch_size*3, num_workers=num_workers, pin_memory=False, shuffle=True)
    return train_dloader,  val_dloader,  test_dloader  #


def get_source_fourdomainnet_dloader(base_path, domain_name, batch_size, num_workers):
    dataset_path = path.join(base_path, 'FourDomainNet')
    train_data_paths, train_data_labels = read_domainnet_data(dataset_path, domain_name, split="train")
    test_data_paths, test_data_labels = read_domainnet_data(dataset_path, domain_name, split="test")
    transforms_train = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.75, 1)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor()
    ])
    transforms_test = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor()
    ])

    train_dataset = DomainNet(train_data_paths, train_data_labels, transforms_train, domain_name)
    test_dataset = DomainNet(test_data_paths, test_data_labels, transforms_test, domain_name)
    print('length of training data:'+str(len(train_dataset)), ', test data:'+str( len(test_dataset)))
    train_dloader = DataLoader(train_dataset, batch_size=batch_size, num_workers=num_workers, pin_memory=True,
                               shuffle=True)
    test_dloader = DataLoader(test_dataset, batch_size=batch_size, num_workers=num_workers, pin_memory=True,
                              shuffle=True)
    return train_dloader, test_dloader

if __name__ == '__main__':
    domain_name_list = ['clipart', 'painting', 'real', 'sketch']
    base_path = '/data/rrshao/datasets'
    batch_size = 64
    num_workers = 8
    for domain_name in domain_name_list:
        train_dloader = get_source_fourdomainnet_dloader(base_path, domain_name, batch_size, num_workers)
        for indx, data in enumerate(train_dloader):
            labels, images = data[0], data[1]
            print(len(labels), len(images))
            print(labels, images)