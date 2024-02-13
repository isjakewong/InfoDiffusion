import os
import torch
import torchvision
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import ImageFolder

class Crop:
    def __init__(self, x1, x2, y1, y2):
        self.x1 = x1
        self.x2 = x2
        self.y1 = y1
        self.y2 = y2

    def __call__(self, img):
        return Ftrans.crop(img, self.x1, self.y1, self.x2 - self.x1,
                           self.y2 - self.y1)

    def __repr__(self):
        return self.__class__.__name__ + "(x1={}, x2={}, y1={}, y2={})".format(
            self.x1, self.x2, self.y1, self.y2)


def d2c_crop():
    # from D2C paper for CelebA dataset.
    cx = 89
    cy = 121
    x1 = cy - 64
    x2 = cy + 64
    y1 = cx - 64
    y2 = cx + 64
    return Crop(x1, x2, y1, y2)

class CustomTensorDataset(Dataset):
    def __init__(self, data, latents_values, latents_classes):
        self.data = data
        self.latents_values = latents_values
        self.latents_classes = latents_classes

    def __getitem__(self, index):
        return (torch.from_numpy(self.data[index]).float(), 
                torch.from_numpy(self.latents_values[index]).float(), 
                torch.from_numpy(self.latents_classes[index]).int())

    def __len__(self):
        return self.data.shape[0]

class CustomImageFolder(ImageFolder):
    def __init__(self, root, transform=None):
        super(CustomImageFolder, self).__init__(root, transform)

    def __getitem__(self, index):
        path = self.imgs[index][0]
        img = self.loader(path)
        if self.transform is not None:
            img = self.transform(img)

        return img

def get_dataset_config(args):
    if args.dataset == 'fmnist':
        args.input_channels = 1
        args.unets_channels = 32
        args.encoder_channels = 32
        args.input_size = 32
    elif args.dataset == 'dsprites':
        args.input_channels = 1
        args.unets_channels = 32
        args.encoder_channels = 32
        args.input_size = 32
    elif args.dataset == 'celeba':
        args.input_channels = 3
        args.unets_channels = 64
        args.encoder_channels = 64 #[1,2,4,8,8]
        args.input_size = 64
    elif args.dataset == 'cifar10':
        args.input_channels = 3
        args.unets_channels = 64
        args.encoder_channels = 64
        args.input_size = 32
    elif args.dataset == 'chairs':
        args.input_channels = 3
        args.unets_channels = 32
        args.encoder_channels = 32
        args.input_size = 64

    shape = (args.input_channels, args.input_size, args.input_size)

    return shape

def get_dataset(args):
    if args.dataset == 'fmnist':
        return get_fmnist(args)
    elif args.dataset == 'celeba':
        return get_celeba(args)
    elif args.dataset == 'cifar10':
        return get_cifar10(args)
    elif args.dataset == 'dsprites':
        return get_dsprites(args)
    elif args.dataset == 'chairs':
        return get_chairs(args)

def get_fmnist(args):
    transform = torchvision.transforms.Compose([
        torchvision.transforms.Resize((args.input_size, args.input_size)),
        torchvision.transforms.RandomHorizontalFlip(),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Lambda(lambda t: (t * 2) - 1),
    ])

    dataset = torchvision.datasets.FashionMNIST(root = args.data_dir, train=True, download=True, transform = transform)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size = args.batch_size, drop_last = True, num_workers = 4)
    
    return dataloader

def get_celeba(args,
               as_tensor: bool = True,
               do_augment: bool = True,
               do_normalize: bool = True,
               crop_d2c: bool = False):
    if crop_d2c:
        transform = [
            d2c_crop(),
            torchvision.transforms.Resize(args.input_size),
        ]
    else:
        transform = [
            torchvision.transforms.Resize(args.input_size),
            torchvision.transforms.CenterCrop(args.input_size),
        ]

    if do_augment:
        transform.append(torchvision.transforms.RandomHorizontalFlip())
    if as_tensor:
        transform.append(torchvision.transforms.ToTensor())
    if do_normalize:
        transform.append(
            torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)))
    transform = torchvision.transforms.Compose(transform)

    if args.mode in ['attr_classification', 'eval_fid']:
        train_set = torchvision.datasets.CelebA(root = args.data_dir, split = "train", download = True, transform = transform)
        valid_set = torchvision.datasets.CelebA(root = args.data_dir, split = "valid", download = True, transform = transform)
        test_set = torchvision.datasets.CelebA(root = args.data_dir, split = "test", download = True, transform = transform)
        train_loader = torch.utils.data.DataLoader(train_set, batch_size = args.batch_size, drop_last = True, shuffle = True, num_workers = 4)
        valid_loader = torch.utils.data.DataLoader(valid_set, batch_size = args.batch_size, drop_last = True, shuffle = True, num_workers = 4)
        test_loader = torch.utils.data.DataLoader(test_set, batch_size = args.batch_size, drop_last = True, shuffle = True, num_workers = 4)
        return (train_loader, valid_loader, test_loader)
    else:
        dataset = torchvision.datasets.CelebA(root = args.data_dir, split = "train", download = True, transform = transform)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size = args.batch_size, drop_last = True, shuffle = False, num_workers = 4)
        
        return dataloader

def get_cifar10(args):
    transform = torchvision.transforms.Compose([
        torchvision.transforms.RandomHorizontalFlip(),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    dataset = torchvision.datasets.CIFAR10(root = args.data_dir, train = True, download = True, transform = transform)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size = args.batch_size, drop_last = True, shuffle = True, num_workers = 4)
    return dataloader

def get_dsprites(args):
    root = os.path.join(args.data_dir+'/dsprites-dataset/dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz')
    file = np.load(root, encoding='latin1')
    data = file['imgs'][:, np.newaxis, :, :]
    latents_values = file['latents_values']
    latents_classes = file['latents_classes']
    # print(data.shape, latents_values.shape, latents_classes.shape)
    # data = torch.from_numpy(data['imgs']).unsqueeze(1).float()
    # latents_values = torch.from_numpy(data['latents_values']).float()
    # latents_classes = torch.from_numpy(data['latents_classes']).int()
    train_kwargs = {'data':data, 'latents_values':latents_values, 'latents_classes':latents_classes}
    dset = CustomTensorDataset
    dataset = dset(**train_kwargs)

    dataloader = DataLoader(dataset,
                            batch_size=args.batch_size,
                            shuffle=True,
                            num_workers=4,
                            pin_memory=True,
                            drop_last=True)

    return dataloader

def get_chairs(args):
    transform = torchvision.transforms.Compose([
        torchvision.transforms.Resize((args.input_size, args.input_size)),
        torchvision.transforms.RandomHorizontalFlip(),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    dataset = CustomImageFolder(root = args.data_dir+'/3DChairs', transform = transform)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size = args.batch_size, drop_last = True, shuffle = True, num_workers = 4)
    return dataloader