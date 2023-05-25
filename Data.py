from random import shuffle
import torch
import numpy as np
import os.path
from torchvision.datasets import utils, MNIST, CIFAR10, CIFAR100
from torchvision import transforms
from torch.utils.data import Subset, DataLoader, random_split
from PIL import Image
# from noisify import noisify_label


class FEMNIST(MNIST):
    """
    This dataset is derived from the Leaf repository
    (https://github.com/TalwalkarLab/leaf) pre-processing of the Extended MNIST
    dataset, grouping examples by writer. Details about Leaf were published in
    "LEAF: A Benchmark for Federated Settings" https://arxiv.org/abs/1812.01097.
    """
    resources = [
        ('https://raw.githubusercontent.com/tao-shen/FEMNIST_pytorch/master/femnist.tar.gz',
         '59c65cec646fc57fe92d27d83afdf0ed')]

    def __init__(self, root, train=True, transform=None, target_transform=None,
                 download=True):
        super(MNIST, self).__init__(root, transform=transform,
                                    target_transform=target_transform)
        self.train = train

        if download:
            self.download()

        if not self._check_exists():
            raise RuntimeError('Dataset not found.' +
                               ' You can use download=True to download it')
        if self.train:
            data_file = self.training_file
        else:
            data_file = self.test_file

        self.data, self.targets, self.users_index = torch.load(os.path.join(self.processed_folder, data_file))

    def __getitem__(self, index):
        img, target = self.data[index], int(self.targets[index])
        img = Image.fromarray(img.numpy(), mode='F')
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return img, target

    def download(self):
        """Download the FEMNIST data if it doesn't exist in processed_folder already."""
        import shutil

        if self._check_exists():
            return

        utils.makedir_exist_ok(self.raw_folder)
        utils.makedir_exist_ok(self.processed_folder)

        # download files
        for url, md5 in self.resources:
            filename = url.rpartition('/')[2]
            utils.download_and_extract_archive(url, download_root=self.raw_folder, filename=filename, md5=md5)

        # process and save as torch files
        print('Processing...')
        shutil.move(os.path.join(self.raw_folder, self.training_file), self.processed_folder)
        shutil.move(os.path.join(self.raw_folder, self.test_file), self.processed_folder)


def Dataset(args):
    trainset, testset = None, None

    if args.dataset == 'cifar10':
        tra_trans = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        val_trans = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        trainset = CIFAR10(root="./data", train=True, download=True, transform=tra_trans)
        testset = CIFAR10(root="./data", train=False, download=True, transform=val_trans)

    if args.dataset == 'femnist' or 'mnist':
        tra_trans = transforms.Compose([
            transforms.Pad(2, padding_mode='edge'),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ])
        val_trans = transforms.Compose([
            transforms.Pad(2, padding_mode='edge'),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ])
        if args.dataset == 'femnist':
            trainset = FEMNIST(root='./data', train=True, transform=tra_trans)
            testset = FEMNIST(root='./data', train=False, transform=val_trans)
        if args.dataset == 'mnist':
            trainset = MNIST(root='./data', train=True, transform=tra_trans, download= True)
            testset = MNIST(root='./data', train=False, transform=val_trans, download= True)
    if args.dataset == 'cifar100':
        tra_trans = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4824, 0.4467), (0.2471, 0.2435, 0.2616)),
        ])
        val_trans = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4824, 0.4467), (0.2471, 0.2435, 0.2616)),
        ])
        trainset = CIFAR100(root="./data", train=True, download=True, transform=tra_trans)
        testset = CIFAR100(root="./data", train=False, download=True, transform=val_trans)
        

    return trainset, testset


class Data(object):

    def __init__(self, args):
        self.args = args
        self.trainset, self.testset = None, None
        trainset, testset = Dataset(args)
        
        if args.sampler == 'iid':
            # 分割train set分配给边缘节点
            num_train = [int(len(trainset) / args.split) for _ in range(args.split)]
            # cumsum_train = torch.tensor(list(num_train)).cumsum(dim=0).tolist()
            # idx_train = range(len(trainset.targets))
            # splited_trainset = [Subset(trainset, idx_train[off - l:off]) for off, l in zip(cumsum_train, num_train)]
            splited_trainset = random_split(trainset, num_train, generator=torch.Generator().manual_seed(42))

            # 分割test set分配给边缘节点 
            num_test = [int(len(testset) / args.split) for _ in range(args.split)]
            # cumsum_test = torch.tensor(list(num_test)).cumsum(dim=0).tolist()
            # idx_test = range(len(testset.targets))
            # splited_testset = [Subset(testset, idx_test[off - l:off]) for off, l in zip(cumsum_test, num_test)]
            splited_testset = random_split(testset, num_test, generator=torch.Generator().manual_seed(42))
            
        elif args.sampler == 'non-iid':
            # 分割train set分配给边缘节点
            targets = np.array(trainset.targets)
            num_train = [int(len(trainset) / args.split) for _ in range(args.split)]
            idx = [list(np.where(targets==1)[0]) for q in range(10)]
            kk = [idx[j][:int(args.mu*len(idx[1]))]for j in range(args.node_num)]    # 得到前百分之mu的对应数据
            num_data = list(range(len(trainset)))    # 数据集总数量
            kk2 = [b for a in kk for b in a]
            ll = list(set(num_data)-set(kk2))   # 将已经被选走的移除
            shuffle(ll)                # 打乱
            ii = [ll[int(o*num_train[o]*(1-args.mu)):int((o+1)*num_train[o]*(1-args.mu))] for o in range(args.split)]  # 随机将剩下的补齐
            zz = [kk[p]+ii[p] for p in range(args.split)]   ##得到最终索引
            
            splited_trainset = [Subset(trainset, zz[q]) for q in range(args.split)]
            
            # 分割test set分配给边缘节点，没必要
            num_test = [int(len(testset) / args.split) for _ in range(args.split)]
            splited_testset = random_split(testset, num_test, generator=torch.Generator().manual_seed(42))


        self.test_all = DataLoader(testset, batch_size=args.batchsize, shuffle=True, num_workers=4)
        self.train_loader = [DataLoader(splited_trainset[i], batch_size=args.batchsize, shuffle=True, num_workers=4)
                             for i in range(args.node_num)]
#         self.test_loader = [DataLoader(splited_testset[i], batch_size=args.batchsize, shuffle=True, num_workers=4)
#                             for i in range(args.node_num)]
        self.test_loader = DataLoader(testset, batch_size=args.batchsize, shuffle=True, num_workers=4)
