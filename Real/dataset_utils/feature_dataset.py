import numpy as np
from torch.utils.data import DataLoader, Dataset
import pickle


def read_pickle(name):
    with open(name, 'rb') as f:
        data = pickle.load(f)
    return data


class FeatureDataset(Dataset):
    def __init__(self, pkl, domain_id, sudo_len, opt=None):
        idx = pkl['domain'] == domain_id
        self.data = pkl['data'][idx].astype(np.float32)
        self.label = pkl['label'][idx].astype(np.int64)
        self.domain = domain_id
        self.real_len = len(self.data)
        self.sudo_len = sudo_len

        # if opt.normalize_domain:
        #     print('===> Normalize in every domain')
        #     self.data_m, self.data_s = self.data.mean(0, keepdims=True), self.data.std(0, keepdims=True)
        #     self.data = (self.data - self.data_m) / self.data_s

    def __getitem__(self, idx):
        idx %= self.real_len
        return self.data[idx], self.label[idx], self.domain

    def __len__(self):
        # return len(self.data)
        return self.sudo_len


class FeatureDataloader(DataLoader):
    def __init__(self, opt):
        self.opt = opt
        self.src_domain = opt.src_domain
        self.tgt_domain = opt.tgt_domain
        self.all_domain = opt.all_domain

        self.pkl = read_pickle(opt.data_path)
        sudo_len = 0
        for i in self.all_domain:
            idx = self.pkl['domain'] == i
            sudo_len = max(sudo_len, idx.sum())
        self.sudo_len = sudo_len

        print("sudo len: {}".format(sudo_len))

        self.train_datasets = [
            FeatureDataset(
                self.pkl,
                domain_id=i,
                opt=opt,
                sudo_len=self.sudo_len,
            ) for i in self.all_domain
        ]

        if self.opt.test_on_all_dmn:
            self.test_datasets = [
                FeatureDataset(
                    self.pkl,
                    domain_id=i,
                    opt=opt,
                    sudo_len=self.sudo_len,
                ) for i in self.all_domain
            ]
        else:
            self.test_datasets = [
                FeatureDataset(
                    self.pkl,
                    domain_id=i,
                    opt=opt,
                    sudo_len=self.sudo_len,
                ) for i in self.tgt_domain
            ]

        self.train_data_loader = [
            DataLoader(dataset,
                       batch_size=opt.batch_size,
                       shuffle=opt.shuffle,
                       # consider if necessary
                       num_workers=0,
                       pin_memory=True,
                       #persistent_workers=True
                       )
            for dataset in self.train_datasets
        ]

        self.test_data_loader = [
            DataLoader(dataset,
                       batch_size=opt.batch_size,
                       shuffle=opt.shuffle,
                       num_workers=0,
                       pin_memory=True,
                       #persistent_workers=True
                       )
            for dataset in self.test_datasets
        ]

    def get_train_data(self):
        # this is return a iterator for the whole dataset
        return zip(*self.train_data_loader)

    def get_test_data(self):
        return zip(*self.test_data_loader)

# class SeqToyDataset(Dataset):
#     # the size may change because of the toy dataset!!
#     def __init__(self, datasets, size=3 * 200):
#         self.datasets = datasets
#         self.size = size
#         print('SeqDataset Size {} Sub Size {}'.format(
#             size, [len(ds) for ds in datasets]
#         ))

#     def __len__(self):
#         return self.size

#     def __getitem__(self, i):
#         return [ds[i] for ds in self.datasets]