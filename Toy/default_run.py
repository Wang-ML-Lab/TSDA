import torch
import numpy as np
import random
import pickle
from utils.utils import *
import os.path
from os import path
# import the experiment setting

from configs.config_tree_14 import opt

# load the data
from torch.utils.data import DataLoader
from dataset_utils.dataset import ToyDataset, SeqToyDataset
# from result_plot import *
# from plot_pca_simple import *


rstate = np.random.default_rng(opt.seed)

# opt.print_switch = True
# opt.use_visdom = False
# opt.model = 'TSDA'
tax_only = False
if opt.model == "DANN":
    from model.model import DANN as Model
elif opt.model == "GDA":
    from model.model import GDA as Model
elif opt.model == "CDANN":
    from model.model import CDANN as Model
    opt.cond_disc = True
elif opt.model == "ADDA":
    from model.model import ADDA as Model
elif opt.model == "MDD":
    from model.model import MDD as Model
elif opt.model == "TSDA":
    from model.model import TSDA as Model
if opt.model == "TDSA" and tax_only:
    from model.model import TSDA as Model
    opt.lambda_d = 0

data_source = opt.dataset
with open(data_source, "rb") as data_file:
    data_pkl = pickle.load(data_file)

print(opt.src_domain)
print(opt.dataset)
# build dataset
opt.A = data_pkl["A"]
opt.A_root = data_pkl["A_root"]
#print(opt.A)
if opt.model == "GDA":
    for i in range(opt.num_domain):
        for j in range(opt.num_domain):
            if i != j:
                if 0 <= opt.A[i][j] <= 0.1:
                    opt.A[i][j] = 1
                else:
                    opt.A[i][j] = 0

trim_percentage = 0.5
data_num = len(data_pkl['data'])
trim_idx = np.linspace(0, data_num-1, num=int(data_num * trim_percentage), dtype=int)
data_pkl["data"] = np.delete(data_pkl["data"], trim_idx, 0)
data_pkl["label"] = np.delete(data_pkl["label"], trim_idx, 0)
data_pkl["domain"] = np.delete(data_pkl["domain"], trim_idx, 0)

data = data_pkl["data"]
data_mean = data.mean(0, keepdims=True)
data_std = data.std(0, keepdims=True)
data_pkl["data"] = (data - data_mean) / data_std  # normalize the raw data
datasets = [
    ToyDataset(data_pkl, i, opt) for i in range(opt.num_domain)
]  # sub dataset for each domain

dataset = SeqToyDataset(
    datasets, size=len(datasets[0])
)  # mix sub dataset to a large one


src_domain = [str(i) for i in opt.src_domain]
exp_id = opt.model + '_' + opt.dataset.split('_')[2] + '_' + ''.join(src_domain)
#spiral = '_full'
spiral = '_half'
percentage = '_' + str(trim_percentage)
outf_path = opt.outf + "/" + exp_id + spiral + percentage
outr_path = opt.outr + '/' + exp_id + spiral + percentage

BO_loss = []
BO_params = []
BO_data = {'loss': BO_loss, 'params': BO_params}
count = 0

# opt.lr_d = 1e-06
# opt.lr_e = 3e-05
# opt.lr_r = 3e-05

def run(search_space=None):
    print(search_space)
    if search_space is not None and opt.model == "TSDA":
        opt.lambda_r = search_space['lambda_r']
        opt.lambda_d = search_space['lambda_d']
        opt.lambda_e = search_space['lambda_e']
        opt.batch_size = search_space['batch_size']
        opt.lr_d = search_space['lr_d']  
        opt.lr_e = search_space['lr_e'] 
        opt.lr_r = search_space['lr_r']
        opt.adj_default = search_space['adj_default']
        if not opt.adj_default:
            opt.A = opt.A_root
    elif search_space is not None and opt.model == "DANN":
        opt.lr_d = search_space['lr_d']
        opt.lr_e = search_space['lr_e']
        opt.batch_size = search_space['batch_size']
    else:
        search_space = {}
        search_space['lr_d'] = opt.lr_d 
        search_space['lr_e'] = opt.lr_e
        search_space['lr_g'] = opt.lr_g


    np.random.seed(opt.seed)
    random.seed(opt.seed)
    torch.manual_seed(opt.seed)
    dataloader = DataLoader(
        dataset=dataset, shuffle=True, batch_size=int(opt.batch_size)
    )
    pred_path = outf_path + "_pred.pkl"
    if path.exists(pred_path):
        info = read_pickle(pred_path)
    else:
        info = {'acc': 0}
    model = Model(opt).to(opt.device)
    for epoch in range(opt.num_epoch):
        model.learn(epoch, dataloader)
        if (epoch + 1) % opt.test_interval == 0 or (epoch + 1) == opt.num_epoch:
            d_all = model.test(epoch, dataloader)
            if (epoch + 1) == opt.num_epoch:
                if d_all['acc'] > info['acc']:
                    model.save()
                    write_pickle(d_all, pred_path)
                    write_pickle(search_space, outf_path + "_best_params_backup.pkl")
                if opt.use_visdom and d_all['acc'] < info['acc']:
                    model.vis_close()

    return d_all['acc']


run()
best_hp = read_pickle(outf_path + '_best_params_backup.pkl')
# best_hp = read_pickle("dump/TSDA_l6l14_0123_half_test_best_params.pkl")
print(best_hp)