import torch
import numpy as np
import random
import argparse
import pickle
from utils.utils import *
import os.path
from os import path
import importlib.util


# load the config files
parser = argparse.ArgumentParser(description='Choose the configs to run.')
parser.add_argument('-c', '--config', type=str, required=True)
args = parser.parse_args()

use_config_spec = importlib.util.spec_from_file_location(
    args.config, "configs/{}.py".format(args.config))
config_module = importlib.util.module_from_spec(use_config_spec)
use_config_spec.loader.exec_module(config_module)
opt = config_module.opt

# from configs.config_imagenet_11 import opt
# #from configs.config_cub_18 import opt

opt.print_switch = True
#opt.use_visdom = True
opt.model = 'TSDA'
tax_only = False
if opt.model == "TSDA":
    from model.model import TSDA as Model
if opt.model == "TDSA" and tax_only:
    from model.model import TSDA as Model
    opt.lambda_d = 0
if opt.model == "DANN":
    from model.model import DANN as Model

if opt.model == "NAIVE":
    from model.model import NAIVE as Model
elif opt.model == "GDA":
    opt.A = read_pickle(opt.data_src + "A_brown_grda.pkl")
    from model.model import GDA as Model
elif opt.model == "CDANN":
    from model.model import CDANN as Model
    opt.cond_disc = True
elif opt.model == "ADDA":
    from model.model import ADDA as Model
elif opt.model == "MDD":
    from model.model import MDD as Model

model = Model(opt).to(opt.device)


# opt.src_domain = [6,7,8,9,10]
opt.num_source = len(opt.src_domain)
opt.num_target = opt.num_domain - opt.num_source
src_domain = [str(i) for i in opt.src_domain]
exp_id = opt.model + '_' + opt.dataset.split('/')[1].split('.')[0] + '_' + ''.join(src_domain)
outf_path = opt.outf + "/" + exp_id
outr_path = opt.outr + '/' + exp_id

search_space = {}

def run(search_space=None):
    print(search_space)
    if search_space is not None and opt.model == "TSDA":
        opt.lr_d = search_space['lr_d'] 
        opt.lr_e = search_space['lr_e']
        opt.lr_r = search_space['lr_r']
        if not opt.adj_default:
            opt.A = opt.A_root
    elif search_space is not None and opt.model == "DANN":
        opt.lr_d = search_space['lr_d']
        opt.lr_e = search_space['lr_e']
        opt.batch_size = search_space['batch_size']

    from dataset_utils.feature_dataset import FeatureDataloader

    dataloader = FeatureDataloader(opt)

    np.random.seed(opt.seed)
    random.seed(opt.seed)
    torch.manual_seed(opt.seed)
    model = Model(opt).to(opt.device)
    pred_path = outf_path + "_pred.pkl"


    for epoch in range(opt.num_epoch):
        model.learn(epoch, dataloader)
        if (epoch + 1) % opt.save_interval == 0 or (epoch + 1) == opt.num_epoch:
            model.save()
        if (epoch + 1) % opt.save_interval == 0 or (epoch + 1) == opt.num_epoch:
            d_all = model.test(epoch, dataloader)
            if (epoch + 1) == opt.num_epoch:
                write_pickle(d_all, pred_path)
                print(d_all['acc_msg'])



    return d_all

run()