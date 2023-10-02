from easydict import EasyDict
import numpy as np
import pickle


def read_pickle(name):
    with open(name, "rb") as f:
        data = pickle.load(f)
    return data


# load/output dir
opt = EasyDict()
opt.loadf = "./dump"
opt.outf = "./dump"
opt.outr = './data/test result'

# normalize each data domain
# opt.normalize_domain = False
opt.print_switch = False
opt.adj_default = True
# now it is half circle
opt.num_domain = 40
# the specific source and target domain:
opt.src_domain = np.array([14,17,20,23,26,29])  # tight_boundary
opt.num_source = opt.src_domain.shape[0]
opt.num_target = opt.num_domain - opt.num_source
opt.test_on_all_dmn = True




#opt.model = "DANN"
#opt.model = "CDANN"
#opt.model = "ADDA"
opt.model = 'TSDA'
# opt.model = "GDA"
#opt.model = "MDD"
opt.cond_disc = (
    False  # whether use conditional discriminator or not (for CDANN)
)


opt.use_visdom = False
opt.visdom_port = 2000

opt.use_g_encode = True # False  # True
if opt.use_g_encode:
    opt.g_encode = read_pickle("g_encode_l7l40.pkl")


opt.device = "cuda"
opt.seed = 2333  # 1# 101 # 1 # 233 # 1

# opt.lambda_gan = 0.5  # 0.5 # 0.3125 # 0.5 # 0.5
opt.lambda_gan = 0

# for MDD use only
opt.lambda_src = 0.5
opt.lambda_tgt = 0.5

# for TDDA use only
opt.lambda_r = 0.594795294351678
opt.lambda_d = 0.2488505602149487
opt.lambda_e = 0.7786359421472335
opt.lambda_c = 1

opt.num_epoch = 300
opt.batch_size = 10
opt.lr_d = 1e-5 # 3e-5 # 1e-4 # 2.9 * 1e-5 #3e-5  # 1e-4
opt.lr_e = 3e-5  # 3e-5 # 1e-4 # 2.9 * 4e-6
opt.lr_g = 1e-5
opt.lr_r = 1e-4
opt.gamma = 100
opt.beta1 = 0.9
opt.weight_decay = 5e-4
opt.wgan = False  # do not use wgan to train
opt.no_bn = True  # do not use batch normalization # True

# model size configs, used for D, E, F
opt.nx = 2  # dimension of the input data
opt.nt = 2  # dimension of the vertex embedding
opt.nh = 512  # dimension of hidden # 512
opt.nc = 2  # number of label class
opt.nd_out = 2  # dimension of D's output
opt.nr_out = 2

# sample how many vertices for training R
opt.sample_v = opt.num_domain

# # sample how many vertices for training G
opt.sample_v_g = opt.num_domain

opt.test_interval = 20
opt.save_interval = 100
# drop out rate
opt.p = 0.2
opt.shuffle = True


# dataset
opt.dataset = "data/toy_d128_l7l40.pkl"
