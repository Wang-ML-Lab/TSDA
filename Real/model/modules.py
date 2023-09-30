import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict


class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


# TODO: G now is a fixed embedding:
class GNet(nn.Module):
    def __init__(self, opt):
        super(GNet, self).__init__()
        # for cluster
        # self.G = torch.FloatTensor(
        #     [
        #         [1, 0],
        #         [1, 1],
        #         [0, 1],
        #     ]
        # ).to(opt.device)

        # for quarter
        # num_domain = opt.num_domain
        # G = np.eye(num_domain, num_domain - 1) + np.eye(num_domain, num_domain - 1, k=-1)
        # self.G = torch.from_numpy(G).float().to(device=opt.device)

        # the following code is for testing G only:

        # self.G = torch.randn(opt.num_domain, opt.nd_out).to(opt.device)
        # self.G.requires_grad=True
        # self.bias = torch.tensor(-1.9476923942565918).to(opt.device)# torch.randn(1).to(opt.device)
        # self.bias = torch.tensor(-2.0).to(opt.device)
        # self.bias.requires_grad=True

        # # self.weight = torch.tensor(2.4483554363250732).to(opt.device)# torch.randn(1).to(opt.device)
        # self.weight = torch.tensor(2.02).to(opt.device)
        # self.weight.requires_grad=True
        self.use_g_encode = opt.use_g_encode

        # self.bias = torch.tensor(-1.9476923942565918).to(opt.device)# torch.randn(1).to(opt.device)
        # self.bias = torch.tensor(-2.0).to(opt.device)
        # self.bias.requires_grad=True

        # # self.weight = torch.tensor(2.4483554363250732).to(opt.device)# torch.randn(1).to(opt.device)
        # self.weight = torch.tensor(2.02).to(opt.device)
        # self.weight.requires_grad=True
        if self.use_g_encode:
            G = np.zeros((opt.num_domain, opt.nv_embed))
            for i in range(opt.num_domain):
                G[i] = opt.g_encode[str(i)]
            self.G = torch.from_numpy(G).float().to(device=opt.device)
        else:
            self.fc1 = nn.Linear(opt.num_domain, opt.nh)
            # self.fc2 = nn.Linear(opt.nh, opt.nh)
            self.fc_final = nn.Linear(opt.nh, opt.nv_embed)

    def forward(self, x):
        re = x.dim() == 3
        if re:
            T, B, C = x.shape
            x = x.reshape(T * B, -1)

        if self.use_g_encode:
            x = torch.matmul(x.float(), self.G)
        else:
            x = F.relu(self.fc1(x.float()))
            # x = F.relu(self.fc1(x))
            # x = F.relu(self.fc1(x))
            # x = F.relu(self.fc2(x))
            # x = nn.Dropout(p=p)(x)
            x = self.fc_final(x)

        return x
        # return torch.matmul(x.float(), self.G)
        # drop out:
        # p = self.opt.p
        # x = nn.Dropout(p=p)(x.float())
        # x = F.relu(self.fc1(x.float()))
        # # x = F.relu(self.fc1(x))
        # # x = F.relu(self.fc2(x))
        # # x = nn.Dropout(p=p)(x)
        # x = self.fc_final(x)
        # return x


class AlexNet_BVLC_Feature(nn.Module):
    def __init__(self, opt):
        super(AlexNet_BVLC_Feature, self).__init__()
        # nh = opt.nh

        self.features = nn.Sequential(OrderedDict([
            ("conv1", nn.Conv2d(3, 96, kernel_size=11, stride=4)),
            ("relu1", nn.ReLU(inplace=True)),
            ("pool1", nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True)),
            ("norm1", nn.LocalResponseNorm(5, 1.e-4, 0.75)),
            ("conv2", nn.Conv2d(96, 256, kernel_size=5, padding=2, groups=2)),
            ("relu2", nn.ReLU(inplace=True)),
            ("pool2", nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True)),
            ("norm2", nn.LocalResponseNorm(5, 1.e-4, 0.75)),
            ("conv3", nn.Conv2d(256, 384, kernel_size=3, padding=1)),
            ("relu3", nn.ReLU(inplace=True)),
            ("conv4", nn.Conv2d(384, 384, kernel_size=3, padding=1, groups=2)),
            ("relu4", nn.ReLU(inplace=True)),
            ("conv5", nn.Conv2d(384, 256, kernel_size=3, padding=1, groups=2)),
            ("relu5", nn.ReLU(inplace=True)),
            ("pool5", nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True)),
        ]))

        self.g_embed_fc = nn.Sequential(
            # be careful about this code!
            nn.Linear(opt.nv_embed, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 512),
            nn.ReLU(inplace=True)
        )
        # self.fc1_var = nn.Linear(nt, nh)
        # self.fc2_var = nn.Linear(nh, nh)

        self.fuse_fc = nn.Sequential(
            nn.Linear(256 * 6 * 6 + 512, 4096),
            nn.ReLU(inplace=True),
            # tmp no drop out
            nn.Linear(4096, 4096),
            # nn.ReLU(inplace=True),
            # nn.Linear(4096, 4096)
        )

    def forward(self, x, t):
        T, B, C, W, H = x.shape
        x = x.reshape(T * B, C, W, H)
        t = t.reshape(T * B, -1)

        x = self.features(x)
        t = self.g_embed_fc(t)

        x = x.view(x.size(0), 256 * 6 * 6)
        x = torch.cat((x, t), dim=1)

        x = self.fuse_fc(x)
        return x.reshape(T, B, -1)


class AlexNet_BVLC(nn.Module):
    def __init__(self, num_classes=1000, dropout=False):
        super(AlexNet_BVLC, self).__init__()
        self.features = nn.Sequential(OrderedDict([
            ("conv1", nn.Conv2d(3, 96, kernel_size=11, stride=4)),
            ("relu1", nn.ReLU(inplace=True)),
            ("pool1", nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True)),
            ("norm1", nn.LocalResponseNorm(5, 1.e-4, 0.75)),
            ("conv2", nn.Conv2d(96, 256, kernel_size=5, padding=2, groups=2)),
            ("relu2", nn.ReLU(inplace=True)),
            ("pool2", nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True)),
            ("norm2", nn.LocalResponseNorm(5, 1.e-4, 0.75)),
            ("conv3", nn.Conv2d(256, 384, kernel_size=3, padding=1)),
            ("relu3", nn.ReLU(inplace=True)),
            ("conv4", nn.Conv2d(384, 384, kernel_size=3, padding=1, groups=2)),
            ("relu4", nn.ReLU(inplace=True)),
            ("conv5", nn.Conv2d(384, 256, kernel_size=3, padding=1, groups=2)),
            ("relu5", nn.ReLU(inplace=True)),
            ("pool5", nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True)),
        ]))

        self.classifier = nn.Sequential(OrderedDict([
            ("fc6", nn.Linear(256 * 6 * 6, 4096)),
            ("relu6", nn.ReLU(inplace=True)),
            ("drop6", nn.Dropout() if dropout else Identity(sub=0.5)),
            ("fc7", nn.Linear(4096, 4096)),
            ("relu7", nn.ReLU(inplace=True)),
            ("drop7", nn.Dropout() if dropout else Identity(sub=0.5)),
            ("fc8", nn.Linear(4096, 1000))
        ]))

        self.final = nn.Linear(4096, num_classes)

    def forward(self, x, t):
        x = self.features(x)
        x = x.view(x.size(0), 256 * 6 * 6)
        x = self.classifier._modules['fc6'](x)
        x = self.classifier._modules['relu6'](x)
        x = self.classifier._modules['drop6'](x)
        x = self.classifier._modules['fc7'](x)
        x = self.classifier._modules['relu7'](x)
        x = self.classifier._modules['drop7'](x)
        x = self.final(x)
        return x

    def fix(self, alpha):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
                m.momentum = alpha

    def to_classes(self, num_classes):
        num_ftrs = self.final.in_features
        self.final = nn.Linear(num_ftrs, num_classes)
        self.final.weight.data.normal_(0, 0.001)


class AlexNet_BN(AlexNet_BVLC):

    def __init__(self, num_classes=1000, dropout=False):
        super(AlexNet_BN, self).__init__(num_classes=1000, dropout=False)
        self.bn = nn.BatchNorm1d(4096)
        for m in self.modules():
            if isinstance(m, nn.BatchNorm1d) or isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x, t):
        x = self.features(x)
        x = x.view(x.size(0), 256 * 6 * 6)
        x = self.classifier._modules['fc6'](x)
        x = self.classifier._modules['relu6'](x)
        x = self.classifier._modules['drop6'](x)
        x = self.classifier._modules['fc7'](x)
        x = self.bn(x)
        x = self.classifier._modules['relu7'](x)
        x = self.final(x)
        return x

    # def reset_edges(self):
    #     return

    # def set_bn_from_edges(self,idx, ew=None):
    #     return

    # def copy_source(self,idx):
    #     return

    # def init_edges(self,edges):
    #     return


class AlexNet_GraphBN(AlexNet_BVLC):

    def __init__(self, num_classes=1000, dropout=False, domains=30):
        super(AlexNet_GraphBN, self).__init__(num_classes=1000, dropout=False)
        self.bns = GraphBN(4096, domains=domains)

    def forward(self, x, t):
        x = self.features(x)
        x = x.view(x.size(0), 256 * 6 * 6)
        x = self.classifier._modules['fc6'](x)
        x = self.classifier._modules['relu6'](x)
        x = self.classifier._modules['drop6'](x)
        x = self.classifier._modules['fc7'](x)
        x = x.view(x.shape[0], -1, 1, 1)
        x = self.bns(x, t)
        x = x.view(x.shape[0], -1)
        x = self.classifier._modules['relu7'](x)
        x = self.final(x)
        return x

    def reset_edges(self):
        self.bns.reset_edges()

    def set_bn_from_edges(self, idx, ew=None):
        self.bns.set_bn_from_edges(idx, ew=ew)

    def copy_source(self, idx):
        self.bns.copy_source(idx)

    def init_edges(self, edges):
        self.bns.init_edges(edges)


##############################
##### INSTANTIATE MODELS #####
##############################

# def get_graph_net(classes=3, domains=30, url=None):

# 	model=AlexNet_GraphBN(domains=domains)
# 	state = model.state_dict()
# 	state.update(torch.load(url))
# 	model.load_state_dict(state)

# 	model.to_classes(classes)

# 	return model

class FeatureNet(nn.Module):
    def __init__(self, opt):
        super(FeatureNet, self).__init__()

        nx, nh, nt, p = opt.num_input, opt.nh, opt.nv_embed, opt.p
        self.p = p
        self.nh = nh

        self.fc1 = nn.Linear(nx, nh)
        self.fc2 = nn.Linear(nh * 2, nh * 2)
        # self.fc3 = nn.Linear(nh * 2, nh * 2)
        # self.fc4 = nn.Linear(nh * 2, nh * 2)
        self.fc_final = nn.Linear(nh * 2, nh)

        # here I change the input to fit the change dimension
        self.fc1_var = nn.Linear(nt, nh)
        self.fc2_var = nn.Linear(nh, nh)

        # self.bn=nn.BatchNorm1d(nh * 2)
        # for m in self.modules():
        #     if isinstance(m, nn.BatchNorm1d) or isinstance(m, nn.BatchNorm2d):
        #         m.weight.data.fill_(1)
        #         m.bias.data.zero_()

    def forward(self, x, t):
        re = x.dim() == 3
        if re:
            T, B, C = x.shape
            x = x.reshape(T * B, -1)
            t = t.reshape(T * B, -1)

        x = F.relu(self.fc1(x))
        t = F.relu(self.fc1_var(t))
        t = F.relu(self.fc2_var(t))

        # combine feature in the middle
        x = torch.cat((x, t), dim=1)

        # main
        x = F.relu(self.fc2(x))
        # x = F.relu(self.fc3(x))
        # x = F.relu(self.fc4(x))

        # bn
        # x = nn.BatchNorm1d(self.nh * 2)(x)
        # x = self.bn(x)

        x = self.fc_final(x)

        if re:
            return x.reshape(T, B, -1)
        else:
            return x


class GraphDNet(nn.Module):
    """
    Generate z' for connection loss
    """

    def __init__(self, opt):
        super(GraphDNet, self).__init__()
        nh = opt.nh
        nin = nh
        self.fc3 = nn.Linear(nin, nh)
        self.bn3 = nn.BatchNorm1d(nh)

        self.fc4 = nn.Linear(nh, nh)
        self.bn4 = nn.BatchNorm1d(nh)

        self.fc5 = nn.Linear(nh, nh)
        self.bn5 = nn.BatchNorm1d(nh)

        self.fc6 = nn.Linear(nh, nh)
        self.bn6 = nn.BatchNorm1d(nh)

        self.fc7 = nn.Linear(nh, nh)
        self.bn7 = nn.BatchNorm1d(nh)

        # be careful!! here use dimension of vertex embedding to encode; originally do not have this code
        # self.fc_final = nn.Linear(nh, opt.nd_out)
        self.fc_final = nn.Linear(nh, opt.nd_out)
        # self.fc_final = nn.Linear(nh, nh)

        if opt.no_bn:
            self.bn3 = Identity()
            self.bn4 = Identity()
            self.bn5 = Identity()
            self.bn6 = Identity()
            self.bn7 = Identity()

    def forward(self, x):
        re = x.dim() == 3

        if re:
            T, B, C = x.shape
            x = x.reshape(T * B, -1)

        x = F.relu(self.bn3(self.fc3(x)))
        x = F.relu(self.bn4(self.fc4(x)))
        x = F.relu(self.bn5(self.fc5(x)))
        x = F.relu(self.bn6(self.fc6(x)))
        x = F.relu(self.bn7(self.fc7(x)))

        # be careful!! here use the dimension of vertex embedding to encode; originally do not have this code
        x = self.fc_final(x)

        if re:
            return x.reshape(T, B, -1)
        else:
            return x


class ResGraphDNet(nn.Module):
    """
    Generate z' for connection loss
    """

    def __init__(self, opt):
        super(ResGraphDNet, self).__init__()
        nh = opt.nh
        nin = nh
        self.fc3 = nn.Linear(nin, nh)
        self.bn3 = nn.BatchNorm1d(nh)

        self.fc4 = nn.Linear(nh, nh)
        self.bn4 = nn.BatchNorm1d(nh)

        self.fc5 = nn.Linear(nh, nh)
        self.bn5 = nn.BatchNorm1d(nh)

        self.fc6 = nn.Linear(nh, nh)
        self.bn6 = nn.BatchNorm1d(nh)

        self.fc7 = nn.Linear(nh, nh)
        self.bn7 = nn.BatchNorm1d(nh)

        self.fc8 = nn.Linear(nh, nh)
        self.bn8 = nn.BatchNorm1d(nh)

        self.fc9 = nn.Linear(nh, nh)
        self.bn9 = nn.BatchNorm1d(nh)

        self.fc10 = nn.Linear(nh, nh)
        self.bn10 = nn.BatchNorm1d(nh)

        self.fc11 = nn.Linear(nh, nh)
        self.bn11 = nn.BatchNorm1d(nh)

        # be careful!! here use 2 to encode; originally do not have this code
        self.fc_final = nn.Linear(nh, opt.nd_out)

        if opt.no_bn:
            self.bn3 = Identity()
            self.bn4 = Identity()
            self.bn5 = Identity()
            self.bn6 = Identity()
            self.bn7 = Identity()
            self.bn8 = Identity()
            self.bn9 = Identity()
            self.bn10 = Identity()
            self.bn11 = Identity()

    def forward(self, x):
        re = x.dim() == 3

        if re:
            T, B, C = x.shape
            x = x.reshape(T * B, -1)

        x = F.relu(self.bn3(self.fc3(x)))
        id1 = x
        out = F.relu(self.bn4(self.fc4(x)))
        out = self.bn5(self.fc5(out))
        x = F.relu(out + id1)

        id2 = x
        # out = F.relu(self.bn5(self.fc5(x)))
        out = F.relu(self.bn6(self.fc6(x)))
        out = self.bn7(self.fc7(out))
        x = F.relu(out + id2)

        id3 = x
        out = F.relu(self.bn8(self.fc8(x)))
        out = self.bn9(self.fc9(out))
        x = F.relu(out + id3)

        id4 = x
        out = F.relu(self.bn10(self.fc10(x)))
        out = self.bn11(self.fc11(out))
        x = F.relu(out + id4)

        # be careful!! here use 2 to encode; originally do not have this code
        x = self.fc_final(x)
        # print(x.detach().cpu().numpy())

        if re:
            return x.reshape(T, B, -1)
        else:
            return x


class DiscNet(nn.Module):
    """
    Discriminator doing binary classification: source v.s. target
    """

    def __init__(self, opt):
        super(DiscNet, self).__init__()
        nh = opt.nh

        nin = nh
        self.fc3 = nn.Linear(nin, nh)
        self.bn3 = nn.BatchNorm1d(nh)

        self.fc4 = nn.Linear(nh, nh)
        self.bn4 = nn.BatchNorm1d(nh)

        self.fc5 = nn.Linear(nh, nh)
        self.bn5 = nn.BatchNorm1d(nh)

        self.fc6 = nn.Linear(nh, nh)
        self.bn6 = nn.BatchNorm1d(nh)

        self.fc7 = nn.Linear(nh, nh)
        self.bn7 = nn.BatchNorm1d(nh)

        if opt.no_bn:
            self.bn3 = Identity()
            self.bn4 = Identity()
            self.bn5 = Identity()
            self.bn6 = Identity()
            self.bn7 = Identity()

        self.fc_final = nn.Linear(nh, 1)
        if opt.model in ['ADDA', 'CUA']:
            print('===> Discrinimator Output Activation: sigmoid')
            self.output = lambda x: torch.sigmoid(x)
        else:
            print('===> Discrinimator Output Activation: identity')
            self.output = lambda x: x

    def forward(self, x):
        re = x.dim() == 3

        if re:
            T, B, C = x.shape
            x = x.reshape(T * B, -1)

        x = F.relu(self.bn3(self.fc3(x)))
        x = F.relu(self.bn4(self.fc4(x)))
        x = F.relu(self.bn5(self.fc5(x)))
        x = F.relu(self.bn6(self.fc6(x)))
        x = F.relu(self.bn7(self.fc7(x)))
        x = self.output(self.fc_final(x))

        if re:
            return x.reshape(T, B, -1)
        else:
            return x


class ClassDiscNet(nn.Module):
    """
    Discriminator doing multi-class classification on the domain
    """

    def __init__(self, opt):
        super(ClassDiscNet, self).__init__()
        nh = opt.nh
        nc = opt.nc
        nin = nh
        nout = opt.num_domain

        if opt.cond_disc:
            print('===> Conditioned Discriminator')
            nmid = nh * 2
            self.cond = nn.Sequential(
                nn.Linear(nc, nh),
                nn.ReLU(True),
                nn.Linear(nh, nh),
                nn.ReLU(True),
            )
        else:
            #print('===> Unconditioned Discriminator')
            nmid = nh
            self.cond = None

        #print(f'===> Discriminator will distinguish {nout} domains')

        self.fc3 = nn.Linear(nin, nh)
        self.bn3 = nn.BatchNorm1d(nh)

        self.fc4 = nn.Linear(nmid, nh)
        self.bn4 = nn.BatchNorm1d(nh)

        self.fc5 = nn.Linear(nh, nh)
        self.bn5 = nn.BatchNorm1d(nh)

        self.fc6 = nn.Linear(nh, nh)
        self.bn6 = nn.BatchNorm1d(nh)

        self.fc7 = nn.Linear(nh, nh)
        self.bn7 = nn.BatchNorm1d(nh)

        self.fc8 = nn.Linear(nh, nh)
        self.bn8 = nn.BatchNorm1d(nh)

        self.fc9 = nn.Linear(nh, nh)
        self.bn9 = nn.BatchNorm1d(nh)

        if opt.no_bn:
            self.bn3 = Identity()
            self.bn4 = Identity()
            self.bn5 = Identity()
            self.bn6 = Identity()
            self.bn7 = Identity()
            self.bn8 = Identity()
            self.bn9 = Identity()

        self.fc_final = nn.Linear(nh, nout)

    def forward(self, x):
        re = x.dim() == 3

        if re:
            T, B, C = x.shape
            x = x.reshape(T * B, -1)
            # f_exp = f_exp.reshape(T * B, -1)

        x = F.relu(self.bn3(self.fc3(x)))
        x = F.relu(self.bn4(self.fc4(x)))
        x = F.relu(self.bn5(self.fc5(x)))
        x = F.relu(self.bn6(self.fc6(x)))
        x = F.relu(self.bn7(self.fc7(x)))
        x = F.relu(self.bn8(self.fc8(x)))
        x = F.relu(self.bn8(self.fc9(x)))
        # x = self.fc_final(x)
        x = F.relu(self.fc_final(x))
        x = torch.log_softmax(x, dim=1)
        if re:
            return x.reshape(T, B, -1)
        else:
            return x


class CondClassDiscNet(nn.Module):
    """
    Discriminator doing multi-class classification on the domain
    """

    def __init__(self, opt):
        super(CondClassDiscNet, self).__init__()
        nh = opt.nh
        nc = opt.nc
        nin = nh
        nout = opt.num_domain

        if opt.cond_disc:
            print('===> Conditioned Discriminator')
            nmid = nh * 2
            self.cond = nn.Sequential(
                nn.Linear(nc, nh),
                nn.ReLU(True),
                nn.Linear(nh, nh),
                nn.ReLU(True),
            )
        else:
            print('===> Unconditioned Discriminator')
            nmid = nh
            self.cond = None

        print(f'===> Discriminator will distinguish {nout} domains')

        self.fc3 = nn.Linear(nin, nh)
        self.bn3 = nn.BatchNorm1d(nh)

        self.fc4 = nn.Linear(nmid, nh)
        self.bn4 = nn.BatchNorm1d(nh)

        self.fc5 = nn.Linear(nh, nh)
        self.bn5 = nn.BatchNorm1d(nh)

        self.fc6 = nn.Linear(nh, nh)
        self.bn6 = nn.BatchNorm1d(nh)

        self.fc7 = nn.Linear(nh, nh)
        self.bn7 = nn.BatchNorm1d(nh)

        if opt.no_bn:
            self.bn3 = Identity()
            self.bn4 = Identity()
            self.bn5 = Identity()
            self.bn6 = Identity()
            self.bn7 = Identity()

        self.fc_final = nn.Linear(nh, nout)

    def forward(self, x, f_exp):
        re = x.dim() == 3

        if re:
            T, B, C = x.shape
            x = x.reshape(T * B, -1)
            f_exp = f_exp.reshape(T * B, -1)

        x = F.relu(self.bn3(self.fc3(x)))
        if self.cond is not None:
            f = self.cond(f_exp)
            x = torch.cat([x, f], dim=1)
        x = F.relu(self.bn4(self.fc4(x)))
        x = F.relu(self.bn5(self.fc5(x)))
        x = F.relu(self.bn6(self.fc6(x)))
        x = F.relu(self.bn7(self.fc7(x)))
        x = self.fc_final(x)
        x = torch.log_softmax(x, dim=1)
        if re:
            return x.reshape(T, B, -1)
        else:
            return x


class PredNet(nn.Module):
    def __init__(self, opt):
        super(PredNet, self).__init__()

        nh, nc = opt.nh, opt.nc
        nin = nh
        self.fc3 = nn.Linear(nin, nh)
        self.bn3 = nn.BatchNorm1d(nh)
        self.fc4 = nn.Linear(nh, nh)
        self.bn4 = nn.BatchNorm1d(nh)
        self.fc_final = nn.Linear(nh, nc)
        if opt.no_bn:
            self.bn3 = Identity()
            self.bn4 = Identity()

    def forward(self, x, return_softmax=False):
        re = x.dim() == 3
        if re:
            T, B, C = x.shape
            x = x.reshape(T * B, -1)

        x = F.relu(self.bn3(self.fc3(x)))
        x = F.relu(self.bn4(self.fc4(x)))
        x = self.fc_final(x)
        x_softmax = F.softmax(x, dim=1)

        # just a test !!!
        # x = F.log_softmax(x, dim=1)
        # x = torch.clamp_max(x_softmax + 1e-4, 1)
        # x = torch.log(x)
        x = torch.log(x_softmax + 1e-4)

        if re:
            x = x.reshape(T, B, -1)
            x_softmax = x_softmax.reshape(T, B, -1)

        if return_softmax:
            return x, x_softmax
        else:
            return x


class RegressorNet(nn.Module):
    """

    """

    def __init__(self, opt):
        super(RegressorNet, self).__init__()
        nh = opt.nh
        nin = nh
        self.fc3 = nn.Linear(nin, nh)
        self.bn3 = nn.BatchNorm1d(nh)

        self.fc4 = nn.Linear(nh, nh)
        self.bn4 = nn.BatchNorm1d(nh)

        self.fc5 = nn.Linear(nh, nh)
        self.bn5 = nn.BatchNorm1d(nh)

        self.fc6 = nn.Linear(nh, nh)
        self.bn6 = nn.BatchNorm1d(nh)

        self.fc7 = nn.Linear(nh, nh)
        self.bn7 = nn.BatchNorm1d(nh)

        self.fc_final = nn.Linear(nh, opt.nr_out)

        if opt.no_bn:
            self.bn3 = Identity()
            self.bn4 = Identity()
            self.bn5 = Identity()
            self.bn6 = Identity()
            self.bn7 = Identity()

    def forward(self, x):
        re = x.dim() == 3

        if re:
            T, B, C = x.shape
            x = x.reshape(T * B, -1)

        x = F.relu(self.bn3(self.fc3(x)))
        x = F.relu(self.bn4(self.fc4(x)))
        x = F.relu(self.bn5(self.fc5(x)))
        x = F.relu(self.bn6(self.fc6(x)))
        x = F.relu(self.bn7(self.fc7(x)))

        x = self.fc_final(x)

        if re:
            return x.reshape(T, B, -1)
        else:
            return x

# ======================================================================================================================
