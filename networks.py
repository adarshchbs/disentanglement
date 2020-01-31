import torch
from torch import nn
import params


class encoder(nn.Module):
    def __init__(self, in_dim, z_dim):
        super( encoder, self ).__init__()
        self.layer_1 = nn.Linear(in_features = in_dim, out_features = z_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        self.batch_norm = nn.BatchNorm1d(z_dim)
        self.layer_2 = nn.Linear(in_features = z_dim, out_features = z_dim)

    def forward(self, input_):
        ret = self.layer_1(input_)
        ret = self.batch_norm(ret)
        ret = self.dropout(ret)
        ret = self.relu(ret)
        ret = self.layer_2(ret)
        return ret

class encoder_without_dropout(nn.Module):
    def __init__(self, in_dim, z_dim):
        super( encoder_without_dropout, self ).__init__()
        self.layer_1 = nn.Linear(in_features = in_dim, out_features = z_dim)
        self.relu = nn.ReLU()
        # self.dropout = nn.Dropout(0.5)
        self.batch_norm = nn.BatchNorm1d(z_dim)
        self.layer_2 = nn.Linear(in_features = z_dim, out_features = z_dim)

    def forward(self, input_):
        ret = self.layer_1(input_)
        ret = self.batch_norm(ret)
        # ret = self.dropout(ret)
        ret = self.relu(ret)
        ret = self.layer_2(ret)
        return ret


class decoder(nn.Module):
    def __init__(self, feat_dim):
        super( decoder, self ).__init__()
        self.layer_1 = nn.Linear(in_features = 2*feat_dim, out_features = feat_dim//2)
        self.relu = nn.ReLU()
        self.layer_2 = nn.Linear(in_features = feat_dim//2, out_features = params.x_dim )

    def forward(self, z_input, s_input):
        concat_feature = torch.cat( ( z_input, s_input ), 1 )
        ret = self.layer_1(concat_feature)
        ret = self.relu(ret)
        ret = self.layer_2(ret)
        return ret

class GradientReversal(torch.autograd.Function):

    @staticmethod
    def forward(ctx, inp):
        return inp.clone()
    @staticmethod
    def backward(ctx, grad_out):
        return -1 * grad_out.clone()


class adv_classifier(nn.Module):
    def __init__(self, feat_dim, num_classes):
        super( adv_classifier, self ).__init__()
        self.grad_rev = GradientReversal()
        self.layer_1 = nn.Linear(in_features = feat_dim, out_features = num_classes)
        # self.bn = nn.BatchNorm1d(num_features = num_classes)
        # self.relu = nn.ReLU()

    def forward(self, input_):
        ret = self.grad_rev.apply(input_)
        ret = self.layer_1(ret)
        # ret = self.bn(ret)
        # ret = self.relu(ret)
        return ret

