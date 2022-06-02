from PIL import Image
import matplotlib.pyplot as plt

from utils.loadimg import *
from utils.warp import WarpOpticalFlow
from utils.warp import WarpDisparity

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import torchvision.transforms as transforms
import torchvision.models as models

def gram_matrix(input):
    n, c, h, w = input.size() 
    # n = batch size
    # c = channels, number of featue map
    # (h,w) = height and width of feature map

    features = input.view(n, c, h*w)
    # resise F_XL into \hat F_XL

    G = torch.bmm(features, torch.transpose(features, dim0 = 1,dim1 = 2))  
    # compute the gram product

    # we 'normalize' the values of the gram matrix
    # by dividing by the number of element in each feature maps.
    return G.div(c*h*w)

class ContentLoss(nn.Module):

    def __init__(self, target,):
        super(ContentLoss, self).__init__()
        # we 'detach' the target content from the tree used
        # to dynamically compute the gradient: this is a stated value,
        # not a variable. Otherwise the forward method of the criterion
        # will throw an error.
        self.target = target.detach()

    def forward(self, input):
        tmp_input = input.expand(self.target.shape)
        self.loss = F.mse_loss(tmp_input, self.target)
        return input

class StyleLoss(nn.Module):

    def __init__(self, target_feature):
        super(StyleLoss, self).__init__()
        self.target = gram_matrix(target_feature).detach()

    def forward(self, input):

        G = gram_matrix(input)
        tmp_target = self.target.expand(input.size(0),-1,-1)
        self.loss = F.mse_loss(G, tmp_target)
        return input


class OpticalFlowLoss(nn.Module):

    def __init__(self, flows_t, flowt_s, masks_t, maskt_s, bidir=False):
        super(OpticalFlowLoss, self).__init__()
        self.flows_t = flows_t.detach()
        self.flowt_s = flowt_s.detach()
        self.bidir = bidir

        B, C, H, W = self.flowt_s.shape

        # out-of-boundary check
        self.occ_maskt_s = WarpOpticalFlow(mask_loader().expand((B,3,H,W)),self.flowt_s).detach()
        self.occ_masks_t = WarpOpticalFlow(mask_loader().expand((B,3,H,W)),self.flows_t).detach()

        self.occ_maskt_s *= (1-maskt_s)
        self.occ_maskt_s *= (1-masks_t)

        #print(self.occ_maskt_s.shape)

    def forward(self, source, target):
        """
        source: original view 1*C*H*W
        target: novel views N*C*H*W
        """
        # target -> source
        s_warped = WarpOpticalFlow(target, self.flowt_s)
        difft_s = torch.abs(s_warped - source.detach().expand(target.shape))
        difft_s = difft_s * self.occ_maskt_s
        
        # source -> target

        if self.bidir:
            t_warped = WarpOpticalFlow(source.expand(target.shape), self.flows_t)
            diffs_t = torch.abs(t_warped - target.detach())
            diffs_t = diffs_t * self.occ_masks_t
            diff = torch.cat([difft_s, diffs_t])
        else:   
            diff = difft_s

        self.loss = F.mse_loss(diff, torch.zeros(diff.shape).to(device))

        return self.loss
