import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from utils.warp import *
from utils.loadimg import *
from utils.modules import *

import torchvision.transforms as transforms
import torchvision.models as models

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Default Model Parameters
content_layers_default = ['conv_4']
style_layers_default = ['conv_1', 'conv_2', 'conv_3', 'conv_4', 'conv_5']
"""
CNN Normalization
VGG networks are trained on images with each channel normalized by 
mean=[0.485, 0.456, 0.406], and std=[0.229, 0.224, 0.225].

"""
cnn_normalization_mean = torch.tensor([0.485, 0.456, 0.406]).to(device)
cnn_normalization_std = torch.tensor([0.229, 0.224, 0.225]).to(device)

class Normalization(nn.Module):
    def __init__(self, mean, std):
        super(Normalization, self).__init__()
        self.mean = mean.clone().detach().view(-1, 1, 1)
        self.std  = std.clone().detach().view(-1, 1, 1)

    def forward(self, img):
        # normalize img
        return (img - self.mean) / self.std

def get_style_model_and_losses(style_img, content_img, 
                               content_layers = content_layers_default,
                               style_layers   = style_layers_default):

    cnn = models.vgg19(pretrained=True).features.to(device).eval()
    cnn = copy.deepcopy(cnn)

    # normalization module
    normalization  = Normalization(cnn_normalization_mean, cnn_normalization_std).to(device)

    content_losses = []
    style_losses   = []

    # Create a new nn sequential from pretrained VGG network:
    model = nn.Sequential(normalization)

    i = 0

    for layer in cnn.children():
        if isinstance(layer, nn.Conv2d):
            i += 1
            name = 'conv_{}'.format(i)
        elif isinstance(layer, nn.ReLU):
            name = 'relu_{}'.format(i)
            layer = nn.ReLU(inplace=False)
            """
            The in-place version doesn't play very nicely with the ContentLoss
            and StyleLoss we insert below. So we replace with out-of-place
            ones here.
            """
        elif isinstance(layer, nn.MaxPool2d):
            name = 'pool_{}'.format(i)
        elif isinstance(layer, nn.BatchNorm2d):
            name = 'bn_{}'.format(i)
        else:
            raise RuntimeError('Unrecognized layer: {}'.formnum_stepsat(layer.__class__.__name__))

        model.add_module(name, layer)

        if name in content_layers:
            # add content loss:
            target = model(content_img).detach()
            content_loss = ContentLoss(target)
            model.add_module("content_loss_{}".format(i), content_loss)
            content_losses.append(content_loss)

        if name in style_layers:
            # add style loss:
            target_feature = model(style_img).detach()
            style_loss = StyleLoss(target_feature)
            model.add_module("style_loss_{}".format(i), style_loss)
            style_losses.append(style_loss)

    # trim off the layers after the last content and style losses
    for i in range(len(model) - 1, -1, -1):
        if isinstance(model[i], ContentLoss) or isinstance(model[i], StyleLoss):
            break

    model = model[:(i + 1)]

    return model, style_losses, content_losses
