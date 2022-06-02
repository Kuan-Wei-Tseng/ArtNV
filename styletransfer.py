from PIL import Image
from pathlib import Path
import matplotlib.pyplot as plt

from utils.warp import *
from utils.loadimg import *
from utils.modules import *
from utils.VGGmodel import *

from tqdm import tqdm 

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import torchvision.transforms as transforms
import torchvision.models as models

import os
import sys
import copy
import glob
import argparse

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
parser = argparse.ArgumentParser(description='PyTorch Artistic View Synthesis')

# Training parameters
parser.add_argument('--num_steps', type=int, default=300, 
                     help='Number of optimization iterations')

parser.add_argument('--StylePath', type=str, default="../images/style/0.jpg", 
                     help='Path to the style image')        
parser.add_argument('--ContentDir', type=str, default="../images/content", 
                     help='Path to the content images')

parser.add_argument('--flow', action="store_true")
parser.add_argument('--FlowDir', type=str, default="../images/flow/", 
                     help='Path to the optical flow')
parser.add_argument('--MaskDir', type=str, default="../images/mask/", 
                     help='Path to the optical flow')
# parser.add_argument('--FlowDirect', type=int, default=1,
#                      help='0: Use bi-directional optical flow for warping\n\
#                            1: Use source to target optical flow\n\
#                           -1: Use target to source optical flow\n')           
parser.add_argument('--OutDir', type=str, default="../images/result/", 
                     help='Path to the save the results')

parser.add_argument('--style_weight'  , type=int, default=1000000)
parser.add_argument('--content_weight', type=int, default=1)
parser.add_argument('--flow_weight', type=int, default=3000)

FLAGS = parser.parse_args()
torch.manual_seed(0)

if __name__ == '__main__':

    style_img = image_loader(Path(FLAGS.StylePath))
    anchor_path  = sorted(Path(FLAGS.ContentDir).glob("*r.png"))
    n = len(anchor_path)

    for index in range(n):

        prefix = anchor_path[index].stem.split('-')[0]
        source_img = image_loader(anchor_path[index])
        input_img = [source_img]

        for j in range(12):
            target_img = image_loader(Path(FLAGS.ContentDir)/f"{prefix}-{j:02}s.png")
            input_img.append(target_img)

        input_img = torch.cat(input_img)

        # Load Optical Flow
        if FLAGS.flow:
            # Source to Target Flow
            flows_t_path = sorted(Path(FLAGS.FlowDir).glob("flows_t*"))
            flows_t = []
            for flow_path in flows_t_path:
                flows_t.append(torch.load(flow_path))

            # Source to Target Flow Mask
            masks_t_path = sorted(Path(FLAGS.MaskDir).glob("masks_t*"))
            masks_t = []
            for mask_path in masks_t_path:
                masks_t.append(torch.load(mask_path))

            # Target to Source Flow
            flowt_s_path = sorted(Path(FLAGS.FlowDir).glob("flowt_s*"))
            flowt_s = []
            for flow_path in flowt_s_path:
                flowt_s.append(torch.load(flow_path))           

            # Target to Source Flow Mask 
            maskt_s_path = sorted(Path(FLAGS.MaskDir).glob("maskt_s*"))
            maskt_s = []
            for mask_path in maskt_s_path:
                maskt_s.append(torch.load(mask_path))

            flows_t = torch.cat(flows_t)
            masks_t = torch.stack(masks_t)
            flowt_s = torch.cat(flowt_s)
            maskt_s = torch.stack(maskt_s)

            flowlosses = OpticalFlowLoss(flows_t, flowt_s, masks_t, maskt_s)
            
        # Build the style transfer model:
        model, style_losses, content_losses = get_style_model_and_losses(style_img, input_img)

        # Optimizer
        optimizer = optim.LBFGS([input_img.requires_grad_()])

        # Start Optimization:
        with tqdm(total=FLAGS.num_steps, file=sys.stdout) as iterator:
            i = [0]
            while i[0] <= FLAGS.num_steps-20:
                def closure():
                    input_img.data.clamp_(0, 1)
                    optimizer.zero_grad()
                    model(input_img)

                    style_score = 0 
                    content_score = 0
                    flow_loss= torch.tensor([0])
                    
                    for sl in style_losses:
                        style_score += sl.loss
                    for cl in content_losses:
                        content_score += cl.loss
                
                    style_loss = FLAGS.style_weight * style_score
                    content_loss = FLAGS.content_weight * content_score

                    loss = style_loss + content_loss
                    
                    if FLAGS.flow:
                        flow_loss = FLAGS.flow_weight * flowlosses(input_img[0], input_img[1:])
                        loss += flow_loss

                    iterator.set_postfix_str(f"style_loss:{style_loss.item():4f}, " +
                                             f"content_loss:{content_loss.item():4f}, "+
                                             f"flow_loss:{flow_loss.item():4f}")
                    i[0] += 1
                    iterator.update()
                    loss.backward()
                    return loss

                optimizer.step(closure)
            



        # # Start Optimization:
        # print('Optimizing...')
        # run = [0]
        # while run[0] <= FLAGS.num_steps:
        #     def closure():
        #         # Correct the values of updated input image
        #         input_img.data.clamp_(0, 1)
        #         optimizer.zero_grad()
        #         model(input_img)

        #         style_score = 0
        #         content_score = 0
        #         flow_score = 0
        #         depth_score = 0
        #         disparity_score = 0 

        #         for sl in style_losses:
        #             style_score += sl.loss
        #         for cl in content_losses:
        #             content_score += cl.loss

        #         style_score *= FLAGS.style_weight
        #         content_score *= FLAGS.content_weight

        #         loss = style_score + content_score 

        #         if FLAGS.depth:
        #             depth_score += FLAGS.depth_weight * depthloss(input_img[0], input_img[1:])
        #             loss += depth_score
        #         else:
        #             depth_score = torch.tensor([0])

        #         if FLAGS.flow:
        #             flow_score += FLAGS.flow_weight * flowloss(input_img[0], input_img[1:])
        #             loss += flow_score            
        #         else:
        #             flow_score = torch.tensor([0])


        #         loss.backward()

        #         run[0] += 1

        #         if run[0] % 10 == 0:
        #             print(f"Run {run}")
        #             print(f"Style Loss :  {style_score.item():4f}\
        #                     Content Loss: {content_score.item():4f}\
        #                     Depth Loss: {depth_score.item():4f}\
        #                     Flow Loss:{flow_score.item()}")

        #         return loss

        #     optimizer.step(closure)

        # output = input_img.data.clamp_(0, 1)
        # B, C, H, W = output.shape

        # for i in range(1,B):
        #     image_writer(output[i,:,:,:], Path(FLAGS.OutDir)/f"{FLAGS.Method}"/f"{index+1:05}-{i:02}s-{styleID}.png")

        # os.system(f"ffmpeg -y -f image2 -framerate 10 -i {Path(FLAGS.OutDir)}/{FLAGS.Method}/{index+1:05}-%2ds-{styleID}.png -s 512x512 \
        # -vcodec libx264 -crf 25 -pix_fmt yuv420p {Path(FLAGS.OutDir)}/{FLAGS.Method}/video/{index+1:05}-{styleID}.mp4")


