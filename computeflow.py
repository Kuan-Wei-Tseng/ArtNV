import os
import cv2
import glob
import argparse
import numpy as np

import torch
import torch.nn.functional as F

from tqdm import tqdm 
from PIL import Image
from pathlib import Path

import sys
sys.path.append('RAFT/core')

from raft import RAFT
from utils import flow_viz
from utils import frame_utils
from utils.utils import InputPadder

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_image(imfile):
    img = np.array(Image.open(imfile).resize((512,512))).astype(np.uint8)
    img = torch.from_numpy(img).permute(2, 0, 1).float()
    return img[None].to(device)

def viz(img, flo):
    img = img[0].permute(1,2,0).cpu().numpy()
    flo = flo[0].permute(1,2,0).cpu().numpy()
    
    # map flow to rgb image
    flo = flow_viz.flow_to_image(flo)
    img_flo = np.concatenate([img, flo], axis=0)

    cv2.imshow('image', img_flo[:, :, [2,1,0]]/255.0)
    cv2.waitKey(0)

def saveflow(flow, fname):
    flow = flow[0].permute(1,2,0).cpu().numpy()
    flow = flow_viz.flow_to_image(flow)
    cv2.imwrite(fname, flow)

def get_consistency_map(flow12, flow21, consistency_thresh=1.0):
    flow21_warped = F.grid_sample(flow21, normalize_for_grid_sample(flow12))
    diff = flow21_warped - get_grid(flow12)
    diff = torch.sqrt(torch.sum(torch.pow(diff, 2), 1))
    mask1 = diff < consistency_thresh
    mask1 = mask1.float()
    return mask1

def normalize_for_grid_sample(flow):
    warp_map = flow.clone().detach()
    H, W = warp_map.size()[-2:]
    warp_map[:, 0, :, :] = warp_map[:, 0, :, :] / (W - 1) * 2 - 1
    warp_map[:, 1, :, :] = warp_map[:, 1, :, :] / (H - 1) * 2 - 1
    return warp_map.permute(0, 2, 3, 1)

def get_grid(tensor, homogeneous=False):
    B, _, H, W = tensor.size()
    xx = torch.arange(0, W).view(1, -1).repeat(H, 1)
    yy = torch.arange(0, H).view(-1, 1).repeat(1, W)
    xx = xx.view(1, 1, H, W).repeat(B, 1, 1, 1)
    yy = yy.view(1, 1, H, W).repeat(B, 1, 1, 1)
    grid = torch.cat((xx, yy), 1).float().to(tensor.device)
    if homogeneous:
        ones = torch.ones(B, 1, H, W).float().to(tensor.device)
        grid = torch.cat((grid, ones), 1)
    return grid

def demo(args):
    model = torch.nn.DataParallel(RAFT(args))
    model.load_state_dict(torch.load(args.model))
    model = model.module
    model.to(device)
    model.eval()

    with torch.no_grad():

        anchor_path  = sorted(Path(args.ContentDir).glob("*r.png"))
        n = len(anchor_path)

        for index in range(n):
            prefix = anchor_path[index].stem.split('-')[0]
            source_img = load_image(anchor_path[index])

            for j in tqdm(range(args.Views)):
                target_img = load_image(Path(args.ContentDir)/f"{prefix}-{j:02}.png")
                _, flows_t = model(target_img, source_img, iters=20, test_mode=True)
                _, flowt_s = model(source_img, target_img, iters=20, test_mode=True)

                torch.save(flows_t, Path(args.FlowDir)/f"flows_t{prefix}-{j:02}.pt")
                torch.save(flowt_s, Path(args.FlowDir)/f"flowt_s{prefix}-{j:02}.pt")

                # Compute occulsion masks:
                masks_t = get_consistency_map(flows_t, flowt_s)
                maskt_s = get_consistency_map(flowt_s, flows_t)

                torch.save(masks_t, Path(args.MaskDir)/f"masks_t{prefix}-{j:02}.pt")
                torch.save(maskt_s, Path(args.MaskDir)/f"maskt_s{prefix}-{j:02}.pt")     

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default="RAFT/models/raft-things.pth", help="restore checkpoint")
    parser.add_argument('--path', help="dataset for evaluation")
    parser.add_argument('--small', action='store_true', help='use small model')
    parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')
    parser.add_argument('--alternate_corr', action='store_true', help='use efficent correlation implementation')
    parser.add_argument('--Views', type=int, default=14, help='Number of Novel Views')
    parser.add_argument('--ContentDir', type=str, default="../images/content/synthesized", help='Path to the content images')
    parser.add_argument('--FlowDir', type=str, default="../images/flow", help='Path to save the optical flows')
    parser.add_argument('--MaskDir', type=str, default="../images/mask", help='Path to save the optical flows')
    args = parser.parse_args()

    demo(args)
