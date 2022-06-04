import os
import sys
os.environ['DEBUG'] = '0'
sys.path.append('./synsin')

import cv2
import glob
import argparse
import quaternion
import numpy as np

import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
from pathlib import Path

from models.networks.sync_batchnorm import convert_model
from models.base_model import BaseModel
from options.options import get_model

torch.backends.cudnn.enabled = True

parser = argparse.ArgumentParser(description='PyTorch Artistic View Synthesis')

# Training parameters
parser.add_argument('--MODEL_PATH', type=str, default='synsin/modelcheckpoints/realestate/zbufferpts.pth', 
                     help='View synthesis pretrained model path')
parser.add_argument('--RawDir', type=str, default="../images/content/raw", 
                     help='Path to raw inputs')
parser.add_argument('--OutDir', type=str, default="../images/content/", 
                     help='Path to the save the results')

FLAGS = parser.parse_args()

# Parameters
BATCH_SIZE = 1

def output2img(img_tensor, path):
    im2 = img_tensor.squeeze().cpu().permute(1,2,0).numpy()
    im2 = (im2*0.5 + 0.5)*255
    im2 = transforms.ToPILImage()(im2.astype(np.uint8))
    im2 = im2.resize((512, 512))
    im2.save(path)

# Load the model
opts = torch.load(FLAGS.MODEL_PATH)['opts']
opts.render_ids = [1]
model = get_model(opts).cuda()

torch_devices = [int(gpu_id.strip()) for gpu_id in opts.gpu_ids.split(",")]
device = 'cuda:' + str(torch_devices[0])

if 'sync' in opts.norm_G:
    model = convert_model(model)
    model = nn.DataParallel(model, torch_devices[0:1]).cuda()
else:
    model = nn.DataParallel(model, torch_devices[0:1]).cuda()

#  Load the original model to be tested
model_to_test = BaseModel(model, opts)
model_to_test.load_state_dict(torch.load(FLAGS.MODEL_PATH)['state_dict'])
model_to_test.eval()

# Load Image
transform = transforms.Compose([
    transforms.Resize((256,256)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

ims = sorted(Path(FLAGS.RawDir).glob('*'))

for index, im_path in enumerate(ims):

    prefix = im_path.stem.split('-')[0]

    im = Image.open(im_path)
    im = transform(im)

    # Input Batch
    batch = {
        'images' : [im.unsqueeze(0)],
        'cameras' : [{
            'K' : torch.eye(4).unsqueeze(0),
            'Kinv' : torch.eye(4).unsqueeze(0)
        }]
    }
    # Generate a new view at the new transformation

    phi, theta, ksi, tx, ty, tz = [0, 0, 0, 0, 0, 0]

    RTs = [torch.eye(4).unsqueeze(0)]
    r = 0.05

    for i in range(14):
        RT = torch.eye(4).unsqueeze(0)
        RT[0,0:3,0:3] = torch.Tensor(quaternion.as_rotation_matrix(quaternion.from_rotation_vector([phi, theta, ksi])))
        RT[0,0:3,3] = torch.Tensor([r*np.sin(2*np.pi*i/7), 0, i*0.03])
        #RT[0,0:3,3] = torch.Tensor([r*np.cos(2*np.pi*i/12), r*np.sin(2*np.pi*i/12), tz])
        RTs.append(RT)

    with torch.no_grad():
        pred_imgs = model_to_test.model.module.forward_angle(batch, RTs)
        depth = nn.Sigmoid()(model_to_test.model.module.pts_regressor(batch['images'][0].cuda()))

    output2img(pred_imgs[0], Path(FLAGS.OutDir)/"synthesized"/f"{prefix}-r.png")

    for i in range(1,15):
        output2img(pred_imgs[i], Path(FLAGS.OutDir)/"synthesized"/f"{prefix}-{i-1:02}.png")

    os.system(f"ffmpeg -hide_banner -y -f image2 -framerate 10 -i {Path(FLAGS.OutDir)}/synthesized/{prefix}-%2d.png -s 512x512 \
    -vcodec libx264 -crf 25 -pix_fmt yuv420p {Path(FLAGS.OutDir)}/video/{prefix}.mp4")


