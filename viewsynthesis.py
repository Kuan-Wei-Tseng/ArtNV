import os
import sys
import argparse
os.environ['DEBUG'] = '0'
sys.path.append('./thirdparties/synsin')

import cv2
import quaternion
import numpy as np

import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image

from models.networks.sync_batchnorm import convert_model
from models.base_model import BaseModel
from options.options import get_model

torch.backends.cudnn.enabled = True

parser = argparse.ArgumentParser(description='PyTorch Artistic View Synthesis')

# Training parameters
parser.add_argument('--MODEL_PATH', type=str, default='./thirdparties/synsin/modelcheckpoints/realestate/zbufferpts.pth', 
                     help='View synthesis pretrained model path')
parser.add_argument('--SeqName', type=str, default='Temple', 
                     help='Tanks and Temple data sequence')
FLAGS = parser.parse_args()


# Parameters
BATCH_SIZE = 1

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

import glob
ims = sorted(glob.glob(f"./images/tank_and_temple/{FLAGS.SeqName}/content/*.jpg"))

print(f"./images/tank_and_temple/{FLAGS.SeqName}/content/*.jpg")

for ImageNo, im_path in enumerate(ims):
    im = Image.open(im_path)
    im = transform(im)

    # # Target Pose
    # phi, theta, ksi, tx, ty, tz = [0, 0, 0, 0.1, 0, 0]
    # RT = torch.eye(4).unsqueeze(0)
    # RT[0,0:3,0:3] = torch.Tensor(quaternion.as_rotation_matrix(quaternion.from_rotation_vector([phi, theta, ksi])))
    # RT[0,0:3,3] = torch.Tensor([tx, ty, tz])

    # phi, theta, ksi, tx, ty, tz = [0, 0, 0, -0.1, 0, 0]
    # RT2 = torch.eye(4).unsqueeze(0)
    # RT2[0,0:3,0:3] = torch.Tensor(quaternion.as_rotation_matrix(quaternion.from_rotation_vector([phi, theta, ksi])))
    # RT2[0,0:3,3] = torch.Tensor([tx, ty, tz])

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
    RT = torch.eye(4).unsqueeze(0)
    RT[0,0:3,0:3] = torch.Tensor(quaternion.as_rotation_matrix(quaternion.from_rotation_vector([phi, theta, ksi])))
    RT[0,0:3,3] = torch.Tensor([tx, ty, tz])

    with torch.no_grad():
        pred_imgs = model_to_test.model.module.forward_angle(batch, [RT])
        depth = nn.Sigmoid()(model_to_test.model.module.pts_regressor(batch['images'][0].cuda()))

        # import cv2
        im2 = pred_imgs[0].squeeze().cpu().permute(1,2,0).numpy() * 0.5 + 0.5
        cv2.imwrite(f"{os.path.splitext(im_path)[0]}-r.png", cv2.resize(im2*255,(512,512)))


    for dt in range(0,12):
        r = 0.1
        tx = r*np.cos(2*np.pi*dt/12)
        ty = r*np.sin(2*np.pi*dt/12)
        # theta = 0.1*dt/11 - 0.05
        
        
        # Target Pose   
        RT = torch.eye(4).unsqueeze(0)
        RT[0,0:3,0:3] = torch.Tensor(quaternion.as_rotation_matrix(quaternion.from_rotation_vector([phi, theta, ksi])))
        RT[0,0:3,3] = torch.Tensor([tx, ty, tz])

        with torch.no_grad():
            pred_imgs = model_to_test.model.module.forward_angle(batch, [RT])
            depth = nn.Sigmoid()(model_to_test.model.module.pts_regressor(batch['images'][0].cuda()))

            # import cv2
            im2 = pred_imgs[0].squeeze().cpu().permute(1,2,0).numpy() * 0.5 + 0.5
            cv2.imwrite(f"{os.path.splitext(im_path)[0]}-{dt:02}s.png", cv2.resize(im2*255,(512,512)))


    RT = torch.eye(4).unsqueeze(0)


    os.system(f"ffmpeg -hide_banner -y -f image2 -framerate 10 -i {os.path.splitext(im_path)[0]}-%2ds.png -s 512x512 \
    -vcodec libx264 -crf 25 -pix_fmt yuv420p ./images/tank_and_temple/{FLAGS.SeqName}/video/{ImageNo:05}.mp4")

    
        # cv2.imshow("windows", cv2.resize(im1,(1200,1200)))
        # cv2.waitKey(0)
        # cv2.imshow("windows", cv2.resize(im2,(1200,1200)))
        # cv2.waitKey(0)    

        # resize = transforms.Resize([512,512])
        # depth = resize(depth.squeeze(0))

        # torch.save(depth, "images/depth.depth")

        # dimg = depth.permute(1,2,0).cpu().numpy()
        # dimg = dimg*255/dimg.max()
        # dimg = cv2.applyColorMap(dimg.astype(np.uint8), cv2.COLORMAP_JET)

        # cv2.imshow("window",cv2.resize(dimg,(1200,1200)))
        # cv2.waitKey(0)

        # cv2.imwrite("images/depth.png", cv2.resize(dimg,(512,512)))
        

        #torch.save(depth,f"../images/{method}/depth/{seqname}/depth.depth")

