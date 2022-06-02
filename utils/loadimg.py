import torch
import torchvision.transforms as transforms

from PIL import Image
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
imsize = 512 if torch.cuda.is_available() else 128
loader = transforms.Compose([
         transforms.Resize(imsize),
         transforms.ToTensor()]) 

def image_loader(image_name):
    image = Image.open(image_name)
    image = image.resize((512, 512))
    image = loader(image).unsqueeze(0)
    return image.to(device, torch.float)

def mask_loader():
    image = Image.new("1",(512,512),color=1)
    image = loader(image).unsqueeze(0)
    return image.to(device, torch.float)

def image_writer(tensor, file_name):
    image = tensor.cpu().clone()
    image = image.squeeze(0)
    image = transforms.ToPILImage()(image)
    image.save(file_name)
