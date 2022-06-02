import glob
import torch
import torch.nn as nn
import torch.nn.functional as F

def WarpOpticalFlow(x, flow=None):

    """
    warp an image/tensor (im2) back to im1, according to the optical flow
    x: [B, C, H, W] (im2)
    flow: [B, 2, H, W] flow
    """

    B, C, H, W = x.size()
    # mesh grid 
    xx = torch.arange(0, W).view(1,-1).repeat(H,1)
    yy = torch.arange(0, H).view(-1,1).repeat(1,W)
    xx = xx.view(1,1,H,W).repeat(B,1,1,1)
    yy = yy.view(1,1,H,W).repeat(B,1,1,1)
    grid = torch.cat((xx,yy),1).float()

    if x.is_cuda:
        grid = grid.cuda()

    vgrid = grid + flow

    # scale grid to [-1,1] 
    vgrid[:,0,:,:] = 2.0*vgrid[:,0,:,:].clone() / max(W-1,1)-1.0
    vgrid[:,1,:,:] = 2.0*vgrid[:,1,:,:].clone() / max(H-1,1)-1.0

    vgrid = vgrid.permute(0,2,3,1)        
    output = nn.functional.grid_sample(x, vgrid)
    mask = torch.ones(x.size()).cuda()
    mask = nn.functional.grid_sample(mask, vgrid)

    # if W==128:
        # np.save('mask.npy', mask.cpu().data.numpy())
        # np.save('warp.npy', output.cpu().data.numpy())
    
    mask[mask<0.9999] = 0
    mask[mask>0] = 1
    
    return output*mask

def WarpDisparity(img, depth, baseline, fl, scale=1.356122448979592):
    """
    Args:
        img     : input image [B, 3, H, W]
        depth   : depth map of the source image [B, H, W]
        baseline: steroscopic camera baseline [B,1]
        fl      : focal length
    """
    B, C, H, W = img.size()

    # Create Empty Mesh Grid
    xx = torch.arange(0, W).view(1,-1).repeat(H,1)
    yy = torch.arange(0, H).view(-1,1).repeat(1,W)
    xx = xx.view(1,1,H,W).repeat(B,1,1,1)
    yy = yy.view(1,1,H,W).repeat(B,1,1,1)
    # grid shape: [B,2 H,W]
    vgrid = torch.cat((xx,yy),1).float().cuda()

    bf = baseline * fl * scale
    disp = bf.view(-1,1,1,1) / depth

    vgrid[:,0] += disp.squeeze(1)

    # scale grid to [-1,1] 
    vgrid[:,0,:,:] = 2.0*vgrid[:,0,:,:].clone() / max(W-1,1)-1.0
    vgrid[:,1,:,:] = 2.0*vgrid[:,1,:,:].clone() / max(H-1,1)-1.0

    vgrid = vgrid.permute(0,2,3,1)        
    output = nn.functional.grid_sample(img, vgrid)
    mask = torch.ones(img.size()).cuda()
    mask = nn.functional.grid_sample(mask, vgrid)

    mask[mask<0.9999] = 0
    mask[mask>0] = 1
    
    return output*mask

# def WarpDepth(img, depth, baseline, fl=1, scale=1):

#     """
#     Args:
#         img     : input image [B, 3, H, W]
#         depth   : depth map of the source image [B, H, W]
#         baseline: steroscopic camera baseline [B,1]
#         fl      : focal length
#     """
#     B, C, H, W = img.size()

#     # Create Empty Mesh Grid
#     xx = torch.arange(0, W).view(1,-1).repeat(H,1)
#     yy = torch.arange(0, H).view(-1,1).repeat(1,W)
#     xx = xx.view(1,1,H,W).repeat(B,1,1,1)
#     yy = yy.view(1,1,H,W).repeat(B,1,1,1)
#     # grid shape: [B,2 H,W]
#     vgrid = torch.cat((xx,yy),1).float().cuda()

#     bf = -baseline * fl * scale
#     disp = bf.view(-1,1,1,1) / depth

#     vgrid[:,0] += disp.squeeze(1)

#     # scale grid to [-1,1] 
#     vgrid[:,0,:,:] = 2.0*vgrid[:,0,:,:].clone() / max(W-1,1)-1.0
#     vgrid[:,1,:,:] = 2.0*vgrid[:,1,:,:].clone() / max(H-1,1)-1.0

#     vgrid = vgrid.permute(0,2,3,1)        
#     output = nn.functional.grid_sample(img, vgrid)
#     mask = torch.ones(img.size()).cuda()
#     mask = nn.functional.grid_sample(mask, vgrid)

#     mask[mask<0.9999] = 0
#     mask[mask>0] = 1
    
#     return output*mask
