from warp import *
from loadimg import *
import cv2

left_img = image_loader("../images/test0.png")
#right_img = image_loader("")
depth = torch.load("../images/depth.depth")/200
baseline = torch.tensor([0.2]).cuda()

right_img = WarpDepth(left_img, depth, baseline)
image_writer(right_img,"../images/test1_warped.png")

# dimg = depth.permute(1,2,0).cpu().numpy()
# dimg = dimg*255/dimg.max()
# dimg = cv2.applyColorMap(dimg.astype(np.uint8), cv2.COLORMAP_JET)

# cv2.imshow("window",cv2.resize(dimg,(1200,1200)))
# cv2.waitKey(0)