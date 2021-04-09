import torch
import cv2
import dsacstar
from network import Network
from skimage import io
from torchvision import transforms
from matplotlib import pyplot as plt

scene = 'nclt'
weightsDir = 'network_output/nclt_trial_v2_e2e.pth'
imageDir = 'datasets/nclt/train/rgb/2012-01-08_1326031373334462.color.png'

# hyperparameters
hypotheses = 64 # number of hypotheses, i.e. number of RANSAC iterations
threshold = 10 # inlier threshold in pixels (RGB) or centimeters (RGB-D)
inlieralpha = 100 # alpha parameter of the soft inlier count; controls the softness of the hypotheses score distribution; lower means softer
maxpixelerror = 100 # maximum reprojection (RGB, in px) or 3D distance (RGB-D, in cm) error when checking pose consistency towards all measurements; error is clamped to this value for stability
mode = 1 # test mode: 1 = RGB, 2 = RGB-D

# dataset parameters
focal_length = 100


# load weights
network = Network(torch.zeros((3)), False)
network.load_state_dict(torch.load(weightsDir))
network = network.cuda()
network.eval()

#define image processing elements
image_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize(int(480)),
    transforms.Grayscale(),
    transforms.ColorJitter(brightness=0.1, contrast=0.1),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.4],
        std=[0.25]
        )
])

# load image
image = io.imread(imageDir)
f_scale_factor = 1
image = image_transform(image)
image = image.unsqueeze(0)
image = image.cuda()

# predict
scene_coordinates = network(image)
scene_coordinates = scene_coordinates.cpu()
out_pose = torch.zeros((4, 4))
if mode < 2:
    dsacstar.forward_rgb(
        scene_coordinates, 
        out_pose, 
        hypotheses, 
        threshold,
        focal_length, 
        float(image.size(3) / 2), #principal point assumed in image center
        float(image.size(2) / 2), 
        inlieralpha,
        maxpixelerror,
        network.OUTPUT_SUBSAMPLE)
else:
    pass
    # pose from RGB-D

out_pose = out_pose.inverse()
print(out_pose)