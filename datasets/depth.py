import torch
import cv2
import numpy as np
import imageio
import matplotlib.pyplot as plt
import pathlib
import argparse

parser = argparse.ArgumentParser(
	description='Initialize a scene coordinate regression network.',
	formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('scene', help='name of a scene in the dataset folder')

opt = parser.parse_args()

scene = pathlib.Path(opt.scene)

train_rgb = scene/'train'/'rgb'
train_depth = scene/'train'/'depth'
train_depth.mkdir()
test_rgb = scene/'test'/'rgb'
test_depth = scene/'test'/'depth'
test_depth.mkdir()

use_large_model = True

if use_large_model:
    midas = torch.hub.load("intel-isl/MiDaS", "MiDaS")
else:
    midas = torch.hub.load("intel-isl/MiDaS", "MiDaS_small")

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
midas.to(device)
midas.eval()

midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")

if use_large_model:
    transform = midas_transforms.default_transform
else:
    transform = midas_transforms.small_transform

for rgb, depth in [(train_rgb, train_depth), (test_rgb, test_depth)]:
    for img_path in rgb.glob('frame*'):
        img = cv2.imread(str(img_path))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        input_batch = transform(img).to(device)

        with torch.no_grad():
            prediction = midas(input_batch)

            prediction = torch.nn.functional.interpolate(
                prediction.unsqueeze(1),
                size=img.shape[:2],
                mode="bicubic",
                align_corners=False,
            ).squeeze()
            
        output = prediction.cpu().numpy()

        midas_depth = 4000000 / output
        midas_depth = midas_depth.astype(np.uint16)
        fname = img_path.name
        depth_filename = fname.split('.')[0] + '.depth.png'
        save_path = depth/depth_filename
        print('saving depth file to {}'.format(str(save_path)))
        imageio.imwrite(str(save_path), midas_depth)
