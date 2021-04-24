from __future__ import absolute_import, division, print_function
import sys
import numpy as np
import torch
#from layers import disp_to_depth, ScaleRecovery
import matplotlib.pyplot as plt
import cv2
import math
import torch.nn as nn
import torch.nn.functional as F
import imageio
from tqdm import tqdm
import os

def disp_to_depth(disp, min_depth, max_depth):
    """Convert network's sigmoid output into depth prediction
    The formula for this conversion is given in the 'additional considerations'
    section of the paper.
    """
    min_disp = 1 / max_depth
    max_disp = 1 / min_depth
    scaled_disp = min_disp + (max_disp - min_disp) * disp
    depth = 1 / scaled_disp
    return scaled_disp, depth

class BackprojectDepth(nn.Module):
    """Layer to transform a depth image into a point cloud
    """
    def __init__(self, batch_size, height, width):
        super(BackprojectDepth, self).__init__()

        self.batch_size = batch_size
        self.height = height
        self.width = width

        meshgrid = np.meshgrid(range(self.width), range(self.height), indexing='xy')
        self.id_coords = np.stack(meshgrid, axis=0).astype(np.float32)
        self.id_coords = nn.Parameter(
            torch.from_numpy(self.id_coords),
            requires_grad=False)

        self.ones = nn.Parameter(
            torch.ones(self.batch_size, 1, self.height * self.width),
                       requires_grad=False)

        self.pix_coords = torch.unsqueeze(torch.stack(
            [self.id_coords[0].view(-1), self.id_coords[1].view(-1)], 0), 0)
        self.pix_coords = self.pix_coords.repeat(batch_size, 1, 1)
        self.pix_coords = nn.Parameter(
            torch.cat([self.pix_coords, self.ones], 1), requires_grad=False)

    def forward(self, depth, inv_K):
        cam_points = torch.matmul(inv_K[:, :3, :3], self.pix_coords)
        cam_points = depth.view(self.batch_size, 1, -1) * cam_points
        cam_points = torch.cat([cam_points, self.ones], 1).reshape(
            self.batch_size, 4, self.height, self.width)

        return cam_points


class ScaleRecovery(nn.Module):
    """Layer to estimate scale through dense geometrical constrain
    """
    def __init__(self, batch_size, height, width):
        super(ScaleRecovery, self).__init__()
        self.backproject_depth = BackprojectDepth(batch_size, height, width)
        self.batch_size = batch_size
        self.height = height
        self.width = width

    # derived from https://github.com/zhenheny/LEGO
    def get_surface_normal(self, cam_points, nei=1):
        cam_points_ctr  = cam_points[:, :-1, nei:-nei, nei:-nei]
        cam_points_x0   = cam_points[:, :-1, nei:-nei, 0:-(2*nei)]
        cam_points_y0   = cam_points[:, :-1, 0:-(2*nei), nei:-nei]
        cam_points_x1   = cam_points[:, :-1, nei:-nei, 2*nei:]
        cam_points_y1   = cam_points[:, :-1, 2*nei:, nei:-nei]
        cam_points_x0y0 = cam_points[:, :-1, 0:-(2*nei), 0:-(2*nei)]
        cam_points_x0y1 = cam_points[:, :-1, 2*nei:, 0:-(2*nei)]
        cam_points_x1y0 = cam_points[:, :-1, 0:-(2*nei), 2*nei:]
        cam_points_x1y1 = cam_points[:, :-1, 2*nei:, 2*nei:]

        vector_x0   = cam_points_x0   - cam_points_ctr
        vector_y0   = cam_points_y0   - cam_points_ctr
        vector_x1   = cam_points_x1   - cam_points_ctr
        vector_y1   = cam_points_y1   - cam_points_ctr
        vector_x0y0 = cam_points_x0y0 - cam_points_ctr
        vector_x0y1 = cam_points_x0y1 - cam_points_ctr
        vector_x1y0 = cam_points_x1y0 - cam_points_ctr
        vector_x1y1 = cam_points_x1y1 - cam_points_ctr

        normal_0 = F.normalize(torch.cross(vector_x0,   vector_y0,   dim=1), dim=1).unsqueeze(0)
        normal_1 = F.normalize(torch.cross(vector_x1,   vector_y1,   dim=1), dim=1).unsqueeze(0)
        normal_2 = F.normalize(torch.cross(vector_x0y0, vector_x0y1, dim=1), dim=1).unsqueeze(0)
        normal_3 = F.normalize(torch.cross(vector_x1y0, vector_x1y1, dim=1), dim=1).unsqueeze(0)

        normals = torch.cat((normal_0, normal_1, normal_2, normal_3), dim=0).mean(0)
        normals = F.normalize(normals, dim=1)

        refl = nn.ReflectionPad2d(nei)
        normals = refl(normals)

        return normals

    def get_ground_mask(self, cam_points, normal_map, threshold=10):
        b, _, h, w = normal_map.size()
        cos = nn.CosineSimilarity(dim=1, eps=1e-6)

        threshold = math.cos(math.radians(threshold))
        ones, zeros = torch.ones(b, 1, h, w).cuda(), torch.zeros(b, 1, h, w).cuda()
        vertical = torch.cat((zeros, ones, zeros), dim=1)

        cosine_sim = cos(normal_map, vertical).unsqueeze(1)
        vertical_mask = (cosine_sim > threshold) | (cosine_sim < -threshold)

        y = cam_points[:,1,:,:].unsqueeze(1)
        ground_mask = vertical_mask.masked_fill(y <= 0, False)

        return ground_mask

    def forward(self, depth, K, real_cam_height):
        inv_K = torch.inverse(K)

        cam_points = self.backproject_depth(depth, inv_K)
        surface_normal = self.get_surface_normal(cam_points)
        ground_mask = self.get_ground_mask(cam_points, surface_normal)

        cam_heights = (cam_points[:,:-1,:,:] * surface_normal).sum(1).abs().unsqueeze(1)
        cam_heights_masked = torch.masked_select(cam_heights, ground_mask)
        cam_height = torch.median(cam_heights_masked).unsqueeze(0)

        scale = torch.reciprocal(cam_height).mul_(real_cam_height)

        return scale, ground_mask

def rel_to_depth(rel_depth):
    original_height, original_width = rel_depth.shape
    cam_height = torch.tensor([1.350]).cuda()
    # / 2.57
    # K = np.array([[0.399, 0, 0.826, 0],
    #               [0, 0.399, 0.621, 0],
    #               [0, 0, 1, 0],
    #               [0, 0, 0, 1]], dtype=np.float32)
    K = np.array([[0.399, 0, 0.600, 0],
                  [0, 0.399, 0.395, 0],
                  [0, 0, 1, 0],
                  [0, 0, 0, 1]], dtype=np.float32)
    ######### mm #############
    # K = np.array([[399, 0, 600, 0],
    #               [0, 399, 395, 0],
    #               [0, 0, 1, 0],
    #               [0, 0, 0, 1]], dtype=np.float32)
    tensor_K = K.copy()
    tensor_K[0, :] *= original_width
    tensor_K[1, :] *= original_height
    tensor_K = torch.from_numpy(tensor_K).unsqueeze(0).cuda()
    scale_recovery = ScaleRecovery(1, original_height, original_width).cuda()
    relative_depth = 1 / rel_depth
    relative_depth = torch.from_numpy(relative_depth)
    relative_depth = relative_depth.cuda()
    scale, ground_mask = scale_recovery(relative_depth, tensor_K, cam_height)
    absolute_depth = (relative_depth * scale).cpu().numpy()
    # Saving colormapped depth image
    return absolute_depth

def main(args):
    # Train
    # src_folder = './undis_raw_train/rgb/'
    # target_folder = './datasets/nclt/train/'

    # Test
    src_folder = './undis_raw_test/rgb/'
    target_folder = './datasets/nclt/test/'

    # midas load
    midas = torch.hub.load("intel-isl/MiDaS", "MiDaS")
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    midas.to(device)
    midas.eval()
    midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
    transform = midas_transforms.default_transform

    pbar = tqdm(total=len(os.listdir(src_folder)))
    for imgFile in os.listdir(src_folder):
        img = cv2.imread(src_folder + imgFile)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # img = img[90:398, 76:550]
        img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
        img = img[200:1400, 210:1000]
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
        output[output==0] = 1e-5
        abs_depth_mm = rel_to_depth(output)
        # eye file
        abs_depth_m = abs_depth_mm / 1000
        offsetX = 4
        offsetY = 4
        abs_depth_m = abs_depth_m[offsetY::8, offsetX::8]
        prediction_grid = np.zeros((2,
        				            math.ceil(5000 / 8), 
				                    math.ceil(5000 / 8)))
        for x in range(0, prediction_grid.shape[2]):
            for y in range(0, prediction_grid.shape[1]):
                prediction_grid[0, y, x] = x * 8
                prediction_grid[1, y, x] = y * 8
        xy = prediction_grid[:,:abs_depth_m.shape[0],:abs_depth_m.shape[1]].copy()
        xy[0] += offsetX
        xy[1] += offsetY
        xy[0] -= img.shape[1] / 2 # before rotation
        xy[1] -= img.shape[2] / 2 # so flipped
        focal_length = 341.78875220088753 # copied
        xy /= focal_length
        xy[0] *= abs_depth_m
        xy[1] *= abs_depth_m

        eye = np.ndarray((4, abs_depth_m.shape[0], abs_depth_m.shape[1]))
        eye[0:2] = xy
        eye[2] = abs_depth_m
        eye[3] = 1
        eye = eye.astype(np.float32)
        eyeTensor = torch.from_numpy(eye)

        # saving to files
        img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
        abs_depth_mm = np.rot90(abs_depth_mm)
        time_stamp = imgFile[:-10]
        imageio.imwrite(target_folder + '/depth/' + time_stamp + '.depth.tiff', abs_depth_mm)
        imageio.imwrite(target_folder + '/rgb/' + time_stamp + '.color.png', img)
        torch.save(eyeTensor, target_folder + '/eye/' + time_stamp + '.dat')
        pbar.update(1)
    pbar.close()



if __name__ == '__main__':
    sys.exit(main(sys.argv))