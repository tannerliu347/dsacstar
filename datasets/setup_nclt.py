import os
import numpy as np
import sys
import cv2
import csv
from tqdm import tqdm

def mkdir(directory):
	"""Checks whether the directory exists and creates it if necessacy."""
	if not os.path.exists(directory):
		os.makedirs(directory)

def nearest(ts, img_list):
    """given a timestamp, find the nearest timestamp in img_list"""
    # ts - timestamp
    min_diff = 1000000000000
    min_ts = img_list[0]
    for t in img_list:
        if abs(int(t) - ts) < min_diff:
            min_diff = abs(int(t) - ts)
            min_ts = t
    return min_ts

def save_img(src, des):
    """ rotates and saves img to des"""
    img = cv2.imread(src)
    img_rot = cv2.rotate(img,rotateCode=cv2.ROTATE_90_CLOCKWISE)
    cv2.imwrite(des, img_rot)

def ssc_to_homo(ssc):

    # Convert 6-DOF ssc coordinate transformation to 4x4 homogeneous matrix
    # transformation

    sr = np.sin(np.pi/180.0 * ssc[3])
    cr = np.cos(np.pi/180.0 * ssc[3])

    sp = np.sin(np.pi/180.0 * ssc[4])
    cp = np.cos(np.pi/180.0 * ssc[4])

    sh = np.sin(np.pi/180.0 * ssc[5])
    ch = np.cos(np.pi/180.0 * ssc[5])

    H = np.zeros((4, 4))

    H[0, 0] = ch*cp
    H[0, 1] = -sh*cr + ch*sp*sr
    H[0, 2] = sh*sr + ch*sp*cr
    H[1, 0] = sh*cp
    H[1, 1] = ch*cr + sh*sp*sr
    H[1, 2] = -ch*sr + sh*sp*cr
    H[2, 0] = -sp
    H[2, 1] = cp*sr
    H[2, 2] = cp*cr

    H[0, 3] = ssc[0]
    H[1, 3] = ssc[1]
    H[2, 3] = ssc[2]

    H[3, 3] = 1

    return H


sys.path.append('/datadrive/dsacstar/datasets')
src_folder = '/datadrive/dsacstar/datasets/nclt_source'
focallength = 409.719024
target_folder = '/datadrive/dsacstar/datasets/nclt'
# bounds for masking a subregion of trajectory
x_l_bound = -375
x_u_bound = -75
y_l_bound = -175
y_u_bound = -45
train_size = 650
match_threshold = 10000

# camera calibration matrices
x_lb3_c = [0.014543, 0.039337, 0.000398, -138.449751, 89.703877, -66.518051]
x_rob_lb3 = [0.035, 0.002, -1.23, -179.93, -0.23, 0.50]
H_lb3_c = ssc_to_homo(x_lb3_c)
H_rob_lb3 = ssc_to_homo(x_rob_lb3)
H_c_lb3 = np.linalg.inv(H_lb3_c)
H_lb3_rob = np.linalg.inv(H_rob_lb3)

trainSplit = ["2012-01-08", "2012-03-17", "2012-10-28", "2012-11-04"]
testSplit = ["2013-04-05"]
print(os.listdir(src_folder))
sampled_image_csv = []
img_target_folder = f'{target_folder}/train/rgb'
cali_target_folder = f'{target_folder}/train/calibration'
poses_target_folder = f'{target_folder}/train/poses'

for train_seq in tqdm(trainSplit):
    current_seq_src = f'{src_folder}/{train_seq}'
    # loads ground truth
    current_gt = np.loadtxt(f'{current_seq_src}/groundtruth_{train_seq}.csv', delimiter=',')
    current_gt = current_gt[(current_gt[:,2] >= x_l_bound) & (current_gt[:,2] <= x_u_bound) \
        & (current_gt[:,1] <= y_u_bound) & (current_gt[:,1] >= y_l_bound)]

    current_img_folder = f'{current_seq_src}/lb3/Cam1'
    img_list = [os.path.splitext(filename)[0] for filename in os.listdir(current_img_folder)]
    print(np.array(img_list, dtype=int))
    
    # use np.arange to approximate sampling, since cam and gt are async
    sample_idx = np.arange(0, current_gt.shape[0]-1, int(current_gt.shape[0]/train_size))
    img_to_sample = current_gt[sample_idx, 0]
    img_to_sample = img_to_sample[:train_size]
    
    # with open('sampled_image.csv', 'w', newline='') as csvfile:
    #     csv_writer = csv.writer(csvfile, delimiter=',')
        # enumerate all samples, find nearest in images
        # save to rgb, calibration, poses
    pbar = tqdm(total=train_size)
    for idx, img_ts_expected in enumerate(img_to_sample):
        img_ts = nearest(img_ts_expected, img_list)
        img_src = f'{current_img_folder}/{img_ts}.tiff'
        frame_name = f'{train_seq}_{img_ts}'
        img_dst = f'{img_target_folder}/{frame_name}.color.png'
        # save rgb
        save_img(img_src, img_dst)
        sampled_image_csv = [frame_name, str(current_gt[sample_idx[idx], 2]), \
            str(current_gt[sample_idx[idx], 1]), str(-current_gt[sample_idx[idx], 3]), \
                str(current_gt[sample_idx[idx], 4]), str(current_gt[sample_idx[idx], 5]), \
                    str(current_gt[sample_idx[idx], 6])] #x y -z r p y
        # csv_writer.writerow(sampled_image_csv)
        # save poses
        x_rob = [current_gt[sample_idx[idx], 2], current_gt[sample_idx[idx], 1], -current_gt[sample_idx[idx], 3], current_gt[sample_idx[idx], 4], current_gt[sample_idx[idx], 5], current_gt[sample_idx[idx], 6]]
        H_rob = ssc_to_homo(x_rob)
        H_c = H_c_lb3 @ H_lb3_rob @ H_rob
        np.savetxt(f'{poses_target_folder}/{frame_name}.pose.txt', H_rob)
        # save calibration
        np.savetxt(f'{cali_target_folder}/{frame_name}.calibration.txt', np.array([focallength]))
        pbar.update(1)
    pbar.close()

########################################################################################################
# test set generate
print("test set")
testSplit = ["2013-04-05"]
# print(os.listdir(src_folder))
img_target_folder = f'{target_folder}/test/rgb'
cali_target_folder = f'{target_folder}/test/calibration'
poses_target_folder = f'{target_folder}/test/poses'

for train_seq in tqdm(testSplit):
    current_seq_src = f'{src_folder}/{train_seq}'
    # loads ground truth
    current_gt = np.loadtxt(f'{current_seq_src}/groundtruth_{train_seq}.csv', delimiter=',')
    current_gt = current_gt[(current_gt[:,2] >= x_l_bound) & (current_gt[:,2] <= x_u_bound) \
        & (current_gt[:,1] <= y_u_bound) & (current_gt[:,1] >= y_l_bound)]

    current_img_folder = f'{current_seq_src}/lb3/Cam1'
    img_list = [os.path.splitext(filename)[0] for filename in os.listdir(current_img_folder)]
    # print(np.array(img_list, dtype=int))
    
    # use np.arange to approximate sampling, since cam and gt are async
    sample_idx = np.arange(0, current_gt.shape[0]-1, int(current_gt.shape[0]/train_size))
    img_to_sample = current_gt[sample_idx, 0]
    img_to_sample = img_to_sample[:train_size]
    
    # with open('sampled_image.csv', 'w', newline='') as csvfile:
    #     csv_writer = csv.writer(csvfile, delimiter=',')
        # enumerate all samples, find nearest in images
        # save to rgb, calibration, poses
    pbar = tqdm(total=train_size)
    for idx, img_ts_expected in enumerate(img_to_sample):
        img_ts = nearest(img_ts_expected, img_list)
        img_src = f'{current_img_folder}/{img_ts}.tiff'
        frame_name = f'{train_seq}_{img_ts}'
        img_dst = f'{img_target_folder}/{frame_name}.color.png'
        # save rgb
        save_img(img_src, img_dst)
        sampled_image_csv = [frame_name, str(current_gt[sample_idx[idx], 2]), \
            str(current_gt[sample_idx[idx], 1]), str(-current_gt[sample_idx[idx], 3]), \
                str(current_gt[sample_idx[idx], 4]), str(current_gt[sample_idx[idx], 5]), \
                    str(current_gt[sample_idx[idx], 6])] #x y -z r p y
        # csv_writer.writerow(sampled_image_csv)
        # save poses
        x_rob = [current_gt[sample_idx[idx], 2], current_gt[sample_idx[idx], 1], -current_gt[sample_idx[idx], 3], current_gt[sample_idx[idx], 4], current_gt[sample_idx[idx], 5], current_gt[sample_idx[idx], 6]]
        H_rob = ssc_to_homo(x_rob)
        H_c = H_c_lb3 @ H_lb3_rob @ H_rob
        np.savetxt(f'{poses_target_folder}/{frame_name}.pose.txt', H_rob)
        # save calibration
        np.savetxt(f'{cali_target_folder}/{frame_name}.calibration.txt', np.array([focallength]))
        pbar.update(1)
    pbar.close()