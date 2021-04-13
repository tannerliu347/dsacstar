import os
import numpy as np
import sys
import cv2
from tqdm import tqdm

img_scale = 1

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
    """rescale height to 480"""
    global img_scale
    img = cv2.imread(src)
    src_height = int(img.shape[0])
    src_width = int(img.shape[1])
    new_height = 480
    img_scale =  new_height / src_height
    img_new = cv2.resize(img, (int(src_width / src_height * new_height), new_height))
    cv2.imwrite(des, img_new)

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

############################################################################
# TODO: Check these items and edit    #
sys.path.append('/datadrive/dsacstar/datasets')
src_folder = './nclt_source'
focallength = 399.433184
target_folder = './nclt'
# bounds for masking a subregion of trajectory
x_l_bound = -175
x_u_bound = -45
y_l_bound = -375
y_u_bound = -80
train_size = 1500
match_threshold = 10000
cam = 'Cam5'

# camera calibration matrices
x_lb3_c = [0.041862, -0.001905, -0.000212, 160.868615, 89.914152, 160.619894]
x_rob_lb3 = [0.035, 0.002, -1.23, -179.93, -0.23, 0.50]
H_lb3_c = ssc_to_homo(x_lb3_c)
H_rob_lb3 = ssc_to_homo(x_rob_lb3)
H_c_lb3 = np.linalg.inv(H_lb3_c)
H_lb3_rob = np.linalg.inv(H_rob_lb3)

trainSplit = ["2012-01-08", "2012-03-17", "2012-10-28"]
############################################################################

img_target_folder = f'{target_folder}/train/rgb'
cali_target_folder = f'{target_folder}/train/calibration'
poses_target_folder = f'{target_folder}/train/poses'

for train_seq in tqdm(trainSplit):
    sampled_image_csv = []
    current_seq_src = f'{src_folder}/{train_seq}'
    # loads ground truth
    current_gt = np.loadtxt(f'{current_seq_src}/groundtruth_{train_seq}.csv', delimiter=',')
    current_gt = current_gt[(current_gt[:,1] >= x_l_bound) & (current_gt[:,1] <= x_u_bound) \
        & (current_gt[:,2] <= y_u_bound) & (current_gt[:,2] >= y_l_bound)]
    print(current_gt.shape)
    current_gt = current_gt[current_gt[:,0] < current_gt[0,0]+1e9]
    print(current_gt.shape)
    current_img_folder = f'{current_seq_src}/lb3/{cam}'
    img_list = [os.path.splitext(filename)[0] for filename in os.listdir(current_img_folder)]
    
    # use np.arange to approximate sampling, since cam and gt are async
    sample_idx = np.arange(0, current_gt.shape[0]-1, int(current_gt.shape[0]/train_size))
    img_to_sample = current_gt[sample_idx, 0]
    img_to_sample = img_to_sample[:train_size]
    print(img_to_sample.shape)
    
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
        sampled_image_csv.append([current_gt[sample_idx[idx], 0], current_gt[sample_idx[idx], 1], \
            current_gt[sample_idx[idx], 2], current_gt[sample_idx[idx], 3], \
                current_gt[sample_idx[idx], 4], current_gt[sample_idx[idx], 5], \
                    current_gt[sample_idx[idx], 6]])
        # save poses
        x_rob = [current_gt[sample_idx[idx], 1], current_gt[sample_idx[idx], 2], current_gt[sample_idx[idx], 3], current_gt[sample_idx[idx], 4], current_gt[sample_idx[idx], 5], current_gt[sample_idx[idx], 6]]
        x_rob[3] *= 180.0 / np.pi
        x_rob[4] *= 180.0 / np.pi
        x_rob[5] *= 180.0 / np.pi
        H_rob = ssc_to_homo(x_rob)
        # H_c = H_c_lb3 @ H_lb3_rob @ H_rob
        H_c = H_rob @ H_lb3_rob @ H_c_lb3
        np.savetxt(f'{poses_target_folder}/{frame_name}.pose.txt', H_c)
        # save calibration
        np.savetxt(f'{cali_target_folder}/{frame_name}.calibration.txt', np.array([focallength*img_scale]))
        pbar.update(1)
    pbar.close()
    np.savetxt(f'{train_seq}_sample.txt', sampled_image_csv)

########################################################################################################
# # test set generate
print("test set")

############################################################################
# TODO: Check these items and edit    #
train_size = 1000
testSplit = ["2012-03-31"]
############################################################################

# print(os.listdir(src_folder))
img_target_folder = f'{target_folder}/test/rgb'
cali_target_folder = f'{target_folder}/test/calibration'
poses_target_folder = f'{target_folder}/test/poses'

for train_seq in tqdm(testSplit):
    sampled_image_csv = []
    current_seq_src = f'{src_folder}/{train_seq}'
    # loads ground truth
    current_gt = np.loadtxt(f'{current_seq_src}/groundtruth_{train_seq}.csv', delimiter=',')
    current_gt = current_gt[(current_gt[:,1] >= x_l_bound) & (current_gt[:,1] <= x_u_bound) \
        & (current_gt[:,2] <= y_u_bound) & (current_gt[:,2] >= y_l_bound)]
    print(current_gt.shape)
    current_gt = current_gt[current_gt[:,0] < current_gt[0,0]+1e9]
    print(current_gt.shape)
    current_img_folder = f'{current_seq_src}/lb3/{cam}'
    img_list = [os.path.splitext(filename)[0] for filename in os.listdir(current_img_folder)]
    
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
        sampled_image_csv.append([current_gt[sample_idx[idx], 0], current_gt[sample_idx[idx], 1], \
            current_gt[sample_idx[idx], 2], current_gt[sample_idx[idx], 3], \
                current_gt[sample_idx[idx], 4], current_gt[sample_idx[idx], 5], \
                    current_gt[sample_idx[idx], 6]])
        # csv_writer.writerow(sampled_image_csv)
        # save poses
        x_rob = [current_gt[sample_idx[idx], 1], current_gt[sample_idx[idx], 2], current_gt[sample_idx[idx], 3], current_gt[sample_idx[idx], 4], current_gt[sample_idx[idx], 5], current_gt[sample_idx[idx], 6]]
        x_rob[3] *= 180.0 / np.pi
        x_rob[4] *= 180.0 / np.pi
        x_rob[5] *= 180.0 / np.pi
        H_rob = ssc_to_homo(x_rob)
        H_c = H_rob @ H_lb3_rob @ H_c_lb3
        np.savetxt(f'{poses_target_folder}/{frame_name}.pose.txt', H_c)
        # save calibration
        np.savetxt(f'{cali_target_folder}/{frame_name}.calibration.txt', np.array([focallength*img_scale]))
        pbar.update(1)
    pbar.close()
    np.savetxt(f'{train_seq}_sample.txt', sampled_image_csv)