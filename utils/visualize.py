import matplotlib
import numpy as np

matplotlib.use('Agg')
import matplotlib.pyplot as plt

# keypoints: ndarray [1, 51]
def DrawSkeleton(keypoints, offset=None, head1=None, head2=None, image_name='Skeleton.jpg'):
    # pos_x = keypoints[0:45:3]
    # pos_y = keypoints[1:45:3]
    # pos_z = keypoints[2:45:3]
    # head = keypoints[6:9]
    pos_x = keypoints[0:len(keypoints):3]
    pos_y = keypoints[1:len(keypoints):3]
    pos_z = keypoints[2:len(keypoints):3]
    if offset is not None:
        pos_x = pos_x - offset[0]
        pos_y = pos_y - offset[1]
        pos_z = pos_z - offset[2]
    head = keypoints[6:9]

    xp = pos_x.T
    yp = pos_y.T
    zp = pos_z.T
    ax = plt.axes(projection='3d')
    if head1 is None and head2 is None:
        pass
    else:
        f = head1
        f = f/np.sqrt((f[0]**2 + f[1]**2 + f[2]**2))
        print(f)
        u = head2
        u = u/np.sqrt((u[0]**2 + u[1]**2 + u[2]**2))
        print(u)
        #画起点为head,终点为f_end的向量
        ax.quiver(head[0], head[1], head[2], f[0]*10, f[1]*10, f[2]*10, color='green', arrow_length_ratio=0.2)
        #画起点为head,终点为u_end的向量
        ax.quiver(head[0], head[1], head[2], u[0]*10, u[1]*10, u[2]*10, color='blue', arrow_length_ratio=0.2)
    radius = 30
    ax.set_xlim3d([-radius, radius])
    ax.set_zlim3d([-radius, radius])
    ax.set_ylim3d([-radius, radius])
    ax.view_init(elev=15., azim=50)
    
    ax.dist = 7.5

    # 3D scatter
    ax.scatter3D(xp, yp, zp, cmap='Greens')
    
    # hips, neck, head, node [0, 1, 2]
    ax.plot(xp[0:3], yp[0:3], zp[0:3], ls='-', color='gray')
    
    # RightShoulder, RightArm, RightForeArm, RightHand
    ax.plot(xp[3:7], yp[3:7], zp[3:7], ls='-', color='blue')
    # LeftShoulder, LeftArm, LeftForeArm, LeftHand
    ax.plot(xp[7:11], yp[7:11], zp[7:11], ls='-', color='red')
    # RightUpLeg, RightLeg, RightFoot
    ax.plot(xp[11:14], yp[11:14], zp[11:14], ls='-', color='blue')
    
    # LeftUpLeg, LeftLeg, LeftFoot
    ax.plot(xp[14:17], yp[14:17], zp[14:17], ls='-', color='red')

    plt.savefig(image_name, dpi=300)

# keypoints: ndarray [1, 45]
def DrawSkeleton45(keypoints, head1=None, head2=None, image_name='Skeleton.jpg'):
    # pos_x = keypoints[0:45:3]
    # pos_y = keypoints[1:45:3]
    # pos_z = keypoints[2:45:3]
    # head = keypoints[6:9]
    pos_x = keypoints[0:len(keypoints):3]
    pos_y = keypoints[1:len(keypoints):3]
    pos_z = keypoints[2:len(keypoints):3]
    head = keypoints[6:9]

    xp = pos_x.T
    yp = pos_y.T
    zp = pos_z.T
    ax = plt.axes(projection='3d')
    if head1 is None and head2 is None:
        pass
    else:
        f = head1
        f = f/np.sqrt((f[0]**2 + f[1]**2 + f[2]**2))
        print(f)
        u = head2
        u = u/np.sqrt((u[0]**2 + u[1]**2 + u[2]**2))
        print(u)
        #画起点为head,终点为f_end的向量
        ax.quiver(head[0], head[1], head[2], f[0]*10, f[1]*10, f[2]*10, color='green', arrow_length_ratio=0.2)
        #画起点为head,终点为u_end的向量
        ax.quiver(head[0], head[1], head[2], u[0]*10, u[1]*10, u[2]*10, color='blue', arrow_length_ratio=0.2)
    radius = 1
    ax.set_xlim3d([-radius, radius])
    ax.set_zlim3d([-radius, radius])
    ax.set_ylim3d([-radius, radius])
    ax.view_init(elev=15., azim=90)
    
    ax.dist = 7.5

    # 3D scatter
    ax.scatter3D(xp, yp, zp, cmap='Greens')
    
    # hips, neck, head, node [0, 1, 2]
    ax.plot(xp[0:3], yp[0:3], zp[0:3], ls='-', color='gray')
    
    # RightShoulder, RightArm, RightHand
    ax.plot(xp[3:6], yp[3:6], zp[3:6], ls='-', color='blue')
    # LeftShoulder, LeftArm, LeftHand
    ax.plot(xp[6:9], yp[6:9], zp[6:9], ls='-', color='red')
    # RightUpLeg, RightLeg, RightFoot
    ax.plot(xp[9:12], yp[9:12], zp[9:12], ls='-', color='blue')
    
    # LeftUpLeg, LeftLeg, LeftFoot
    ax.plot(xp[12:15], yp[12:15], zp[12:15], ls='-', color='red')

    plt.savefig(image_name, dpi=300)

if __name__=='__main__':
    import torch
    import math
    import os
    from bvh import Bvh
    import csv
    import pandas as pd
    import numpy as np
    # def _load_keypoint_positon(transform):
    #     # hips = rotation['Hips']@rotation['translation']
    #     # hips = rotation['translation']
    #     hips = transform["Hips"]
    #     spine = transform["Spine"]@hips
    #     spine1 = transform["Spine1"]@spine
    #     spine2 = transform["Spine2"]@spine1
    #     spine3 = transform["Spine3"]@spine2
    #     neck = transform["Neck"]@spine3
    #     head = transform["Head"]@neck
    #     RightShoulder = transform["RightShoulder"]@spine3
    #     RightArm = transform["RightArm"]@RightShoulder
    #     RightForeArm = transform["RightForeArm"]@RightArm
    #     RightHand = transform["RightHand"]@RightForeArm
    #     LeftShoulder = transform["LeftShoulder"]@spine3
    #     LeftArm = transform["LeftArm"]@LeftShoulder
    #     LeftForeArm = transform["LeftForeArm"]@LeftArm
    #     LeftHand = transform["LeftHand"]@LeftForeArm
    #     RightUpLeg = transform["RightUpLeg"]@hips
    #     RightLeg = transform["RightLeg"]@RightUpLeg
    #     RightFoot = transform["RightFoot"]@RightLeg
    #     LeftUpLeg = transform["LeftUpLeg"]@hips
    #     LeftLeg = transform["LeftLeg"]@LeftUpLeg
    #     LeftFoot = transform["LeftFoot"]@LeftLeg

    #     return torch.cat([hips[0:3,3].unsqueeze(0), neck[0:3,3].unsqueeze(0), head[0:3,3].unsqueeze(0), RightShoulder[0:3,3].unsqueeze(0), RightArm[0:3,3].unsqueeze(0), RightForeArm[0:3,3].unsqueeze(0), RightHand[0:3,3].unsqueeze(0),
    #                       LeftShoulder[0:3,3].unsqueeze(0), LeftArm[0:3,3].unsqueeze(0), LeftForeArm[0:3,3].unsqueeze(0), LeftHand[0:3,3].unsqueeze(0), RightUpLeg[0:3,3].unsqueeze(0), RightLeg[0:3,3].unsqueeze(0), 
    #                       RightFoot[0:3,3].unsqueeze(0), LeftUpLeg[0:3,3].unsqueeze(0), LeftLeg[0:3,3].unsqueeze(0), LeftFoot[0:3,3].unsqueeze(0)], dim=-1)
    
    # def _load_offset(mocap):
    #     # Hips->Spine->Spine1->Spine2->Spine3->Neck->Head
    #     joint_name = ['Hips', 'Spine', 'Spine1', 'Spine2', 'Spine3', 'Neck', 'Head',
    #                     'RightShoulder', 'RightArm', 'RightForeArm', 'RightHand',
    #                     'LeftShoulder', 'LeftArm', 'LeftForeArm', 'LeftHand',
    #                     'RightUpLeg', 'RightLeg', 'RightFoot',
    #                     'LeftUpLeg', 'LeftLeg', 'LeftFoot']
    #     joint_offset = {}
    #     for joint in joint_name:
    #         offset_ = mocap.joint_offset(joint)
    #         print("joint name: ", joint)
    #         print("offset_: ", offset_)
    #         offset = torch.Tensor([[offset_[0], offset_[1], offset_[2]]])
    #         joint_offset[joint] = offset
    #     return joint_offset
            
    # def _load_rotation(traj, idx):
    #     # bvh rotation: Z axis, X axis, Y axis
    #     # Hip has 6 channels: [translation, rotation]
    #     # idx: current frame in a bvh file
    #     # traj = pickle.load(open(directory, 'rb'))
    #     x_hips = torch.Tensor([[traj[idx][0], traj[idx][1], traj[idx][2]]]).T
    #     r_hips = load_joint_rotation(traj, (3,6), idx) 
    #     r_spine = load_joint_rotation(traj, (6,9), idx)     
    #     r_spine1 = load_joint_rotation(traj, (9,12), idx) 
    #     r_spine2 = load_joint_rotation(traj, (12,15), idx)
    #     r_spine3 = load_joint_rotation(traj, (15,18), idx) 
    #     r_neck = load_joint_rotation(traj, (18,21), idx) 
    #     r_head = load_joint_rotation(traj, (21,24), idx) 
    #     r_RightShoulder = load_joint_rotation(traj, (24,27), idx)  
    #     r_RightArm = load_joint_rotation(traj, (27,30), idx) 
    #     r_RightForeArm = load_joint_rotation(traj, (30,33), idx) 
    #     r_RightHand = load_joint_rotation(traj, (33,36), idx) 
    #     r_LeftShoulder = load_joint_rotation(traj, (36, 39), idx)   
    #     r_LeftArm = load_joint_rotation(traj, (39, 42), idx) 
    #     r_LeftForeArm = load_joint_rotation(traj, (42, 45), idx) 
    #     r_LeftHand = load_joint_rotation(traj, (45, 48), idx)
    #     r_RightUpLeg = load_joint_rotation(traj, (48, 51), idx) 
    #     r_RightLeg = load_joint_rotation(traj, (51, 54), idx) 
    #     r_RightFoot = load_joint_rotation(traj, (54, 57), idx) 
    #     r_LeftUpLeg = load_joint_rotation(traj, (57, 60), idx) 
    #     r_LeftLeg = load_joint_rotation(traj, (60, 63), idx) 
    #     r_LeftFoot = load_joint_rotation(traj, (63, 66), idx) 
    
    #     return {"translation":x_hips, "Hips":r_hips, "Spine":r_spine, "Spine1":r_spine1, "Spine2":r_spine2, "Spine3":r_spine3,
    #             "Neck":r_neck, "Head":r_head, "RightShoulder":r_RightShoulder, "RightArm":r_RightArm, "RightForeArm":r_RightForeArm, 
    #             "RightHand":r_RightHand, "LeftShoulder":r_LeftShoulder, "LeftArm":r_LeftArm, "LeftForeArm":r_LeftForeArm, 
    #             "LeftHand":r_LeftHand, "RightUpLeg":r_RightUpLeg, "RightLeg":r_RightLeg, "RightFoot":r_RightFoot,
    #             "LeftUpLeg":r_LeftUpLeg, "LeftLeg":r_LeftLeg, "LeftFoot":r_LeftFoot}

    # # traj:将要读取的traj文件; index: 该关节在数组中的索引位置; idx:某一帧的帧数
    # def load_joint_rotation(traj, index, idx):
    #     if(index[1]-index[0] == 3):
    #         joint_rotation = traj[idx][index[0]:index[1]]
    #         return get_rotation_matrix(joint_rotation[0], joint_rotation[1], joint_rotation[2])

    # def get_rotation_matrix(x_rad, y_rad, z_rad):
    #     Rx = torch.tensor([[1, 0, 0, 0],
    #                         [0, math.cos(x_rad), -math.sin(x_rad), 0],
    #                         [0, math.sin(x_rad), math.cos(x_rad), 0],
    #                         [0, 0, 0, 1]])
    #     Ry = torch.tensor([[math.cos(y_rad), 0, math.sin(y_rad), 0],
    #                         [0, 1, 0, 0],
    #                         [-math.sin(y_rad), 0, math.cos(y_rad), 0],
    #                         [0, 0, 0, 1]])
    #     Rz = torch.tensor([[math.cos(z_rad), -math.sin(z_rad), 0, 0],
    #                         [math.sin(z_rad), math.cos(z_rad), 0, 0],
    #                         [0, 0, 1, 0],
    #                         [0, 0, 0, 1]])
    #     return Rx@Ry@Rz

    # def get_transform_matrix(joint_rotation, joint_offset):
    #     offset_transform = torch.zeros([4,4])
    #     offset_transform[0,0] = 1
    #     offset_transform[1,1] = 1
    #     offset_transform[2,2] = 1
    #     offset_transform[3,3] = 1
    #     offset_transform[0,3] = joint_offset[0,0]
    #     offset_transform[1,3] = joint_offset[0,1]
    #     offset_transform[2,3] = joint_offset[0,2]
    #     transform = joint_rotation@offset_transform
    #     return transform

    # def load_joint_transform(joint_rotation, joint_offset):
    #     tran_hips = get_transform_matrix(joint_rotation["Hips"], joint_offset["Hips"])
    #     tran_spine = get_transform_matrix(joint_rotation["Spine"], joint_offset["Spine"])
    #     tran_spine1 = get_transform_matrix(joint_rotation["Spine1"], joint_offset["Spine1"])
    #     tran_spine2 = get_transform_matrix(joint_rotation["Spine2"], joint_offset["Spine2"])
    #     tran_spine3 = get_transform_matrix(joint_rotation["Spine3"], joint_offset["Spine3"])
    #     tran_neck = get_transform_matrix(joint_rotation["Neck"], joint_offset["Neck"])
    #     tran_head = get_transform_matrix(joint_rotation["Head"], joint_offset["Head"])
    #     tran_rightshoulder = get_transform_matrix(joint_rotation["RightShoulder"], joint_offset["RightShoulder"])
    #     tran_rightarm = get_transform_matrix(joint_rotation["RightArm"], joint_offset["RightArm"])
    #     tran_rightforearm = get_transform_matrix(joint_rotation["RightForeArm"], joint_offset["RightForeArm"])
    #     tran_righthand = get_transform_matrix(joint_rotation["RightHand"], joint_offset["RightHand"])
    #     tran_leftshoulder = get_transform_matrix(joint_rotation["LeftShoulder"], joint_offset["LeftShoulder"])
    #     tran_leftarm = get_transform_matrix(joint_rotation["LeftArm"], joint_offset["LeftArm"])
    #     tran_leftforearm = get_transform_matrix(joint_rotation["LeftForeArm"], joint_offset["LeftForeArm"])
    #     tran_lefthand = get_transform_matrix(joint_rotation["LeftHand"], joint_offset["LeftHand"])
    #     tran_rightupleg = get_transform_matrix(joint_rotation["RightUpLeg"], joint_offset["RightUpLeg"])
    #     tran_rightleg = get_transform_matrix(joint_rotation["RightLeg"], joint_offset["RightLeg"])
    #     tran_rightfoot = get_transform_matrix(joint_rotation["RightFoot"], joint_offset["RightFoot"])
    #     tran_leftupleg = get_transform_matrix(joint_rotation["LeftUpLeg"], joint_offset["LeftUpLeg"])
    #     tran_leftleg = get_transform_matrix(joint_rotation["LeftLeg"], joint_offset["LeftLeg"])
    #     tran_leftfoot = get_transform_matrix(joint_rotation["LeftFoot"], joint_offset["LeftFoot"])
    #     return {"Hips":tran_hips, "Spine":tran_spine, "Spine1":tran_spine1, "Spine2":tran_spine2, "Spine3":tran_spine3,
    #             "Neck":tran_neck, "Head":tran_head, "RightShoulder": tran_rightshoulder, "RightArm":tran_rightarm,
    #             "RightForeArm":tran_rightforearm, "RightHand":tran_righthand, "LeftShoulder":tran_leftshoulder,
    #             "LeftArm":tran_leftarm, "LeftForeArm":tran_leftforearm, "LeftHand":tran_lefthand, "RightUpLeg":tran_rightupleg,
    #             "RightLeg":tran_rightleg, "RightFoot":tran_rightfoot, "LeftUpLeg":tran_leftupleg, "LeftLeg":tran_leftleg, "LeftFoot":tran_leftfoot}




    # dir = '0213_take_06'
    # bvh_file = dir + ".bvh"
    # ind_frame_in_mocap = 0
    # traj_path = '/home/liumin/litianyi/workspace/data/datasets/data'
    # dataset_path = '/home/liumin/litianyi/workspace/data/datasets'
    # bvh_path = os.path.join(dataset_path, "traj", bvh_file)
    # traj_file = os.path.join(traj_path, dir + '.npy')
    # with open(bvh_path) as f:
    #     mocap = Bvh(f.read())
    # traj = np.load(traj_file)[:-1]
    # print("traj shape: ", traj.shape)
    # rotation = _load_rotation(traj, ind_frame_in_mocap)
    # offset = _load_offset(mocap)
    # transform = load_joint_transform(rotation, offset)
    # keypoints = _load_keypoint_positon(transform)
  
    # # d_max = torch.max(keypoints, dim=1)[0].unsqueeze(1)
    # # # d_max shape: (batch)->(batch, 1)
    # # d_min = torch.min(keypoints, dim=1)[0].unsqueeze(1)
    # # # print("d_min shape: ", d_min.shape)
    # # dst = d_max - d_min
    # # keypoints = ((keypoints - d_min) / dst - 0.5) / 0.5
    # keypoints = keypoints.numpy().squeeze(0)
    # print("keypoints: ", keypoints)
    # DrawSkeleton(keypoints)

    # def _load_keypoint_positon(rotation, offset):
    #     # hips = rotation['Hips']@rotation['translation']
    #     # hips = rotation['translation']
    #     hips = torch.zeros([3,1])
    #     spine3 = hips + rotation['Spine']@offset['Spine'] \
    #                 + rotation['Spine1']@offset['Spine1'] \
    #                 + rotation['Spine2']@offset['Spine2'] \
    #                 + rotation['Spine3']@offset['Spine3'] 
    #     neck = spine3 + rotation['Neck']@offset['Neck']
    #     head = neck + rotation['Head']@offset['Head']
    #     RightShoulder = spine3 + rotation['RightShoulder']@offset['RightShoulder']
    #     RightArm = RightShoulder + rotation['RightArm']@offset['RightArm']
    #     RightForeArm = RightArm + rotation['RightForeArm']@offset['RightForeArm']
    #     RightHand = RightForeArm + rotation['RightHand']@offset['RightHand']
    #     LeftShoulder = spine3 + rotation['LeftShoulder']@offset['LeftShoulder']
    #     LeftArm = LeftShoulder + rotation['LeftArm']@offset['LeftArm']
    #     LeftForeArm = LeftArm + rotation['LeftForeArm']@offset['LeftForeArm']
    #     LeftHand = LeftForeArm + rotation['LeftHand']@offset['LeftHand']
    #     RightUpLeg = hips + rotation['RightUpLeg']@offset['RightUpLeg']
    #     RightLeg = RightUpLeg + rotation['RightLeg']@offset['RightLeg']
    #     RightFoot = RightLeg + rotation['RightFoot']@offset['RightFoot']
    #     LeftUpLeg = hips + rotation['LeftUpLeg']@offset['LeftUpLeg']
    #     LeftLeg = LeftUpLeg + rotation['LeftLeg']@offset['LeftLeg']
    #     LeftFoot = LeftLeg + rotation['LeftFoot']@offset['LeftFoot']
    #     return torch.cat([hips.T, neck.T, head.T, RightShoulder.T, RightArm.T, RightForeArm.T, RightHand.T, LeftShoulder.T, LeftArm.T,
    #                       LeftForeArm.T, LeftHand.T, RightUpLeg.T, RightLeg.T, RightFoot.T, LeftUpLeg.T, LeftLeg.T, LeftFoot.T], dim=-1)
    
    # def _load_offset(mocap):
    #     # Hips->Spine->Spine1->Spine2->Spine3->Neck->Head
    #     joint_name = ['Hips', 'Spine', 'Spine1', 'Spine2', 'Spine3', 'Neck', 'Head',
    #                     'RightShoulder', 'RightArm', 'RightForeArm', 'RightHand',
    #                     'LeftShoulder', 'LeftArm', 'LeftForeArm', 'LeftHand',
    #                     'RightUpLeg', 'RightLeg', 'RightFoot',
    #                     'LeftUpLeg', 'LeftLeg', 'LeftFoot']
    #     joint_offset = {}
    #     for joint in joint_name:
    #         offset_ = mocap.joint_offset(joint)
    #         offset = torch.Tensor([[offset_[0], offset_[1], offset_[2]]]).T
    #         joint_offset[joint] = offset
    #     return joint_offset
            
    # def _load_rotation(traj, idx):
    #     # bvh rotation: Z axis, X axis, Y axis
    #     # Hip has 6 channels: [translation, rotation]
    #     # idx: current frame in a bvh file
    #     # traj = pickle.load(open(directory, 'rb'))
    #     x_hips = torch.Tensor([[traj[idx][0], traj[idx][1], traj[idx][2]]]).T
    #     r_hips = load_joint_rotation(traj, (3,6), idx) 
    #     r_spine = load_joint_rotation(traj, (6,9), idx)     
    #     r_spine1 = load_joint_rotation(traj, (9,12), idx) 
    #     r_spine2 = load_joint_rotation(traj, (12,15), idx)
    #     r_spine3 = load_joint_rotation(traj, (15,18), idx) 
    #     r_neck = load_joint_rotation(traj, (18,21), idx) 
    #     r_head = load_joint_rotation(traj, (21,24), idx) 
    #     r_RightShoulder = load_joint_rotation(traj, (24,27), idx)  
    #     r_RightArm = load_joint_rotation(traj, (27,30), idx) 
    #     r_RightForeArm = load_joint_rotation(traj, (30,33), idx) 
    #     r_RightHand = load_joint_rotation(traj, (33,36), idx) 
    #     r_LeftShoulder = load_joint_rotation(traj, (36, 39), idx)   
    #     r_LeftArm = load_joint_rotation(traj, (39, 42), idx) 
    #     r_LeftForeArm = load_joint_rotation(traj, (42, 45), idx) 
    #     r_LeftHand = load_joint_rotation(traj, (45, 48), idx)
    #     r_RightUpLeg = load_joint_rotation(traj, (48, 51), idx) 
    #     r_RightLeg = load_joint_rotation(traj, (51, 54), idx) 
    #     r_RightFoot = load_joint_rotation(traj, (54, 57), idx) 
    #     r_LeftUpLeg = load_joint_rotation(traj, (57, 60), idx) 
    #     r_LeftLeg = load_joint_rotation(traj, (60, 63), idx) 
    #     r_LeftFoot = load_joint_rotation(traj, (63, 66), idx) 
    #     # 这里在之前犯了一个错误,由于此处的向量是右乘的,所以矩阵的相乘顺序要反过来
    #     # R_hips = r_hips
    #     # R_spine = R_hips@r_spine
    #     # R_spine1 = R_spine@r_spine1
    #     # R_spine2 = R_spine1@r_spine2
    #     # R_spine3 = R_spine2@r_spine3
    #     # R_neck = R_spine3@r_neck
    #     # R_head = R_neck@r_head
    #     # R_rightShoulder = R_spine3@r_RightShoulder
    #     # R_rightArm = R_rightShoulder@r_RightArm
    #     # R_rightForeArm = R_rightArm@r_RightForeArm
    #     # R_rightHand = R_rightForeArm@r_RightHand
    #     # R_leftShoulder = R_spine3@r_LeftShoulder
    #     # R_leftArm = R_leftShoulder@r_LeftArm
    #     # R_leftForeArm = R_leftArm@r_LeftForeArm
    #     # R_leftHand = R_leftForeArm@r_LeftHand
    #     # R_rightUpLeg = R_hips@r_RightUpLeg
    #     # R_rightLeg = R_rightUpLeg@r_RightLeg
    #     # R_rightFoot = R_rightLeg@r_RightFoot
    #     # R_leftUpLeg = R_hips@r_LeftUpLeg
    #     # R_leftLeg = R_leftUpLeg@r_LeftLeg
    #     # R_leftFoot = R_leftLeg@r_LeftFoot

    #     R_hips = r_hips
    #     R_spine = r_spine@R_hips
    #     R_spine1 = r_spine1@R_spine
    #     R_spine2 = r_spine2@R_spine1
    #     R_spine3 = r_spine3@R_spine2
    #     R_neck = r_neck@R_spine3
    #     R_head = r_head@R_neck
    #     R_rightShoulder = r_RightShoulder@R_spine3
    #     R_rightArm = r_RightArm@R_rightShoulder
    #     R_rightForeArm = r_RightForeArm@R_rightArm
    #     R_rightHand = r_RightHand@R_rightForeArm
    #     R_leftShoulder = r_LeftShoulder@R_spine3
    #     R_leftArm = r_LeftArm@R_leftShoulder
    #     R_leftForeArm = r_LeftForeArm@R_leftArm
    #     R_leftHand = r_LeftHand@R_leftForeArm
    #     R_rightUpLeg = r_RightUpLeg@R_hips
    #     R_rightLeg = r_RightLeg@R_rightUpLeg
    #     R_rightFoot = r_RightFoot@R_rightLeg
    #     R_leftUpLeg = r_LeftUpLeg@R_hips
    #     R_leftLeg = r_LeftLeg@R_leftUpLeg
    #     R_leftFoot = r_LeftFoot@R_leftLeg
    #     return {"translation":x_hips, "Hips":R_hips, "Spine":R_spine, "Spine1":R_spine1, "Spine2":R_spine2, "Spine3":R_spine3,
    #             "Neck":R_neck, "Head":R_head, "RightShoulder":R_rightShoulder, "RightArm":R_rightArm, "RightForeArm":R_rightForeArm, 
    #             "RightHand":R_rightHand, "LeftShoulder":R_leftShoulder, "LeftArm":R_leftArm, "LeftForeArm":R_leftForeArm, 
    #             "LeftHand":R_leftHand, "RightUpLeg":R_rightUpLeg, "RightLeg":R_rightLeg, "RightFoot":R_rightFoot,
    #             "LeftUpLeg":R_leftUpLeg, "LeftLeg":R_leftLeg, "LeftFoot":R_leftFoot}

    # # traj:将要读取的traj文件; index: 该关节在数组中的索引位置; idx:某一帧的帧数
    # def load_joint_rotation(traj, index, idx):
    #     if(index[1]-index[0] == 3):
    #         joint_rotation = traj[idx][index[0]:index[1]]
    #         return get_rotation_matrix(joint_rotation[0], joint_rotation[1], joint_rotation[2])

    # def get_rotation_matrix(x_rad, y_rad, z_rad):
    #     Rx = torch.tensor([[1, 0, 0],
    #                         [0, math.cos(x_rad), -math.sin(x_rad)],
    #                         [0, math.sin(x_rad), math.cos(x_rad)]])
    #     Ry = torch.tensor([[math.cos(y_rad), 0, math.sin(y_rad)],
    #                         [0, 1, 0],
    #                         [-math.sin(y_rad), 0, math.cos(y_rad)]])
    #     Rz = torch.tensor([[math.cos(z_rad), -math.sin(z_rad), 0],
    #                         [math.sin(z_rad), math.cos(z_rad), 0],
    #                         [0, 0, 1]])
    #     return Rx@Ry@Rz
    
    # dir = '0213_take_06'
    # bvh_file = dir + ".bvh"
    # ind_frame_in_mocap = 0
    # traj_path = '/home/liumin/litianyi/workspace/data/datasets/data'
    # dataset_path = '/home/liumin/litianyi/workspace/data/datasets'
    # bvh_path = os.path.join(dataset_path, "traj", bvh_file)
    # traj_file = os.path.join(traj_path, dir + '.npy')
    # with open(bvh_path) as f:
    #     mocap = Bvh(f.read())
    # traj = np.load(traj_file)[:-1]
    # print("traj shape: ", traj.shape)
    # rotation = _load_rotation(traj, ind_frame_in_mocap)
    # offset = _load_offset(mocap)
    # keypoints = _load_keypoint_positon(rotation, offset)
  
    # d_max = torch.max(keypoints, dim=1)[0].unsqueeze(1)
    # # d_max shape: (batch)->(batch, 1)
    # d_min = torch.min(keypoints, dim=1)[0].unsqueeze(1)
    # # print("d_min shape: ", d_min.shape)
    # dst = d_max - d_min
    # keypoints = ((keypoints - d_min) / dst - 0.5) / 0.5
    # keypoints = keypoints.numpy().squeeze(0)
    # print("keypoints: ", keypoints)
    # DrawSkeleton(keypoints)
    csv_path = "/home/liumin/litianyi/workspace/data/datasets/traj/0213_take_06_worldpos.csv"
    df = pd.read_csv(csv_path,usecols=[1,2,3,4,5,6,7,8,9,10,11,12,19,20,21,22,23,24,25,26,27,34,35,36,37,38,
                                    39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,121,122,123,
                                    124,125,126,127,128,129,130,131,132,196,197,198,199,200,201,])
    step = 4
    keypoints = df.head(1).values.squeeze(0)
    keypoint = np.concatenate([keypoints[0:3],keypoints[-6:],keypoints[45:57],keypoints[33:45],keypoints[12:21],
                        keypoints[3:12]])
    
    offset = np.zeros(3)
    offset[0] = keypoint[0]
    offset[1] = keypoint[1]
    offset[2] = keypoint[2]

    d_max = np.max(keypoint)
    d_min = np.min(keypoint)
    
    DrawSkeleton(keypoint, offset=offset)