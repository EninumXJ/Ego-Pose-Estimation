import os
import yaml
from torch.utils.data import Dataset
import torch
from PIL import Image
from bvh import Bvh
import pickle
import math
import numpy as np
import torch.nn.functional as F
import pandas as pd
# from ego_pose.camera_pose_recover import *
# from mocap.pose import load_bvh_file, interpolated_traj

subject_config = "meta_subject_01.yaml"

def get_rotation_matrix(x_rad, y_rad, z_rad):
    Rx = torch.tensor([[1, 0, 0],
                        [0, math.cos(x_rad), -math.sin(x_rad)],
                        [0, math.sin(x_rad), math.cos(x_rad)]])
    Ry = torch.tensor([[math.cos(y_rad), 0, math.sin(y_rad)],
                        [0, 1, 0],
                        [-math.sin(y_rad), 0, math.cos(y_rad)]])
    Rz = torch.tensor([[math.cos(z_rad), -math.sin(z_rad), 0],
                        [math.sin(z_rad), math.cos(z_rad), 0],
                        [0, 0, 1]])
    return Rx@Ry@Rz

def load_joint_offset(mocap, joint):
    joint_offset = mocap.joint_offset(joint)
    return torch.Tensor([[joint_offset[0], joint_offset[1], joint_offset[2]]]).T

# traj:将要读取的traj文件; index: 该关节在数组中的索引位置; idx:某一帧的帧数
def load_joint_rotation(traj, index, idx):
    if(index[1]-index[0] == 3):
        joint_rotation = traj[idx][index[0]:index[1]]
        return get_rotation_matrix(joint_rotation[0], joint_rotation[1], joint_rotation[2])

class MoCapDataset(Dataset):
    def __init__(self, dataset_path, config_path, image_tmpl, image_transform=None, mocap_fr=30, test_mode=False):
        with open(config_path, 'r') as f:
            config = yaml.load(f.read(),Loader=yaml.FullLoader)
        self.dataset_path = dataset_path
        self.capture = config['capture']  # frame rate of video
        self.test_mode = test_mode
        if self.test_mode == False:
            self.data_list = config['train']
            self.data_sync = [config['video_mocap_sync'][i] for i in self.data_list]
        else:
            self.data_list = config['test']
            self.data_sync = [config['video_mocap_sync'][i] for i in self.data_list]
        self.mocap_fr = mocap_fr
        self.image_tmpl = image_tmpl
        self.transform = image_transform

        self.data_dict = []
        length = 0
        for i in range(len(self.data_sync)):
            self.data_dict.append(range(length, length + self.data_sync[i][2] - self.data_sync[i][1]))
            length += self.data_sync[i][2] - self.data_sync[i][1]

    def _load_image(self, directory, idx):
        return Image.open(os.path.join(directory, self.image_tmpl.format(idx))).convert('RGB')

    # load mocap offset from bvh file
    def _load_offset(self, mocap):
        # Hips->Spine->Spine1->Spine2->Spine3->Neck->Head
        joint_name = ['Hips', 'Spine', 'Spine1', 'Spine2', 'Spine3', 'Neck', 'Head',
                        'RightShoulder', 'RightArm', 'RightForeArm', 'RightHand',
                        'LeftShoulder', 'LeftArm', 'LeftForeArm', 'LeftHand',
                        'RightUpLeg', 'RightLeg', 'RightFoot',
                        'LeftUpLeg', 'LeftLeg', 'LeftFoot']
        joint_offset = {}
        for joint in joint_name:
            offset_ = mocap.joint_offset(joint)
            offset = torch.Tensor([[offset_[0], offset_[1], offset_[2]]]).T
            joint_offset[joint] = offset
        return joint_offset
            
    def _load_rotation(self, traj, idx):
        # bvh rotation: X axis, Y axis, Z axis
        # Hip has 6 channels: [translation, rotation]
        # idx: current frame in a bvh file
        # traj = pickle.load(open(directory, 'rb'))
        x_hips = torch.Tensor([[traj[idx][0], traj[idx][1], traj[idx][2]]]).T
        r_hips = load_joint_rotation(traj, (3,6), idx) 
        r_spine = load_joint_rotation(traj, (6,9), idx)     
        r_spine1 = load_joint_rotation(traj, (9,12), idx) 
        r_spine2 = load_joint_rotation(traj, (12,15), idx)
        r_spine3 = load_joint_rotation(traj, (15,18), idx) 
        r_neck = load_joint_rotation(traj, (18,21), idx) 
        r_head = load_joint_rotation(traj, (21,24), idx) 
        r_RightShoulder = load_joint_rotation(traj, (24,27), idx)  
        r_RightArm = load_joint_rotation(traj, (27,30), idx) 
        r_RightForeArm = load_joint_rotation(traj, (30,33), idx) 
        r_RightHand = load_joint_rotation(traj, (33,36), idx) 
        r_LeftShoulder = load_joint_rotation(traj, (36, 39), idx)   
        r_LeftArm = load_joint_rotation(traj, (39, 42), idx) 
        r_LeftForeArm = load_joint_rotation(traj, (42, 45), idx) 
        r_LeftHand = load_joint_rotation(traj, (45, 48), idx)
        r_RightUpLeg = load_joint_rotation(traj, (48, 51), idx) 
        r_RightLeg = load_joint_rotation(traj, (51, 54), idx) 
        r_RightFoot = load_joint_rotation(traj, (54, 57), idx) 
        r_LeftUpLeg = load_joint_rotation(traj, (57, 60), idx) 
        r_LeftLeg = load_joint_rotation(traj, (60, 63), idx) 
        r_LeftFoot = load_joint_rotation(traj, (63, 66), idx) 
        
        # R_hips = r_hips
        # R_spine = R_hips@r_spine
        # R_spine1 = R_spine@r_spine1
        # R_spine2 = R_spine1@r_spine2
        # R_spine3 = R_spine2@r_spine3
        # R_neck = R_spine3@r_neck
        # R_head = R_neck@r_head
        # R_rightShoulder = R_spine3@r_RightShoulder
        # R_rightArm = R_rightShoulder@r_RightArm
        # R_rightForeArm = R_rightArm@r_RightForeArm
        # R_rightHand = R_rightForeArm@r_RightHand
        # R_leftShoulder = R_spine3@r_LeftShoulder
        # R_leftArm = R_leftShoulder@r_LeftArm
        # R_leftForeArm = R_leftArm@r_LeftForeArm
        # R_leftHand = R_leftForeArm@r_LeftHand
        # R_rightUpLeg = R_hips@r_RightUpLeg
        # R_rightLeg = R_rightUpLeg@r_RightLeg
        # R_rightFoot = R_rightLeg@r_RightFoot
        # R_leftUpLeg = R_hips@r_LeftUpLeg
        # R_leftLeg = R_leftUpLeg@r_LeftLeg
        # R_leftFoot = R_leftLeg@r_LeftFoot

        R_hips = r_hips
        R_spine = r_spine@R_hips
        R_spine1 = r_spine1@R_spine
        R_spine2 = r_spine2@R_spine1
        R_spine3 = r_spine3@R_spine2
        R_neck = r_neck@R_spine3
        R_head = r_head@R_neck
        R_rightShoulder = r_RightShoulder@R_spine3
        R_rightArm = r_RightArm@R_rightShoulder
        R_rightForeArm = r_RightForeArm@R_rightArm
        R_rightHand = r_RightHand@R_rightForeArm
        R_leftShoulder = r_LeftShoulder@R_spine3
        R_leftArm = r_LeftArm@R_leftShoulder
        R_leftForeArm = r_LeftForeArm@R_leftArm
        R_leftHand = r_LeftHand@R_leftForeArm
        R_rightUpLeg = r_RightUpLeg@R_hips
        R_rightLeg = r_RightLeg@R_rightUpLeg
        R_rightFoot = r_RightFoot@R_rightLeg
        R_leftUpLeg = r_LeftUpLeg@R_hips
        R_leftLeg = r_LeftLeg@R_leftUpLeg
        R_leftFoot = r_LeftFoot@R_leftLeg
        return {"translation":x_hips, "Hips":R_hips, "Spine":R_spine, "Spine1":R_spine1, "Spine2":R_spine2, "Spine3":R_spine3,
                "Neck":R_neck, "Head":R_head, "RightShoulder":R_rightShoulder, "RightArm":R_rightArm, "RightForeArm":R_rightForeArm, 
                "RightHand":R_rightHand, "LeftShoulder":R_leftShoulder, "LeftArm":R_leftArm, "LeftForeArm":R_leftForeArm, 
                "LeftHand":R_leftHand, "RightUpLeg":R_rightUpLeg, "RightLeg":R_rightLeg, "RightFoot":R_rightFoot,
                "LeftUpLeg":R_leftUpLeg, "LeftLeg":R_leftLeg, "LeftFoot":R_leftFoot}

    # traj: .p文件名称
    # idx: 当前的视频帧 
    def _load_transform(self, offset, traj, idx):
        #bvh_file = os.path.join(traj[:-5], '.bvh')  ## '0213_take_01_traj.p'->'0213_take_01'+'.bvh'
        rotation = self._load_rotation(traj, idx)
        # print("frame: ", idx)
        Translation = rotation["translation"] + rotation["Spine"]@offset["Spine"] + rotation["Spine1"]@offset["Spine1"] + \
                      rotation["Spine2"]@offset["Spine2"] + rotation["Spine3"]@offset["Spine3"] + rotation["Neck"]@offset["Neck"] + rotation["Head"]@offset["Head"]
        Rotation = rotation["Head"]
        return Translation.T, Rotation
    
    def _load_f_u(self, rotation, offset):
        # bvh_file = os.path.join(traj_file[:-5], '.bvh')  ## '0213_take_01_traj.p'->'0213_take_01'+'.bvh'
        f = rotation["Head"]@offset["Head"]
        F.normalize(f, dim=1)
        # 我们假设u向量是f向量绕y轴顺时针旋转90度
        u = get_rotation_matrix(0, -90/180*math.pi, 0)@f
        return f.T, u.T

    # 读取身体各个关节在本地坐标系中的坐标位置 一个包含51个元素的tensor
    def _load_keypoint_positon(self, rotation, offset):
        # hips = rotation['translation']@rotation['Hips']
        hips = rotation['translation']
        spine3 = hips + rotation['Spine']@offset['Spine'] \
                    + rotation['Spine1']@offset['Spine1'] \
                    + rotation['Spine2']@offset['Spine2'] \
                    + rotation['Spine3']@offset['Spine3'] 
        neck = spine3 + rotation['Neck']@offset['Neck']
        head = neck + rotation['Head']@offset['Head']
        RightShoulder = spine3 + rotation['RightShoulder']@offset['RightShoulder']
        RightArm = RightShoulder + rotation['RightArm']@offset['RightArm']
        RightForeArm = RightArm + rotation['RightForeArm']@offset['RightForeArm']
        RightHand = RightForeArm + rotation['RightHand']@offset['RightHand']
        LeftShoulder = spine3 + rotation['LeftShoulder']@offset['LeftShoulder']
        LeftArm = LeftShoulder + rotation['LeftArm']@offset['LeftArm']
        LeftForeArm = LeftArm + rotation['LeftForeArm']@offset['LeftForeArm']
        LeftHand = LeftForeArm + rotation['LeftHand']@offset['LeftHand']
        RightUpLeg = hips + rotation['RightUpLeg']@offset['RightUpLeg']
        RightLeg = RightUpLeg + rotation['RightLeg']@offset['RightLeg']
        RightFoot = RightLeg + rotation['RightFoot']@offset['RightFoot']
        LeftUpLeg = hips + rotation['LeftUpLeg']@offset['LeftUpLeg']
        LeftLeg = LeftUpLeg + rotation['LeftLeg']@offset['LeftLeg']
        LeftFoot = LeftLeg + rotation['LeftFoot']@offset['LeftFoot']
        return torch.cat([hips.T, neck.T, head.T, RightShoulder.T, RightArm.T, RightHand.T, LeftShoulder.T, LeftArm.T,
                          LeftHand.T, RightUpLeg.T, RightLeg.T, RightFoot.T, LeftUpLeg.T, LeftLeg.T, LeftFoot.T], dim=-1)

    def _build_motion(self, offset, traj, index):
        R = []
        D = []
        for i in range(index-31,index):
            d, r = self._load_transform(offset, traj, i)  # translation and rotation
            D.append(d)   ### d = 1*3 tensor
            R.append(r)
        
        return torch.stack(R, 0), torch.stack(D, 0)

    def load_keypoints(self, filepath, ind_frame, length):
        df = pd.read_csv(filepath,usecols=[1,2,3,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,
                                   40,41,42,43,44,45,46,47,48,49,50,51,70,71,72,73,74,75,
                                   82,83,84,85,86,87,88,89,90,100,101,102,103,104,105,106,107,108])
        step = 4    # mocap_fr=120, video_fr=30!
        ind_in_csv = (ind_frame-length+1)*step  # [0,4,8,...]
        # print("ind_in_csv: ", (ind_frame-length+1)*step)
        keypoints = df.iloc[ind_in_csv:(ind_frame+1)*step:step].values
        #(hips,spine1,neck1,head,rightarm,rightforearm,righthand,leftarm,leftforearm,
        # lefthand,rightupleg,rigthleg,rightfoot,leftupleg,leftleg,leftfoot)
        keypoint = np.concatenate([keypoints[:,0:3],keypoints[:,6:9],keypoints[:,33:39],keypoints[:,12:21],
                           keypoints[:,24:33],keypoints[:,39:48], keypoints[:,48:57]], axis=1)
        # keypoint shape: (length, 48)
        ## numpy->tensor
        keypoint = torch.from_numpy(keypoint)
        if keypoint.shape[0] > 1 and keypoint.shape[0] < length:
            # print("keypoint shape: ", keypoint.shape)
            zero = torch.zeros(1, 48)
            keypoint = torch.cat([keypoint, zero], dim=0)
            # print("keypoint shape: ", keypoint.shape)
        offset = torch.zeros([16, length, 3])
         ### 将root平移到原点处
        offset[..., 0] = keypoint[..., 0]
        offset[..., 1] = keypoint[..., 1]
        offset[..., 2] = keypoint[..., 2]
        # shape: (length,48)-> 3*(length,16)
        pos_x = keypoint[...,0:keypoint.shape[-1]:3]
        pos_y = keypoint[...,1:keypoint.shape[-1]:3]
        pos_z = keypoint[...,2:keypoint.shape[-1]:3]
        pos_x = pos_x - offset.permute(1,0,2)[...,0]
        pos_y = pos_y - offset.permute(1,0,2)[...,1]
        pos_z = pos_z - offset.permute(1,0,2)[...,2]
        # shape: (length,16)->(length,16,1)->(length,16,3)->(length,48)
        keypoint = torch.cat([pos_x.unsqueeze(-1), pos_y.unsqueeze(-1), pos_z.unsqueeze(-1)], dim=-1).reshape(length,-1)
        ### 将keypoints映射到[-1,1]之间
        d_max = torch.max(keypoint, dim=-1)[0].unsqueeze(1)
        # d_max shape: (batch)->(batch, 1)
        d_min = torch.min(keypoint, dim=-1)[0].unsqueeze(1)
        # print("d_min shape: ", d_min.shape)
        dst = d_max - d_min
        keypoint = ((keypoint - d_min) / (dst+0.00001) - 0.5) / 0.5 
        # shape: (length,48)
        return keypoint

    def __getitem__(self, index):
        ind_bool = [index in i for i in self.data_dict]
        ind = ind_bool.index(True)  # ind表示该index属于第ind个视频
        ind_frame_in_mocap = index - self.data_dict[ind][0] + self.data_sync[ind][1] #确定index所对应的mocap中的帧数
        #ind_frame_in_video = ind_frame_in_mocap - 4
        ind_frame_in_video = ind_frame_in_mocap + self.data_sync[ind][0]
        # print("index: ", index)
        # print("ind_frame_in_video: ", ind_frame_in_video)
        # print("ind_frame_in_mocap: ", ind_frame_in_mocap)
        # if(ind_frame_in_video < 32):
        #     print("index out of bounds")
        #     return
        dir = self.data_list[ind]
        print("dir: ", dir)
        image_dir = os.path.join(self.dataset_path, "fpv_frames", dir)
        feature_path = os.path.join(self.dataset_path, "features", dir, "feature_30frames.npy")
        feature = np.load(feature_path)
        # traj_path = "data/" + traj_file
        # bvh_file = dir + ".bvh"
        # bvh_path = os.path.join(self.dataset_path, "traj", bvh_file)
        # with open(bvh_path) as f:
        #     mocap = Bvh(f.read())
        # traj = np.load(traj_path)[:-1]
        image = self._load_image(image_dir, ind_frame_in_video)
        # rotation = self._load_rotation(traj, ind_frame_in_mocap)
        # offset = self._load_offset(mocap)
       
        motion_np = feature[ind_frame_in_video+1,:]
        motion = torch.from_numpy(motion_np).type(torch.float32).unsqueeze(0)
        # 获取文件夹的数字前缀：91_09_drunk_walk->91_09
        if dir[2] == '_':
            front_num = dir[0:5]
        else:
            front_num = dir[0:6]
        keypoints_path = os.path.join(self.dataset_path, "keypoints", front_num+"_worldpos.csv")
        keypoints_ = self.load_keypoints(keypoints_path, ind_frame_in_mocap +1, 1)
        # print("keypoints_ shape: ", keypoints_.shape)
        keypoints = keypoints_[:, 6:]
        keypoints_x = keypoints[:,0:45:3]
        keypoints_y = keypoints[:,1:45:3]
        keypoints_z = keypoints[:,2:45:3]
        # f = keypoints_[:, 0:3]
        # u = keypoints_[:, 3:6]
        ### 将keypoints映射到[-1,1]之间
        # d_max = torch.max(keypoints, dim=1)[0].unsqueeze(1)
        # # d_max shape: (batch)->(batch, 1)
        # d_min = torch.min(keypoints, dim=1)[0].unsqueeze(1)
        # # print("d_min shape: ", d_min.shape)
        # dst = d_max - d_min
        # # keypoints = torch.cat((keypoints_x, keypoints_y, keypoints_z), dim=-1)
        # keypoints = ((keypoints - d_min) / dst - 0.5) / 0.5 
        # label = torch.cat([f, u, keypoints], dim=1) 
        label = keypoints   
        # print("label shape: ", label.shape)
        return self.transform(image), label, motion

    def __len__(self):
        len = 0 
        for i in self.data_sync:
            temp = i[2] - i[1]
            len += temp
        return len
    

class EgoMotionDataset(Dataset):
    def __init__(self, dataset_path, config_path, image_tmpl, image_transform=None, mocap_fr=30, L=20, test_mode=False, scene='lab'):
        with open(config_path, 'r') as f:
            config = yaml.load(f.read(), Loader=yaml.FullLoader)
        self.dataset_path = dataset_path
        self.test_mode = test_mode
        if self.test_mode == False:
            self.data_list = config['train']
        else:
            self.data_list = config['test']
           
        self.mocap_frames = config['mocap_frames']
        self.video_frames = config['video_frames']
        self.mocap_fr = mocap_fr
        self.image_tmpl = image_tmpl
        self.transform = image_transform
        self.scene = scene
        self.length = L
        self.data_dict = []
        self.dir_name = []   # "02_01_walk/1"
        len = 0
        for i in self.data_list:  # 02_01_walk
            for j in self.video_frames[i][self.scene]: # 1,2,...,6
                self.dir_name.append(i+'_'+str(j))
                self.data_dict.append(range(len, len + min(self.video_frames[i][self.scene][j]+1, self.mocap_frames[i])))
                len += min(self.video_frames[i][self.scene][j]+1, self.mocap_frames[i])

    def _load_image(self, directory, idx):
        return Image.open(os.path.join(directory, self.image_tmpl.format(idx))).convert('RGB')
    
    # 读取人体骨骼位置
    def load_keypoints(self, filepath, ind_frame, length):
        df = pd.read_csv(filepath,usecols=[1,2,3,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,
                                   40,41,42,43,44,45,46,47,48,49,50,51,70,71,72,73,74,75,
                                   82,83,84,85,86,87,88,89,90,100,101,102,103,104,105,106,107,108])
        step = 4    # mocap_fr=120, video_fr=30!
        ind_in_csv = (ind_frame-length+1)*step  # [0,4,8,...]
        # print("ind_in_csv: ", (ind_frame-length+1)*step)
        keypoints = df.iloc[ind_in_csv:(ind_frame+1)*step:step].values
        #(hips,spine1,neck1,head,rightarm,rightforearm,righthand,leftarm,leftforearm,
        # lefthand,rightupleg,rigthleg,rightfoot,leftupleg,leftleg,leftfoot)
        keypoint = np.concatenate([keypoints[:,0:3],keypoints[:,6:9],keypoints[:,33:39],keypoints[:,12:21],
                           keypoints[:,24:33],keypoints[:,39:48], keypoints[:,48:57]], axis=1)
        # keypoint shape: (length, 48)
        ## numpy->tensor
        keypoint = torch.from_numpy(keypoint)
        if keypoint.shape[0] > 1 and keypoint.shape[0] < length:
            # print("keypoint shape: ", keypoint.shape)
            zero = torch.zeros(1, 48)
            keypoint = torch.cat([keypoint, zero], dim=0)
            # print("keypoint shape: ", keypoint.shape)
        offset = torch.zeros([16, length, 3])
         ### 将root平移到原点处 
        offset[..., 0] = keypoint[..., 0]
        offset[..., 1] = keypoint[..., 1]
        offset[..., 2] = keypoint[..., 2]
        # shape: (length,48)-> 3*(length,16)
        pos_x = keypoint[...,0:keypoint.shape[-1]:3]
        pos_y = keypoint[...,1:keypoint.shape[-1]:3]
        pos_z = keypoint[...,2:keypoint.shape[-1]:3]
        pos_x = pos_x - offset.permute(1,0,2)[...,0]
        pos_y = pos_y - offset.permute(1,0,2)[...,1]
        pos_z = pos_z - offset.permute(1,0,2)[...,2]
        # shape: (length,16)->(length,16,1)->(length,16,3)->(length,48)
        keypoint = torch.cat([pos_x.unsqueeze(-1), pos_y.unsqueeze(-1), pos_z.unsqueeze(-1)], dim=-1).reshape(length,-1).float()
        ### 将keypoints映射到[-1,1]之间
        d_max = torch.max(keypoint, dim=-1)[0].unsqueeze(1)
        # d_max shape: (batch)->(batch, 1)
        d_min = torch.min(keypoint, dim=-1)[0].unsqueeze(1)
        # print("d_min shape: ", d_min.shape)
        dst = d_max - d_min
        keypoint = ((keypoint - d_min) / (dst+0.00001) - 0.5) / 0.5 
        # shape: (length,48)
        # print("keypoint: ", keypoint)
        return keypoint

    def __len__(self):
        return self.data_dict[-1][-1]

    def __getitem__(self, index):
        ind_bool = [index in i for i in self.data_dict]
        ind = ind_bool.index(True)  # ind表示该index属于第ind个视频
        ind_frame = index - self.data_dict[ind][0]

        dir = self.dir_name[ind][:-2]
        # print("dir: ", dir)
        sub_dir = self.dir_name[ind][-1]
        # print(dir, sub_dir)
        image_dir = os.path.join(self.dataset_path, self.scene, dir, sub_dir)
        image = self._load_image(image_dir, ind_frame)
        feature_path = os.path.join(self.dataset_path, "features", dir, self.scene, sub_dir, "feature_01frames.npy")
        feature = np.load(feature_path)
        # 获取文件夹的数字前缀：91_09_drunk_walk->91_09
        if dir[2] == '_':
            front_num = dir[0:5]
        else:
            front_num = dir[0:6]
        keypoints_path = os.path.join(self.dataset_path, "keypoints", front_num+"_worldpos.csv")
        
        ## 特征的维度必须是31
        if ind_frame-30 < 0:
            motion_np = feature[0:ind_frame+1,:]
            motion_zero = np.zeros((30-ind_frame,13))
            motion_np = np.concatenate([motion_zero, motion_np], axis=0)
        else:
            motion_np = feature[ind_frame-30:ind_frame+1,:]
        # print("motion_np shape: ", motion_np.shape)
        motion = torch.from_numpy(motion_np).type(torch.float32).unsqueeze(0).permute(0,2,1)
        # print("motion shape: ", motion.shape)
        keypoints = self.load_keypoints(keypoints_path, ind_frame, 1).type(torch.float32)
        # print("keypoints_ shape: ", keypoints_.shape)
        label = keypoints
        # print("label shape: ", label.shape)
        return self.transform(image), label, motion

if __name__=='__main__':
    config_path = '/home/litianyi/data/EgoMotion/meta_remy.yml'
    dataset_path = '/home/litianyi/data/EgoMotion/'
    import torchvision
    
    class Scale():
        def __init__(self, size, interpolation=Image.BILINEAR):
            self.worker = torchvision.transforms.Resize(size, interpolation)

        def __call__(self, img):
            return self.worker(img)
        
    class ToTorchFormatTensor(object):
    # Converts a PIL.Image (RGB) or numpy.ndarray (H x W x C) in the range [0, 255]
    # to a torch.FloatTensor of shape (C x H x W) in the range [0.0, 1.0] """
        def __init__(self, div=True):
            self.div = div

        def __call__(self, pic):
            if isinstance(pic, np.ndarray):
                # handle numpy array
                img = torch.from_numpy(pic).permute(2, 0, 1).contiguous()
            else:
                # handle PIL Image
                img = torch.ByteTensor(torch.ByteStorage.from_buffer(pic.tobytes()))
                img = img.view(pic.size[1], pic.size[0], len(pic.mode))
                # put it from HWC to CHW format
                # yikes, this transpose takes 80% of the loading time/CPU
                img = img.transpose(0, 1).transpose(0, 2).contiguous()
            return img.float().div(255) if self.div else img.float()

    class GroupNormalize(object):
        def __init__(self, mean, std):
            self.mean = mean
            self.std = std

        def __call__(self, tensor):
            rep_mean = self.mean * (tensor.size()[0]//len(self.mean))
            rep_std = self.std * (tensor.size()[0]//len(self.std))

            # TODO: make efficient
            for t, m, s in zip(tensor, rep_mean, rep_std):
                t.sub_(m).div_(s)

            return tensor

    dataset = EgoMotionDataset(dataset_path=dataset_path,
                               config_path=config_path,
                               image_tmpl="{:04d}.jpg",
                               image_transform=torchvision.transforms.Compose([
                                                Scale(256),
                                                ToTorchFormatTensor(),
                                                GroupNormalize(
                                                    mean=[.485, .456, .406],
                                                    std=[.229, .224, .225])
                                                ]), 
                               L=20,
                               test_mode=False)
    from torch.utils.data import DataLoader
    train_loader = DataLoader(dataset=dataset, batch_size=16, 
                              shuffle=True,num_workers=8, pin_memory=True)
    for i, (image, label, motion) in enumerate(train_loader):
        # print("image shape: ", image.shape)
        print("motion shape: ", motion.shape)
        # print("label shape: ", label.shape)
    # with open(config_path, 'r') as f:
    #     config = yaml.load(f.read(), Loader=yaml.FullLoader)
    # data_list = config['train']
    # mocap_frames = config['mocap_frames']
    # video_frames = config['video_frames']
    # data_dict = []
    # scene = 'lab'
    # len = 0
    # for i in data_list:  # 02_01_walk
    #     for j in video_frames[i][scene]: # 1,2,...,6
    #         # print("dir: ", i)
    #         # print("sub_dir: ", j)
    #         # print("video frames: ", video_frames[i][scene][j])
    #         # print("mocap frames: ", mocap_frames[i])
    #         data_dict.append(range(len, len + min(video_frames[i][scene][j]+1, mocap_frames[i])))
    #         len += min(video_frames[i][scene][j]+1, mocap_frames[i])
    # print(data_dict[-1][-1])