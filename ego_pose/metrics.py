import torch

def MPJPE(joint_num, gt, output):
    # print("gt shape: ", gt.shape)
    # print("output shape: ", output.shape)
    # L = gt.shape[1]
    PE = 0
    for i in range(0, joint_num):
        gt_joint = gt[..., i*3:i*3+3]
        output_joint = output[..., i*3:i*3+3]
        PE += torch.sum(torch.norm(gt_joint - output_joint, dim=-1))
    # convert to mm
    # relative_const =  0.056444 * 1000
    relative_const = 30
    return PE/joint_num*relative_const

def accelerate(output, length):
    pass