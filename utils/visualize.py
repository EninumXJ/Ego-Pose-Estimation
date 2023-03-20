import matplotlib
import numpy as np

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
# keypoints: ndarray [1, 51]
def DrawSkeleton(keypoints, offset=None, head1=None, head2=None, 
                 image_name='Skeleton.jpg', dataset='EgoMotion'):
    plt.rcParams['figure.figsize'] = (14.0, 5.0) # 设置figure_size尺寸
    pos_x = keypoints[0:len(keypoints):3]
    pos_z = keypoints[1:len(keypoints):3]
    pos_y = keypoints[2:len(keypoints):3]
    if offset is not None:
        pos_x = pos_x - offset[0]
        pos_y = pos_y - offset[1]
        pos_z = pos_z - offset[2]
    head = keypoints[6:9]

    xp = pos_x.T
    yp = pos_y.T
    zp = pos_z.T
    ax = plt.axes(projection='3d')
    if head1==None and head2==None:
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
    
    radius = 1.2
    ax.set_xlim3d([-radius, radius])
    ax.set_ylim3d([-radius, 20*radius])
    ax.set_zlim3d([-radius, radius])
    ax.view_init(elev=0., azim=0)
    # 拉长y轴
    # x_scale=0.8
    # y_scale=6
    # z_scale=1
    # scale=np.diag([x_scale, y_scale, z_scale, 1.0])
    # scale=scale*(1/scale.max())
    # scale[3,3]=0.2
    scale=np.diag([1, 5, 1, 1.0])
    def short_proj():
        return np.dot(Axes3D.get_proj(ax), scale)

    ax.get_proj=short_proj
    ax.set_box_aspect(aspect = (0.8,3,0.8))
    # ax.set_aspect('equal')

    ax.dist = 6

    if dataset == 'Yuan':
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
    if dataset == 'EgoMotion':
        # 3D scatter
        ax.scatter3D(xp, yp, zp, cmap='Greens')
        # hips, spine1, neck1, head, 
        ax.plot(xp[0:4], yp[0:4], zp[0:4], ls='-', color='gray')
        
        # RightArm, RightForeArm, RightHand
        ax.plot(xp[4:7], yp[4:7], zp[4:7], ls='-', color='blue')
        # connect RightArm and spine1
        ax.plot([xp[1],xp[4]], [yp[1],yp[4]], [zp[1],zp[4]], ls='-', color='blue')
        # LeftArm, LeftForeArm, LeftHand
        ax.plot(xp[7:10], yp[7:10], zp[7:10], ls='-', color='red')
        # connect LeftArm and spine1
        ax.plot([xp[1],xp[7]], [yp[1],yp[7]], [zp[1],zp[7]], ls='-', color='red')
        # RightUpLeg, RightLeg, RightFoot
        ax.plot(xp[10:13], yp[10:13], zp[10:13], ls='-', color='blue')
        # connect hips and RightUpLeg
        ax.plot([xp[0],xp[10]], [yp[0],yp[10]], [zp[0],zp[10]], ls='-', color='blue')
        # LeftUpLeg, LeftLeg, LeftFoot
        ax.plot(xp[13:16], yp[13:16], zp[13:16], ls='-', color='red')
        # connect hips and LeftUpLeg
        ax.plot([xp[0],xp[13]], [yp[0],yp[13]], [zp[0],zp[13]], ls='-', color='red')

    # dx, dy, dz = 5, 30, 5
    # k = 40
    # n = int((dx+10)/k)+1
    # ax.set_xticks(np.linspace(-n, k*(n-1), n))
    # n = int(int(dy+10)/k)+1
    # ax.set_yticks(np.linspace(0, k*(n-1), n))
    # n = int(int(dz+10)/k)+1
    # ax.set_zticks(np.linspace(-n, k*(n-1), n))
    # ax.view_init(azim=0, elev=15)
    # ax.set_box_aspect([dx, dy, dz])
    # ax.auto_scale_xyz([-1, 1], [-1, 21], [-1, 1])
    # ax.set_aspect('equal', adjustable='box')
    # ax.set_box_aspect(aspect = (1,4,1))
    plt.savefig(image_name, dpi=500)

# keypoints: ndarray [1, 45]
def DrawSkeleton45(keypoints, head1=None, head2=None, image_name='Skeleton.jpg'):
    pos_x = keypoints[0:len(keypoints):3]
    pos_y = keypoints[1:len(keypoints):3]
    pos_z = keypoints[2:len(keypoints):3]
    head = keypoints[6:9]

    xp = pos_x.T
    yp = pos_y.T
    zp = pos_z.T
    ax = plt.axes(projection='3d')
    if head1 == None and head2 == None:
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
    ax.view_init(elev=15., azim=70)
    
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


def PlotLPose(keypoints, L, head1=None, head2=None, offset=None, 
              dataset='EgoMotion', image_name='Sequence_Pose.jpg'):
    plt.rcParams['figure.figsize'] = (18.0, 5.0) # 设置figure_size尺寸
    print("keypoints shape", keypoints.shape)
    pos_x = keypoints[:, 0:keypoints.shape[1]:3]  # shape:(20, 16)
    pos_z = keypoints[:, 1:keypoints.shape[1]:3]
    pos_y = keypoints[:, 2:keypoints.shape[1]:3]
    print("pos_x shape: ", pos_x.shape)
    pos_x = pos_x - np.expand_dims(keypoints[:, 0], axis=1)
    pos_y = pos_y - np.expand_dims(keypoints[:, 1], axis=1)
    pos_z = pos_z - np.expand_dims(keypoints[:, 2], axis=1)
    head = keypoints[6:9]

    # xp = pos_x.T
    # yp = pos_y.T
    # zp = pos_z.T
    ax = plt.axes(projection='3d')
    if head1==None and head2==None:
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
    
    radius = 1.2
    ax.set_xlim3d([-radius, radius])
    ax.set_ylim3d([-radius, 20*radius])
    ax.set_zlim3d([-radius, radius])
    ax.view_init(elev=0., azim=0)
    # 拉长y轴
    x_scale=0.8
    y_scale=5
    z_scale=1
    scale=np.diag([x_scale, y_scale, z_scale, 1.0])
    scale=scale*(1.0/scale.max())
    scale[3,3]=1

    def short_proj():
        return np.dot(Axes3D.get_proj(ax), scale)

    ax.get_proj=short_proj
    # ax.set_box_aspect(aspect = (0.8,3,0.8))
    # ax.set_aspect('equal')

    ax.dist = 4

    if dataset == 'Yuan':
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
    if dataset == 'EgoMotion':
        for i in range(1, L, 2):
            # 3D scatter
            xp = pos_x[i, :].T
            yp = pos_y[i, :].T
            zp = pos_z[i, :].T
            yp += i*1
            ax.scatter3D(xp, yp, zp, cmap='Greens')
            # hips, spine1, neck1, head, 
            ax.plot(xp[0:4], yp[0:4], zp[0:4], ls='-', color='gray')
            
            # RightArm, RightForeArm, RightHand
            ax.plot(xp[4:7], yp[4:7], zp[4:7], ls='-', color='blue')
            # connect RightArm and spine1
            ax.plot([xp[1],xp[4]], [yp[1],yp[4]], [zp[1],zp[4]], ls='-', color='blue')
            # LeftArm, LeftForeArm, LeftHand
            ax.plot(xp[7:10], yp[7:10], zp[7:10], ls='-', color='red')
            # connect LeftArm and spine1
            ax.plot([xp[1],xp[7]], [yp[1],yp[7]], [zp[1],zp[7]], ls='-', color='red')
            # RightUpLeg, RightLeg, RightFoot
            ax.plot(xp[10:13], yp[10:13], zp[10:13], ls='-', color='blue')
            # connect hips and RightUpLeg
            ax.plot([xp[0],xp[10]], [yp[0],yp[10]], [zp[0],zp[10]], ls='-', color='blue')
            # LeftUpLeg, LeftLeg, LeftFoot
            ax.plot(xp[13:16], yp[13:16], zp[13:16], ls='-', color='red')
            # connect hips and LeftUpLeg
            ax.plot([xp[0],xp[13]], [yp[0],yp[13]], [zp[0],zp[13]], ls='-', color='red')

    # dx, dy, dz = 5, 30, 5
    # k = 40
    # n = int((dx+10)/k)+1
    # ax.set_xticks(np.linspace(-n, k*(n-1), n))
    # n = int(int(dy+10)/k)+1
    # ax.set_yticks(np.linspace(0, k*(n-1), n))
    # n = int(int(dz+10)/k)+1
    # ax.set_zticks(np.linspace(-n, k*(n-1), n))
    # ax.view_init(azim=0, elev=15)
    # ax.set_box_aspect([dx, dy, dz])
    # ax.auto_scale_xyz([-1, 1], [-1, 21], [-1, 1])
    # ax.set_aspect('equal', adjustable='box')
    # ax.set_box_aspect(aspect = (1,4,1))
    plt.savefig(image_name, dpi=500)

def PlotLPose2D(keypoints, L, head1=None, head2=None, offset=None, 
              dataset='EgoMotion', image_name='Sequence_Pose.jpg', gt_flag=False):
    plt.rcParams['figure.figsize'] = (25.0, 5.0) # 设置figure_size尺寸
    print("keypoints shape", keypoints.shape)
    pos_x = keypoints[:, 0:keypoints.shape[1]:3]  # shape:(20, 16)
    pos_z = keypoints[:, 1:keypoints.shape[1]:3]
    pos_y = keypoints[:, 2:keypoints.shape[1]:3]
    print("pos_x shape: ", pos_x.shape)
    # pos_x = pos_x - np.expand_dims(keypoints[:, 0], axis=1)
    pos_y = pos_y - np.expand_dims(keypoints[:, 1], axis=1)
    # pos_z = pos_z - np.expand_dims(keypoints[:, 2], axis=1)
    head = keypoints[6:9]

    # 取y轴为2D坐标中的x轴，z轴为y轴
    
    plt.figure('Draw')
    if dataset == 'Yuan':
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
    if dataset == 'EgoMotion':
        for i in range(1, L, 2):
            # 3D scatter
            xp = pos_x[i, :].T
            yp = pos_y[i, :].T
            zp = pos_z[i, :].T
            yp += i*1
            if  not gt_flag:
                plt.scatter(yp, zp, c='b', alpha=0.5)
                # hips, spine1, neck1, head, 
                plt.plot(yp[0:4], zp[0:4], ls='-', color='blue')
            
                # RightArm, RightForeArm, RightHand
                plt.plot(yp[4:7], zp[4:7], ls='-', color='blue')
                # connect RightArm and spine1
                plt.plot([yp[1],yp[4]], [zp[1],zp[4]], ls='-', color='blue')
                # LeftArm, LeftForeArm, LeftHand
                plt.plot(yp[7:10], zp[7:10], ls='-', color='blue')
                # connect LeftArm and spine1
                plt.plot([yp[1],yp[7]], [zp[1],zp[7]], ls='-', color='blue')
                # RightUpLeg, RightLeg, RightFoot
                plt.plot(yp[10:13], zp[10:13], ls='-', color='blue')
                # connect hips and RightUpLeg
                plt.plot([yp[0],yp[10]], [zp[0],zp[10]], ls='-', color='blue')
                # LeftUpLeg, LeftLeg, LeftFoot
                plt.plot(yp[13:16], zp[13:16], ls='-', color='blue')
                # connect hips and LeftUpLeg
                plt.plot([yp[0],yp[13]], [zp[0],zp[13]], ls='-', color='blue')
                my_x_ticks = np.arange(0, 20, 1)
                my_y_ticks = np.arange(-2, 2, 0.5)
                plt.xticks(my_x_ticks)
                plt.yticks(my_y_ticks)
                plt.axis("equal")
            else:
                plt.scatter(yp, zp, c='r', alpha=0.5)
                # hips, spine1, neck1, head, 
                plt.plot(yp[0:4], zp[0:4], ls='-', color='red')
            
                # RightArm, RightForeArm, RightHand
                plt.plot(yp[4:7], zp[4:7], ls='-', color='red')
                # connect RightArm and spine1
                plt.plot([yp[1],yp[4]], [zp[1],zp[4]], ls='-', color='red')
                # LeftArm, LeftForeArm, LeftHand
                plt.plot(yp[7:10], zp[7:10], ls='-', color='red')
                # connect LeftArm and spine1
                plt.plot([yp[1],yp[7]], [zp[1],zp[7]], ls='-', color='red')
                # RightUpLeg, RightLeg, RightFoot
                plt.plot(yp[10:13], zp[10:13], ls='-', color='red')
                # connect hips and RightUpLeg
                plt.plot([yp[0],yp[10]], [zp[0],zp[10]], ls='-', color='red')
                # LeftUpLeg, LeftLeg, LeftFoot
                plt.plot(yp[13:16], zp[13:16], ls='-', color='red')
                # connect hips and LeftUpLeg
                plt.plot([yp[0],yp[13]], [zp[0],zp[13]], ls='-', color='red')
                my_x_ticks = np.arange(0, 20, 1)
                my_y_ticks = np.arange(-2, 2, 0.5)
                plt.xticks(my_x_ticks)
                plt.yticks(my_y_ticks)
                plt.axis("equal")
    # dx, dy, dz = 5, 30, 5
    # k = 40
    # n = int((dx+10)/k)+1
    # ax.set_xticks(np.linspace(-n, k*(n-1), n))
    # n = int(int(dy+10)/k)+1
    # ax.set_yticks(np.linspace(0, k*(n-1), n))
    # n = int(int(dz+10)/k)+1
    # ax.set_zticks(np.linspace(-n, k*(n-1), n))
    # ax.view_init(azim=0, elev=15)
    # ax.set_box_aspect([dx, dy, dz])
    # ax.auto_scale_xyz([-1, 1], [-1, 21], [-1, 1])
    # ax.set_aspect('equal', adjustable='box')
    # ax.set_box_aspect(aspect = (1,4,1))
    plt.savefig(image_name, dpi=800)

if __name__=='__main__':
    keypoint = np.array([[ 0.0945,  0.0125,  0.1152,  0.1040,  0.0359,  0.8434,  0.1053,  0.0562,
          1.0000,  0.2202,  0.0418,  0.7179,  0.2825,  0.0251,  0.5322,  0.5138,
          0.1713, -0.0814,  0.0886,  0.1202,  0.7150,  0.0126,  0.1453,  0.5355,
          0.1286,  0.5460,  0.0444,  0.1807, -0.0630,  0.0725,  0.0708, -0.1156,
         -0.4864,  0.0570, -0.1081, -1.0000, -0.0198,  0.0471,  0.0886, -0.0831,
         -0.0822, -0.4649, -0.0998, -0.1110, -0.9777]])
    print(keypoint[0, :].shape)
    # DrawSkeleton(keypoint[0, 6:], keypoint[0, 0:3], keypoint[0, 3:6])
    DrawSkeleton45(keypoint[0, :])