import cv2
import numpy as np
import torch

def detectAndDescribe(image):
    # 将彩色图片转换成灰度图
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 建立SIFT生成器
    descriptor = cv2.SIFT_create()
    # 检测SIFT特征点，并计算描述子
    (kps, features) = descriptor.detectAndCompute(gray, None)

    # 将结果转换成NumPy数组
    kps = np.float32([kp.pt for kp in kps])

    # 返回特征点集，及对应的描述特征
    return (kps, features)

def matchKeypoints(kpsA, kpsB, featuresA, featuresB, camera_matrix, ratio=0.75, reprojThresh=4.0, method=1):  # method=1为BF暴力匹配， method=2为Flann匹配
    if method == 1:
        # 建立暴力匹配器
        matcher = cv2.BFMatcher()
        # 使用KNN检测来自A、B图的SIFT特征匹配对，K=2
        rawMatches = matcher.knnMatch(featuresA, featuresB, 2)
        matches = []
        for m in rawMatches:
            # 当最近距离跟次近距离的比值小于ratio值时，保留此匹配对
            if len(m) == 2 and m[0].distance < m[1].distance * ratio:
                # 存储两个点在featuresA, featuresB中的索引值
                matches.append((m[0].trainIdx, m[0].queryIdx))

        # 当筛选后的匹配对大于4时，计算视角变换矩阵
        if len(matches) > 8:
            # 获取匹配对的点坐标
            ptsA = np.float32([kpsA[i] for (_, i) in matches])
            ptsB = np.float32([kpsB[i] for (i, _) in matches])

            # 计算视角变换矩阵
            E, mask = cv2.findEssentialMat(ptsA, ptsB, camera_matrix, method=cv2.FM_RANSAC, prob = 0.999, threshold = 1.0)
            # 返回结果
            return E, ptsA, ptsB

        # 如果匹配对小于4时，返回None
        return None

# max_frames代表取得的连续视频帧数 默认为30
def poseRecover_1(image_path, frame_in_video, image_tmpl, max_frames=30):
    transform = []
    img_path1 = image_path + image_tmpl%(frame_in_video-max_frames)
    imageA = cv2.imread(img_path1)
    kps1, feature1 = detectAndDescribe(imageA)
    if(feature1 is not None):
        feature1 = feature1.astype(np.float32)
    for i in range(frame_in_video-max_frames+1, frame_in_video+1):
        # img_path2 = image_path + "/%05d"%(i) + ".png"
        img_path2 = image_path + image_tmpl%(i)
        # print("img_path2:", img_path2)
        # imageA = cv2.imread(img_path1)
        imageB = cv2.imread(img_path2)
        W = imageA.shape[1]
        kps2, feature2 = detectAndDescribe(imageB)
        if(feature2 is not None):
            feature2 = feature2.astype(np.float32)
        # f = FocalLength /(sensorsize / Width）
        ### 这里的相机内参都是我们假设的
        cx = 36
        cy = 24
        fx = 20.78 / (cx / W)
        fy = 20.78 / (cy / W)
        camera_matrix = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
        if(matchKeypoints(kps1, kps2, feature1, feature2, camera_matrix) != None):
            (E, ptsA, ptsB) = matchKeypoints(kps1, kps2, feature1, feature2, camera_matrix)
            # E, mask = cv2.findEssentialMat(ptsB, ptsA, focal, pp, cv2.RANSAC, 0.999, 1.0)
            T = cv2.recoverPose(E, ptsA, ptsB, camera_matrix)
            # _, R, t, _ = cv2.recoverPose(E, ptsB, ptsA, focal=focal, pp=pp, mask=mask)
            #print("T: ", T)
            R = T[1]  ### rotation
            R = R.reshape(1, 9)
            # print(R.shape)
            t = T[2]  ### translation
            t = t.T
            # print(t.shape)
        else:
            R = np.zeros((1, 9))
            t = np.zeros((1, 3))
        # g_hat = np.zeros((1, 1))
        # g_hat[0, 0] = 0.3*(1-0.5)
        # transform_ = np.concatenate((R, t, g_hat), axis=1)
        transform_ = np.concatenate((R, t), axis=1)
        transform.append(transform_)
    transform = np.array(transform)
    transform = torch.from_numpy(transform).permute(1,2,0).float()
    return transform
    # print("transform shape:", transform.shape)

# max_frames代表取得的连续视频帧数 默认为30
def poseRecover_2(image_path, frame_in_video, image_tmpl, max_frames=30):
    transform = []
    # img_path1 = image_path + "/%05d"%(frame_in_video-max_frames-1) + ".png"
    # imageA = cv2.imread(img_path1)
    # kps1, feature1 = detectAndDescribe(imageA)
    # if(feature1 is not None):
    #         feature1 = feature1.astype(np.float32)
    for i in range(frame_in_video-max_frames+1, frame_in_video+1):
        img_path1 = image_path + image_tmpl%(i-1)
        imageA = cv2.imread(img_path1)
        img_path2 = image_path + image_tmpl%(i)
        imageB = cv2.imread(img_path2)
        W = imageA.shape[1]
        kps1, feature1 = detectAndDescribe(imageA)
        kps2, feature2 = detectAndDescribe(imageB)
        if(feature1 is not None):
            feature1 = feature1.astype(np.float32)
        if(feature2 is not None):
            feature2 = feature2.astype(np.float32)
        # f = FocalLength /(sensorsize / Width）
        ### 这里的相机内参都是我们假设的
        cx = 36
        cy = 24
        fx = 20.78 / (cx / W)
        fy = 20.78 / (cy / W)
        camera_matrix = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
        if(matchKeypoints(kps1, kps2, feature1, feature2, camera_matrix) != None):
            (E, ptsA, ptsB) = matchKeypoints(kps1, kps2, feature1, feature2, camera_matrix)
            # E, mask = cv2.findEssentialMat(ptsB, ptsA, focal, pp, cv2.RANSAC, 0.999, 1.0)
            T = cv2.recoverPose(E, ptsA, ptsB, camera_matrix)
            # _, R, t, _ = cv2.recoverPose(E, ptsB, ptsA, focal=focal, pp=pp, mask=mask)
            #print("T: ", T)
            R = T[1]  ### rotation
            R = R.reshape(1, 9)
            # print(R.shape)
            t = T[2]  ### translation
            t = t.T
            # print(t.shape)
        else:
            R = np.zeros((1, 9))
            t = np.zeros((1, 3))
        g_hat = np.zeros((1, 1))
        g_hat[0, 0] = 0.3*(1-0.5)
        transform_ = np.concatenate((R, t, g_hat), axis=1)
        # transform_ = np.concatenate((R, t), axis=1)*100
        transform.append(transform_)
    transform = np.array(transform)
    transform = torch.from_numpy(transform).permute(1,2,0).float()
    return transform
    # print("transform shape:", transform.shape)

if __name__=='__main__':
    # image_path = '/home/liumin/litianyi/workspace/data/datasets/fpv_frames/0213_take_01/'
    # frame_in_video = 50
    # L = 15
    # for i in reversed(range(L)):
    #     try:
    #         motion = poseRecover_1(image_path, frame_in_video-i, max_frames=15)
    #     except:
    #         motion = torch.zeros([1, 12, 15], dtype=torch.float)
    #     finally:
    #         if i==L-1:
    #             motion_batch = motion 
    #         else:
    #             motion_batch = torch.cat((motion_batch, motion), 0)
    # print(motion_batch.shape)
    import os
    import yaml
    fpv_path = "/home/liumin/litianyi/workspace/data/EgoMotion/lab/"
    meta_path = "/home/liumin/litianyi/workspace/data/EgoMotion/meta_remy.yml"
    with open(meta_path, 'r') as f:
        config = yaml.load(f.read(),Loader=yaml.FullLoader)
    train_list = config['train']
    test_list = config["test"]
    config_path = "/home/liumin/litianyi/workspace/data/EgoMotion/lab/frame_num.yml"
    with open(config_path, 'r') as f:
        num_config = yaml.load(f.read(),Loader=yaml.FullLoader)
    # dir_image_num = num_config["image_number"]
    # print(dir_image_num)
    dirlist = os.listdir(fpv_path)
    
    for dir in dirlist:
        # print(sub_dir)
        for sub_dir in os.listdir(os.path.join(fpv_path, dir)):
            # 1,2,3,...
            transform = torch.zeros([1, 13, 1])
            image_dir = os.path.join(fpv_path, dir, sub_dir)
            print(image_dir)
            ## 创建文件夹
            feature_path = "/home/liumin/litianyi/workspace/data/EgoMotion/features"
            feature_path = os.path.join(feature_path, dir, "lab", sub_dir)
            print("feature path: ", feature_path)
            if not os.path.exists(feature_path):
                os.makedirs(feature_path)
            max_num = num_config[dir][sub_dir]
            for i in range(max_num):
                # 取第i帧与第i-1帧之间的连续变换参数
                if i == 0:
                    motion = torch.zeros([1, 13, 1])
                    transform = torch.cat([transform, motion], dim=0)
                else:
                    try:
                        motion = poseRecover_2(image_dir, i, image_tmpl="/%04d.jpg", max_frames=1)
                    except:
                        motion = torch.zeros([1, 13, 1])       
                    finally:
                        transform = torch.cat([transform, motion], dim=0)
            features = transform.squeeze(-1).numpy()
            print("features shape: ", features.shape)
            save_path = os.path.join(feature_path, "feature_01frames.npy")
            print("save_path: ", save_path)
            np.save(save_path, features)