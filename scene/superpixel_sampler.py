
import os
os.environ["OPENCV_IO_ENABLE_OPENEXR"]="1"
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import shutil
import math
import torch
from scene.superpixel_computer import SlicSuperPixel

import numpy as np
import cv2
from scipy.spatial.transform import Rotation as R
from scene import Scene, GaussianModel
from argparse import ArgumentParser
from arguments import PipelineParams
from utils.loss_utils import l1_loss, ssim
from gaussian_renderer import render, network_gui
from scene.colmap_loader import read_points3D_binary
from tqdm import tqdm

class SuperpixelSampler:
    def __init__(self, args):
        self.args = args
        self.device = "cuda"

        self.scene_path = args.scene_path
        self.colmap_images_path = args.colmap_images
        self.colmap_cameras_path = args.colmap_cameras
        self.rgb_path = args.rgb
        self.depth_path = args.depth
        self.save_path = args.save_path

        # dataset for pre-processing
        self.rgb_list = []
        self.depth_list = []
        self.xyz_list = []
        self.pose_list = []

        # camera params
        self.width = 640
        self.height = 480
        self.intr = np.eye(3, dtype=np.float32)
        self.intr_t = torch.eye(3, dtype=torch.float32, device=self.device)
        self.inv_intr = torch.zeros((3, 3), dtype=torch.float32, device=self.device)
        self.FoVx = None
        self.FoVy = None

        # keyframes
        self.keyframe_idx_list = []
        self.keyframe_gt_img = []
        self.gt_img_list =[]

        # superpixel sampling
        self.iteration_N = 1
        self.region_size = 16
        self.ruler = 50
        self.SSP = SlicSuperPixel(device=self.device, iteration_N=self.iteration_N, ruler=self.ruler, region_size=self.region_size, IsDrawPolyDecomp=False, rgb_dist_threshold=8, rgb_var_dist_threshold=8)
        self.pointcloud_rgb = None
        self.pointcloud_xyz = None

        self.xy_one = None
        self.projection_matrix = None

        # Gaussian Splatting
        self.gaussians = GaussianModel(3, self.device)
        self.scene = Scene(args, self.gaussians)
        self.pipe = PipelineParams(ArgumentParser(description="Training script parameters"))
        self.background = torch.tensor([1, 1, 1], dtype=torch.float32, device=self.device)


        self.full_proj_transform_list = []
        self.world_view_transform_list = []
        self.camera_center_list = []



    def ReadData(self):
        def GetProjectionMatrix(znear, zfar, fovX, fovY):
            tanHalfFovY = math.tan((fovY / 2))
            tanHalfFovX = math.tan((fovX / 2))

            top = tanHalfFovY * znear
            bottom = -top
            right = tanHalfFovX * znear
            left = -right

            with torch.no_grad():
                P = torch.zeros(4, 4, dtype=torch.float32, device=self.device)

                z_sign = 1.0

                P[0, 0] = 2.0 * znear / (right - left)
                P[1, 1] = 2.0 * znear / (top - bottom)
                P[0, 2] = (right + left) / (right - left)
                P[1, 2] = (top + bottom) / (top - bottom)
                P[3, 2] = z_sign
                P[2, 2] = z_sign * zfar / (zfar - znear)
                P[2, 3] = -(zfar * znear) / (zfar - znear)
            return P
        def ReadColmapCameras():
            flag = True
            with open(self.colmap_cameras_path, 'r') as file:
                for line in file:
                    cameras_info = line.strip()
                    if flag:
                        split_info = cameras_info.split(" ")
                        if split_info[0] == "#":
                            continue
                        self.width = int(split_info[2])
                        self.height = int(split_info[3])

                        fx = float(split_info[4])
                        fy = float(split_info[5])
                        cx = float(split_info[6])
                        cy = float(split_info[7])
                        print("focal:", fx, fy, cx, cy, self.width, self.height)

                        self.intr[0][0] = fx
                        self.intr[0][2] = cx
                        self.intr[1][1] = fy
                        self.intr[1][2] = cy
                        self.intr[2][2] = 1

                        self.intr_t[0][0] = fx
                        self.intr_t[0][2] = cx
                        self.intr_t[1][1] = fy
                        self.intr_t[1][2] = cy
                        self.intr_t[2][2] = 1

                        self.inv_intr[0][0] = 1 / fx
                        self.inv_intr[0][2] = -cx / fx
                        self.inv_intr[1][1] = 1 / fy
                        self.inv_intr[1][2] = -cy / fy
                        self.inv_intr[2][2] = 1

                        FoVx = 2 * math.atan(self.width / (2 * fx))
                        FoVy = 2 * math.atan(self.height / (2 * fy))
                        self.FoVx = FoVx
                        self.FoVy = FoVy
                        self.projection_matrix = GetProjectionMatrix(znear=0.01, zfar=100, fovX=FoVx, fovY=FoVy).transpose(0, 1).type(torch.FloatTensor).to(self.device)

                        print("Read Camera parameters", cameras_info)
                        break
        def ReadColmapImages():
            flag = True
            with open(self.colmap_images_path, 'r') as file:
                for line in file:
                    images_info = line.strip()
                    if flag:
                        split_info = images_info.split(" ")
                        if split_info[0] == "#":
                            continue
                        quat = split_info[1:5]
                        tvec = split_info[5:8]

                        pose = torch.eye(4).to(torch.float32).to(self.device)
                        r = R.from_quat([quat[1], quat[2], quat[3], quat[0]])
                        pose[:3, :3] = torch.from_numpy(r.as_matrix()).to(torch.float32).to(self.device)
                        pose[:3, 3] = torch.from_numpy(np.array([float(tvec[0]), float(tvec[1]),
                                                                 float(tvec[2])])).to(torch.float32).to(self.device)
                        pose_t = torch.linalg.inv(pose)
                        self.pose_list.append(pose_t.detach())

                        flag = False
                    else:
                        flag = True
                print("Read ", len(self.pose_list), "Images and converted into Poses")
        def ReadRGBImages():
            with open(self.rgb_path, 'r') as file:
                for line in file:
                    rgb_path = line.strip()
                    rgb = cv2.cvtColor(cv2.imread(rgb_path), cv2.COLOR_BGR2RGB)
                    self.rgb_list.append(rgb)

                    # rgb_torch = torch.from_numpy(rgb).to(self.device)
                    # img_gt = torch.permute(rgb_torch.type(torch.FloatTensor), (2, 0, 1)).to(self.device) / 255.0
                    # self.gt_img_list.append(img_gt.detach())

            print("Read ", len(self.rgb_list), "RGB images")
        def ReadDepthImages():
            u = torch.arange(self.width, dtype=torch.float32)
            for i in range(self.height - 1):
                u = torch.vstack((u, torch.arange(self.width)))
            v = torch.tile(torch.arange(self.height), (1, 1)).T
            for i in range(self.width - 1):
                v = torch.hstack((v, torch.tile(torch.arange(self.height, dtype=torch.float32), (1, 1)).T))
            uv = torch.stack((u, v), dim=2).to(self.device)
            ones = torch.ones((uv.shape[0], uv.shape[1], 1), dtype=torch.float32).to(self.device)
            uv_one = torch.cat((uv, ones), dim=2).to(self.device)
            uv_one = torch.unsqueeze(uv_one, dim=2)
            self.xy_one = torch.tensordot(uv_one, self.inv_intr, dims=([3], [1])).squeeze()

            with open(self.depth_path, 'r') as file:
                for line in file:
                    depth_path = line.strip()
                    d_16bit = cv2.imread(depth_path, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
                    d_32bit = d_16bit.astype(np.float32)
                    self.depth_list.append(d_32bit)


            print("Read ", len(self.depth_list), "Depth images")

        ReadColmapCameras()
        ReadColmapImages()
        ReadRGBImages()
        ReadDepthImages()

    def SelectKeyframe(self):
        angle_threshold = 0.4  # Radian threshold
        shift_threshold = 0.5  # Meter threshold
        prev_kf_pose = None
        prev_kf_inv_pose = None
        for i, pose in enumerate(self.pose_list):
            if prev_kf_pose is None:
                self.keyframe_idx_list.append(i)
                prev_kf_pose = self.pose_list[i].detach()
                prev_kf_inv_pose = torch.linalg.inv(prev_kf_pose)
            else:
                relative_pose = torch.matmul(prev_kf_inv_pose, pose.detach())
                relative_pose = relative_pose / relative_pose[3, 3]

                # Compute trace
                val = float(relative_pose[0][0] + relative_pose[1][1] + relative_pose[2][2])
                if val > 3.0:
                    val = 3.0
                elif val < -1.0:
                    val = -1.0

                angle = math.acos((val - 1) * 0.5)
                shift = torch.norm(relative_pose[:3, 3])

                if angle_threshold <= angle or shift_threshold <= shift:
                    self.keyframe_idx_list.append(i)
                    prev_kf_pose = self.pose_list[i].detach()
                    prev_kf_inv_pose = torch.linalg.inv(prev_kf_pose)
        for i, kf_idx in enumerate(self.keyframe_idx_list):
            pose = self.pose_list[kf_idx].detach()
            world_view_transform = torch.inverse(pose).T.detach()
            camera_center = torch.inverse(world_view_transform)[3, :3]
            full_proj_transform = torch.matmul(world_view_transform, self.projection_matrix)
            self.full_proj_transform_list.append(full_proj_transform.detach())
            self.world_view_transform_list.append(world_view_transform.detach())
            self.camera_center_list.append(camera_center.detach())

            depth = self.depth_list[kf_idx]
            depth_torch = torch.from_numpy(np.array(depth, dtype=np.float32)).detach().to(self.device)
            kf_xyz = torch.mul(self.xy_one.detach(), depth_torch.unsqueeze(dim=2))
            self.xyz_list.append(kf_xyz.detach())

            rgb_gt = self.rgb_list[kf_idx]
            rgb_torch = torch.from_numpy(rgb_gt)
            img_gt = torch.permute(rgb_torch.type(torch.FloatTensor), (2, 0, 1)).to(self.device) / 255.0
            self.gt_img_list.append(img_gt)


        print("Selected", len(self.keyframe_idx_list), "keyframes")




    def SuperpixelGuidedSampling(self):
        self.superpixel_indices = torch.zeros((2, self.height, self.width), dtype=torch.float32)
        for i in range(self.height):
            for j in range(self.width):
                self.superpixel_indices[0, i, j] = i
                self.superpixel_indices[1, i, j] = j

        pointcloud_xyz = torch.empty((0, 3), dtype=torch.float32, device=self.device)
        pointcloud_rgb = torch.empty((0, 3), dtype=torch.float32, device=self.device)



        with tqdm(total=len(self.keyframe_idx_list)) as pbar:
            for i, kf_idx in enumerate(self.keyframe_idx_list):
                rgb = self.rgb_list[kf_idx]
                rgb_torch = torch.from_numpy(rgb).detach().to(self.device)
                kf_xyz = self.xyz_list[i]
                pose = self.pose_list[kf_idx]

                superpixel_index = self.SSP.ComputeMergedSuperpixel(rgb)
                masked_xyz = kf_xyz[superpixel_index[0, :], superpixel_index[1, :], :]
                masked_rgb = rgb_torch[superpixel_index[0, :], superpixel_index[1, :], :]
                converted_xyz = self.ConvertXYZCamToWorld(torch.transpose(masked_xyz, 1, 0).detach(), pose)

                pointcloud_xyz = torch.cat((pointcloud_xyz, (torch.transpose(converted_xyz, 1, 0))[:, :3]), 0)
                pointcloud_rgb = torch.cat((pointcloud_rgb, masked_rgb), 0)
                pbar.update(1)
                pbar.set_description(f"Superpixel-guided Sampling for keyframes {i + 1}/{len(self.keyframe_idx_list)}")
                pbar.set_postfix({
                    "Sampled points": f"{pointcloud_rgb.shape[0]}"
                })

        self.pointcloud_xyz = pointcloud_xyz
        self.pointcloud_rgb = pointcloud_rgb
        print("Total Sampled points", pointcloud_rgb.shape[0])

    def InitializeGaussian(self):
        self.gaussians = GaussianModel(3, self.device)
        self.scene = Scene(self.args, self.gaussians)
        self.scene.SetSceneInfo(self.pointcloud_xyz, self.pointcloud_rgb, self.save_path)


    def SaveSceneAsPng(self):
        save_path = self.save_path
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        if not os.path.exists(save_path+"/img"):
            os.makedirs(save_path+"/img")

        self.scene.save(save_path)

        for i in range(len(self.pose_list)):
            with torch.no_grad():
                pose = self.pose_list[i].detach()
                world_view_transform = torch.inverse(pose).T.detach()
                camera_center = torch.inverse(world_view_transform)[3, :3]
                full_proj_transform = torch.matmul(world_view_transform, self.projection_matrix)

            render_pkg = render(self.FoVx, self.FoVy, self.height, self.width, world_view_transform,
                                full_proj_transform, camera_center, self.gaussians, self.pipe, self.background,
                                1.0)
            img = render_pkg["render"]
            np_render = torch.permute(img, (1, 2, 0)).detach().cpu().numpy()*255.0
            np_bgr = cv2.cvtColor(np_render, cv2.COLOR_BGR2RGB)
            cv2.imwrite(save_path + f"/img/{i}.png", np_bgr)

    def ConvertXYZWorldToCam(self, xyz, pose):
        with torch.no_grad():
            # ones = torch.ones((1, xyz.shape[1]), dtype=torch.float32, device=self.device)
            # xyz_one = torch.cat((xyz, ones), dim=0)

            world_xyz = torch.matmul(torch.inverse(pose), xyz.T)

            xyz_mask = world_xyz[3, :].ne(0.0)
            masked_world_xyz = world_xyz[:, xyz_mask]

            masked_world_xyz = masked_world_xyz[:, :] / masked_world_xyz[3, :]

        return masked_world_xyz


    def ConvertXYZCamToWorld(self, xyz, pose):
        with torch.no_grad():
            ones = torch.ones((1, xyz.shape[1]), dtype=torch.float32, device=self.device)
            xyz_one = torch.cat((xyz, ones), dim=0)
            world_xyz = torch.matmul(pose, xyz_one)

            xyz_mask = world_xyz[3, :].ne(0.0)
            masked_world_xyz = world_xyz[:, xyz_mask]

            masked_world_xyz = masked_world_xyz[:, :] / masked_world_xyz[3, :]

        return masked_world_xyz
