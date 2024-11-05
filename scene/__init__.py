#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import os
import random
import json

import torch

from utils.system_utils import searchForMaxIteration
from scene.dataset_readers import sceneLoadTypeCallbacks
from scene.gaussian_model import GaussianModel
from arguments import ModelParams
from utils.camera_utils import cameraList_from_camInfos, camera_to_JSON

class Scene:

    gaussians : GaussianModel

    def __init__(self, args : ModelParams, gaussians : GaussianModel, load_iteration=None, resolution_scales=[1.0]):
        self.model_path = ""
        self.loaded_iter = None
        self.gaussians = gaussians
        self.args = args

        self.train_cameras = {}
        self.test_cameras = {}

    def SetSceneInfoBin(self, dest_path):
        scene_info = sceneLoadTypeCallbacks["SuperpixelGuidedBin"](self.args.colmap_images, self.args.colmap_cameras,
                                                                self.args.ply, self.args.images, xyz, rgb)

        with open(scene_info.ply_path, 'rb') as src_file, open(os.path.join(dest_path, "input.ply"), 'wb') as dest_file:
            dest_file.write(src_file.read())
        json_cams = []
        camlist = []
        if scene_info.test_cameras:
            camlist.extend(scene_info.test_cameras)
        if scene_info.train_cameras:
            camlist.extend(scene_info.train_cameras)
        for id, cam in enumerate(camlist):
            json_cams.append(camera_to_JSON(id, cam))
        with open(os.path.join(dest_path, "cameras.json"), 'w') as file:
            json.dump(json_cams, file)

        self.cameras_extent = scene_info.nerf_normalization["radius"]

        for resolution_scale in [1.0]:
            # Load Training Cameras
            self.train_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.train_cameras, resolution_scale,
                                                                            self.args)
            # Load Test Cameras
            self.test_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.test_cameras, resolution_scale,
                                                                           self.args)
        self.gaussians.create_from_pcd(scene_info.point_cloud, self.cameras_extent)
    def SetSceneInfo(self, xyz, rgb, dest_path):
        scene_info = sceneLoadTypeCallbacks["SuperpixelGuided"](self.args.colmap_images, self.args.colmap_cameras, self.args.ply, self.args.images, xyz, rgb)

        with open(scene_info.ply_path, 'rb') as src_file, open(os.path.join(dest_path, "input.ply"), 'wb') as dest_file:
            dest_file.write(src_file.read())
        json_cams = []
        camlist = []
        if scene_info.test_cameras:
            camlist.extend(scene_info.test_cameras)
        if scene_info.train_cameras:
            camlist.extend(scene_info.train_cameras)
        for id, cam in enumerate(camlist):
            json_cams.append(camera_to_JSON(id, cam))
        with open(os.path.join(dest_path, "cameras.json"), 'w') as file:
            json.dump(json_cams, file)

        self.cameras_extent = scene_info.nerf_normalization["radius"]

        for resolution_scale in [1.0]:
            # Load Training Cameras
            self.train_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.train_cameras, resolution_scale, self.args)
            # Load Test Cameras
            self.test_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.test_cameras, resolution_scale, self.args)
        self.gaussians.create_from_pcd(scene_info.point_cloud, self.cameras_extent)


    def save(self, path):
        point_cloud_path = os.path.join(path, "point_cloud")
        self.gaussians.save_ply(os.path.join(point_cloud_path, "point_cloud.ply"))

    def getTrainCameras(self, scale=1.0):
        return self.train_cameras[scale]

    def getTestCameras(self, scale=1.0):
        return self.test_cameras[scale]