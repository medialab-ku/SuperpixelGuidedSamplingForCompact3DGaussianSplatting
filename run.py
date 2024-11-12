from argparse import ArgumentParser, Namespace
import sys
from scene.superpixel_sampler import SuperpixelSampler
import os
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False
import time
import copy

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    parser.add_argument('--scene_path', type=str, default="")
    parser.add_argument('--colmap_images', type=str, default="")
    parser.add_argument('--colmap_cameras', type=str, default="")
    parser.add_argument('--rgb', type=str, default="")
    parser.add_argument('--depth', type=str, default="")

    parser.add_argument('--images', type=str, default="")
    parser.add_argument('--ply', type=str, default="")
    parser.add_argument('--resolution', default=-1)
    parser.add_argument('--data_device', type=str, default="cuda")
    parser.add_argument('--sh_degree', default=3)
    parser.add_argument('--save_path', type=str, default="")


    args = parser.parse_args(sys.argv[1:])
    print("args", args)
    args.colmap_images = args.scene_path + "/images.txt"
    args.colmap_cameras = args.scene_path + "/cameras.txt"
    args.rgb = args.scene_path + "/rgb_list.txt"
    args.depth = args.scene_path + "/depth_list.txt"
    args.images = args.scene_path + "/images"
    args.ply = args.scene_path + "/points3D.ply"
    args.save_path = args.scene_path + "/save"


    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)

    sps = SuperpixelSampler(args)  # Initialize SuperpixelSampler
    sps.ReadData()
    sps.SelectKeyframe()
    sps.SuperpixelGuidedSampling()
    sps.InitializeGaussian()

    sps.SaveSceneAsPng()

