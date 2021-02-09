import os, sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from core.dataset import KITTI_2012, KITTI_2015, KITTI_pose
from core.evaluation import eval_flow_avg, load_gt_flow_kitti
from core.evaluation import eval_depth
# from core.visualize import Visualizer_debug
from core.visualize import Visualizer_debug, resize_flow, flow_to_image, flow_write_png
from core.networks import Model_depth_pose, Model_flow, Model_flowposenet, Model_depth, Model_geometry
from core.evaluation import load_gt_flow_kitti, load_gt_mask
from core.networks.structures.inverse_warp import pose_vec2mat 
import torch
from tqdm import tqdm
# import pdb
import cv2
import numpy as np
import yaml
from path import Path

from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt



def compute_pose_error(gt, pred):
    RE = 0
    snippet_length = gt.shape[0]
    scale_factor = np.sum(gt[:,:,-1] * pred[:,:,-1])/np.sum(pred[:,:,-1] ** 2)
    ATE = np.linalg.norm((gt[:,:,-1] - scale_factor * pred[:,:,-1]).reshape(-1))
    for gt_pose, pred_pose in zip(gt, pred):
        # Residual matrix to which we compute angle's sin and cos
        R = gt_pose[:,:3] @ np.linalg.inv(pred_pose[:,:3])
        s = np.linalg.norm([R[0,1]-R[1,0],
                            R[1,2]-R[2,1],
                            R[0,2]-R[2,0]])
        c = np.trace(R) - 1
        # Note: we actually compute double of cos and sin, but arctan2 is invariant to scale
        RE += np.arctan2(s,c)

    return ATE/snippet_length, RE/snippet_length


def test_pose_odom(cfg, model):
    print('Evaluate pose using kitti odom. Using model in '+cfg.model_dir)
    dataset_dir = Path(cfg.kitti_odom_dir)
    dataset = KITTI_pose(dataset_dir, cfg.sequences, 3)
    print('{} snippets to test'.format(len(dataset)))
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    errors = np.zeros((len(dataset), 2), np.float32)
    for j, sample in enumerate(tqdm(dataset)):
        imgs = sample['imgs']
        imgl = cv2.resize(imgs[0], (cfg.img_hw[1], cfg.img_hw[0])).astype(np.float32)
        img  = cv2.resize(imgs[1], (cfg.img_hw[1], cfg.img_hw[0])).astype(np.float32)
        imgr = cv2.resize(imgs[2], (cfg.img_hw[1], cfg.img_hw[0])).astype(np.float32)
        imgs = np.concatenate([imgl, img, imgr], 2)
        img_input = torch.from_numpy(imgs / 255.0).float().cuda().unsqueeze(0).permute(0,3,1,2)

        poses = model.infer_pose(img_input)

        poses = poses.detach().cpu()[0]
        poses = torch.cat([poses[0].view(-1,6), torch.zeros(1,6).float(), poses[1].view(-1,6)])

        inv_transform_matrices = pose_vec2mat(poses).numpy().astype(np.float64)

        rot_matrices = np.linalg.inv(inv_transform_matrices[:,:,:3])
        tr_vectors = -rot_matrices @ inv_transform_matrices[:,:,-1:]

        transform_matrices = np.concatenate([rot_matrices, tr_vectors], axis=-1)

        first_inv_transform = inv_transform_matrices[0]
        final_poses = first_inv_transform[:,:3] @ transform_matrices
        final_poses[:,:,-1:] += first_inv_transform[:,-1:]

        ATE, RE = compute_pose_error(sample['poses'], final_poses)
        errors[j] = ATE, RE

    mean_errors = errors.mean(0)
    std_errors = errors.std(0)
    error_names = ['ATE','RE']
    print("Results")
    print("\t {:>10}, {:>10}".format(*error_names))
    print("mean \t {:10.4f}, {:10.4f}".format(*mean_errors))
    print("std \t {:10.4f}, {:10.4f}".format(*std_errors))


def draw_pose_odom(cfg, model):
    print('Evaluate pose using kitti odom. Using model in '+cfg.model_dir)
    dataset_dir = Path(cfg.kitti_odom_dir)
    dataset = KITTI_pose(dataset_dir, cfg.sequences, 3)
    print('{} snippets to test'.format(len(dataset)))
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    errors = np.zeros((len(dataset), 2), np.float32)
    gt_xyz = np.zeros((3, len(dataset)), np.float32)
    last_gt_xyz = np.zeros((3,1), np.float32)
    for j, sample in enumerate(tqdm(dataset)):
        imgs = sample['imgs']
        imgl = cv2.resize(imgs[0], (cfg.img_hw[1], cfg.img_hw[0])).astype(np.float32)
        img  = cv2.resize(imgs[1], (cfg.img_hw[1], cfg.img_hw[0])).astype(np.float32)
        imgr = cv2.resize(imgs[2], (cfg.img_hw[1], cfg.img_hw[0])).astype(np.float32)
        imgs = np.concatenate([imgl, img, imgr], 2)
        img_input = torch.from_numpy(imgs / 255.0).float().cuda().unsqueeze(0).permute(0,3,1,2)

        poses = model.infer_pose(img_input)

        poses = poses.detach().cpu()[0]
        poses = torch.cat([poses[0].view(-1,6), torch.zeros(1,6).float(), poses[1].view(-1,6)])

        inv_transform_matrices = pose_vec2mat(poses).numpy().astype(np.float64)

        rot_matrices = np.linalg.inv(inv_transform_matrices[:,:,:3])
        tr_vectors = -rot_matrices @ inv_transform_matrices[:,:,-1:]

        transform_matrices = np.concatenate([rot_matrices, tr_vectors], axis=-1)

        first_inv_transform = inv_transform_matrices[0]
        final_poses = first_inv_transform[:,:3] @ transform_matrices
        final_poses[:,:,-1:] += first_inv_transform[:,-1:]
        gt_xyz[:,[j]] = -sample['poses'][1,:,:3].T @ sample['poses'][1,:,-1:] + last_gt_xyz
        last_gt_xyz = gt_xyz[:,[j]]

    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.set_title("gt")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    figure = ax.plot(gt_xyz[0,:], gt_xyz[1,:], gt_xyz[2,:], c='r')
    plt.show()

    

if __name__ == '__main__':
    import argparse
    arg_parser = argparse.ArgumentParser(
        description="TrianFlow testing."
    )
    arg_parser.add_argument('-c', '--config_file', default=None, help='config file.')
    arg_parser.add_argument('-g', '--gpu', type=str, default=0, help='gpu id.')
    arg_parser.add_argument('--mode', type=str, default='depth', help='mode for testing.')
    arg_parser.add_argument('--task', type=str, default='kitti_depth', help='To test on which task, kitti_depth or kitti_flow or nyuv2 or demo')
    arg_parser.add_argument('--image_path', type=str, default=None, help='Set this only when task==demo. Depth demo for single image.')
    arg_parser.add_argument('--pretrained_model', type=str, default=None, help='directory for loading flow pretrained models')
    arg_parser.add_argument('--result_dir', type=str, default=None, help='directory for saving predictions')

    args = arg_parser.parse_args()
    if not os.path.exists(args.config_file):
        raise ValueError('config file not found.')
    with open(args.config_file, 'r') as f:
        cfg = yaml.safe_load(f)
    cfg['img_hw'] = (cfg['img_hw'][0], cfg['img_hw'][1])
    #cfg['log_dump_dir'] = os.path.join(args.model_dir, 'log.pkl')
    cfg['model_dir'] = args.result_dir

    # copy attr into cfg
    for attr in dir(args):
        if attr[:2] != '__':
            cfg[attr] = getattr(args, attr)
    
    class pObject(object):
        def __init__(self):
            pass
    cfg_new = pObject()
    for attr in list(cfg.keys()):
        setattr(cfg_new, attr, cfg[attr])

    if args.mode == 'flow':
        model = Model_flow(cfg_new)
    elif args.mode == 'depth' or args.mode == 'flow_3stage':
        model = Model_depth(cfg_new)
    elif args.mode == 'geom':
        model = Model_geometry(cfg_new)
    elif args.mode == 'flowposenet':
        model = Model_flowposenet(cfg_new)
    
    # if args.task == 'demo':
    #     model = Model_geometry(cfg_new)

    model.cuda()
    weights = torch.load(args.pretrained_model)
    model.load_state_dict(weights['model_state_dict'], strict=False)
    model.eval()
    print('Model Loaded.')

    if args.task == 'kitti_pose':
        test_pose_odom(cfg_new, model)
    elif args.task == 'draw_pose':
        draw_pose_odom(cfg_new, model)
        