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


def draw_depth_odom(cfg, model, save_dir='./'):
    print('save depth image using kitti odom')
    dataset_dir = Path(cfg.kitti_odom_dir)
    dataset = KITTI_pose(dataset_dir, cfg.sequences, 3)
    print('{} snippets to test'.format(len(dataset)))
    visualizer = Visualizer_debug(dump_dir=save_dir)

    for j, sample in enumerate(tqdm(dataset)):
        imgs = sample['imgs']
        img = cv2.resize(imgs[1], (cfg.img_hw[1], cfg.img_hw[0])).astype(np.float32)
        img_input = torch.from_numpy(img / 255.0).float().cuda().unsqueeze(0).permute(0,3,1,2)
        disp = model.infer_depth(img_input)
        disp = disp[0].detach().cpu().numpy()
        disp = disp.transpose(1,2,0)
        disp = cv2.resize(disp, (cfg.img_hw[1], cfg.img_hw[0]):

def draw_flow_odom(cfg, model, save_dir='./'):
    print('save flow image using kitti odom')
    dataset_dir = Path(cfg.kitti_odom_dir)
    dataset = KITTI_pose(dataset_dir, cfg.sequences, 3)                                                          
    print('{} snippets to test'.format(len(dataset)))

    for j, sample in enumerate(tqdm(dataset)):
        imgs = sample['imgs']
        img1 = cv2.resize(imgs[0], (cfg.img_hw[1], cfg.img_hw[0])).astype(np.float32)
        img2 = cv2.resize(imgs[1], (cfg.img_hw[1], cfg.img_hw[0])).astype(np.float32)
        img1_input = torch.from_numpy(img1 / 255.0).float().cuda().unsqueeze(0).permute(0,3,1,2)
        img2_input = torch.from_numpy(img2 / 255.0).float().cuda().unsqueeze(0).permute(0,3,1,2)
        flow = model.inference_flow(img2_input, img1_input)
        flow_12 = resize_flow(flow, (375,1242))
        np_flow_12 = flow_12[0].detach().cpu().numpy().transpose([1, 2, 0])
        vis_flow = flow_to_image(np_flow_12)
        vis_flow = vis_flow.transpose([1,2,0])
        cv2.imwrite('./results/%d.png' % j, vis_flow)

    print('saved!')
    
    

def draw_pose_odom(cfg, model):
    print('Evaluate pose using kitti odom. Using model in '+cfg.model_dir)
    dataset_dir = Path(cfg.kitti_odom_dir)
    dataset = KITTI_pose(dataset_dir, cfg.sequences, 3)
    print('{} snippets to test'.format(len(dataset)))
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    errors = np.zeros((len(dataset), 2), np.float32)
    gt_xyz = np.zeros((3, len(dataset)), np.float32)
    pre_xyz = np.zeros((3, len(dataset)), np.float32)
    last_gt_xyz = np.zeros((3,1), np.float32)
    last_pre_xyz = np.zeros((3,1), np.float32)
    for j, sample in enumerate(tqdm(dataset)):
        # imgs = sample['imgs']
        # imgl = cv2.resize(imgs[0], (cfg.img_hw[1], cfg.img_hw[0])).astype(np.float32)
        # img  = cv2.resize(imgs[1], (cfg.img_hw[1], cfg.img_hw[0])).astype(np.float32)
        # imgr = cv2.resize(imgs[2], (cfg.img_hw[1], cfg.img_hw[0])).astype(np.float32)
        # imgs = np.concatenate([imgl, img, imgr], 2)
        # img_input = torch.from_numpy(imgs / 255.0).float().cuda().unsqueeze(0).permute(0,3,1,2)

        # poses = model.infer_pose(img_input)

        # poses = poses.detach().cpu()[0]
        # poses = torch.cat([poses[0].view(-1,6), torch.zeros(1,6).float(), poses[1].view(-1,6)])

        # inv_transform_matrices = pose_vec2mat(poses).numpy().astype(np.float64)

        # rot_matrices = np.linalg.inv(inv_transform_matrices[:,:,:3])
        # tr_vectors = -rot_matrices @ inv_transform_matrices[:,:,-1:]

        # transform_matrices = np.concatenate([rot_matrices, tr_vectors], axis=-1)

        # first_inv_transform = inv_transform_matrices[0]
        # final_poses = first_inv_transform[:,:3] @ transform_matrices
        # final_poses[:,:,-1:] += first_inv_transform[:,-1:]
        gt_xyz[:,[j]] = -sample['poses'][1,:,:3].T @ sample['poses'][1,:,-1:] + last_gt_xyz
        pre_xyz[:,[j]] = tr_vectors[2,:,-1:] + last_pre_xyz 
        last_pre_xyz = pre_xyz[:,[j]]

    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.set_title("gt")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    figure = ax.plot(pre_xyz[0,:], pre_xyz[1,:], pre_xyz[2,:], c='r')
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

    if args.task == 'draw_pose':
        draw_pose_odom(cfg_new, model)
    elif args.task == 'draw_depth':
        draw_depth_odom(cfg_new, model, args.result_dir)
    else:
        draw_flow_odom(cfg_new, model, args.result_dir)
        