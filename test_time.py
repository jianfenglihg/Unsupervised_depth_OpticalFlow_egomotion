import os, sys

from cv2 import data
from core import dataset
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
import time


def test_flow_time(cfg, model):
    dataset = KITTI_2015(cfg.gt_2015_dir)
    time_sum = 0
    for index, inputs in enumerate(tqdm(dataset)):
        img, _, _ = inputs
        img = img[None,:,:,:]
        img_h = int(img.shape[2]/2)
        img1, img2 = img[:,:,:img_h,:], img[:,:,img_h:,:]
        img1 = img1.cuda()
        img2 = img2.cuda()
        time_start = time.time()
        flow = model.inference_flow(img1,img2)
        time_end = time.time()
        time_sum += (time_end - time_start)
    print('the time cost of per imageL: {}'.format(time_sum/len(dataset)))
    print('the num of images is: {}'.format(len(dataset)))



def test_depth_time(cfg, model):
    print('Evaluate depth inference time. using model in ' + cfg.model_dir)
    filenames = open('./data/eigen/test_files.txt').readlines()
    time_sum = 0
    for i in range(len(filenames)):
        path1, idx, _ = filenames[i].strip().split(' ')
        img = cv2.imread(os.path.join(os.path.join(cfg.raw_base_dir, path1), 'image_02/data/'+str(idx)+'.png'))
        img_resize = cv2.resize(img, (cfg.img_hw[1], cfg.img_hw[0]))
        img_input = torch.from_numpy(img_resize / 255.0).float().cuda().unsqueeze(0).permute(0,3,1,2)
        time_start = time.time()
        disp = model.infer_depth(img_input)
        time_end = time.time()
        time_sum += (time_end - time_start)
    print('per image time cost is: ')
    print(time_sum/len(filenames))
    print('the num of images is :')
    print(len(filenames))
    return time_sum/len(filenames)




def test_pose_time(cfg, model):
    print('Evaluate pose using kitti odom. Using model in '+cfg.model_dir)
    dataset_dir = Path(cfg.kitti_odom_dir)
    dataset = KITTI_pose(dataset_dir, cfg.sequences, 3)
    print('{} snippets to test'.format(len(dataset)))
    time_sum = 0

    for j, sample in enumerate(tqdm(dataset)):
        imgs = sample['imgs']
        imgl = cv2.resize(imgs[0], (cfg.img_hw[1], cfg.img_hw[0])).astype(np.float32)
        img  = cv2.resize(imgs[1], (cfg.img_hw[1], cfg.img_hw[0])).astype(np.float32)
        imgr = cv2.resize(imgs[2], (cfg.img_hw[1], cfg.img_hw[0])).astype(np.float32)
        imgs = np.concatenate([imgl, img, imgr], 2)
        img_input = torch.from_numpy(imgs / 255.0).float().cuda().unsqueeze(0).permute(0,3,1,2)
        time_start = time.time()
        poses = model.infer_pose(img_input)
        time_end = time.time()
        time_sum += (time_end - time_start)

    print('pose time cost of per image is: {}'.format(time_sum/len(dataset)))



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

    if args.task == 'depth_time':
        test_depth_time(cfg_new, model)
    elif args.task == 'flow_time':
        test_flow_time(cfg_new, model)
    elif args.task == 'pose_time':
        test_pose_time(cfg_new, model)
