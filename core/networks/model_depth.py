import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from pytorch_ssim import SSIM
from .structures import *

class Model_depth(nn.Module):
    """
    monodepth2 + dynamic mask
    """
    def __init__(self, cfg):
        super(Model_depth, self).__init__()

        self.dataset = cfg.dataset
        self.num_scales = cfg.num_scales
        self.depth_net = Depth_Model(cfg.depth_scale)
        self.pose_net = PoseCNN(cfg.num_input_frames)

    def reconstruction(self, ref_img, intrinsics, depth, depth_ref, pose, padding_mode='zeros'):
        reconstructed_img = []
        valid_mask = []
        projected_depth = []
        computed_depth = []

        for scale in range(self.num_scales):
            
            depth_scale = depth[scale]
            depth_ref_scale = depth_ref[scale]
            b,_,h,w = depth_scale.size()
            ref_img_scaled = F.interpolate(ref_img, (h, w), mode='area')
            downscale = ref_img.size(2)/h
            intrinsics_scaled = torch.cat((intrinsics[:, 0:2]/downscale, intrinsics[:, 2:]), dim=1)

            reconstructed_img_scale, valid_mask_scale, projected_depth_scale, computed_depth_scale = \
                 inverse_warp2(ref_img_scaled, depth_scale, depth_ref_scale, pose, intrinsics_scaled)

            reconstructed_img.append(reconstructed_img_scale)
            valid_mask.append(valid_mask_scale)
            projected_depth.append(projected_depth_scale)
            computed_depth.append(computed_depth_scale)

        return reconstructed_img, valid_mask, projected_depth, computed_depth

    def compute_photometric_loss(self, img_list, img_warped_list, mask_list):
        loss_list = []
        for scale in range(self.num_scales):
            img, img_warped, mask = img_list[scale], img_warped_list[scale], mask_list[scale]
            divider = mask.mean((1,2,3))
            img_diff = torch.abs((img - img_warped)) * mask.repeat(1,3,1,1)
            loss_pixel = img_diff.mean((1,2,3)) / (divider + 1e-12) # (B)
            loss_list.append(loss_pixel[:,None])
        loss = torch.cat(loss_list, 1).sum(1) # (B)
        return loss

    def compute_consis_loss(self, predicted_depth_list, computed_depth_list):
        loss_list = []
        for scale in range(self.num_scales):
            predicted_depth, computed_depth = predicted_depth_list[scale], computed_depth_list[scale]
            depth_diff = ((computed_depth - predicted_depth).abs() /
                    (computed_depth + predicted_depth).abs()).clamp(0, 1)
            loss_consis = depth_diff.mean()
            loss_list.append(loss_consis)
        loss = torch.cat(loss_list, 1).sum(1) # (B)
        return loss

    def compute_ssim_loss(self, img_list, img_warped_list, mask_list):
        loss_list = []
        for scale in range(self.num_scales):
            img, img_warped, occ_mask = img_list[scale], img_warped_list[scale], mask_list[scale]
            divider = occ_mask.mean((1,2,3))
            occ_mask_pad = occ_mask.repeat(1,3,1,1)
            ssim = SSIM(img * occ_mask_pad, img_warped * occ_mask_pad)
            loss_ssim = torch.clamp((1.0 - ssim) / 2.0, 0, 1).mean((1,2,3))
            loss_ssim = loss_ssim / (divider + 1e-12)
            loss_list.append(loss_ssim[:,None])
        loss = torch.cat(loss_list, 1).sum(1)
        return loss
            

    def forward(self, inputs):
        # initialization
        images, K_ms, K_inv_ms = inputs
        K, K_inv = K_ms[:,0,:,:], K_inv_ms[:,0,:,:]
        assert (images.shape[1] == 3)

        img_h, img_w = int(images.shape[2] / 3), images.shape[3] 
        img_l, img, img_r = images[:,:,:img_h,:], images[:,:,img_h:2*img_h,:], images[:,:,2*img_h:3*img_h,:]

        # depth infer
        disp_l_list = self.depth_net(img_l) # Nscales * [B, 1, H, W]
        disp_list = self.depth_net(img) 
        disp_r_list = self.depth_net(img_r)

        # pose infer
        pose_inputs = torch.cat([img_l,img,img_r],1)
        pose_vectors = self.pose_net(pose_inputs)
        pose_vec_fwd = pose_vectors[:,1,:]
        pose_vec_bwd = pose_vectors[:,0,:]

        # calculate reconstructed image
        reconstructed_imgs_from_l, valid_masks_to_l, predicted_depths_to_l, computed_depths_to_l = \
            self.reconstruction(img_l, K, disp_list, disp_l_list, pose_vec_bwd)
        reconstructed_imgs_from_r, valid_masks_to_r, predicted_depths_to_r, computed_depths_to_r = \
            self.reconstruction(img_r, K, disp_list, disp_r_list, pose_vec_fwd)

        loss_pack = {}

        loss_pack['loss_depth_pixel'] = self.compute_photometric_loss(img,reconstructed_imgs_from_l,valid_masks_to_l) + \
            self.compute_photometric_loss(img,reconstructed_imgs_from_r,valid_masks_to_r)

        loss_pack['loss_depth_ssim'] = self.compute_ssim_loss(img,reconstructed_imgs_from_l,valid_masks_to_l) + \
            self.compute_ssim_loss(img,reconstructed_imgs_from_r,valid_masks_to_r)

        loss_pack['loss_depth_consis'] =  self.compute_consis_loss(predicted_depths_to_l, computed_depths_to_l) + \
            self.compute_consis_loss(predicted_depths_to_r, computed_depths_to_r)

        return loss_pack




