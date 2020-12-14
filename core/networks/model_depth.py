import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from pytorch_ssim import SSIM
from structures import *

class Model_depth(nn.Module):
    """
    monodepth2 + dynamic mask
    """
    def __init__(self, cfg):
        super(Model_depth, self).__init__()

        self.dataset = cfg.dataset
        self.num_scales = cfg.num_scales
        self.depth_net = Depth_Model(cfg.num_scales)
        self.pose_net = PoseCNN(cfg.num_input_frames)

    def generate_img_pyramid(self, img, num_pyramid):
        img_h, img_w = img.shape[2], img.shape[3]
        img_pyramid = []
        for s in range(num_pyramid):
            # img_new = F.adaptive_avg_pool2d(img, [int(img_h / (2**s)), int(img_w / (2**s))]).data
            img_new = F.interpolate(img, (int(img_h / (2**s)), int(img_w / (2**s))), mode='bilinear')
            img_pyramid.append(img_new)
        return img_pyramid
    
    def reconstruction_up(self, ref_img, intrinsics, depth, depth_ref, pose, padding_mode='zeros'):
        reconstructed_img = []
        valid_mask = []
        projected_depth = []
        computed_depth = []
        b,_,h,w = ref_img.size()

        for scale in range(self.num_scales):
            
            depth_scale = depth[scale]
            depth_ref_scale = depth_ref[scale]
            depth_up = F.interpolate(depth_scale, (h, w), mode='bilinear')
            depth_ref_up = F.interpolate(depth_ref_scale, (h, w), mode='bilinear')

            reconstructed_img_scale, valid_mask_scale, projected_depth_scale, computed_depth_scale = \
                 inverse_warp2(ref_img, depth_up, depth_ref_up, pose, intrinsics)

            reconstructed_img.append(reconstructed_img_scale)
            valid_mask.append(valid_mask_scale)
            projected_depth.append(projected_depth_scale)
            computed_depth.append(computed_depth_scale)

        return reconstructed_img, valid_mask, projected_depth, computed_depth

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

    def compute_texture_mask(self, img_list, img_warped_list, img_list_source):
        texture_masks = []
        for scale in range(self.num_scales):
            img, img_warped, img_source = img_list[scale], img_warped_list[scale], img_list_source[scale]
            texture_mask = (torch.abs(img-img_warped).mean(1, keepdim=True) < torch.abs(img-img_source).mean(1, keepdim=True)).float()
            texture_masks.append(texture_mask)
        return texture_masks

    def compute_photometric_loss(self, img_list, img_warped_list, mask_list):
        loss_list = []
        for scale in range(self.num_scales):
            img, img_warped, mask = img_list[scale], img_warped_list[scale], mask_list[scale]
            # texture_mask = F.interpolate(compute_texture_mask(img), size=(mask.shape[2], mask.shape[3]), mode='bilinear')
            # mask = mask*texture_mask
            divider = mask.mean((1,2,3))
            img_diff = torch.abs((img - img_warped)) * mask.repeat(1,3,1,1)
            loss_pixel = img_diff.mean((1,2,3)) / (divider + 1e-12) # (B)
            loss_list.append(loss_pixel[:,None])
        loss = torch.cat(loss_list, 1).sum(1) # (B)
        return loss
    
    
    def compute_photometric_depth_loss_up(self, img, img_warped_list, img_list_source, mask_list):
        loss_list = []
        weight_alpha = 2.7
        for scale in range(self.num_scales):
            img_warped, img_source, mask = img_warped_list[scale], img_list_source[scale], mask_list[scale]
            # texture_mask = F.interpolate(compute_texture_mask(img), size=(mask.shape[2], mask.shape[3]), mode='bilinear')
            texture_mask = (torch.abs(img-img_warped).mean(1, keepdim=True) < torch.abs(img-img_source).mean(1, keepdim=True)).float()
            mask = mask*texture_mask
            divider = mask.mean((1,2,3))
            img_diff = torch.abs((img - img_warped)) * mask.repeat(1,3,1,1)
            loss_pixel = img_diff.mean((1,2,3)) / (divider + 1e-12) # (B)
            loss_pixel = loss_pixel / math.pow(weight_alpha, scale)
            loss_list.append(loss_pixel[:,None])
        loss = torch.cat(loss_list, 1).sum(1) # (B)
        return loss


    def compute_photometric_depth_loss(self, img_list, img_warped_list, img_list_source, mask_list):
        loss_list = []
        for scale in range(self.num_scales):
            img, img_warped, img_source, mask = img_list[scale], img_warped_list[scale], img_list_source[scale], mask_list[scale]
            # texture_mask = F.interpolate(compute_texture_mask(img), size=(mask.shape[2], mask.shape[3]), mode='bilinear')
            texture_mask = (torch.abs(img-img_warped).mean(1, keepdim=True) < torch.abs(img-img_source).mean(1, keepdim=True)).float()
            mask = mask*texture_mask
            divider = mask.mean((1,2,3))
            img_diff = torch.abs((img - img_warped)) * mask.repeat(1,3,1,1)
            loss_pixel = img_diff.mean((1,2,3)) / (divider + 1e-12) # (B)
            loss_list.append(loss_pixel[:,None])
        loss = torch.cat(loss_list, 1).sum(1) # (B)
        return loss

    def compute_photometric_loss_min(self, img_list, img_warped_from_l_list, img_warped_from_r_list):
        loss_list = []
        for scale in range(self.num_scales):
            img, img_warped_from_l, img_warped_from_r = img_list[scale], img_warped_from_l_list[scale], img_warped_from_r_list[scale]
            # mask = compute_texture_mask(img)
            # divider = mask.mean((1,2,3))
            img_diff_l = torch.abs((img - img_warped_from_l))
            img_diff_r = torch.abs((img - img_warped_from_r))
            img_diff   = torch.cat([img_diff_l, img_diff_r], 1)
            img_diff   = torch.min(img_diff,1,True)[0]
            loss_pixel = img_diff.mean((1,2,3)) # (B)
            loss_list.append(loss_pixel[:,None])
        loss = torch.cat(loss_list,1).sum(1) # (B)
        return loss

            

    def compute_consis_loss(self, predicted_depth_list, computed_depth_list):
        loss_list = []
        for scale in range(self.num_scales):
            predicted_depth, computed_depth = predicted_depth_list[scale], computed_depth_list[scale]
            depth_diff = ((computed_depth - predicted_depth).abs() /
                    (computed_depth + predicted_depth).abs()).clamp(0, 1)
            loss_consis = depth_diff.mean((1,2,3)) # (B)
            loss_list.append(loss_consis[:,None])
        loss = torch.cat(loss_list, 1).sum(1) # (B)
        return loss
    
    def compute_ssim_loss_up(self, img, img_warped_list, mask_list):
        loss_list = []
        for scale in range(self.num_scales):
            img_warped, occ_mask = img_warped_list[scale], mask_list[scale]
            divider = occ_mask.mean((1,2,3))
            occ_mask_pad = occ_mask.repeat(1,3,1,1)
            ssim = SSIM(img * occ_mask_pad, img_warped * occ_mask_pad)
            loss_ssim = torch.clamp((1.0 - ssim) / 2.0, 0, 1).mean((1,2,3))
            loss_ssim = loss_ssim / (divider + 1e-12)
            loss_list.append(loss_ssim[:,None])
        loss = torch.cat(loss_list, 1).sum(1)
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
    
    def compute_smooth_loss_up(self, img, disps):
        # img: [b,3,h,w] depth: [b,1,h,w]
        """Computes the smoothness loss for a disparity image
        The color image is used for edge-aware smoothness
        """
        loss_list = []
        img_h, img_w = img.shape[2], img.shape[3] 
        for scale in range(self.num_scales):
            disp = disps[scale]

            disp = F.interpolate(disp, size=(img_h, img_w), mode='bilinear')
            # depth_h, depth_w = disp.shape[2], disp.shape[3]
            # img = F.interpolate(img, size=(img_h, img_w), mode='bilinear')

            grad_disp_x = torch.abs(disp[:, :, :, :-1] - disp[:, :, :, 1:])
            grad_disp_y = torch.abs(disp[:, :, :-1, :] - disp[:, :, 1:, :])

            grad_img_x = torch.mean(torch.abs(img[:, :, :, :-1] - img[:, :, :, 1:]), 1, keepdim=True)
            grad_img_y = torch.mean(torch.abs(img[:, :, :-1, :] - img[:, :, 1:, :]), 1, keepdim=True)

            grad_disp_x *= torch.exp(-grad_img_x)
            grad_disp_y *= torch.exp(-grad_img_y)

            grad_disp = grad_disp_x.mean((1,2,3)) + grad_disp_y.mean((1,2,3))
            loss_list.append(grad_disp_x[:,None]) # (B)
        
        loss = torch.cat(loss_list, 1).sum(1)
        return loss

    def compute_smooth_loss(self, img, disps):
        # img: [b,3,h,w] depth: [b,1,h,w]
        """Computes the smoothness loss for a disparity image
        The color image is used for edge-aware smoothness
        """
        loss_list = []
        img_h, img_w = img.shape[2], img.shape[3] 
        for scale in range(self.num_scales):
            disp = disps[scale]

            disp = F.interpolate(disp, size=(img_h, img_w), mode='bilinear')
            # depth_h, depth_w = disp.shape[2], disp.shape[3]
            # img = F.interpolate(img, size=(img_h, img_w), mode='bilinear')

            grad_disp_x = torch.abs(disp[:, :, :, :-1] - disp[:, :, :, 1:])
            grad_disp_y = torch.abs(disp[:, :, :-1, :] - disp[:, :, 1:, :])

            grad_img_x = torch.mean(torch.abs(img[:, :, :, :-1] - img[:, :, :, 1:]), 1, keepdim=True)
            grad_img_y = torch.mean(torch.abs(img[:, :, :-1, :] - img[:, :, 1:, :]), 1, keepdim=True)

            grad_disp_x *= torch.exp(-grad_img_x)
            grad_disp_y *= torch.exp(-grad_img_y)

            grad_disp = grad_disp_x.mean((1,2,3)) + grad_disp_y.mean((1,2,3))
            loss_list.append(grad_disp[:,None]) # (B)
        
        loss = torch.cat(loss_list, 1).sum(1)
        return loss


    def disp2depth(self, disp, min_depth=0.1, max_depth=100.0):
        min_disp = 1 / max_depth
        max_disp = 1 / min_depth
        scaled_disp = min_disp + (max_disp - min_disp) * disp
        depth = 1 / scaled_disp
        return depth

    def infer_depth(self, img):
        disp_list = self.depth_net(img)
        depth = self.disp2depth(disp_list[0])
        return depth

    def fusion_mask(self, valid_mask, texture_mask):
        masks = []
        for scale in range(self.num_scales):
            valid, texture = valid_mask[scale], texture_mask[scale]
            mask = valid * texture
            masks.append(mask)

        return masks


    def forward(self, inputs):
        # initialization
        images, K_ms, K_inv_ms = inputs
        K, K_inv = K_ms[:,0,:,:], K_inv_ms[:,0,:,:]
        assert (images.shape[1] == 3)

        img_h, img_w = int(images.shape[2] / 3), images.shape[3] 
        img_l, img, img_r = images[:,:,:img_h,:], images[:,:,img_h:2*img_h,:], images[:,:,2*img_h:3*img_h,:]
        
        img_list = self.generate_img_pyramid(img, self.num_scales)
        img_l_list = self.generate_img_pyramid(img_l, self.num_scales)
        img_r_list = self.generate_img_pyramid(img_r, self.num_scales)

        # depth infer
        # disp_l_list = self.depth_net(img_l) # Nscales * [B, 1, H, W]
        # disp_list = self.depth_net(img) 
        # disp_r_list = self.depth_net(img_r)

        depth_l_list = self.depth_net(img_l) # Nscales * [B, 1, H, W]
        depth_list = self.depth_net(img) 
        depth_r_list = self.depth_net(img_r)

        # depth_l_list = [self.disp2depth(disp) for disp in disp_l_list]
        # depth_list   = [self.disp2depth(disp) for disp in disp_list]
        # depth_r_list = [self.disp2depth(disp) for disp in disp_r_list]

        # pose infer
        pose_inputs = torch.cat([img_l,img,img_r],1)
        pose_vectors = self.pose_net(pose_inputs)
        pose_vec_fwd = pose_vectors[:,1,:]
        pose_vec_bwd = pose_vectors[:,0,:]

        # calculate reconstructed image
        reconstructed_imgs_from_l, valid_masks_to_l, predicted_depths_to_l, computed_depths_to_l = \
            self.reconstruction(img_l, K, depth_list, depth_l_list, pose_vec_bwd)
        reconstructed_imgs_from_r, valid_masks_to_r, predicted_depths_to_r, computed_depths_to_r = \
            self.reconstruction(img_r, K, depth_list, depth_r_list, pose_vec_fwd)

        # compute texture mask
        texture_mask_bwd = self.compute_texture_mask(img_list, reconstructed_imgs_from_l, img_l_list)
        texture_mask_fwd = self.compute_texture_mask(img_list, reconstructed_imgs_from_r, img_r_list)

        fusion_mask_bwd = self.fusion_mask(valid_masks_to_l, texture_mask_bwd)
        fusion_mask_fwd = self.fusion_mask(valid_masks_to_r, texture_mask_fwd)

        loss_pack = {}
        mask_pack = {}

        loss_pack['loss_depth_pixel'] = self.compute_photometric_loss(img_list,reconstructed_imgs_from_l,fusion_mask_bwd) + \
            self.compute_photometric_loss(img_list,reconstructed_imgs_from_r,fusion_mask_fwd)
        # loss_pack['loss_depth_pixel'] = self.compute_photometric_depth_loss(img_list, reconstructed_imgs_from_l,img_l_list,valid_masks_to_l) + \
            # self.compute_photometric_depth_loss(img_list, reconstructed_imgs_from_r, img_r_list, valid_masks_to_r)
        # loss_pack['loss_depth_pixel'] = self.compute_photometric_loss_min(img_list, reconstructed_imgs_from_l, reconstructed_imgs_from_r)

        # loss_pack['loss_depth_ssim'] = self.compute_ssim_loss(img_list,reconstructed_imgs_from_l,fusion_mask_bwd) + \
        #    self.compute_ssim_loss(img_list,reconstructed_imgs_from_r,fusion_mask_fwd)
        loss_pack['loss_depth_ssim'] = torch.zeros([2]).to(img_l.get_device()).requires_grad_()

        loss_pack['loss_depth_smooth'] = self.compute_smooth_loss(img, depth_list) + self.compute_smooth_loss(img_l, depth_l_list) + \
            self.compute_smooth_loss(img_r, depth_r_list)

        # loss_pack['loss_depth_consis'] =  self.compute_consis_loss(predicted_depths_to_l, computed_depths_to_l) + \
        #    self.compute_consis_loss(predicted_depths_to_r, computed_depths_to_r)
        loss_pack['loss_depth_consis'] = torch.zeros([2]).to(img_l.get_device()).requires_grad_()

        return loss_pack,mask_pack