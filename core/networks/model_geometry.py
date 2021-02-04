import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from pytorch_ssim import SSIM
from structures import *

import cv2

class Model_geometry(nn.Module):
    """
    monodepth2 + dynamic mask
    """
    def __init__(self, cfg):
        super(Model_geometry, self).__init__()

        self.dataset = cfg.dataset
        self.num_scales = cfg.num_scales
        
        self.flow_consist_alpha = cfg.flow_consist_alpha
        self.flow_consist_beta = cfg.flow_consist_beta
        
        self.depth_net = Depth_Model(cfg.num_scales)
        self.pose_net = PoseCNN(cfg.num_input_frames)
        self.fpyramid = FeaturePyramid()
        self.pwc_model = PWC_tf()
        
        self.PnP_ransac_iter = 1000
        self.PnP_ransac_thre = 1
        self.PnP_ransac_times = 5

        self.inlier_thres = 0.1
        self.rigid_thres = 0.5

        self.ratio = cfg.geometric_ratio
        self.num = cfg.geometric_num

        self.beta = cfg.pose_beta
        # self.bpnp = BPnP_m3d.apply
        # self.filter = reduced_ransac(check_num=cfg.ransac_points, thres=self.inlier_thres, dataset=cfg.dataset)
    
    def get_flow_norm(self, flow, p=2):
        '''
        Inputs:
        flow (bs, 2, H, W)
        '''
        flow_norm = torch.norm(flow, p=p, dim=1).unsqueeze(1) + 1e-12
        return flow_norm

    def get_flow_normalization(self, flow, p=2):
        '''
        Inputs:
        flow (bs, 2, H, W)
        '''
        flow_norm = torch.norm(flow, p=p, dim=1).unsqueeze(1) + 1e-12
        flow_normalization = flow / flow_norm.repeat(1,2,1,1)
        return flow_normalization

    def generate_img_pyramid(self, img, num_pyramid):
        img_h, img_w = img.shape[2], img.shape[3]
        img_pyramid = []
        for s in range(num_pyramid):
            # img_new = F.adaptive_avg_pool2d(img, [int(img_h / (2**s)), int(img_w / (2**s))]).data
            img_new = F.interpolate(img, (int(img_h / (2**s)), int(img_w / (2**s))), mode='bilinear')
            img_pyramid.append(img_new)
        return img_pyramid

    def warp_flow_pyramid(self, img_pyramid, flow_pyramid):
        img_warped_pyramid = []
        for img, flow in zip(img_pyramid, flow_pyramid):
            img_warped_pyramid.append(warp_flow(img, flow, use_mask=True))
        return img_warped_pyramid

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

    def compute_occ_weight(self, img_pyramid_from_l, img_pyramid, img_pyramid_from_r):
        weight_fwd = []
        weight_bwd = []
        valid_bwd = []
        valid_fwd = []
        for scale in range(self.num_scales):
            img_from_l, img, img_from_r = img_pyramid_from_l[scale], img_pyramid[scale], img_pyramid_from_r[scale]

            valid_pixels_fwd = 1 - (img_from_r == 0).prod(1, keepdim=True).type_as(img_from_r)
            valid_pixels_bwd = 1 - (img_from_l == 0).prod(1, keepdim=True).type_as(img_from_l)

            valid_bwd.append(valid_pixels_bwd)
            valid_fwd.append(valid_pixels_fwd)

            img_diff_l = torch.abs((img-img_from_l)).mean(1, True)
            img_diff_r = torch.abs((img-img_from_r)).mean(1, True)

            diff_cat = torch.cat((img_diff_l, img_diff_r),1)
            weight = 1 - nn.functional.softmax(diff_cat,1)
            # weight = Variable(weight.data,requires_grad=False)

            with torch.no_grad():
                weight = (weight > 0.48).float()
                # weight = 2*torch.exp(-(weight-0.5)**2/0.03).float()
                weight_bwd.append(torch.unsqueeze(weight[:,0,:,:],1))
                weight_fwd.append(torch.unsqueeze(weight[:,1,:,:],1))
                
        return weight_bwd, weight_fwd, valid_bwd, valid_fwd

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
            divider = mask.mean((1,2,3))
            img_diff = torch.abs((img - img_warped)) * mask.repeat(1,3,1,1)
            loss_pixel = img_diff.mean((1,2,3)) / (divider + 1e-12) # (B)
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

    # def compute_photometric_flow_loss(self, img_list, img_warped_list, valid_mask_list, occ_mask_list):
    #     loss_list = []
    #     for scale in range(self.num_scales):
    #         img, img_warped, valid_mask, occ_mask = img_list[scale], img_warped_list[scale], valid_mask_list[scale], occ_mask_list[scale]
    #         # texture_mask = F.interpolate(compute_texture_mask(img), size=(mask.shape[2], mask.shape[3]), mode='bilinear')
    #         mask = valid_mask * occ_mask
    #         divider = mask.mean((1,2,3))
    #         img_diff = torch.abs((img - img_warped)) * mask.repeat(1,3,1,1)
    #         loss_pixel = img_diff.mean((1,2,3)) / (divider + 1e-12) # (B)
    #         loss_list.append(loss_pixel[:,None])
    #     loss = torch.cat(loss_list, 1).sum(1) # (B)
    #     return loss

    def compute_consis_loss(self, predicted_depth_list, computed_depth_list, mask_list):
        loss_list = []
        for scale in range(self.num_scales):
            predicted_depth, computed_depth, mask = predicted_depth_list[scale], computed_depth_list[scale], mask_list[scale]
            divider = mask.mean((1,2,3))
            depth_diff = ((computed_depth - predicted_depth).abs() /
                    (computed_depth + predicted_depth).abs()).clamp(0, 1)
            depth_diff = depth_diff * mask
            loss_consis = depth_diff.mean((1,2,3)) / (divider + 1e-12) # (B)
            loss_list.append(loss_consis[:,None])
        loss = torch.cat(loss_list, 1).sum(1) # (B)
        return loss

    def compute_loss_flow_consis(self, fwd_flow_pyramid, bwd_flow_pyramid, occ_mask_list):
        loss_list = []
        for scale in range(self.num_scales):
            fwd_flow, bwd_flow, occ_mask = fwd_flow_pyramid[scale], bwd_flow_pyramid[scale], occ_mask_list[scale]
            fwd_flow_norm = self.get_flow_normalization(fwd_flow)
            bwd_flow_norm = self.get_flow_normalization(bwd_flow).float()
            bwd_flow_norm = Variable(bwd_flow_norm.data,requires_grad=False)
            occ_mask = 1-occ_mask

            divider = occ_mask.mean((1,2,3))
            
            loss_consis = (torch.abs(fwd_flow_norm+bwd_flow_norm) * occ_mask).mean((1,2,3))
            loss_consis = loss_consis / (divider + 1e-12)
            loss_list.append(loss_consis[:,None])
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

    def gradients(self, img):
        dy = img[:,:,1:,:] - img[:,:,:-1,:]
        dx = img[:,:,:,1:] - img[:,:,:,:-1]
        return dx, dy

    def cal_grad2_error(self, flow, img):
        img_grad_x, img_grad_y = self.gradients(img)
        w_x = torch.exp(-10.0 * torch.abs(img_grad_x).mean(1).unsqueeze(1))
        w_y = torch.exp(-10.0 * torch.abs(img_grad_y).mean(1).unsqueeze(1))

        dx, dy = self.gradients(flow)
        dx2, _ = self.gradients(dx)
        _, dy2 = self.gradients(dy)
        error = (w_x[:,:,:,1:] * torch.abs(dx2)).mean((1,2,3)) + (w_y[:,:,1:,:] * torch.abs(dy2)).mean((1,2,3))
        #error = (w_x * torch.abs(dx)).mean((1,2,3)) + (w_y * torch.abs(dy)).mean((1,2,3))
        return error / 2.0

    def compute_loss_flow_smooth(self, optical_flows, img_pyramid):
        loss_list = []
        for scale in range(self.num_scales):
            flow, img = optical_flows[scale], img_pyramid[scale]
            #error = self.cal_grad2_error(flow, img)
            error = self.cal_grad2_error(flow/20.0, img)
            loss_list.append(error[:,None])
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

    def inference_flow(self, img1, img2):
        img_hw = [img1.shape[2], img1.shape[3]]
        feature_list_1, feature_list_2 = self.fpyramid(img1), self.fpyramid(img2)
        optical_flow = self.pwc_model(feature_list_1, feature_list_2, img_hw)[0]
        return optical_flow

    def infer_pose(self, imgs):
        pose = self.pose_net(imgs)
        return pose

    # def meshgrid(self, h, w):
    #     xx, yy = np.meshgrid(np.arange(0,w), np.arange(0,h))
    #     meshgrid = np.transpose(np.stack([xx,yy], axis=-1), [2,0,1]) # [2,h,w]
    #     meshgrid = torch.from_numpy(meshgrid)
    #     return meshgrid

    def meshgrid(self, B, H, W):
        xx = torch.arange(0, W).view(1,-1).repeat(H,1)
        yy = torch.arange(0, H).view(-1,1).repeat(1,W)
        xx = xx.view(1,1,H,W).repeat(B,1,1,1)
        yy = yy.view(1,1,H,W).repeat(B,1,1,1)
        grid = torch.cat((xx,yy),1).float()
        return grid

    def compute_epipolar_map_8pF(self, matches, pose, flow, intrinsics, intrinsics_inverse):
        b,_,h,w = flow.size()
        batch_size, flow_h, flow_w = flow.shape[0], flow.shape[2], flow.shape[3]
        grid = self.meshgrid(b, h, w).to(flow.get_device()) #[b,2,h,w]
        corres = grid + flow
        # corres = torch.cat([(grid[:,0,:,:] + flow[:,0,:,:]).unsqueeze(1), (grid[:,1,:,:] + flow[:,1,:,:]).unsqueeze(1)], 1)
        match = torch.cat([grid, corres], 1) # [b,4,h,w]
        match = match.view([b,4,-1]) # [b,4,n]
        # mask = mask.view([b,1,-1]) # [b,1,n] 

        num_batch = match.shape[0]
        match_num = match.shape[-1]

        points1 = match[:,:2,:]
        points2 = match[:,2:,:]
        ones = torch.ones(num_batch, 1, match_num).to(points1.get_device())
        points1 = torch.cat([points1, ones], 1) # [b,3,n]
        points2 = torch.cat([points2, ones], 1).transpose(1,2) # [b,n,3]

        F = self.compute_fundmental_mat(matches, pose, intrinsics, intrinsics_inverse)
        F = F.unsqueeze(1)
        F_tiles = F.view([-1,3,3])


        epi_line = F_tiles.bmm(points1) # [b,3,n]
        
        dist_p2l = torch.abs((epi_line.permute([0, 2, 1]) * points2).sum(-1, keepdim=True)) # [b,n,1]

        a = epi_line[:,0,:].unsqueeze(1).transpose(1,2) # [b,n,1]
        b = epi_line[:,1,:].unsqueeze(1).transpose(1,2) # [b,n,1]
        dist_div = torch.sqrt(a*a + b*b) + 1e-6

        epipolar = dist_p2l / dist_div #[b,n,1]
        epipolar = epipolar.view([batch_size,flow_h,flow_w,1]) 

        return epipolar.permute(0,3,1,2)
    
    def compute_epipolar_map(self, pose, flow, intrinsics, intrinsics_inverse):
        b,_,h,w = flow.size()
        batch_size, flow_h, flow_w = flow.shape[0], flow.shape[2], flow.shape[3]
        grid = self.meshgrid(b, h, w).to(flow.get_device()) #[b,2,h,w]
        corres = grid + flow
        # corres = torch.cat([(grid[:,0,:,:] + flow[:,0,:,:]).unsqueeze(1), (grid[:,1,:,:] + flow[:,1,:,:]).unsqueeze(1)], 1)
        match = torch.cat([grid, corres], 1) # [b,4,h,w]
        match = match.view([b,4,-1]) # [b,4,n]
        # mask = mask.view([b,1,-1]) # [b,1,n] 

        num_batch = match.shape[0]
        match_num = match.shape[-1]

        points1 = match[:,:2,:]
        points2 = match[:,2:,:]
        ones = torch.ones(num_batch, 1, match_num).to(points1.get_device())
        points1 = torch.cat([points1, ones], 1) # [b,3,n]
        points2 = torch.cat([points2, ones], 1) # [b,3,n]

        # compute fundmental matrix
        E = compute_essential_matrix(pose)
        F_meta = E.bmm(intrinsics_inverse)
        # F = torch.inverse(intrinsics.permute([0, 2, 1])).bmm(F_meta)  # T and then -1
        F = intrinsics_inverse.transpose(1,2).bmm(F_meta)  # T and then -1

        epi_line = F.bmm(points1) # [b,3,n]
        a = epi_line[:,0,:].unsqueeze(1) # [b,1,n]
        b = epi_line[:,1,:].unsqueeze(1) # [b,1,n]
        dist_div = torch.sqrt(a*a + b*b) + 1e-6

        geom_dist = torch.abs(torch.sum(points2 * epi_line, dim=1, keepdim=True)) #[b,1,n]
        epipolar = geom_dist / dist_div #[b,1,n]
        # epipolar = geom_dist  #[b,1,n]
        epipolar = epipolar.view([batch_size,1,flow_h,flow_w]) 


        # points2 = points2.transpose(1,2)
        # epi_line = F.bmm(points1) # [b,3,n]
        # dist_p2l = torch.abs((epi_line.permute([0, 2, 1]) * points2).sum(-1, keepdim=True)) # [b,n,1]

        # a = epi_line[:,0,:].unsqueeze(1).transpose(1,2) # [b,n,1]
        # b = epi_line[:,1,:].unsqueeze(1).transpose(1,2) # [b,n,1]
        # dist_div = torch.sqrt(a*a + b*b) + 1e-6
        # dist_map = dist_p2l / dist_div # [B, n, 1]
        # dist_map = dist_map.transpose(1,2)
        # dist_map = dist_map.view([batch_size,1,flow_h,flow_w])

        # loss = (dist_map * mask.transpose(1,2)).mean([1,2]) / mask.mean([1,2])
        return epipolar
        # return dist_map, epipolar

    # def compute_epipolar_loss(self, dist_map, rigid_mask, inlier_mask):

    #     loss = (dist_map * (rigid_mask - inlier_mask)).mean((1,2,3)) / \
    #          ((rigid_mask - inlier_mask).mean((1,2,3)) + 1e-4)

    #     return loss

    def compute_epipolar_loss(self, dist_map, rigid_mask):
    
        loss = (dist_map*rigid_mask).mean((1,2,3)) / (rigid_mask.mean((1,2,3)) + 1e-12)
        loss = dist_map.mean((1,2,3))

        return loss

    def get_rigid_mask(self, dist_map):
        with torch.no_grad():
            rigid_mask = (dist_map < self.rigid_thres).float()
            inlier_mask = (dist_map < self.inlier_thres).float()
            rigid_score = rigid_mask * 1.0 / (1.0 + dist_map)
        return rigid_mask, inlier_mask, rigid_score

    def top_ratio_sample(self, match, depth, mask, ratio):
        # match: [b, 4, -1] mask: [b, 1, -1]
        b, total_num = match.shape[0], match.shape[-1]
        scores, indices = torch.topk(mask, int(ratio*total_num), dim=-1) # [B, 1, ratio*tnum]
        select_match = torch.gather(match.transpose(1,2), index=indices.squeeze(1).unsqueeze(-1).repeat(1,1,4), dim=1).transpose(1,2) # [b, 4, ratio*tnum]
        select_depth = torch.gather(depth.transpose(1,2), index=indices.squeeze(1).unsqueeze(-1), dim=1).transpose(1,2) # [b, 1, ratio*tnum]
        return select_match, select_depth, scores

    def robust_rand_sample(self, match, depth, mask, num):
        # match: [b, 4, -1] mask: [b, 1, -1]
        b, n = match.shape[0], match.shape[2]
        nonzeros_num = torch.min(torch.sum(mask > 0, dim=-1)) # []
        if nonzeros_num.detach().cpu().numpy() == n:
            rand_int = torch.randint(0, n, [num])
            select_match = match[:,:,rand_int]
            select_depth = depth[:,:,rand_int]
        else:
            # If there is zero score in match, sample the non-zero matches.
            num = np.minimum(nonzeros_num.detach().cpu().numpy(), num)
            select_idxs = []
            for i in range(b):
                nonzero_idx = torch.nonzero(mask[i,0,:]) # [nonzero_num,1]
                rand_int = torch.randint(0, nonzero_idx.shape[0], [int(num)])
                select_idx = nonzero_idx[rand_int, :] # [num, 1]
                select_idxs.append(select_idx)
            select_idxs = torch.stack(select_idxs, 0) # [b,num,1]
            select_match = torch.gather(match.transpose(1,2), index=select_idxs.repeat(1,1,4), dim=1).transpose(1,2) # [b, 4, num]
            select_depth = torch.gather(depth.transpose(1,2), index=select_idxs, dim=1).transpose(1,2) # [b, 1, num]
        return select_match, select_depth, num

    def sample_match(self, flow, depth, mask):
        b,_,h,w = flow.size()
        depth = depth.view([b,1,-1]).contiguous()
        grid = self.meshgrid(b, h, w).to(flow.get_device()) #[b,2,h,w]
        corres = torch.cat([(grid[:,0,:,:] + flow[:,0,:,:]).unsqueeze(1), (grid[:,1,:,:] + flow[:,1,:,:]).unsqueeze(1)], 1)
        match = torch.cat([grid, corres], 1) # [b,4,h,w]
        match = match.view([b,4,-1]).contiguous() # [b,4,n]

        mask = mask.view([b,1,-1]).contiguous()    # jianfeng add contiguous

        match_by_top_ratio, depth_by_top_ratio, top_ratio_mask = self.top_ratio_sample(match, depth, mask, self.ratio)
        sampled_match, sampled_depth, end_num = self.robust_rand_sample(match_by_top_ratio, depth_by_top_ratio, top_ratio_mask, self.num)
        
        return sampled_match, sampled_depth


    def pnp(self, pts2d, pts3d, K, ini_pose=None):
            bs = pts2d.size(0)
            n = pts2d.size(1)
            device = pts2d.device
            K_np = np.array(K.detach().cpu())
            P_6d = torch.zeros(bs,6,device=device)

            for i in range(bs):
                pts2d_i_np = np.ascontiguousarray(pts2d[i].detach().cpu()).reshape((n,1,2))
                pts3d_i_np = np.ascontiguousarray(pts3d[i].detach().cpu()).reshape((n,3))
                if ini_pose is None:
                    _, rvec0, T0, _ = cv2.solvePnPRansac(objectPoints=pts3d_i_np, imagePoints=pts2d_i_np, cameraMatrix=K_np, distCoeffs=None, flags=cv2.SOLVEPNP_ITERATIVE, confidence=0.9999 ,reprojectionError=1)
                else:
                    rvec0 = np.array(ini_pose[i, 0:3].cpu().view(3, 1))
                    T0 = np.array(ini_pose[i, 3:6].cpu().view(3, 1))
                _, rvec, T = cv2.solvePnP(objectPoints=pts3d_i_np, imagePoints=pts2d_i_np, cameraMatrix=K_np, distCoeffs=None, flags=cv2.SOLVEPNP_ITERATIVE, useExtrinsicGuess=True, rvec=rvec0, tvec=T0)
                angle_axis = torch.tensor(rvec,device=device,dtype=torch.float).view(1, 3)
                T = torch.tensor(T,device=device,dtype=torch.float).view(1, 3)
                P_6d[i,:] = torch.cat((T, angle_axis),dim=-1)

            return P_6d


    def compute_pnp_loss(self, depth, matches, pose_vec, K, K_inv):
        # world point
        b, _, n = matches.size()
        xy = matches[:,:2,:] # [b,2,n]
        ones = torch.ones([b,1,n]).float().to(depth.get_device()) # [b,1,n]
        pts_3d = K_inv.bmm(torch.cat([xy, ones], 1)) * depth # [b,3,n]
        pts_3d = pts_3d.transpose(1,2)
        
        # image point
        corres = matches[:,2:,:]
        corres = corres.transpose(1,2)

        # calculate pose by pnp
        pose_pred = self.pnp(corres, pts_3d, K[0]) # [b,6]
        # pose_pred = self.bpnp(corres, pts_3d, K[0]) # [b,6]

        # TO DO: pose_net's loss function, watch out the difference between pnp and bpnp class
        # position = pose_pred[:, 3:]
        # orientation = pose_pred[:, :3]
        # position_target = pose_vec[:, :3]
        # orientation_target = pose_vec[:, 3:]

        position = pose_pred[:, :3]
        orientation = pose_pred[:, 3:]
        position_target = pose_vec[:, :3]
        orientation_target = pose_vec[:, 3:]


        position_loss = F.l1_loss(position, position_target, reduction='none')
        # position_loss = F.mse_loss(position, position_target, reduction='none')
        orientation_loss = F.l1_loss(orientation, orientation_target, reduction='none')
        # orientation_loss = F.mse_loss(orientation, orientation_target, reduction='none')
        loss = position_loss + self.beta * orientation_loss

        return loss

    def compute_fundmental_mat(self, matches, pose_vec, intrinsics, intrinsics_inverse):
        b = matches.shape[0]
        check_match = matches.contiguous()

        cv_f = []
        for i in range(b):
            if self.dataset == 'nyuv2':
                f, m = cv2.findFundamentalMat(check_match[i,:2,:].transpose(0,1).detach().cpu().numpy(), check_match[i,2:,:].transpose(0,1).detach().cpu().numpy(), cv2.FM_LMEDS, 0.99)
            else:
                f, m = cv2.findFundamentalMat(check_match[i,:2,:].transpose(0,1).detach().cpu().numpy(), check_match[i,2:,:].transpose(0,1).detach().cpu().numpy(), cv2.FM_RANSAC, 0.1, 0.99)
            cv_f.append(f)
        cv_f = np.stack(cv_f, axis=0)
        cv_f = torch.from_numpy(cv_f).float().to(matches.get_device())

        return cv_f

    def compute_eight_point_loss(self, matches, pose_vec, intrinsics, intrinsics_inverse):
        b = matches.shape[0]
        check_match = matches.contiguous()

        cv_f = []
        for i in range(b):
            if self.dataset == 'nyuv2':
                f, m = cv2.findFundamentalMat(check_match[i,:2,:].transpose(0,1).detach().cpu().numpy(), check_match[i,2:,:].transpose(0,1).detach().cpu().numpy(), cv2.FM_LMEDS, 0.99)
            else:
                f, m = cv2.findFundamentalMat(check_match[i,:2,:].transpose(0,1).detach().cpu().numpy(), check_match[i,2:,:].transpose(0,1).detach().cpu().numpy(), cv2.FM_RANSAC, 0.1, 0.99)
            cv_f.append(f)
        cv_f = np.stack(cv_f, axis=0)
        cv_f = torch.from_numpy(cv_f).float().to(matches.get_device())

        E = compute_essential_matrix(pose_vec)
        F_meta = E.bmm(intrinsics_inverse)
        F_pred = torch.inverse(intrinsics.permute([0, 2, 1])).bmm(F_meta)
        loss = F.smooth_l1_loss(F_pred, cv_f)
        return loss

    
    def midpoint_triangulate(self, match, K, K_inv, P1, P2):
        # match: [b, 4, num] P1: [b, 3, 4]
        # Match is in the image coordinates. P1, P2 is camera parameters. [B, 3, 4] match: [B, M, 4]
        b, n = match.shape[0], match.shape[2]
        RT1 = K_inv.bmm(P1) # [b, 3, 4]
        RT2 = K_inv.bmm(P2)
        ones = torch.ones([b,1,n]).to(match.get_device())
        pts1 = torch.cat([match[:,:2,:], ones], 1)
        pts2 = torch.cat([match[:,2:,:], ones], 1)
        
        ray1_dir = (RT1[:,:,:3].transpose(1,2)).bmm(K_inv).bmm(pts1)# [b,3,n]
        ray1_dir = ray1_dir / (torch.norm(ray1_dir, dim=1, keepdim=True, p=2) + 1e-12)
        ray1_origin = (-1) * RT1[:,:,:3].transpose(1,2).bmm(RT1[:,:,3].unsqueeze(-1)) # [b, 3, 1]
        ray2_dir = (RT2[:,:,:3].transpose(1,2)).bmm(K_inv).bmm(pts2) # [b,3,n]
        ray2_dir = ray2_dir / (torch.norm(ray2_dir, dim=1, keepdim=True, p=2) + 1e-12)
        ray2_origin = (-1) * RT2[:,:,:3].transpose(1,2).bmm(RT2[:,:,3].unsqueeze(-1)) # [b, 3, 1]
    
        dir_cross = torch.cross(ray1_dir, ray2_dir, dim=1) # [b,3,n]
        denom = 1.0 / (torch.sum(dir_cross * dir_cross, dim=1, keepdim=True)+1e-12) # [b,1,n]
        origin_vec = (ray2_origin - ray1_origin).repeat(1,1,n) # [b,3,n]
        a1 = origin_vec.cross(ray2_dir, dim=1) # [b,3,n]
        a1 = torch.sum(a1 * dir_cross, dim=1, keepdim=True) * denom # [b,1,n]
        a2 = origin_vec.cross(ray1_dir, dim=1) # [b,3,n]
        a2 = torch.sum(a2 * dir_cross, dim=1, keepdim=True) * denom # [b,1,n]
        p1 = ray1_origin + a1 * ray1_dir
        p2 = ray2_origin + a2 * ray2_dir
        point = (p1 + p2) / 2.0 # [b,3,n]
        # Convert to homo coord to get consistent with other functions.
        point_homo = torch.cat([point, ones], dim=1).transpose(1,2) # [b,n,4]
        return point_homo


    def reproject(self, P, point3d):
        # P: [b,3,4] point3d: [b,n,4]
        point2d = P.bmm(point3d.transpose(1,2)) # [b,4,n]
        point2d_coord = (point2d[:,:2,:] / (point2d[:,2,:].unsqueeze(1) + 1e-12)).transpose(1,2) # [b,n,2]
        point2d_depth = point2d[:,2,:].unsqueeze(1).transpose(1,2) # [b,n,1]
        return point2d_coord, point2d_depth
    
    def scale_adapt(self, depth1, depth2, eps=1e-12):
        with torch.no_grad():
            A = torch.sum((depth1 ** 2) / (depth2 ** 2 + eps), dim=1) # [b,1]
            C = torch.sum(depth1 / (depth2 + eps), dim=1) # [b,1]
            a = C / (A + eps)
        return a

    def affine_adapt(self, depth1, depth2, use_translation=True, eps=1e-12):
        a_scale = self.scale_adapt(depth1, depth2, eps=eps)
        if not use_translation: # only fit the scale parameter
            return a_scale, torch.zeros_like(a_scale)
        else:
            with torch.no_grad():
                A = torch.sum((depth1 ** 2) / (depth2 ** 2 + eps), dim=1) # [b,1]
                B = torch.sum(depth1 / (depth2 ** 2 + eps), dim=1) # [b,1]
                C = torch.sum(depth1 / (depth2 + eps), dim=1) # [b,1]
                D = torch.sum(1.0 / (depth2 ** 2 + eps), dim=1) # [b,1]
                E = torch.sum(1.0 / (depth2 + eps), dim=1) # [b,1]
                a = (B*E - D*C) / (B*B - A*D + 1e-12)
                b = (B*C - A*E) / (B*B - A*D + 1e-12)

                # check ill condition
                cond = (B*B - A*D)
                valid = (torch.abs(cond) > 1e-4).float()
                a = a * valid + a_scale * (1 - valid)
                b = b * valid
            return a, b

    def register_depth(self, depth_pred, coord_tri, depth_tri):
        # depth_pred: [b, 1, h, w] coord_tri: [b,n,2] depth_tri: [b,n,1]
        batch, _, h, w = depth_pred.shape[0], depth_pred.shape[1], depth_pred.shape[2], depth_pred.shape[3]
        n = depth_tri.shape[1]
        coord_tri_nor = torch.stack([2.0*coord_tri[:,:,0] / (w-1.0) - 1.0, 2.0*coord_tri[:,:,1] / (h-1.0) - 1.0], -1)
        depth_inter = F.grid_sample(depth_pred, coord_tri_nor.view([batch,n,1,2]), padding_mode='reflection').squeeze(-1).transpose(1,2) # [b,n,1]

        # Normalize
        scale = torch.median(depth_inter, 1)[0] / (torch.median(depth_tri, 1)[0] + 1e-12)
        scale = scale.detach() # [b,1]
        scale_depth_inter = depth_inter / (scale.unsqueeze(-1) + 1e-12)
        scale_depth_pred = depth_pred / (scale.unsqueeze(-1).unsqueeze(-1) + 1e-12)
        
        # affine adapt
        a, b = self.affine_adapt(scale_depth_inter, depth_tri, use_translation=False)
        affine_depth_inter = a.unsqueeze(1) * scale_depth_inter + b.unsqueeze(1) # [b,n,1]
        affine_depth_pred = a.unsqueeze(-1).unsqueeze(-1) * scale_depth_pred + b.unsqueeze(-1).unsqueeze(-1) # [b,1,h,w]
        return affine_depth_pred, affine_depth_inter
    
    def get_trian_loss(self, tri_depth, pred_tri_depth):
        # depth: [b,n,1]
        # To Do: mask the dynamic part 
        loss = torch.pow(1.0 - pred_tri_depth / (tri_depth + 1e-12), 2).mean((1,2))
        return loss


    # def compute_triangulate_loss(self, match, pose, K, K_inv, depth):
    #     depth_pred = depth.transpose(1,2) # [b, n, 1]
    #     P1, P2 = compute_projection_matrix(pose,K)
    #     triangulated_point = self.midpoint_triangulate(match,K,K_inv,P1,P2) # [b,n,4]
    #     depth_calc = triangulated_point[:,:,2].unsqueeze(-1)
    #     pt_depth_loss = self.get_trian_loss(depth_calc, depth_pred)
    #     return pt_depth_loss

    def compute_triangulate_loss(self, match, pose, K, K_inv, depth_pred1, depth_pred2):
        depth_pred1 = depth_pred1[0]
        depth_pred2 = depth_pred2[0]
        
        P1, P2 = compute_projection_matrix(pose,K)
        triangulated_point = self.midpoint_triangulate(match,K,K_inv,P1,P2) # [b,n,4]
        point2d_1_coord, point2d_1_depth = self.reproject(P1, triangulated_point) # [b,n,2], [b,n,1]
        point2d_2_coord, point2d_2_depth = self.reproject(P2, triangulated_point)

        rescaled_pred1, inter_pred1 = self.register_depth(depth_pred1, point2d_1_coord, point2d_1_depth)
        rescaled_pred2, inter_pred2 = self.register_depth(depth_pred2, point2d_2_coord, point2d_2_depth)

        pt_depth_loss = self.get_trian_loss(point2d_1_depth, inter_pred1) + self.get_trian_loss(point2d_2_depth, inter_pred2)
        return pt_depth_loss 

    def compute_dynamic_mask(self, intrinsics, depth, pose, flow):
        depth_0 = depth[0]
        dynamic_masks = []
        flow_diffs = []
        flow_diff_scores = []
        for scale in range(self.num_scales):
            
            depth_scale = depth[scale]
            flow_scale = flow[scale]

            b,_,h,w = depth_scale.size()
            downscale = depth_0.size(2)/h
            intrinsics_scaled = torch.cat((intrinsics[:, 0:2]/downscale, intrinsics[:, 2:]), dim=1)
            rigid_flow = calculate_rigid_flow(depth_scale, pose, intrinsics_scaled)

            # consist_bound = torch.max(self.flow_consist_beta * (self.get_flow_norm(flow_scale)+self.get_flow_norm(rigid_flow))/2, torch.from_numpy(np.array([self.flow_consist_alpha])).float().to(flow_scale.get_device()))
            consist_bound = self.flow_consist_alpha * (torch.pow(self.get_flow_norm(flow_scale),2) + torch.pow(self.get_flow_norm(rigid_flow),2)) + self.flow_consist_beta
            flow_diff = torch.abs(rigid_flow - flow_scale)
            flow_diffs.append(flow_diff)

            with torch.no_grad():
                # dyna_mask = (self.get_flow_norm(flow_diff) < consist_bound).float()
                dyna_mask = (torch.pow(self.get_flow_norm(flow_diff),2) < consist_bound).float()
                dynamic_masks.append(dyna_mask)
                flow_diff_score = (1.0 / (1e-4 + self.get_flow_norm(flow_diff)))
                # fwd_mask = self.fusion_mask(valid_mask_fwd, occ_mask_fwd, rigid_mask_fwd_list)
                flow_diff_scores.append(flow_diff_score)

        return flow_diffs, dynamic_masks, flow_diff_scores


    def compute_depth_flow_consis_loss(self, flow_diffs, masks=None, scales=3):
        loss_list = []
        for scale in range(scales):
            flow_diff = flow_diffs[scale]
            b,_,w,h = flow_diff.size()
            # flow_diff_normalized = torch.cat([flow_diff[:,0,:,:].unsqueeze(1)*2/(w-1), flow_diff[:,1,:,:].unsqueeze(1)*2/(h-1)],1)
            if masks == None:
                mask = torch.ones(b,1,w,h).to(flow_diff.get_device())
            else:
                mask = masks[scale]
            
            divider = mask.mean((1,2,3))
            error = (flow_diff*mask.repeat(1,2,1,1)).mean((1,2,3)) / (divider + 1e-12)
            # error = (flow_diff_normalized*mask.repeat(1,2,1,1)).mean((1,2,3)) / (divider + 1e-12)
            loss_list.append(error[:,None])
        loss = torch.cat(loss_list, 1).sum(1) # (B)
        return loss


    def fusion_mask(self, valid_mask, occ_mask, dynamic_mask):
        fusion_mask = []
        for scale in range(self.num_scales):
            valid = valid_mask[scale]
            occ = occ_mask[scale]
            dyna = dynamic_mask[scale]
            mask = valid * occ * dyna
            fusion_mask.append(mask)

        return fusion_mask

    def fusion_mask_4item(self, valid_mask, occ_mask, dynamic_mask, texture_mask):
        fusion_mask = []
        for scale in range(self.num_scales):
            valid = valid_mask[scale]
            occ = occ_mask[scale]
            dyna = dynamic_mask[scale]
            texture = texture_mask[scale]
            mask = valid * occ * dyna * texture
            fusion_mask.append(mask)
        return fusion_mask

    def fusion_mask_2item(self, valid_mask, occ_mask):
        fusion_mask = []
        for scale in range(self.num_scales):
            valid = valid_mask[scale]
            occ = occ_mask[scale]
            mask = valid * occ
            fusion_mask.append(mask)

        return fusion_mask
            

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
        disp_l_list = self.depth_net(img_l) # Nscales * [B, 1, H, W]
        disp_list = self.depth_net(img) 
        disp_r_list = self.depth_net(img_r)

        # pose infer
        pose_inputs = torch.cat([img_l,img,img_r],1)
        pose_vectors = self.pose_net(pose_inputs)
        pose_vec_fwd = pose_vectors[:,1,:]
        pose_vec_bwd = pose_vectors[:,0,:]

        # flow infer
        feature_list_l, feature_list, feature_list_r = self.fpyramid(img_l), self.fpyramid(img), self.fpyramid(img_r)
        optical_flows_bwd = self.pwc_model(feature_list, feature_list_l, [img_h, img_w])
        optical_flows_fwd = self.pwc_model(feature_list, feature_list_r, [img_h, img_w])

        # calculate reconstructed image using depth and pose
        reconstructed_imgs_from_l, valid_masks_to_l, predicted_depths_to_l, computed_depths_to_l = \
            self.reconstruction(img_l, K, disp_list, disp_l_list, pose_vec_bwd)
        reconstructed_imgs_from_r, valid_masks_to_r, predicted_depths_to_r, computed_depths_to_r = \
            self.reconstruction(img_r, K, disp_list, disp_r_list, pose_vec_fwd)

        # calculate texture mask
        texture_mask_bwd = self.compute_texture_mask(img_list, reconstructed_imgs_from_l, img_l_list)
        texture_mask_fwd = self.compute_texture_mask(img_list, reconstructed_imgs_from_r, img_r_list)

        # calculate reconstructed image using flow
        img_warped_pyramid_from_l = self.warp_flow_pyramid(img_l_list, optical_flows_bwd)
        img_warped_pyramid_from_r = self.warp_flow_pyramid(img_r_list, optical_flows_fwd)
        occ_mask_bwd, occ_mask_fwd, valid_mask_bwd, valid_mask_fwd = self.compute_occ_weight(img_warped_pyramid_from_l, img_list, img_warped_pyramid_from_r)

        # dynamic mask by d f consis
        flow_diff_bwd, dynamic_masks_bwd, flow_diff_scores_bwd = self.compute_dynamic_mask(K, disp_list, pose_vec_bwd, optical_flows_bwd)
        flow_diff_fwd, dynamic_masks_fwd, flow_diff_scores_fwd = self.compute_dynamic_mask(K, disp_list, pose_vec_fwd, optical_flows_fwd)

        # rigid mask by epipolar
        dist_map_bwd = self.compute_epipolar_map(pose_vec_bwd, optical_flows_bwd[0], K, K_inv)
        # dist_map_bwd, epipolar_bwd = self.compute_epipolar_map(pose_vec_bwd, optical_flows_bwd[0], K, K_inv)
        dist_map_fwd = self.compute_epipolar_map(pose_vec_fwd, optical_flows_fwd[0], K, K_inv)
        # dist_map_fwd, epipolar_fwd = self.compute_epipolar_map(pose_vec_fwd, optical_flows_fwd[0], K, K_inv)
        rigid_mask_bwd, inlier_mask_bwd, rigid_score_bwd = self.get_rigid_mask(dist_map_bwd)
        rigid_mask_fwd, inlier_mask_fwd, rigid_score_fwd = self.get_rigid_mask(dist_map_fwd)

        # select points for geometry calculation
        # filtered_matches_fwd, filtered_depth_fwd = self.sample_match(optical_flows_fwd[0], disp_list[0], rigid_score_fwd)
        # filtered_matches_bwd, filtered_depth_bwd = self.sample_match(optical_flows_bwd[0], disp_list[0], rigid_score_bwd)
        filtered_matches_fwd, filtered_depth_fwd = self.sample_match(optical_flows_fwd[0], disp_list[0], flow_diff_scores_fwd[0])
        filtered_matches_bwd, filtered_depth_bwd = self.sample_match(optical_flows_bwd[0], disp_list[0], flow_diff_scores_bwd[0])


        # compute epipolar loss by 8 point method
        # dist_map_bwd = self.compute_epipolar_map_8pF(filtered_matches_bwd, pose_vec_bwd, optical_flows_bwd[0], K, K_inv)
        # dist_map_fwd = self.compute_epipolar_map_8pF(filtered_matches_fwd, pose_vec_fwd, optical_flows_fwd[0], K, K_inv)
        # rigid_mask_bwd, inlier_mask_bwd, rigid_score_bwd = self.get_rigid_mask(dist_map_bwd)
        # rigid_mask_fwd, inlier_mask_fwd, rigid_score_fwd = self.get_rigid_mask(dist_map_fwd)

        # rigid_mask_bwd_list = self.generate_img_pyramid(rigid_mask_bwd,self.num_scales)
        # rigid_mask_fwd_list = self.generate_img_pyramid(rigid_mask_fwd,self.num_scales)
        
        # fusion mask
        # fwd_mask = self.fusion_mask(valid_masks_to_r, occ_mask_fwd, dynamic_masks_fwd)
        # bwd_mask = self.fusion_mask(valid_masks_to_l, occ_mask_bwd, dynamic_masks_bwd)

        # fwd_mask_valid_occ = self.fusion_mask_occ_valid(valid_masks_to_r, occ_mask_fwd)
        # bwd_mask_valid_occ = self.fusion_mask_occ_valid(valid_masks_to_l, occ_mask_bwd)

        fwd_mask = self.fusion_mask(valid_mask_fwd, occ_mask_fwd, dynamic_masks_fwd)
        bwd_mask = self.fusion_mask(valid_mask_bwd, occ_mask_bwd, dynamic_masks_bwd)
        # fwd_mask = self.fusion_mask_4item(valid_mask_fwd, occ_mask_fwd, rigid_mask_fwd_list, texture_mask_fwd)
        # bwd_mask = self.fusion_mask_4item(valid_mask_bwd, occ_mask_bwd, rigid_mask_bwd_list, texture_mask_bwd)
        # fwd_mask = self.fusion_mask(valid_mask_fwd, occ_mask_fwd, rigid_mask_fwd_list)
        # bwd_mask = self.fusion_mask(valid_mask_bwd, occ_mask_bwd, rigid_mask_bwd_list)

        fwd_mask_texture = self.fusion_mask_2item(fwd_mask, texture_mask_fwd)
        bwd_mask_texture = self.fusion_mask_2item(bwd_mask, texture_mask_bwd)

        fwd_mask_valid_occ = self.fusion_mask_2item(valid_mask_fwd, occ_mask_fwd)
        bwd_mask_valid_occ = self.fusion_mask_2item(valid_mask_bwd, occ_mask_bwd)

        fwd_mask_valid_occ_rigid = self.fusion_mask_2item(fwd_mask_valid_occ, dynamic_masks_fwd)
        bwd_mask_valid_occ_rigid = self.fusion_mask_2item(bwd_mask_valid_occ, dynamic_masks_bwd)

        fwd_mask_valid_occ_dyna = self.fusion_mask_2item(fwd_mask_valid_occ, [1-mask for mask in dynamic_masks_fwd])
        bwd_mask_valid_occ_dyna = self.fusion_mask_2item(bwd_mask_valid_occ, [1-mask for mask in dynamic_masks_bwd])


        # loss function
        loss_pack = {}
        mask_pack = {}
        
        mask_pack['occ_fwd_mask'] = 255*occ_mask_fwd[0][0].cpu().detach().numpy().astype(np.uint8)
        mask_pack['rigid_fwd_mask'] = 255*rigid_mask_fwd[0].cpu().detach().numpy().astype(np.uint8)
        mask_pack['inlier_fwd_mask'] = 255*inlier_mask_fwd[0].cpu().detach().numpy().astype(np.uint8)
        mask_pack['dyna_fwd_mask'] = 255*dynamic_masks_fwd[0][0].cpu().detach().numpy().astype(np.uint8)
        mask_pack['valid_fwd_mask'] = 255*valid_masks_to_r[0][0].cpu().detach().numpy().astype(np.uint8)
        mask_pack['fwd_mask'] = 255*fwd_mask[0][0].cpu().detach().numpy().astype(np.uint8)
        mask_pack['texture_mask_fwd'] = 255*texture_mask_fwd[0][0].cpu().detach().numpy().astype(np.uint8)
        mask_pack['pred_depth_img'] = disp_list[0][0]
        mask_pack['pred_flow_img'] = optical_flows_fwd[0][0].detach().cpu().numpy().transpose([1, 2, 0])
        mask_pack['origin_middle_image'] = img[0].cpu().detach().numpy()

        # depth and pose
        loss_pack['loss_depth_pixel'] = self.compute_photometric_loss(img_list,reconstructed_imgs_from_l,bwd_mask_texture) + \
            self.compute_photometric_loss(img_list,reconstructed_imgs_from_r,fwd_mask_texture)
        # loss_pack['loss_depth_pixel'] = self.compute_photometric_depth_loss(img_list,reconstructed_imgs_from_l,img_l_list,bwd_mask) + \
            # self.compute_photometric_depth_loss(img_list, reconstructed_imgs_from_r, img_r_list, fwd_mask)
        #loss_pack['loss_depth_pixel'] = torch.zeros([2]).to(img_l.get_device()).requires_grad_()

        # loss_pack['loss_depth_ssim'] = self.compute_ssim_loss(img_list,reconstructed_imgs_from_l,bwd_mask_texture) + \
            # self.compute_ssim_loss(img_list,reconstructed_imgs_from_r,fwd_mask_texture)
        loss_pack['loss_depth_ssim'] = torch.zeros([2]).to(img_l.get_device()).requires_grad_()

        loss_pack['loss_depth_smooth'] = self.compute_smooth_loss(img, disp_list) + self.compute_smooth_loss(img_l, disp_l_list) + \
            self.compute_smooth_loss(img_r, disp_r_list)
        #loss_pack['loss_depth_smooth'] = torch.zeros([2]).to(img_l.get_device()).requires_grad_()

        # loss_pack['loss_depth_consis'] =  self.compute_consis_loss(predicted_depths_to_l, computed_depths_to_l, bwd_mask_texture) + \
            # self.compute_consis_loss(predicted_depths_to_r, computed_depths_to_r, fwd_mask_texture)
        loss_pack['loss_depth_consis'] = torch.zeros([2]).to(img_l.get_device()).requires_grad_()


        # flow
        # loss_pack['loss_flow_pixel'] = self.compute_photometric_loss(img_list,img_warped_pyramid_from_l,bwd_mask_valid_occ) + \
        #     self.compute_photometric_loss(img_list,img_warped_pyramid_from_r,fwd_mask_valid_occ)
        loss_pack['loss_flow_pixel'] = self.compute_photometric_loss(img_list,img_warped_pyramid_from_l,bwd_mask_valid_occ_rigid) + \
            self.compute_photometric_loss(img_list,img_warped_pyramid_from_r,fwd_mask_valid_occ_rigid) + \
            2 * self.compute_photometric_loss(img_list,img_warped_pyramid_from_l,bwd_mask_valid_occ_dyna) + \
            2 * self.compute_photometric_loss(img_list,img_warped_pyramid_from_r,fwd_mask_valid_occ_dyna)
        # loss_pack['loss_flow_pixel'] = torch.zeros([2]).to(img_l.get_device()).requires_grad_()

        loss_pack['loss_flow_ssim'] = self.compute_ssim_loss(img_list,img_warped_pyramid_from_l,bwd_mask_valid_occ) + \
            self.compute_ssim_loss(img_list,img_warped_pyramid_from_r,fwd_mask_valid_occ)
        # loss_pack['loss_flow_ssim'] = torch.zeros([2]).to(img_l.get_device()).requires_grad_()

        loss_pack['loss_flow_smooth'] = self.compute_loss_flow_smooth(optical_flows_fwd, img_list)  + \
            self.compute_loss_flow_smooth(optical_flows_bwd, img_list)
        # loss_pack['loss_flow_smooth'] = torch.zeros([2]).to(img_l.get_device()).requires_grad_()

        loss_pack['loss_flow_consis'] = self.compute_loss_flow_consis(optical_flows_fwd, optical_flows_bwd, occ_mask_fwd)
        # loss_pack['loss_flow_consis'] = torch.zeros([2]).to(img_l.get_device()).requires_grad_()
        
        # fusion geom
        # loss_pack['loss_depth_flow_consis'] = self.compute_depth_flow_consis_loss(flow_diff_bwd, bwd_mask_texture, self.num_scales) + \
        #     self.compute_depth_flow_consis_loss(flow_diff_fwd, fwd_mask_texture, self.num_scales)
        loss_pack['loss_depth_flow_consis'] = self.compute_depth_flow_consis_loss(flow_diff_bwd, bwd_mask, 1) + \
            self.compute_depth_flow_consis_loss(flow_diff_fwd, fwd_mask, 1)
        # loss_pack['loss_depth_flow_consis'] = self.compute_depth_flow_consis_loss(flow_diff_bwd, valid_masks_to_l, 1) + \
        #     self.compute_depth_flow_consis_loss(flow_diff_fwd, valid_masks_to_r, 1)
        # loss_pack['loss_depth_flow_consis'] = torch.zeros([2]).to(img_l.get_device()).requires_grad_()

        # loss_pack['loss_epipolar'] = self.compute_epipolar_loss(dist_map_bwd, rigid_mask_bwd, inlier_mask_bwd) + \
            # self.compute_epipolar_loss(dist_map_fwd, rigid_mask_fwd, inlier_mask_fwd)
        loss_pack['loss_epipolar'] = self.compute_epipolar_loss(dist_map_bwd, dynamic_masks_bwd[0]) + \
            self.compute_epipolar_loss(dist_map_fwd, dynamic_masks_fwd[0])
        # loss_pack['loss_epipolar'] = self.compute_epipolar_loss(dist_map_bwd, valid_mask_bwd[0]) + \
        #     self.compute_epipolar_loss(dist_map_fwd, valid_mask_fwd[0])
        # loss_pack['loss_epipolar'] = torch.zeros([2]).to(img_l.get_device()).requires_grad_()

        # loss_pack['loss_triangle'] = self.compute_triangulate_loss(filtered_matches_bwd, pose_vec_bwd, K, K_inv, disp_list, disp_l_list) + \
            # self.compute_triangulate_loss(filtered_matches_fwd, pose_vec_fwd, K, K_inv, disp_list, disp_r_list)
        # loss_pack['loss_triangle'] = self.compute_triangulate_loss(filtered_matches_bwd, pose_vec_bwd, K, K_inv, filtered_depth_bwd) + \
        #     self.compute_triangulate_loss(filtered_matches_fwd, pose_vec_fwd, K, K_inv, filtered_depth_fwd)
        loss_pack['loss_triangle'] = torch.zeros([2]).to(img_l.get_device()).requires_grad_()

        # loss_pack['loss_pnp'] = self.compute_pnp_loss(filtered_depth_bwd, filtered_matches_bwd, pose_vec_bwd, K, K_inv) + \
        #     self.compute_pnp_loss(filtered_depth_fwd, filtered_matches_fwd, pose_vec_fwd, K, K_inv)
        loss_pack['loss_pnp'] = torch.zeros([2]).to(img_l.get_device()).requires_grad_()

        # loss_pack['loss_eight_point'] = self.compute_eight_point_loss(filtered_matches_bwd, pose_vec_bwd, K, K_inv) + \
            # self.compute_eight_point_loss(filtered_matches_fwd, pose_vec_fwd, K, K_inv)
        loss_pack['loss_eight_point'] = torch.zeros([2]).to(img_l.get_device()).requires_grad_()

        return loss_pack, mask_pack