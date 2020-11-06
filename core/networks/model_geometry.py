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
            img_new = F.adaptive_avg_pool2d(img, [int(img_h / (2**s)), int(img_w / (2**s))]).data
            img_pyramid.append(img_new)
        return img_pyramid

    def warp_flow_pyramid(self, img_pyramid, flow_pyramid):
        img_warped_pyramid = []
        for img, flow in zip(img_pyramid, flow_pyramid):
            img_warped_pyramid.append(warp_flow(img, flow, use_mask=True))
        return img_warped_pyramid

    def reconstruction(self, ref_img_list, intrinsics, depth, depth_ref, pose, padding_mode='zeros'):
        reconstructed_img = []
        valid_mask = []
        projected_depth = []
        computed_depth = []
        ref_img = ref_img_list[0]

        for scale in range(self.num_scales):
            
            depth_scale = depth[scale]
            depth_ref_scale = depth_ref[scale]
            b,_,h,w = depth_scale.size()
            # ref_img_scaled = F.interpolate(ref_img, (h, w), mode='area')
            ref_img_scaled = ref_img_list[scale]
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
        for scale in range(self.num_scales):
            img_from_l, img, img_from_r = img_pyramid_from_l[scale], img_pyramid[scale], img_pyramid_from_r[scale]

            img_diff_l = torch.abs((img-img_from_l)).mean(1, True)
            img_diff_r = torch.abs((img-img_from_r)).mean(1, True)

            diff_cat = torch.cat((img_diff_l, img_diff_r),1)
            weight = 1 - nn.functional.softmax(diff_cat,1)
            # weight = Variable(weight.data,requires_grad=False)

            with torch.no_grad():
                weight = 2*torch.exp(-(weight-0.5)**2/0.03).float()
                weight_bwd.append(torch.unsqueeze(weight[:,0,:,:],1))
                weight_fwd.append(torch.unsqueeze(weight[:,1,:,:],1))
                
        return weight_bwd, weight_fwd

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
            loss_list.append(grad_disp_x[:,None]) # (B)
        
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

    def meshgrid(self, h, w):
        xx, yy = np.meshgrid(np.arange(0,w), np.arange(0,h))
        meshgrid = np.transpose(np.stack([xx,yy], axis=-1), [2,0,1]) # [2,h,w]
        meshgrid = torch.from_numpy(meshgrid)
        return meshgrid

    def compute_epipolar_loss(self, pose, flow, intrinsics, intrinsics_inverse, mask):
        b,_,h,w = flow.size()
        grid = self.meshgrid(h, w).float().to(flow.get_device()).unsqueeze(0).repeat(b,1,1,1) #[b,2,h,w]
        corres = torch.cat([(grid[:,0,:,:] + flow[:,0,:,:]).unsqueeze(1), (grid[:,1,:,:] + flow[:,1,:,:]).unsqueeze(1)], 1)
        match = torch.cat([grid, corres], 1) # [b,4,h,w]
        match = match.view([b,4,-1]) # [b,4,n]
        mask = mask.view([b,1,-1]) # [b,1,n] 

        num_batch = match.shape[0]
        match_num = match.shape[-1]

        points1 = match[:,:2,:]
        points2 = match[:,2:,:]
        ones = torch.ones(num_batch, 1, match_num).to(points1.get_device())
        points1 = torch.cat([points1, ones], 1) # [b,3,n]
        points2 = torch.cat([points2, ones], 1).transpose(1,2) # [b,n,3]

        # compute fundmental matrix
        E = compute_essential_matrix(pose)
        F_meta = E.bmm(intrinsics_inverse)
        F = torch.inverse(intrinsics.permute([0, 2, 1])).bmm(F_meta)  # T and then -1 

        epi_line = F.bmm(points1) # [b,3,n]
        dist_p2l = torch.abs((epi_line.permute([0, 2, 1]) * points2).sum(-1, keepdim=True)) # [b,n,1]

        a = epi_line[:,0,:].unsqueeze(1).transpose(1,2) # [b,n,1]
        b = epi_line[:,1,:].unsqueeze(1).transpose(1,2) # [b,n,1]
        dist_div = torch.sqrt(a*a + b*b) + 1e-6
        dist_map = dist_p2l / dist_div # [B, n, 1]

        loss = (dist_map * mask.transpose(1,2)).mean([1,2]) / mask.mean([1,2])
        return loss

    def compute_pnp_loss(self, depth, flow, pose_vec, K, K_inv):
        # world point
        b, h, w = depth.shape[0], depth.shape[2], depth.shape[3]
        xy = self.meshgrid(h,w).unsqueeze(0).repeat(b,1,1,1).float().to(flow.get_device()) # [b,2,h,w]
        ones = torch.ones([b,1,h,w]).float().to(flow.get_device())
        pts_3d = K_inv.bmm(torch.cat([xy, ones], 1).view([b,3,-1])) * depth.view([b,1,-1]) # [b,3,h*w]
        pts_3d = pts_3d.transpose(1,2) # [b,,n,3]

        # image point
        corres = torch.cat([(xy[:,0,:,:] + flow[:,0,:,:]).unsqueeze(1), (xy[:,1,:,:] + flow[:,1,:,:]).unsqueeze(1)], 1) # [b,2,h,w]
        corres = corres.view([b,2,-1]).transpose(1,2) # [b,n,2]

        losses = []
        for i in range(b):
            pts_3d_ = pts_3d[i,:,:,:]
            corres_ = corres[i,:,:,:]
            K_ = K[i].cpu().detach().numpy() # [3,3]
            pts_3d_ = pts_3d_.cpu().detach().numpy() # [n,3]
            corres_ = corres_.cpu().detach().numpy() # [n,2]
            # flag, r, t, inlier = cv2.solvePnP(objectPoints=pts_3d_, imagePoints=corres_, cameraMatrix=K_, distCoeffs=None, iterationsCount=self.PnP_ransac_iter, reprojectionError=self.PnP_ransac_thre)
            retval,rvec,tvec = cv2.solvePnP(pts_3d_,corres_,K_)
            tvec = torch.from_numpy(tvec).to(flow.get_device())
            pose_tvec = pose_vec[i,:3]
            loss = torch.abs(tvec-pose_tvec).mean(1)
            losses.append(loss)
        return torch.cat(losses, 0).sum(0) 

    
    def midpoint_triangulate(self, flow, K, K_inv, P1, P2):
        # match: [b, 4, num] P1: [b, 3, 4]
        # Match is in the image coordinates. P1, P2 is camera parameters. [B, 3, 4] match: [B, M, 4]
        b,_,h,w = flow.size()
        grid = self.meshgrid(h, w).float().to(flow.get_device()).unsqueeze(0).repeat(b,1,1,1) #[b,2,h,w]
        corres = torch.cat([(grid[:,0,:,:] + flow[:,0,:,:]).unsqueeze(1), (grid[:,1,:,:] + flow[:,1,:,:]).unsqueeze(1)], 1)
        match = torch.cat([grid, corres], 1) # [b,4,h,w]
        match = match.view([b,4,-1]) # [b,4,n]

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
        # point_homo = torch.cat([point, ones], dim=1).transpose(1,2) # [b,n,4]
        return point


    def get_reproj_fdp_loss(self, pred1, pred2, P2, K, K_inv, valid_mask, rigid_mask, flow, visualizer=None):
        # pred: [b,1,h,w] Rt: [b,3,4] K: [b,3,3] mask: [b,1,h,w] flow: [b,2,h,w]
        b, h, w = pred1.shape[0], pred1.shape[2], pred1.shape[3]
        xy = self.meshgrid(h,w).unsqueeze(0).repeat(b,1,1,1).float().to(flow.get_device()) # [b,2,h,w]
        ones = torch.ones([b,1,h,w]).float().to(flow.get_device())
        pts1_3d = K_inv.bmm(torch.cat([xy, ones], 1).view([b,3,-1])) * pred1.view([b,1,-1]) # [b,3,h*w]
        pts2_coord, pts2_depth = self.reproject(P2, torch.cat([pts1_3d, ones.view([b,1,-1])], 1).transpose(1,2)) # [b,h*w, 2]
        # TODO Here some of the reprojection coordinates are invalid. (<0 or >max)
        reproj_valid_mask = (pts2_coord > torch.Tensor([0,0]).to(pred1.get_device())).all(-1, True).float() * \
            (pts2_coord < torch.Tensor([w-1,h-1]).to(pred1.get_device())).all(-1, True).float() # [b,h*w, 1]
        reproj_valid_mask = (valid_mask * reproj_valid_mask.view([b,h,w,1]).permute([0,3,1,2])).detach()
        rigid_mask = rigid_mask.detach()
        pts2_depth = pts2_depth.transpose(1,2).view([b,1,h,w])

        # Get the interpolated depth prediction2
        pts2_coord_nor = torch.cat([2.0 * pts2_coord[:,:,0].unsqueeze(-1) / (w - 1.0) - 1.0, 2.0 * pts2_coord[:,:,1].unsqueeze(-1) / (h - 1.0) - 1.0], -1)
        inter_depth2 = F.grid_sample(pred2, pts2_coord_nor.view([b, h, w, 2]), padding_mode='reflection') # [b,1,h,w]
        pj_loss_map = (torch.abs(1.0 - pts2_depth / (inter_depth2 + 1e-12)) * rigid_mask * reproj_valid_mask)
        pj_loss = pj_loss_map.mean((1,2,3)) / ((reproj_valid_mask * rigid_mask).mean((1,2,3))+1e-12)
        #pj_loss = (valid_mask * mask * torch.abs(pts2_depth - inter_depth2) / (torch.abs(pts2_depth + inter_depth2)+1e-12)).mean((1,2,3)) / ((valid_mask * mask).mean((1,2,3))+1e-12) # [b]
        flow_loss = (rigid_mask * torch.abs(flow + xy - pts2_coord.detach().permute(0,2,1).view([b,2,h,w]))).mean((1,2,3)) / (rigid_mask.mean((1,2,3)) + 1e-12)
        return pj_loss, flow_loss

    def reproject(self, P, point3d):
        # P: [b,3,4] point3d: [b,n,4]
        point2d = P.bmm(point3d.transpose(1,2)) # [b,4,n]
        point2d_coord = (point2d[:,:2,:] / (point2d[:,2,:].unsqueeze(1) + 1e-12)).transpose(1,2) # [b,n,2]
        point2d_depth = point2d[:,2,:].unsqueeze(1).transpose(1,2) # [b,n,1]
        return point2d_coord, point2d_depth

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
        loss = torch.pow(1.0 - pred_tri_depth / (tri_depth + 1e-12), 2).mean((1,2))
        return loss


    def compute_triangulate_loss(self, flow, pose, K, K_inv, depth_pred1, depth_pred2):
        P1, P2 = compute_projection_matrix(pose,K)
        triangulated_point = self.midpoint_triangulate(flow,K,K_inv,P1,P2)
        point2d_1_coord, point2d_1_depth = self.reproject(P1, triangulated_point) # [b,n,2], [b,n,1]
        point2d_2_coord, point2d_2_depth = self.reproject(P2, triangulated_point)

        rescaled_pred1, inter_pred1 = self.register_depth(depth_pred1, point2d_1_coord, point2d_1_depth)
        rescaled_pred2, inter_pred2 = self.register_depth(depth_pred2, point2d_2_coord, point2d_2_depth)

        pt_depth_loss = self.get_trian_loss(point2d_1_depth, inter_pred1) + self.get_trian_loss(point2d_2_depth, inter_pred2)
        pj_depth, flow_loss = self.get_reproj_fdp_loss(rescaled_pred1, rescaled_pred2, P2, K, K_inv, img1_valid_mask, img1_rigid_mask, fwd_flow, visualizer=visualizer)
        return pt_depth_loss + pj_depth + flow_loss

    def compute_dynamic_mask(self, intrinsics, depth, pose, flow):
        depth_0 = depth[0]
        dynamic_masks = []
        flow_diffs = []
        for scale in range(self.num_scales):
            
            depth_scale = depth[scale]
            flow_scale = flow[scale]

            b,_,h,w = depth_scale.size()
            downscale = depth_0.size(2)/h
            intrinsics_scaled = torch.cat((intrinsics[:, 0:2]/downscale, intrinsics[:, 2:]), dim=1)
            rigid_flow = calculate_rigid_flow(depth_scale, pose, intrinsics_scaled)

            consist_bound = torch.max(self.flow_consist_beta * (self.get_flow_norm(flow_scale)+self.get_flow_norm(rigid_flow))/2, torch.from_numpy(np.array([self.flow_consist_alpha])).float().to(flow_scale.get_device()))
            flow_diff = torch.abs(rigid_flow - flow_scale)
            flow_diffs.append(flow_diff)

            with torch.no_grad():
                dyna_mask = (self.get_flow_norm(flow_diff) < consist_bound).float()
                dynamic_masks.append(dyna_mask)

        return flow_diffs, dynamic_masks


    def compute_depth_flow_consis_loss(self, flow_diffs, masks=None, scales=3):
        loss_list = []
        for scale in range(scales):
            flow_diff = flow_diffs[scale]
            b,_,w,h = flow_diff.size()
            flow_diff_normalized = torch.cat([flow_diff[:,0,:,:].unsqueeze(1)*2/(w-1), flow_diff[:,1,:,:].unsqueeze(1)*2/(h-1)],1)
            if masks == None:
                mask = torch.ones(b,1,w,h).to(flow_diff.get_device())
            else:
                mask = masks[scale]
            
            divider = mask.mean((1,2,3))
            # error = (flow_diff*mask.repeat(1,2,1,1)).mean((1,2,3)) / (divider + 1e-12)
            error = (flow_diff_normalized*mask.repeat(1,2,1,1)).mean((1,2,3)) / (divider + 1e-12)
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

    def fusion_mask_occ_valid(self, valid_mask, occ_mask):
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

        # print(len(optical_flows_fwd))


        # calculate reconstructed image using depth and pose
        reconstructed_imgs_from_l, valid_masks_to_l, predicted_depths_to_l, computed_depths_to_l = \
            self.reconstruction(img_l_list, K, disp_list, disp_l_list, pose_vec_bwd)
        reconstructed_imgs_from_r, valid_masks_to_r, predicted_depths_to_r, computed_depths_to_r = \
            self.reconstruction(img_r_list, K, disp_list, disp_r_list, pose_vec_fwd)

        # calculate reconstructed image using flow
        img_warped_pyramid_from_l = self.warp_flow_pyramid(img_l_list, optical_flows_bwd)
        img_warped_pyramid_from_r = self.warp_flow_pyramid(img_r_list, optical_flows_fwd)
        occ_mask_bwd, occ_mask_fwd = self.compute_occ_weight(img_warped_pyramid_from_l, img_list, img_warped_pyramid_from_r)

        # dynamic mask
        flow_diff_bwd, dynamic_masks_bwd = self.compute_dynamic_mask(K, disp_list, pose_vec_bwd, optical_flows_bwd)
        flow_diff_fwd, dynamic_masks_fwd = self.compute_dynamic_mask(K, disp_list, pose_vec_fwd, optical_flows_fwd)

        fwd_mask = self.fusion_mask(valid_masks_to_r, occ_mask_fwd, dynamic_masks_fwd)
        bwd_mask = self.fusion_mask(valid_masks_to_l, occ_mask_bwd, dynamic_masks_bwd)

        fwd_mask_valid_occ = self.fusion_mask_occ_valid(valid_masks_to_r, occ_mask_fwd)
        bwd_mask_valid_occ = self.fusion_mask_occ_valid(valid_masks_to_l, occ_mask_bwd)



        # cv2.imwrite('./meta/occ_mask.png', np.transpose(255*occ_mask_fwd[0][0].cpu().detach().numpy(), [1,2,0]).astype(np.uint8))
        # cv2.imwrite('./meta/dyna_mask.png', np.transpose(255*dynamic_masks_fwd[0][0].cpu().detach().numpy(), [1,2,0]).astype(np.uint8))
        # cv2.imwrite('./meta/valid_mask.png', np.transpose(255*valid_masks_to_r[0][0].cpu().detach().numpy(), [1,2,0]).astype(np.uint8))

        # loss function
        loss_pack = {}
        

        # depth and pose
        loss_pack['loss_depth_pixel'] = self.compute_photometric_loss(img_list,reconstructed_imgs_from_l,bwd_mask) + \
            self.compute_photometric_loss(img_list,reconstructed_imgs_from_r,fwd_mask)
        #loss_pack['loss_depth_pixel'] = torch.zeros([2]).to(img_l.get_device()).requires_grad_()

        loss_pack['loss_depth_ssim'] = self.compute_ssim_loss(img_list,reconstructed_imgs_from_l,bwd_mask) + \
            self.compute_ssim_loss(img_list,reconstructed_imgs_from_r,fwd_mask)
        # loss_pack['loss_depth_ssim'] = torch.zeros([2]).to(img_l.get_device()).requires_grad_()

        loss_pack['loss_depth_smooth'] = self.compute_smooth_loss(img, disp_list) + self.compute_smooth_loss(img_l, disp_l_list) + \
            self.compute_smooth_loss(img_r, disp_r_list)
        #loss_pack['loss_depth_smooth'] = torch.zeros([2]).to(img_l.get_device()).requires_grad_()

        loss_pack['loss_depth_consis'] =  self.compute_consis_loss(predicted_depths_to_l, computed_depths_to_l) + \
            self.compute_consis_loss(predicted_depths_to_r, computed_depths_to_r)
        # loss_pack['loss_depth_consis'] = torch.zeros([2]).to(img_l.get_device()).requires_grad_()


        # flow
        loss_pack['loss_flow_pixel'] = self.compute_photometric_loss(img_list,img_warped_pyramid_from_l,bwd_mask_valid_occ) + \
            self.compute_photometric_loss(img_list,img_warped_pyramid_from_r,fwd_mask_valid_occ)
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
        loss_pack['loss_depth_flow_consis'] = self.compute_depth_flow_consis_loss(flow_diff_bwd, bwd_mask, 1) + \
            self.compute_depth_flow_consis_loss(flow_diff_fwd, fwd_mask, 1)
        # loss_pack['loss_depth_flow_consis'] = self.compute_depth_flow_consis_loss(flow_diff_bwd, valid_masks_to_l, 1) + \
        #     self.compute_depth_flow_consis_loss(flow_diff_fwd, valid_masks_to_r, 1)
        # loss_pack['loss_depth_flow_consis'] = torch.zeros([2]).to(img_l.get_device()).requires_grad_()

        loss_pack['loss_epipolar'] = self.compute_epipolar_loss(pose_vec_bwd,optical_flows_bwd[0],K,K_inv,bwd_mask[0]) + \
            self.compute_epipolar_loss(pose_vec_fwd,optical_flows_fwd[0],K,K_inv,fwd_mask[0])
        # loss_pack['loss_epipolar'] = torch.zeros([2]).to(img_l.get_device()).requires_grad_()

        return loss_pack