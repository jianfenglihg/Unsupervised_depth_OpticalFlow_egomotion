import os, sys

def generate_loss_weights_dict(cfg):
    weight_dict = {}

    weight_dict['loss_flow_pixel']   =  cfg.w_flow_pixel
    weight_dict['loss_flow_ssim']    =  cfg.w_flow_ssim
    weight_dict['loss_flow_smooth']  =  cfg.w_flow_smooth
    weight_dict['loss_flow_consis']  =  cfg.w_flow_consis

    weight_dict['loss_depth_pixel']  =  cfg.w_depth_pixel
    weight_dict['loss_depth_ssim' ]   =  cfg.w_depth_ssim
    weight_dict['loss_depth_smooth'] =  cfg.w_depth_smooth
    weight_dict['loss_depth_consis'] =  cfg.w_depth_consis

    return weight_dict

