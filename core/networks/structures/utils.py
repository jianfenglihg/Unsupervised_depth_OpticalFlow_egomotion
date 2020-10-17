import torch
import torch.nn.functional as F


def compute_texture_mask(img):
    """
    we filter out the area of non-gradient
    """

    grad_img_x = torch.mean(torch.abs(img[:, :, :, :-1] - img[:, :, :, 1:]), 1, keepdim=True)
    grad_img_y = torch.mean(torch.abs(img[:, :, :-1, :] - img[:, :, 1:, :]), 1, keepdim=True)
    
    texture_mask = (grad_img_x > 0).float() * (grad_img_y > 0).float()

    return texture_mask
