import torch
import torch.nn.functional as F


def compute_texture_mask(img):
    """
    we filter out the area of non-gradient
    """

    grad_img_x = torch.mean(torch.abs(img[:, :, :, :-1] - img[:, :, :, 1:]), 1, keepdim=True)
    grad_img_y = torch.mean(torch.abs(img[:, :, :-1, :] - img[:, :, 1:, :]), 1, keepdim=True)

    grad_img_x = F.interpolate(grad_img_x, size=(img.shape[2], img.shape[3]), mode='bilinear')
    grad_img_y = F.interpolate(grad_img_y, size=(img.shape[2], img.shape[3]), mode='bilinear')

    texture_mask = (grad_img_x > 0).float() * (grad_img_y > 0).float()

    return texture_mask
