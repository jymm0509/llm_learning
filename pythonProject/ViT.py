import torch
import torch.nn as nn
import torch.nn.functional as F



def image2embed_naive(image, patch_size, weight):
    patch = F.unfold(image, kernel_size=patch_size, stride=patch_size).transpose(-1, -2)
    patch_embedding = patch @ weight
    # print(patch)
    # print(patch_embedding.shape)
    return patch_embedding

bs, ic, image_height, image_width = 1, 3, 8, 8
patch_size = 4
model_dim = 8
patch_depth = patch_size*patch_size*ic
image = torch.randn(bs, ic, image_height, image_width)
weight = torch.randn(patch_depth, model_dim)

patch_embedding = image2embed_naive(image, patch_size, weight)

cls_token = torch.randn(bs, 1, model_dim, requires_grad=True)
torch.cat([cls_token, patch_embedding], dim=1)
