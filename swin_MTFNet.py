import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
import numpy as np
from timm.models.layers import DropPath, trunc_normal_


import torch.distributed as dist

from functools import reduce, lru_cache
from operator import mul
from einops import rearrange

import logging
import os
from PIL import Image
from torch.utils import data

from torch.nn import init
import torchvision.models as models
import torchvision.transforms as transforms
from tqdm import tqdm
import cv2
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

from matplotlib import pylab
from sklearn.metrics import f1_score,precision_score,recall_score,roc_auc_score,roc_curve,confusion_matrix,accuracy_score,log_loss
from sklearn.metrics import precision_recall_curve, average_precision_score,roc_curve, auc, confusion_matrix, accuracy_score, classification_report

from torch import einsum
from einops import rearrange, repeat

import torch.optim as optim
from torch.optim import lr_scheduler
import torchvision
from torchvision import datasets, models, transforms

import time
import os
import copy
import csv

MAX_frame = 30
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")   # use CPU or GPU


class Dataset_3DCNN(data.Dataset):
    "Characterizes a dataset for PyTorch"

    def __init__(self, data_path, transform=None):
        "Initialization"
        self.data_path = data_path
        self.transform = transform

    def __len__(self):
        "Denotes the total number of samples"
        return len(os.listdir(self.data_path))

    def read_images(self, sub_path, use_transform=None):
        X = []
        pic_name_required = []
        pic_name_list = os.listdir(sub_path)

        pic_name_list = sorted(pic_name_list, key=lambda x: str(x.split('.')[0].split('_')[-1]))
        pic_name_list = sorted(pic_name_list, key=lambda x: len(x.split('.')[0].split('_')[-1]))

        num_frame = len(pic_name_list)
        if num_frame > MAX_frame:
            step_need = (num_frame) / (MAX_frame)
            for step in range(MAX_frame):
                required = int(step_need * step)
                if int(step_need * (step + 1)) >= num_frame:
                    pic_name_required.append(pic_name_list[-1])
                else:
                    pic_name_required.append(pic_name_list[required])
        else:
            pic_name_required = pic_name_list.copy()

        for frame_name in pic_name_required:
            frame_path = os.path.join(sub_path, frame_name)
            image = Image.open(frame_path)
            if use_transform is not None:
                image = use_transform(image)
            #             image = torch.from_numpy(image)
            X.append(image)
        X = torch.stack(X, dim=0)

        #         print(X.shape)

        return X

    def __getitem__(self, index):
        "Generates one sample of data"
        self.one_sequence_dir = os.listdir(self.data_path)
        selected_sequence = self.one_sequence_dir[index]
        one_sequence_path = os.path.join(self.data_path, selected_sequence) + '/'
        #         X = self.read_images(one_sequence_path, self.transform).unsqueeze_(0)
        X = self.read_images(one_sequence_path, self.transform)
        X = torch.transpose(X, 0, 1)
        if os.listdir(one_sequence_path)[0].split("_")[0] == "A":
            y = torch.Tensor([1])
        else:
            y = torch.Tensor([0])

        return X, y


def get_root_logger(log_file=None, log_level=logging.INFO):
    """Use ``get_logger`` method in mmcv to get the root logger.
    The logger will be initialized if it has not been initialized. By default a
    StreamHandler will be added. If ``log_file`` is specified, a FileHandler
    will also be added. The name of the root logger is the top-level package
    name, e.g., "mmaction".
    Args:
        log_file (str | None): The log filename. If specified, a FileHandler
            will be added to the root logger.
        log_level (int): The root logger level. Note that only the process of
            rank 0 is affected, while other processes will set the level to
            "Error" and be silent most of the time.
    Returns:
        :obj:`logging.Logger`: The root logger.
    """
    return get_logger(__name__.split('.')[0], log_file, log_level)


class Mlp(nn.Module):
    """ Multilayer perceptron."""

    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.inplanes = 64
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


def window_partition(x, window_size):
    """
    Args:
        x: (B, D, H, W, C)
        window_size (tuple[int]): window size
    Returns:
        windows: (B*num_windows, window_size*window_size, C)
    """
    B, D, H, W, C = x.shape
    x = x.view(B, D // window_size[0], window_size[0], H // window_size[1], window_size[1], W // window_size[2],
               window_size[2], C)
    windows = x.permute(0, 1, 3, 5, 2, 4, 6, 7).contiguous().view(-1, reduce(mul, window_size), C)
    return windows


def window_reverse(windows, window_size, B, D, H, W):
    """
    Args:
        windows: (B*num_windows, window_size, window_size, C)
        window_size (tuple[int]): Window size
        H (int): Height of image
        W (int): Width of image
    Returns:
        x: (B, D, H, W, C)
    """
    x = windows.view(B, D // window_size[0], H // window_size[1], W // window_size[2], window_size[0], window_size[1],
                     window_size[2], -1)
    x = x.permute(0, 1, 4, 2, 5, 3, 6, 7).contiguous().view(B, D, H, W, -1)
    return x


def get_window_size(x_size, window_size, shift_size=None):
    use_window_size = list(window_size)
    if shift_size is not None:
        use_shift_size = list(shift_size)
    for i in range(len(x_size)):
        if x_size[i] <= window_size[i]:
            use_window_size[i] = x_size[i]
            if shift_size is not None:
                use_shift_size[i] = 0

    if shift_size is None:
        return tuple(use_window_size)
    else:
        return tuple(use_window_size), tuple(use_shift_size)

class WindowAttention3D(nn.Module):
    """ Window based multi-head self attention (W-MSA) module with relative position bias.
    It supports both of shifted and non-shifted window.
    Args:
        dim (int): Number of input channels.
        window_size (tuple[int]): The temporal length, height and width of the window.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
    """

    def __init__(self, dim, window_size, num_heads, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):

        super().__init__()
        self.dim = dim
        self.window_size = window_size  # Wd, Wh, Ww
        self.num_heads = num_heads
        head_dim = dim // num_heads   
        self.scale = qk_scale or head_dim ** -0.5

        # define a parameter table of relative position bias

        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1) * (2 * window_size[2] - 1),
                        num_heads))  # 2*Wd-1 * 2*Wh-1 * 2*Ww-1, nH


        # get pair-wise relative position index for each token inside the window
        coords_d = torch.arange(self.window_size[0])
        coords_h = torch.arange(self.window_size[1])
        coords_w = torch.arange(self.window_size[2])
        coords = torch.stack(torch.meshgrid(coords_d, coords_h, coords_w))  # 3, Wd, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 3, Wd*Wh*Ww

        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 3, Wd*Wh*Ww, Wd*Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wd*Wh*Ww, Wd*Wh*Ww, 3
 
        relative_coords[:, :, 0] += self.window_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 2] += self.window_size[2] - 1
 
        relative_coords[:, :, 0] *= (2 * self.window_size[1] - 1) * (2 * self.window_size[2] - 1)
        relative_coords[:, :, 1] *= (2 * self.window_size[2] - 1)

        relative_position_index = relative_coords.sum(-1)  # Wd*Wh*Ww, Wd*Wh*Ww
        self.register_buffer("relative_position_index", relative_position_index)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)


        trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask=None):
        """ Forward function.
        Args:
            x: input features with shape of (num_windows*B, N, C)
            mask: (0/-inf) mask with shape of (num_windows, N, N) or None
        """

        B_, N, C = x.shape

        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # B_, nH, N, C

        q = q * self.scale
        attn = q @ k.transpose(-2, -1)

        relative_position_bias = self.relative_position_bias_table[
            self.relative_position_index[:N, :N].reshape(-1)].reshape(
            N, N, -1)  # Wd*Wh*Ww,Wd*Wh*Ww,nH
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wd*Wh*Ww, Wd*Wh*Ww
        attn = attn + relative_position_bias.unsqueeze(0)  # B_, nH, N, N



            nW = mask.shape[0]

            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x




class SwinTransformerBlock3D(nn.Module):
    """ Swin Transformer Block.
    Args:
        dim (int): Number of input channels.
        num_heads (int): Number of attention heads.
        window_size (tuple[int]): Window size.
        shift_size (tuple[int]): Shift size for SW-MSA.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float, optional): Stochastic depth rate. Default: 0.0
        act_layer (nn.Module, optional): Activation layer. Default: nn.GELU
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, dim, num_heads, window_size=(2, 7, 7), shift_size=(0, 0, 0),
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm, use_checkpoint=False):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        self.use_checkpoint = use_checkpoint

        assert 0 <= self.shift_size[0] < self.window_size[0], "shift_size must in 0-window_size"
        assert 0 <= self.shift_size[1] < self.window_size[1], "shift_size must in 0-window_size"
        assert 0 <= self.shift_size[2] < self.window_size[2], "shift_size must in 0-window_size"

        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention3D(
            dim, window_size=self.window_size, num_heads=num_heads,
            qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        # self.cbam = CBAMBlock(channel=96)
        # self.aft = AFT_FULL()


    def forward_part1(self, x, mask_matrix):
        B, D, H, W, C = x.shape

        window_size, shift_size = get_window_size((D, H, W), self.window_size, self.shift_size)
        x = self.norm1(x)

        # pad feature maps to multiples of window size
        pad_l = pad_t = pad_d0 = 0
        pad_d1 = (window_size[0] - D % window_size[0]) % window_size[0]
        pad_b = (window_size[1] - H % window_size[1]) % window_size[1]
        pad_r = (window_size[2] - W % window_size[2]) % window_size[2]
        x = F.pad(x, (0, 0, pad_l, pad_r, pad_t, pad_b, pad_d0, pad_d1))
        _, Dp, Hp, Wp, _ = x.shape

        # cyclic shift
        if any(i > 0 for i in shift_size):
            shifted_x = torch.roll(x, shifts=(-shift_size[0], -shift_size[1], -shift_size[2]), dims=(1, 2, 3))
            attn_mask = mask_matrix
        else:
            shifted_x = x
            attn_mask = None


        # partition windows
        x_windows = window_partition(shifted_x, window_size)  # B*nW, Wd*Wh*Ww, C


        # W-MSA/SW-MSA
        attn_windows = self.attn(x_windows, mask=attn_mask)  # B*nW, Wd*Wh*Ww, C


        # merge windows
        attn_windows = attn_windows.view(-1, *(window_size + (C,)))
        shifted_x = window_reverse(attn_windows, window_size, B, Dp, Hp, Wp)  # B D' H' W' C


        # reverse cyclic shift
        if any(i > 0 for i in shift_size):
            x = torch.roll(shifted_x, shifts=(shift_size[0], shift_size[1], shift_size[2]), dims=(1, 2, 3))
        else:
            x = shifted_x


        if pad_d1 > 0 or pad_r > 0 or pad_b > 0:
            x = x[:, :D, :H, :W, :].contiguous()
        return x

    def forward_part2(self, x):

        return self.drop_path(self.mlp(self.norm2(x)))

    def forward(self, x, mask_matrix):
        """ Forward function.
        Args:
            x: Input feature, tensor size (B, D, H, W, C).
            mask_matrix: Attention mask for cyclic shift.
        """

        shortcut = x
        # print(x.shape)
        if self.use_checkpoint:
            x = checkpoint.checkpoint(self.forward_part1, x, mask_matrix)
        else:
            x = self.forward_part1(x, mask_matrix)
        x = shortcut + self.drop_path(x)

        if self.use_checkpoint:
            x = x + checkpoint.checkpoint(self.forward_part2, x)
        else:
            x = x + self.forward_part2(x)

        return x

class PatchMerging(nn.Module):
    """ Patch Merging Layer
    Args:
        dim (int): Number of input channels.
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim

        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = norm_layer(4 * dim)

    def forward(self, x):
        """ Forward function.
        Args:
            x: Input feature, tensor size (B, D, H, W, C).
        """
        B, D, H, W, C = x.shape

        # padding
        pad_input = (H % 2 == 1) or (W % 2 == 1)
        if pad_input:
            x = F.pad(x, (0, 0, 0, W % 2, 0, H % 2))


        x0 = x[:, :, 0::2, 0::2, :]  
        x1 = x[:, :, 1::2, 0::2, :]  
        x2 = x[:, :, 0::2, 1::2, :]  
        x3 = x[:, :, 1::2, 1::2, :]  

        x = torch.cat([x0, x1, x2, x3], -1)  

        x = self.norm(x) 
        x = self.reduction(x) 

        return x


# cache each stage results

@lru_cache()
def compute_mask(D, H, W, window_size, shift_size, device):
    img_mask = torch.zeros((1, D, H, W, 1), device=device)  
    cnt = 0

    for d in slice(-window_size[0]), slice(-window_size[0], -shift_size[0]), slice(-shift_size[0], None):
        for h in slice(-window_size[1]), slice(-window_size[1], -shift_size[1]), slice(-shift_size[1], None):
            for w in slice(-window_size[2]), slice(-window_size[2], -shift_size[2]), slice(-shift_size[2], None):
                img_mask[:, d, h, w, :] = cnt
                cnt += 1
    mask_windows = window_partition(img_mask, window_size)  
    mask_windows = mask_windows.squeeze(-1)  

    attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
    attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
    return attn_mask


class BasicLayer(nn.Module):
    """ A basic Swin Transformer layer for one stage.
    Args:
        dim (int): Number of feature channels
        depth (int): Depths of this stage.
        num_heads (int): Number of attention head.
        window_size (tuple[int]): Local window size. Default: (1,7,7).
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 4.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        downsample (nn.Module | None, optional): Downsample layer at the end of the layer. Default: None
    """

    def __init__(self,
                 dim,  
                 depth, 
                 num_heads,  
                 window_size=(1, 7, 7),   
                 mlp_ratio=4.,
                 qkv_bias=False,   
                 qk_scale=None,
                 drop=0.,
                 attn_drop=0.,
                 drop_path=0.,   
                 norm_layer=nn.LayerNorm,
                 downsample=None,  
                 use_checkpoint=False):
        super().__init__()
        self.window_size = window_size 
        self.shift_size = tuple(i // 2 for i in window_size)  
        self.depth = depth 
        self.use_checkpoint = use_checkpoint

        # build blocks
        self.blocks = nn.ModuleList([
            SwinTransformerBlock3D(
                dim=dim,   # 96
                num_heads=num_heads,   # 3
                window_size=window_size,

                shift_size=(0, 0, 0) if (i % 2 == 0) else self.shift_size,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,   # True
                qk_scale=qk_scale,   # None
                drop=drop,
                attn_drop=attn_drop,
                drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                norm_layer=norm_layer,
                use_checkpoint=use_checkpoint,
            )
            for i in range(depth)])   # depth=2

        self.downsample = downsample
        if self.downsample is not None:
            self.downsample = downsample(dim=dim, norm_layer=norm_layer)

    def forward(self, x):
        """ Forward function.
        Args:
            x: Input feature, tensor size (B, C, D, H, W).
        """
        # calculate attention mask for SW-MSA
        B, C, D, H, W = x.shape
        window_size, shift_size = get_window_size((D, H, W), self.window_size, self.shift_size)
        x = rearrange(x, 'b c d h w -> b d h w c')  
        Dp = int(np.ceil(D / window_size[0])) * window_size[0]  
        Hp = int(np.ceil(H / window_size[1])) * window_size[1] 
        Wp = int(np.ceil(W / window_size[2])) * window_size[2]  

        attn_mask = compute_mask(Dp, Hp, Wp, window_size, shift_size, x.device)


        for blk in self.blocks:
            x = blk(x, attn_mask)

        x = x.view(B, D, H, W, -1)

        if self.downsample is not None:
            x = self.downsample(x)
        x = rearrange(x, 'b d h w c -> b c d h w')
        return x



class PatchEmbed3D(nn.Module):
    """ Video to Patch Embedding.
    Args:
        patch_size (int): Patch token size. Default: (2,4,4).
        in_chans (int): Number of input video channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        norm_layer (nn.Module, optional): Normalization layer. Default: None
    """

  
    def __init__(self, patch_size=(2, 4, 4), in_chans=3, embed_dim=96, norm_layer=None):
        super().__init__()
        self.patch_size = patch_size

        self.in_chans = in_chans
        self.embed_dim = embed_dim


        self.proj = nn.Conv3d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        """Forward function."""
        # padding
        _, _, D, H, W = x.size()  # B,C,D,H,W

        if W % self.patch_size[2] != 0:
            x = F.pad(x, (0, self.patch_size[2] - W % self.patch_size[2]))
        if H % self.patch_size[1] != 0:
            x = F.pad(x, (0, 0, 0, self.patch_size[1] - H % self.patch_size[1]))
        if D % self.patch_size[0] != 0:
            x = F.pad(x, (0, 0, 0, 0, 0, self.patch_size[0] - D % self.patch_size[0]))

        x = self.proj(x) 
        if self.norm is not None:   
            D, Wh, Ww = x.size(2), x.size(3), x.size(4)
           
            x = x.flatten(2).transpose(1, 2)  # B, C, D, Wh, Ww -> B, C, D*Wh*Ww ->B,D*Wh*Ww, C
            x = self.norm(x)
            x = x.transpose(1, 2).view(-1, self.embed_dim, D, Wh, Ww)  

        return x



class MultiScaleTemporalAttention(nn.Module):
    def __init__(self, in_channels, scales=[2, 4, 8], reduction=16):
        super(MultiScaleTemporalAttention, self).__init__()
        self.scales = scales
        self.attention_layers = nn.ModuleList([
            nn.Sequential(
                nn.AdaptiveAvgPool3d((scale, 1, 1)),
                nn.Conv3d(in_channels, in_channels // reduction, kernel_size=1, bias=False),
                nn.ReLU(),
                nn.Conv3d(in_channels // reduction, in_channels, kernel_size=1, bias=False),
                nn.Sigmoid()
            )
            for scale in scales
        ])

    def forward(self, x):
        batch_size, channels, time, height, width = x.size()

        attention_weights = []
        for layer in self.attention_layers:
            pooled = layer[0](x)
            pooled = F.interpolate(pooled, size=(time, height, width), mode='trilinear', align_corners=True)
            attention = layer[1:](pooled)
            attention_weights.append(attention)


        out = sum(attention_weights) / len(self.scales)

        return x * out



class FrequencyAttention(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super(FrequencyAttention, self).__init__()
        self.conv1 = nn.Conv3d(in_channels, in_channels // reduction, kernel_size=1, bias=False)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv3d(in_channels // reduction, in_channels, kernel_size=1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):

        x_freq = torch.fft.fftn(x, dim=2)  


        x_real = x_freq.real
        x_imag = x_freq.imag

        real_features = self.conv1(x_real)
        real_features = self.relu(real_features)
        real_features = self.conv2(real_features)


        imag_features = self.conv1(x_imag)
        imag_features = self.relu(imag_features)
        imag_features = self.conv2(imag_features)


        freq_attention_real = self.sigmoid(real_features)
        freq_attention_imag = self.sigmoid(imag_features)


        x_freq_real = x_real * freq_attention_real
        x_freq_imag = x_imag * freq_attention_imag


        x_freq = torch.complex(x_freq_real, x_freq_imag)
        x = torch.fft.ifftn(x_freq, dim=2).real

        return x


# class FrequencyAttention(nn.Module):
#     def __init__(self, in_channels, reduction=16):
#         super(FrequencyAttention, self).__init__()
#         self.conv1 = nn.Conv3d(in_channels, in_channels // reduction, kernel_size=1, bias=False)
#         self.relu = nn.ReLU()
#         self.conv2 = nn.Conv3d(in_channels // reduction, in_channels, kernel_size=1, bias=False)
#         self.sigmoid = nn.Sigmoid()
#
#     def forward(self, x):

#         x_freq = torch.fft.fftn(x, dim=2)  
#
#         
#         x_amp = torch.abs(x_freq)
#         x_phase = torch.angle(x_freq)
#
#         
#         amp_features = self.conv1(x_amp)
#         amp_features = self.relu(amp_features)
#         amp_features = self.conv2(amp_features)
#
#         
#         amp_attention = self.sigmoid(amp_features)
#
#        
#         x_amp = x_amp * amp_attention
#

#         x_freq = torch.polar(x_amp, x_phase)
#         x = torch.fft.ifftn(x_freq, dim=2).real
#
#         return x




class ChannelAttention(nn.Module):
    def __init__(self, channel, reduction=16):
        super().__init__()
        self.maxpool = nn.AdaptiveMaxPool3d(1)
        self.avgpool = nn.AdaptiveAvgPool3d(1)
        self.se = nn.Sequential(
            nn.Conv3d(channel, channel // reduction, 1, bias=False),
            nn.ReLU(),
            nn.Conv3d(channel // reduction, channel, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        max_result = self.maxpool(x)
        avg_result = self.avgpool(x)
        max_out = self.se(max_result)
        avg_out = self.se(avg_result)
        output = self.sigmoid(max_out + avg_out)
        return output


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv = nn.Conv3d(2, 1, kernel_size=kernel_size, padding=kernel_size // 2)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        max_result, _ = torch.max(x, dim=1, keepdim=True)
        avg_result = torch.mean(x, dim=1, keepdim=True)
        result = torch.cat([max_result, avg_result], 1)
        output = self.conv(result)
        output = self.sigmoid(output)
        return output



class MTFBlock(nn.Module):
    def __init__(self, order=['T', 'F', 'C', 'S'], channel=512, reduction=16, kernel_size=49):
        super().__init__()
        self.order = order
        self.attn_dict = {
            'T': MultiScaleTemporalAttention(in_channels=96, reduction=reduction),
            'F': FrequencyAttention(in_channels=96, reduction=reduction),
            'C': ChannelAttention(channel=channel, reduction=reduction),
            'S': SpatialAttention(kernel_size=kernel_size)
        }

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm3d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, x):
        out = x
        for attn_type in self.order:
            out = out * self.attn_dict[attn_type](out)
        return out + x



class Swin_MTFNet(nn.Module):
    """ Swin Transformer backbone.
        A PyTorch impl of : `Swin Transformer: Hierarchical Vision Transformer using Shifted Windows`  -
          https://arxiv.org/pdf/2103.14030
    Args:
        patch_size (int | tuple(int)): Patch size. Default: (4,4,4).
        in_chans (int): Number of input image channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        depths (tuple[int]): Depths of each Swin Transformer stage.
        num_heads (tuple[int]): Number of attention head of each stage.
        window_size (int): Window size. Default: 7.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 4.
        qkv_bias (bool): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float): Override default qk scale of head_dim ** -0.5 if set.
        drop_rate (float): Dropout rate.
        attn_drop_rate (float): Attention dropout rate. Default: 0.
        drop_path_rate (float): Stochastic depth rate. Default: 0.2.
        norm_layer: Normalization layer. Default: nn.LayerNorm.
        patch_norm (bool): If True, add normalization after patch embedding. Default: False.
        frozen_stages (int): Stages to be frozen (stop grad and set eval mode).
            -1 means not freezing any parameters.
    """

    def __init__(self,
                 pretrained=None,
                 pretrained2d=True,

                 patch_size=(4, 4, 4),     
                 in_chans=3,
                 embed_dim=96,
                 depths=[2, 2, 6, 2],   
                 num_heads=[3, 6, 12, 24],
                 window_size=(2, 7, 7),
                 mlp_ratio=4.,
                 qkv_bias=True,
                 qk_scale=None,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.2,
                 norm_layer=nn.LayerNorm,
                 patch_norm=False,
                 frozen_stages=-1,
                 use_checkpoint=False,
                 num_classes=1):
        super().__init__()

        self.pretrained = pretrained
        self.pretrained2d = pretrained2d
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.patch_norm = patch_norm
        self.frozen_stages = frozen_stages
        self.window_size = window_size
        self.patch_size = patch_size


        # split image into non-overlapping patches
        self.patch_embed = PatchEmbed3D(
            patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None)


        self.pos_drop = nn.Dropout(p=drop_rate)

        # stochastic depth
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule

        # build layers
        self.layers = nn.ModuleList()  
        for i_layer in range(self.num_layers):  
            layer = BasicLayer(
                dim=int(embed_dim * 2 ** i_layer),  
                depth=depths[i_layer],   # [2,2,6,2]
                num_heads=num_heads[i_layer],  
                window_size=window_size,  
                mlp_ratio=mlp_ratio,  
                qkv_bias=qkv_bias,   
                qk_scale=qk_scale,  
                drop=drop_rate,   
                attn_drop=attn_drop_rate,   
                drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],   
                norm_layer=norm_layer,  
                downsample=PatchMerging if i_layer < self.num_layers - 1 else None,   
                use_checkpoint=use_checkpoint)
            self.layers.append(layer)

        self.num_features = int(embed_dim * 2 ** (self.num_layers - 1))   

        # add a norm layer for each output
        self.norm = norm_layer(self.num_features)

        self._freeze_stages()


        self.head = nn.Linear(self.num_features, num_classes)
        self.sigmoid = nn.Sigmoid()




        self.mtf =MTFBlock(channel=96)

    def _freeze_stages(self):
        if self.frozen_stages >= 0:
            self.patch_embed.eval()
            for param in self.patch_embed.parameters():
                param.requires_grad = False

        if self.frozen_stages >= 1:
            self.pos_drop.eval()
            for i in range(0, self.frozen_stages):
                m = self.layers[i]
                m.eval()
                for param in m.parameters():
                    param.requires_grad = False

    def inflate_weights(self, logger):
        """Inflate the swin2d parameters to swin3d.
        The differences between swin3d and swin2d mainly lie in an extra
        axis. To utilize the pretrained parameters in 2d model,
        the weight of swin2d models should be inflated to fit in the shapes of
        the 3d counterpart.
        Args:
            logger (logging.Logger): The logger used to print
                debugging infomation.
        """
        checkpoint = torch.load(self.pretrained, map_location='cpu')
        state_dict = checkpoint['model']

        # delete relative_position_index since we always re-init it
        relative_position_index_keys = [k for k in state_dict.keys() if "relative_position_index" in k]
        for k in relative_position_index_keys:
            del state_dict[k]

        # delete attn_mask since we always re-init it
        attn_mask_keys = [k for k in state_dict.keys() if "attn_mask" in k]
        for k in attn_mask_keys:
            del state_dict[k]

        state_dict['patch_embed.proj.weight'] = state_dict['patch_embed.proj.weight'].unsqueeze(2).repeat(1, 1,
                                                                                                          self.patch_size[
                                                                                                              0], 1,
                                                                                                          1) / \
                                                self.patch_size[0]

        # bicubic interpolate relative_position_bias_table if not match
        relative_position_bias_table_keys = [k for k in state_dict.keys() if "relative_position_bias_table" in k]
        for k in relative_position_bias_table_keys:
            relative_position_bias_table_pretrained = state_dict[k]
            relative_position_bias_table_current = self.state_dict()[k]
            L1, nH1 = relative_position_bias_table_pretrained.size()
            L2, nH2 = relative_position_bias_table_current.size()
            L2 = (2 * self.window_size[1] - 1) * (2 * self.window_size[2] - 1)
            wd = self.window_size[0]
            if nH1 != nH2:
                logger.warning(f"Error in loading {k}, passing")
            else:
                if L1 != L2:
                    S1 = int(L1 ** 0.5)
                    relative_position_bias_table_pretrained_resized = torch.nn.functional.interpolate(
                        relative_position_bias_table_pretrained.permute(1, 0).view(1, nH1, S1, S1),
                        size=(2 * self.window_size[1] - 1, 2 * self.window_size[2] - 1),
                        mode='bicubic')
                    relative_position_bias_table_pretrained = relative_position_bias_table_pretrained_resized.view(nH2,
                                                                                                                   L2).permute(
                        1, 0)
            state_dict[k] = relative_position_bias_table_pretrained.repeat(2 * wd - 1, 1)

        msg = self.load_state_dict(state_dict, strict=False)
        logger.info(msg)
        logger.info(f"=> loaded successfully '{self.pretrained}'")
        del checkpoint
        torch.cuda.empty_cache()

    def init_weights(self, pretrained=None):
        """Initialize the weights in backbone.
        Args:
            pretrained (str, optional): Path to pre-trained weights.
                Defaults to None.
        """

        def _init_weights(m):
            if isinstance(m, nn.Linear):
                trunc_normal_(m.weight, std=.02)
                if isinstance(m, nn.Linear) and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)

        if pretrained:
            self.pretrained = pretrained
        if isinstance(self.pretrained, str):
            self.apply(_init_weights)
            logger = get_root_logger()
            logger.info(f'load model from: {self.pretrained}')

            if self.pretrained2d:
                # Inflate 2D model into 3D model.
                self.inflate_weights(logger)
            else:
                # Directly load 3D model.
                load_checkpoint(self, self.pretrained, strict=False, logger=logger)
        elif self.pretrained is None:
            self.apply(_init_weights)
        else:
            raise TypeError('pretrained must be a str or None')

    def forward(self, x):
        """Forward function."""
        # x = self.aft(x)
        # print(x.shape)  
        x = self.patch_embed(x)
        # print('x1', x.shape) 
        x = self.mtf(x)

        x = self.pos_drop(x)
        # print('x2', x.shape) 
        # x = self.mtf(x)

        for layer in self.layers:
            x = layer(x.contiguous())

        x = rearrange(x, 'n c d h w -> n d h w c')
        x = self.norm(x)
        x = rearrange(x, 'n d h w c -> n c d h w')
        # print(x.shape)
        x = x.mean(dim=[2, 3, 4])
        x = self.head(x)
        x = self.sigmoid(x)
        # print('x=', x)


        return x

    def train(self, mode=True):
        """Convert the model into training mode while keep layers freezed."""
        super(Swin_MTFNet, self).train(mode)
        self._freeze_stages()



def train_model(model, optimizer, scheduler, train_loader, valid_loader, dataset_sizes, path, num_epochs=150):
    since = time.time()
    best_acc = 0
    Loss_list = []
    Acc_list = []
    epoch_metrics_train = []
    epoch_metrics_val = []
    val_Loss_list = []
    val_Acc_list = []


    for epoch in range(num_epochs):

        # train
        Pred = []
        Label = []

        # val
        Pred_val = []
        Label_val = []


        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)
        for phase in ['train', 'val']:
            if phase == 'train':
                scheduler.step()
                model.train()
            else:
                model.eval()
            running_loss = 0.0
            running_corrects = 0

            if phase == 'train':
                for inputs, labels in train_loader:
                    inputs = inputs.to(device)
                    # labels = torch.unsqueeze(labels,1).type(torch.float)
                    labels = labels.to(device)
                    optimizer.zero_grad()
                    with torch.set_grad_enabled(phase == 'train'):
                        outputs = model(inputs)
                        loss = F.binary_cross_entropy(outputs, labels)

                        if phase == 'train':
                            loss.backward()
                            optimizer.step()
                    running_loss += loss.item() * inputs.size(0)


                    preds = torch.where(outputs > 0.5, torch.tensor(1).to('cuda:0'), torch.tensor(0).to('cuda:0'))

                    running_corrects += torch.sum(preds == labels.data)

                    preds = preds.cpu().numpy().tolist()  
                    labels = labels.cpu().numpy().tolist()

                    Pred.append(preds)
                    Label.append(labels)
                    # print('outputs.shape=', outputs.shape)
                    # print('outputs=', outputs)
                    # print('preds.shape=', preds)
                    # print('label=', labels.data)

                epoch_loss = running_loss / dataset_sizes[phase]
                # print(running_corrects)
                # print(dataset_sizes[phase])
                epoch_Acc = running_corrects / dataset_sizes[phase]

                Out_Pred = sum(Pred, [])
                Out_Label = sum(Label, [])


                epoch_precision = precision_score(Out_Label, Out_Pred)
                epoch_recall = recall_score(Out_Label, Out_Pred)
                epoch_f1 = f1_score(Out_Label, Out_Pred)
                epoch_auc = roc_auc_score(Out_Label, Out_Pred)
                tn, fp, fn, tp = confusion_matrix(Out_Label, Out_Pred).ravel()
                epoch_specificity = tn / (tn + fp)

                epoch_metrics_train.append({
                'phase': phase,
                'epoch': epoch + 1,
                'loss': epoch_loss,
                'accuracy': epoch_Acc.item(),
                'precision': epoch_precision,
                'recall': epoch_recall,
                'f1': epoch_f1,
                'auc': epoch_auc,
                'specificity': epoch_specificity
                })


                Loss_list.append(epoch_loss)
                Acc_list.append(100*epoch_Acc)
                # print(Loss_list)
                # print(Acc_list)

                # with open(r'/yourpath', "a", newline='') as f:  
                #     writer = csv.writer(f)
                #     for row in rows:
                #         writer.writerow(row)


                print('{} Loss: {:.4f}  Acc:{:.4f}  Precision:{:.4f}  Recall: {}  F1_score: {}， AUC: {}, Specificity: {}'.format(phase, epoch_loss, epoch_Acc, epoch_precision, epoch_recall, epoch_f1, epoch_auc, epoch_specificity))


                # torch.save(model.state_dict(), "/yourpath" % epoch)


            if phase == 'val':
                for inputs, labels in valid_loader:
                    inputs = inputs.to(device)
                    # labels = torch.unsqueeze(labels,1).type(torch.float)
                    labels = labels.to(device)
                    optimizer.zero_grad()
                    with torch.set_grad_enabled(phase == 'val'):
                        outputs = model(inputs)
                        loss = F.binary_cross_entropy(outputs, labels)
                        if phase == 'val':
                            loss.backward()
                            optimizer.step()
                    running_loss += loss.item() * inputs.size(0)
                    # _, preds = torch.max(outputs, 1)
                    preds = torch.where(outputs > 0.5, torch.tensor(1).to('cuda:0'), torch.tensor(0).to('cuda:0'))
                    running_corrects += torch.sum(preds == labels.data)


                    preds = preds.cpu().numpy().tolist()  
                    labels = labels.cpu().numpy().tolist()
                    # print(preds)
                    # print('label = ', labels.data)



                    Pred_val.append(preds)
                    Label_val.append(labels)

                epoch_loss = running_loss / dataset_sizes[phase]
                # print(running_corrects)
                # print(dataset_sizes[phase])
                epoch_Acc = running_corrects / dataset_sizes[phase]

                Out_Pred_val = sum(Pred_val, [])
                Out_Label_val = sum(Label_val, [])

                epoch_precision = precision_score(Out_Label_val, Out_Pred_val)
                epoch_recall = recall_score(Out_Label_val, Out_Pred_val)
                epoch_f1 = f1_score(Out_Label_val, Out_Pred_val)
                epoch_auc = roc_auc_score(Out_Label_val, Out_Pred_val)
                tn, fp, fn, tp = confusion_matrix(Out_Label_val, Out_Pred_val).ravel()
                epoch_specificity = tn / (tn + fp)

                epoch_metrics_val.append({
                'phase': phase,
                'epoch': epoch + 1,
                'loss': epoch_loss,
                'accuracy': epoch_Acc.item(),
                'precision': epoch_precision,
                'recall': epoch_recall,
                'f1': epoch_f1,
                'auc': epoch_auc,
                'specificity': epoch_specificity
                })


                # if epoch_Acc > best_acc:
                #     best_acc = epoch_Acc
                #     print("save model")
                #     torch.save(model.state_dict(), "/yourpath" + path + "/checkpoint-%i.pth" % epoch)


                val_Loss_list.append(epoch_loss)
                val_Acc_list.append(100 * epoch_Acc)
                # val_Accs.append(epoch_Acc)
                print('{} Loss: {:.4f}  Acc:{:.4f}  Precision:{:.4f}  Recall: {}  F1_score: {}， AUC: {}, Specificity: {}'.format(phase, epoch_loss, epoch_Acc, epoch_precision, epoch_recall, epoch_f1, epoch_auc, epoch_specificity))



    save_metrics_to_csv('/data/yourpath', epoch_metrics_train)
    save_metrics_to_csv('/data/yourpath', epoch_metrics_val)




    # for i in range(0, len(Loss_list)):
    #     Loss_list[i] = float(Loss_list[i])
    #     Acc_list[i] = float(Acc_list[i])
    #
    # for i in range(0, len(val_Loss_list)):
    #     val_Loss_list[i] = float(val_Loss_list[i])
    #     val_Acc_list[i] = float(val_Acc_list[i])
    #

    # x1 = range(0, 20)
    # x2 = range(0, 20)
    # y1 = Acc_list
    # y2 = Loss_list
    #

    # # plt.rcParams['font.sans-serif'] = ['SimHei']
    # # plt.rcParams['axes.unicode_minus'] = False
    #

    # plt.subplot(2, 1, 1)
    # plt.plot(x1, y1, 'o-')
    # plt.title('model accuracy')
    # plt.ylabel('accuracy unit:%')
    # my_yTicks1 = np.arange(50, 80, 5)
    # plt.yticks(my_yTicks1)

    # plt.subplot(2, 1, 2)
    # plt.plot(x2, y2, '.-')
    # plt.xlabel('model loss')
    # plt.ylabel('loss')
    # my_yTicks2 = np.arange(0, 2, 0.5)
    # plt.yticks(my_yTicks2)
    #
    # plt.savefig("train_accuracy_loss.jpg")
    #

    # x3 = range(0, 20)
    # x4 = range(0, 20)
    # print('x3:', x3)
    # y3 = val_Acc_list
    # y4 = val_Loss_list
    #
    # # plt.rcParams['font.sans-serif'] = ['SimHei']
    # # plt.rcParams['axes.unicode_minus'] = False
    #

    # plt.subplot(2, 1, 1)
    # plt.plot(x3, y3, 'o-')
    # plt.title('model accuracy')
    # plt.ylabel('accuracy unit:%')
    # my_yTicks1 = np.arange(50, 80, 5)
    # plt.yticks(my_yTicks1)

    # plt.subplot(2, 1, 2)
    # plt.plot(x4, y4, '.-')
    # plt.xlabel('model loss')
    # plt.ylabel('loss')
    # my_yTicks2 = np.arange(0, 2, 0.5)
    # plt.yticks(my_yTicks2)
    #
    # plt.savefig("val_accuracy_loss.jpg")

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    return model

# Acc = test(swin_3D)
# model_metrics(swin_3D, train_loader, valid_loader)


def save_metrics_to_csv(filename, metrics):

    fieldnames = ['phase', 'epoch', 'loss', 'accuracy', 'precision', 'recall', 'f1', 'auc', 'specificity']


    file_exists = os.path.isfile(filename)

    with open(filename, mode='a', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)

        if not file_exists:
            writer.writeheader()


        for metric in metrics:
            writer.writerow(metric)

