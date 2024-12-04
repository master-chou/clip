from collections import OrderedDict
from typing import Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
import loralib as lora
import math
import collections
import torch.nn.init as init
import spconv.pytorch as spconv

class CPEconv(nn.Module):
    def __init__(self, in_channels, spatial_shape, kernel_size=(3, 3, 3), padding=(1, 1, 1)):
        super(CPEconv, self).__init__()
        self.in_channels = in_channels
        self.spatial_shape = 6
        self.conv3d = nn.Conv3d(in_channels, in_channels, kernel_size=kernel_size, padding=padding,groups=in_channels)
        nn.init.zeros_(self.conv3d.weight)
        if self.conv3d.bias is not None:
            nn.init.zeros_(self.conv3d.bias)
        
        self.register_buffer('target_tensor_template', torch.zeros(1, in_channels, self.spatial_shape, 1, 1))

    def generate_3d_coords_from_depth(self, depth_maps):
        # 假设 depth_maps 形状为 (B, H, W)
        B, H, W = depth_maps.shape
        z_min = depth_maps.min(dim=-1, keepdim=True)[0].min(dim=-2, keepdim=True)[0]  # (B, 1, 1)
        z_max = depth_maps.max(dim=-1, keepdim=True)[0].max(dim=-2, keepdim=True)[0]  # (B, 1, 1)
        z = (depth_maps - z_min) / (z_max - z_min + 1e-8)
        # z = depth_maps  # z 坐标为深度值，形状为 (B, H, W)

        return z

    def forward(self, features, depth):
        #features [197,256,768] depth [256,14,14]
        B,h,w=depth.shape
        _,_,C=features.shape
        D = self.spatial_shape
        features = features[1:,:,:]
        features = features.permute(1,0,2)
        coord=self.generate_3d_coords_from_depth(depth)
        bnd=self.spatial_shape - 1
        coord = (coord *bnd).to(torch.int64)
        coord = (
            coord.clamp(0, bnd)  # clamp into bnd
           )
        target_tensor = self.target_tensor_template.expand(B, C, D, h, w).clone()
        # target_tensor = torch.zeros(B, C, D, h, w).to(device=features.device)
        # return 0

        coord = coord.unsqueeze(1).expand(-1, C, -1, -1)  # [B, C, H, W]
        # reshape features 以便与 coord 进行操作
        features = features.view(B, h, w, C)  # [B, H, W, C]
        features = features.permute(0, 3, 1, 2)  # [B, C, H, W]
        features = features.unsqueeze(2).to(dtype=target_tensor.dtype)
        coord = coord.unsqueeze(2)
        # import pdb;pdb.set_trace()

        # scatter features into target_tensor
        target_tensor = target_tensor.scatter_(2, coord, features)
        # 2. 使用 b 的值作为下标，将 features 的值复制到目标张量的相应位置
        # 3. 使用 for 循环将 features 的值复制到目标张量
        # for i in range(B):
        #     for j in range(h):
        #         for k in range(w):
        #             # 获取在 features 中的索引
        #             index = coord[i, j, k]  # 从 b 中获取索引
        #             target_tensor[i, :,index, j, k] = features[i, j * 14 + k, :]  # 复制对应的 features 值
        output = self.conv3d(target_tensor).mean(dim=2) #(B,768,14,14)
        output = output.reshape(-1,output.size(0),output.size(1))
        cls_feat = torch.zeros(1,output.size(-2), output.size(-1)).to(device=output.device,dtype=output.dtype)
        out_feat = torch.cat([cls_feat,output],dim=0)

        return out_feat
class RPE(torch.nn.Module):
    def __init__(self, patch_num, num_heads):
        super(RPE, self).__init__()
        self.num_heads = num_heads
        self.pos_bnd = patch_num
        self.rpe_num = 2 * self.pos_bnd + 1
        self.rpe_table = torch.nn.Parameter(torch.zeros(3 * self.rpe_num, num_heads))
        # torch.nn.init.trunc_normal_(self.rpe_table, std=0.02)

    def generate_3d_coords_from_depth(self,depth_maps):
        # 假设 depth_maps 形状为 (B, H, W)
        B, H, W = depth_maps.shape

        # 生成网格 i, j，形状为 (H, W)
        i, j = torch.meshgrid(torch.arange(H, device=depth_maps.device), torch.arange(W, device=depth_maps.device), indexing='ij')

        # 归一化 x 和 y 坐标
        x = j.float() / (W - 1)  # (H, W)
        y = i.float() / (H - 1)  # (H, W)

        # 将 x 和 y 扩展到 (B, H, W) 以匹配 depth_maps
        x = x.unsqueeze(0).expand(B, -1, -1)  # (B, H, W)
        y = y.unsqueeze(0).expand(B, -1, -1)  # (B, H, W)
        
        z_min = depth_maps.min(dim=-1, keepdim=True)[0].min(dim=-2, keepdim=True)[0]  # (B, 1, 1)
        z_max = depth_maps.max(dim=-1, keepdim=True)[0].max(dim=-2, keepdim=True)[0]  # (B, 1, 1)
        z = (depth_maps - z_min) / (z_max - z_min + 1e-8)
        # z = depth_maps  # z 坐标为深度值，形状为 (B, H, W)

        # 组合成 (B, H, W, 3) 的三维坐标
        coords = torch.stack([x, y, z], dim=-1)  # (B, H, W, 3)

        return coords


    def compute_relative_positions(self,absolute_coords):
        """
        计算相对位置编码
        参数:
        absolute_coords: 形状为 (N, 3) 的绝对三维坐标张量
        返回:
        相对位置编码，形状为 (N, N, 3)
        """
        # 确保输入是一个张量
        if not isinstance(absolute_coords, torch.Tensor):
            raise ValueError("Input must be a PyTorch tensor.")
        N = absolute_coords.shape[1]
        relative_positions = absolute_coords.unsqueeze(2) - absolute_coords.unsqueeze(1)

        return relative_positions


    def forward(self,depth):
        # B,K,K,3
        # import pdb;pdb.set_trace()

        depth=self.generate_3d_coords_from_depth(depth).squeeze(0)
        depth=depth.reshape(depth.size(0),-1,depth.size(-1))
        # zeros_tensor = torch.zeros(depth.size(0), 1, depth.size(-1))
        # depth = torch.cat((zeros_tensor,depth), dim=1)
        coord=self.compute_relative_positions(depth)
        # 将 coord 从 [0, 1] 范围转换为 [0, width] 或 [0, height]
        # coord = coord.reshape(coord.size(0),-1,coord.size(-1))
        # import pdb;pdb.set_trace()
        coord = (coord * torch.tensor([self.pos_bnd, self.pos_bnd, self.pos_bnd], device=coord.device)).round().long()
        idx = (
            coord.clamp(-self.pos_bnd, self.pos_bnd)  # clamp into bnd
            + self.pos_bnd  # relative position to positive index
            + torch.arange(3, device=coord.device) * self.rpe_num  # x, y, z stride
        )
        out = self.rpe_table.index_select(0, idx.reshape(-1))
        # out = out.reshape(coord.size(0) ,coord.size(1) ,coord.size(2) , -1)
        out = out.view(idx.shape + (-1,)).sum(3)

        out = out.permute(0, 3, 1, 2)  # (N, K, K, H) -> (N, H, K, K)
        # out_new=torch.zeros(out.size(0),out.size(1),out.size(2)+1,out.size(3)+1)
        # out_new[:, :, 1:, 1:] = out
        return out

class PositionEmbeddingCoordsSine(nn.Module):
    def __init__(
        self,
        temperature=10000,
        normalize=False,
        scale=None,
        pos_type="fourier",
        d_pos=None,
        d_in=3,
        gauss_scale=1.0,
    ):
        super().__init__()
        self.temperature = temperature
        self.normalize = normalize
        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passed")
        if scale is None:
            scale = 2 * math.pi
        assert pos_type in ["sine", "fourier"]
        self.pos_type = pos_type
        self.scale = scale
        self.ln = LayerNorm(768)
        if pos_type == "fourier":
            assert d_pos is not None
            assert d_pos % 2 == 0
            # define a gaussian matrix input_ch -> output_ch
            B = torch.empty((d_in, d_pos // 2)).normal_()
            B *= gauss_scale
            # self.gauss_B = nn.Parameter(B)  
            self.register_buffer("gauss_B", B)
            self.d_pos = d_pos
        self.trans3d=nn.Conv1d(in_channels=3, out_channels=768, kernel_size=1)
        init.zeros_(self.trans3d.weight)
        if self.trans3d.bias is not None:
            init.zeros_(self.trans3d.bias)
    def get_sine_embeddings(self, xyz, num_channels, input_range):
        ncoords = xyz.shape[1]
        ndim = num_channels // xyz.shape[2]
        if ndim % 2 != 0:
            ndim -= 1
        # automatically handle remainder by assiging it to the first dim
        rems = num_channels - (ndim * xyz.shape[2])

        assert (
            ndim % 2 == 0
        ), f"Cannot handle odd sized ndim={ndim} where num_channels={num_channels} and xyz={xyz.shape}"

        final_embeds = []
        prev_dim = 0

        for d in range(xyz.shape[2]):
            cdim = ndim
            if rems > 0:
                # add remainder in increments of two to maintain even size
                cdim += 2
                rems -= 2

            if cdim != prev_dim:
                dim_t = torch.arange(cdim, dtype=torch.float32, device=xyz.device)
                dim_t = self.temperature ** (2 * (dim_t // 2) / cdim)

            # create batch x cdim x nccords embedding
            raw_pos = xyz[:, :, d]
            if self.scale:
                raw_pos *= self.scale
            pos = raw_pos[:, :, None] / dim_t
            pos = torch.stack(
                (pos[:, :, 0::2].sin(), pos[:, :, 1::2].cos()), dim=3
            ).flatten(2)
            final_embeds.append(pos)
            prev_dim = cdim

        final_embeds = torch.cat(final_embeds, dim=2)
        return final_embeds
    def get_fourier_embeddings(self, xyz, num_channels=None, input_range=None): 
        if num_channels is None:
            num_channels = self.gauss_B.shape[1] * 2
        bsize, npoints = xyz.shape[0], xyz.shape[1]
        assert num_channels > 0 and num_channels % 2 == 0
        d_in, max_d_out = self.gauss_B.shape[0], self.gauss_B.shape[1]
        d_out = num_channels // 2
        # assert d_out <= max_d_out
        assert d_in == xyz.shape[-1]

        # clone coords so that shift/scale operations do not affect original tensor
        # import pdb;pdb.set_trace()
        ncoords = xyz.shape[1]
        if self.normalize:
            # xyz = shift_scale_points(xyz, src_range=input_range)
            pass

        xyz *= 2 * torch.pi
        xyz_proj = torch.mm(xyz.view(-1, d_in), self.gauss_B[:, :d_out]).view(
            bsize, npoints, d_out
        )
        final_embeds = [xyz_proj.sin(), xyz_proj.cos()]

        # return batch x d_pos x npoints embedding
        final_embeds = torch.cat(final_embeds, dim=2)
        # import pdb;pdb.set_trace()
        # final_embeds = self.ln(final_embeds)
        final_embeds = F.normalize(final_embeds, p=2, dim=2)

        # If necessary, you can permute it back to [batch, 196, 768]
        return final_embeds

    def forward(self, depth_map, num_channels=None, input_range=None):
        cam_coords_tensor = self.generate_3d_coords_from_depth(depth_map)  # (B, H, W, 3)
        # cam_coords_tensor = torch.tensor(cam_coords, dtype=torch.float16)  # (B, H, W, 3)
        cam_coords_tensor = cam_coords_tensor.view(cam_coords_tensor.size(0), -1, 3)  # (B, H*W, 3)
        xyz=cam_coords_tensor
        # import pdb;pdb.set_trace()
        assert xyz.ndim == 3
        # xyz is batch x npoints x 3
        if self.pos_type == "sine":
            with torch.no_grad():
                return self.get_sine_embeddings(xyz, 768, input_range)
        elif self.pos_type == "fourier":
            with torch.no_grad():
                return self.get_fourier_embeddings(xyz, num_channels, input_range)
        else:
            raise ValueError(f"Unknown {self.pos_type}")

    def positiontrans3d(self,depth_map):
        cam_coords_tensor = self.generate_3d_coords_from_depth(depth_map)  # (B, H, W, 3)
        # cam_coords_tensor = torch.tensor(cam_coords, dtype=torch.float16)  # (B, H, W, 3)
        cam_coords_tensor = cam_coords_tensor.view(cam_coords_tensor.size(0), -1, 3)  # (B, H*W, 3)
        x=cam_coords_tensor
        x = x.permute(0, 2, 1)  # (B, H*W, 3) -> (B, 3, H*W)
        x = self.trans3d(x)      # 1D卷积映射 (B, 768, H*W)
        x = x.permute(0, 2, 1)   # 转换回 (B, H*W, 768)
        return x
    def generate_3d_coords_from_depth(self, depth_maps):
        # 假设 depth_maps 形状为 (B, H, W)
        B, H, W = depth_maps.shape

        # 生成网格 i, j，形状为 (H, W)
        i, j = torch.meshgrid(torch.arange(H, device=depth_maps.device), torch.arange(W, device=depth_maps.device), indexing='ij')

        # 归一化 x 和 y 坐标
        x = j.float() / (W - 1)  # (H, W)
        y = i.float() / (H - 1)  # (H, W)

        # 将 x 和 y 扩展到 (B, H, W) 以匹配 depth_maps
        x = x.unsqueeze(0).expand(B, -1, -1)  # (B, H, W)
        y = y.unsqueeze(0).expand(B, -1, -1)  # (B, H, W)
        
        z = depth_maps  # z 坐标为深度值，形状为 (B, H, W)

        # 组合成 (B, H, W, 3) 的三维坐标
        coords = torch.stack([x, y, z], dim=-1)  # (B, H, W, 3)

        return coords


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1):
        super().__init__()

        # all conv layers have stride 1. an avgpool is performed after the second convolution when stride > 1
        self.conv1 = nn.Conv2d(inplanes, planes, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu1 = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(planes, planes, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.relu2 = nn.ReLU(inplace=True)

        self.avgpool = nn.AvgPool2d(stride) if stride > 1 else nn.Identity()

        self.conv3 = nn.Conv2d(planes, planes * self.expansion, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu3 = nn.ReLU(inplace=True)

        self.downsample = None
        self.stride = stride

        if stride > 1 or inplanes != planes * Bottleneck.expansion:
            # downsampling layer is prepended with an avgpool, and the subsequent convolution has stride 1
            self.downsample = nn.Sequential(OrderedDict([
                ("-1", nn.AvgPool2d(stride)),
                ("0", nn.Conv2d(inplanes, planes * self.expansion, 1, stride=1, bias=False)),
                ("1", nn.BatchNorm2d(planes * self.expansion))
            ]))

    def forward(self, x: torch.Tensor):
        identity = x

        out = self.relu1(self.bn1(self.conv1(x)))
        out = self.relu2(self.bn2(self.conv2(out)))
        out = self.avgpool(out)
        out = self.bn3(self.conv3(out))

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu3(out)
        return out


class AttentionPool2d(nn.Module):
    def __init__(self, spacial_dim: int, embed_dim: int, num_heads: int, output_dim: int = None):
        super().__init__()
        self.positional_embedding = nn.Parameter(torch.randn(spacial_dim ** 2 + 1, embed_dim) / embed_dim ** 0.5)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.c_proj = nn.Linear(embed_dim, output_dim or embed_dim)
        self.num_heads = num_heads

    def forward(self, x):
        x = x.flatten(start_dim=2).permute(2, 0, 1)  # NCHW -> (HW)NC
        x = torch.cat([x.mean(dim=0, keepdim=True), x], dim=0)  # (HW+1)NC
        x = x + self.positional_embedding[:, None, :].to(x.dtype)  # (HW+1)NC
        x, _ = F.multi_head_attention_forward(
            query=x[:1], key=x, value=x,
            embed_dim_to_check=x.shape[-1],
            num_heads=self.num_heads,
            q_proj_weight=self.q_proj.weight,
            k_proj_weight=self.k_proj.weight,
            v_proj_weight=self.v_proj.weight,
            in_proj_weight=None,
            in_proj_bias=torch.cat([self.q_proj.bias, self.k_proj.bias, self.v_proj.bias]),
            bias_k=None,
            bias_v=None,
            add_zero_attn=False,
            dropout_p=0,
            out_proj_weight=self.c_proj.weight,
            out_proj_bias=self.c_proj.bias,
            use_separate_proj_weight=True,
            training=self.training,
            need_weights=False
        )
        return x.squeeze(0)


class ModifiedResNet(nn.Module):
    """
    A ResNet class that is similar to torchvision's but contains the following changes:
    - There are now 3 "stem" convolutions as opposed to 1, with an average pool instead of a max pool.
    - Performs anti-aliasing strided convolutions, where an avgpool is prepended to convolutions with stride > 1
    - The final pooling layer is a QKV attention instead of an average pool
    """

    def __init__(self, layers, output_dim, heads, input_resolution=224, width=64):
        super().__init__()
        self.output_dim = output_dim
        self.input_resolution = input_resolution

        # the 3-layer stem
        self.conv1 = nn.Conv2d(3, width // 2, kernel_size=3, stride=2, padding=1, bias=False)
        self.conv1_alpha = nn.Conv2d(in_channels=1, out_channels=width // 2, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(width // 2)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(width // 2, width // 2, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(width // 2)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv3 = nn.Conv2d(width // 2, width, kernel_size=3, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(width)
        self.relu3 = nn.ReLU(inplace=True)
        self.avgpool = nn.AvgPool2d(2)

        # residual layers
        self._inplanes = width  # this is a *mutable* variable used during construction
        self.layer1 = self._make_layer(width, layers[0])
        self.layer2 = self._make_layer(width * 2, layers[1], stride=2)
        self.layer3 = self._make_layer(width * 4, layers[2], stride=2)
        self.layer4 = self._make_layer(width * 8, layers[3], stride=2)

        embed_dim = width * 32  # the ResNet feature dimension
        self.attnpool = AttentionPool2d(input_resolution // 32, embed_dim, heads, output_dim)

    def _make_layer(self, planes, blocks, stride=1):
        layers = [Bottleneck(self._inplanes, planes, stride)]

        self._inplanes = planes * Bottleneck.expansion
        for _ in range(1, blocks):
            layers.append(Bottleneck(self._inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x, alpha=None):
        def stem(x):
            x = self.relu1(self.bn1(self.conv1(x) + self.conv1_alpha(alpha)))
            x = self.relu2(self.bn2(self.conv2(x)))
            x = self.relu3(self.bn3(self.conv3(x)))
            x = self.avgpool(x)
            return x

        x = x.type(self.conv1.weight.dtype)
        x = stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.attnpool(x)

        return x


class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16."""

    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)


class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)

class Attention(nn.Module):
    def __init__(
            self,
            dim,
            num_heads=8,
            qkv_bias=True,
            scaled_cosine=False,
            scale_heads=False,
            logit_scale_max=math.log(1. / 0.01),
            attn_drop=0.,
            proj_drop=0.,
            lora_adapt=False, 
            rank=16,
            patch_num=16
    ):
        super().__init__()
        self.scaled_cosine = scaled_cosine
        self.scale_heads = scale_heads
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.logit_scale_max = logit_scale_max
        self.use_rel_pos = True  # 保存相对位置编码的使用状态
        self.rpe = RPE(patch_num=patch_num,num_heads=self.num_heads)
        self.rpe.requires_grad=True
        # import pdb;pdb.set_trace()
        # keeping in_proj in this form (instead of nn.Linear) to match weight scheme of original
        if lora_adapt:
            print("!!!!!!!!!!using lora for qkv projection!!!!!!!!!!")
            self.in_proj = lora.MergedLinear(dim, 3*dim, r=rank, enable_lora=[True, False, True])
        else:
            self.in_proj = nn.Linear(dim, dim * 3)
        # self.in_proj_weight = nn.Parameter(torch.randn((dim * 3, dim)) * self.scale)
        # if qkv_bias:
        #     self.in_proj_bias = nn.Parameter(torch.zeros(dim * 3))
        # else:
        #     self.in_proj_bias = None

        if self.scaled_cosine:
            self.logit_scale = nn.Parameter(torch.log(10 * torch.ones((num_heads, 1, 1))))
        else:
            self.logit_scale = None
        self.attn_drop = nn.Dropout(attn_drop)
        if self.scale_heads:
            self.head_scale = nn.Parameter(torch.ones((num_heads, 1, 1)))
        else:
            self.head_scale = None
        self.out_proj = nn.Linear(dim, dim) if not lora_adapt else lora.Linear(dim, dim, r=rank)
        self.out_drop = nn.Dropout(proj_drop)

    def forward(self, x, attn_mask = None,depth=None):
        L, N, C = x.shape
        q, k, v = self.in_proj(x).chunk(3, dim=-1)
        q = q.contiguous().view(L, N * self.num_heads, -1).transpose(0, 1)
        k = k.contiguous().view(L, N * self.num_heads, -1).transpose(0, 1)
        v = v.contiguous().view(L, N * self.num_heads, -1).transpose(0, 1)

        if self.logit_scale is not None:
            attn = torch.bmm(F.normalize(q, dim=-1), F.normalize(k, dim=-1).transpose(-1, -2))
            logit_scale = torch.clamp(self.logit_scale, max=self.logit_scale_max).exp()
            attn = attn.view(N, self.num_heads, L, L) * logit_scale
            attn = attn.view(-1, L, L)
        else:
            q = q * self.scale
            attn = torch.bmm(q, k.transpose(-2, -1))
        
        if depth is not None:
            depth=depth.squeeze(1)
            res= self.rpe(depth)
            res=res.reshape(-1,res.size(-2),res.size(-1))
            # import pdb;pdb.set_trace()
            attn[:,1:,1:]=attn[:,1:,1:]+res

        if attn_mask is not None:
            if attn_mask.dtype == torch.bool:
                new_attn_mask = torch.zeros_like(attn_mask, dtype=q.dtype)
                new_attn_mask.masked_fill_(attn_mask, float("-inf"))
                attn_mask = new_attn_mask
            attn += attn_mask

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = torch.bmm(attn, v)
        if self.head_scale is not None:
            x = x.view(N, self.num_heads, L, C) * self.head_scale
            x = x.view(-1, L, C)
        x = x.transpose(0, 1).reshape(L, N, C)
        x = self.out_proj(x)
        x = self.out_drop(x)
        return x, attn


class CustomResidualAttentionBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int, attn_mask: torch.Tensor = None, lora_adapt=False, rank=16,patch_num=16):
        super().__init__()
        
        self.attn = Attention(d_model, n_head, lora_adapt=lora_adapt, rank=rank,patch_num=patch_num)
        self.ln_1 = LayerNorm(d_model)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4) if not lora_adapt else lora.Linear(d_model, d_model*4, r=rank)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model) if not lora_adapt else lora.Linear(d_model*4, d_model, r=rank))
        ]))
        self.ln_2 = LayerNorm(d_model)
        self.ln_cpe = LayerNorm(d_model)
        self.attn_mask = attn_mask
        self.cpe=CPEconv(d_model,patch_num)


    def attention(self, x: torch.Tensor,depth=None):
        self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device) if self.attn_mask is not None else None
        return self.attn(x, attn_mask=self.attn_mask,depth=depth)


    def forward(self, x: torch.Tensor, return_attn=False,depth=None):
        # import pdb;pdb.set_trace()
        # x ([577, 50, 1024])
        # if None:
        shortcut=x
        # import pdb;pdb.set_trace()
        # shapes=x.shape
        # x=  x.reshape(-1,x.size(-1))
        # import pdb;pdb.set_trace()
        # cposi = self.cpe(x, depth).reshape(shapes)
        cposi = self.cpe(self.ln_cpe(x), depth)
        x =shortcut+cposi

        attn_out, attn = self.attention(self.ln_1(x),depth)
        x = x + attn_out
        x = x + self.mlp(self.ln_2(x))
        if return_attn:
            return x, attn
        else:
            return x

class ResidualAttentionBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int, attn_mask: torch.Tensor = None):
        super().__init__()
        
        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = LayerNorm(d_model)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model))
        ]))
        self.ln_2 = LayerNorm(d_model)
        self.attn_mask = attn_mask

    def attention(self, x: torch.Tensor):
        self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device) if self.attn_mask is not None else None
        return self.attn(x, x, x, attn_mask=self.attn_mask)[0]

    def forward(self, x: torch.Tensor):
        x = x + self.attention(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x

class Transformer(nn.Module):
    def __init__(self, width: int, layers: int, heads: int, attn_mask: torch.Tensor = None):
        super().__init__()
        self.width = width
        self.layers = layers
        self.resblocks = nn.Sequential(*[ResidualAttentionBlock(width, heads, attn_mask) for _ in range(layers)])

    def forward(self, x: torch.Tensor):
        return self.resblocks(x)

class CustomTransformer(nn.Module):
    def __init__(self, width: int, layers: int, heads: int, attn_mask: torch.Tensor = None, lora_adapt=False, rank=16,patch_num=16):
        super().__init__()
        self.width = width
        self.layers = layers
        self.resblocks = nn.Sequential(*[CustomResidualAttentionBlock(width, heads, attn_mask, lora_adapt=lora_adapt, rank=rank,patch_num=patch_num) for _ in range(layers)])

    def forward(self, x: torch.Tensor, return_attn=False,depth=None):
        # import pdb;pdb.set_trace()
        if return_attn:
            for i, block in enumerate(self.resblocks):
                if i == len(self.resblocks) - 1:
                    return block(x, return_attn=True,depth=depth)
                else:
                    x = block(x,depth=depth)
            assert False    
        for block in self.resblocks:
            # import pdb;pdb.set_trace()
            x = block(x, depth=depth)  # 将 depth 传递给每个模块
        return x
        # return self.resblocks(x)

# ////////////////////////////////////////////////////////////////////////////////////////////
class VisionTransformer(nn.Module):
    def __init__(self, input_resolution: int, patch_size: int, width: int, layers: int, heads: int, output_dim: int, lora_adapt=False, rank=16):
        super().__init__()
        self.input_resolution = input_resolution
        self.output_dim = output_dim
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=width, kernel_size=patch_size, stride=patch_size, bias=False)
        self.conv1_alpha = nn.Conv2d(in_channels=1, out_channels=width, kernel_size=patch_size, stride=patch_size, bias=False)
        nn.init.zeros_(self.conv1_alpha.weight)
        scale = width ** -0.5
        self.class_embedding = nn.Parameter(scale * torch.randn(width))
        self.positional_embedding = nn.Parameter(scale * torch.randn((input_resolution // patch_size) ** 2 + 1, width))
        # self.depth_positional_embedding = nn.Parameter(scale * torch.zeros((input_resolution // patch_size) ** 2, width))  # 用于alpha的深度编码
        # self.depth_positional_embedding = PositionEmbeddingCoordsSine(temperature=10000,
        #             normalize=True,
        #             scale=2 * torch.pi,
        #             pos_type="fourier",
        #             d_pos=768,  # 示例输出维度
        #             d_in=3,
        #             gauss_scale=1.0
        #         )
        # self.sine_positional_embedding = PositionEmbeddingCoordsSine(temperature=10000,
        #             normalize=True,
        #             scale=2 * torch.pi,
        #             pos_type="sine",
        #             d_pos=768,  # 示例输出维度
        #             d_in=3,
        #             gauss_scale=1.0
        #         )
        # self.large_positional_embedding = PositionEmbeddingCoordsSine(temperature=10000,
        #             normalize=True,
        #             scale=2 * torch.pi,
        #             pos_type="sine",
        #             d_pos=1024,  # 示例输出维度
        #             d_in=3,
        #             gauss_scale=1.0
        #         )
        # self.depth_mlp=nn.Linear(768,768)
        # nn.init.zeros_(self.depth_mlp.weight)
        # if self.depth_mlp.bias is not None:
        #     nn.init.zeros_(self.depth_mlp.bias)
        self.patch_size=patch_size

        self.ln_pre = LayerNorm(width)
        self.transformer = CustomTransformer(width, layers, heads, lora_adapt=lora_adapt, rank=rank,patch_num=input_resolution // patch_size)

        self.ln_post = LayerNorm(width)
        self.proj = nn.Parameter(scale * torch.randn(width, output_dim))

    def forward(self, x: torch.Tensor, alpha=None, return_attn=False,pos_embed=None):
        # import pdb;pdb.set_trace()
        x = self.conv1(x)  # shape = [*, width, grid, grid]
        # ASSUME alpha is always not None!
        # import pdb;pdb.set_trace()
        # if pos_embed == "nodepth":
        #     pass
        # else:
        #     x = x + self.conv1_alpha(alpha)
        # import pdb;pdb.set_trace()

        x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
        x = torch.cat([self.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device), x], dim=1)  # shape = [*, grid ** 2 + 1, width]
        # import pdb;pdb.set_trace()
        alpha_resized = F.adaptive_avg_pool2d(alpha, (self.input_resolution // self.patch_size, self.input_resolution // self.patch_size))
        # alpha_flattened = alpha_resized.flatten(start_dim=2).permute(0, 2, 1) 
        alpha_resized = alpha_resized.squeeze(1)
        # x[:, 1:] += self.depth_positional_embedding.to(x.dtype) * alpha_flattened
        # import pdb;pdb.set_trace()
        # if pos_embed == "fourier":
        #     depth_embedding = self.depth_positional_embedding(alpha_resized)
        #     x[:, 1:] +=self.depth_mlp(depth_embedding)
        # elif pos_embed == "sine":
        #     depth_embedding = self.sine_positional_embedding(alpha_resized)
        #     x[:, 1:] +=self.depth_mlp(depth_embedding)
        # elif pos_embed == "3d":
        #     depth_embedding = self.depth_positional_embedding.positiontrans3d(alpha_resized)
        #     x[:, 1:] +=self.depth_mlp(depth_embedding)
        
        x = x + self.positional_embedding.to(x.dtype)
        x = self.ln_pre(x)
        # import pdb;pdb.set_trace()
        x = x.permute(1, 0, 2)  # NLD -> LND
        if return_attn:
            x, attn_last = self.transformer(x, return_attn=True,depth=alpha_resized)
        else:
            x = self.transformer(x, return_attn=False,depth=alpha_resized)
        x = x.permute(1, 0, 2)  # LND -> NLD

        x = self.ln_post(x[:, 0, :])

        if self.proj is not None:
            x = x @ self.proj
        if return_attn:
            return x, attn_last
        else:
            return x
# /////////////////////////////////////////////////////////////////////////////////////////////////////

class CLIP(nn.Module):
    def __init__(self,
                 embed_dim: int,
                 # vision
                 image_resolution: int,
                 vision_layers: Union[Tuple[int, int, int, int], int],
                 vision_width: int,
                 vision_patch_size: int,
                 # text
                 context_length: int,
                 vocab_size: int,
                 transformer_width: int,
                 transformer_heads: int,
                 transformer_layers: int,
                 lora_adapt = False,
                 rank = 16,
                 ):
        super().__init__()

        self.context_length = context_length

        if isinstance(vision_layers, (tuple, list)):
            vision_heads = vision_width * 32 // 64
            self.visual = ModifiedResNet(
                layers=vision_layers,
                output_dim=embed_dim,
                heads=vision_heads,
                input_resolution=image_resolution,
                width=vision_width
            )
        else:
            vision_heads = vision_width // 64
            self.visual = VisionTransformer(
                input_resolution=image_resolution,
                patch_size=vision_patch_size,
                width=vision_width,
                layers=vision_layers,
                heads=vision_heads,
                output_dim=embed_dim,
                lora_adapt=lora_adapt,
                rank=rank
            )

        self.transformer = Transformer(
            width=transformer_width,
            layers=transformer_layers,
            heads=transformer_heads,
            attn_mask=self.build_attention_mask()
        )

        self.vocab_size = vocab_size
        self.token_embedding = nn.Embedding(vocab_size, transformer_width)
        self.positional_embedding = nn.Parameter(torch.empty(self.context_length, transformer_width))
        self.ln_final = LayerNorm(transformer_width)

        self.text_projection = nn.Parameter(torch.empty(transformer_width, embed_dim))
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

        self.initialize_parameters()

    def initialize_parameters(self):
        nn.init.normal_(self.token_embedding.weight, std=0.02)
        nn.init.normal_(self.positional_embedding, std=0.01)

        if isinstance(self.visual, ModifiedResNet):
            if self.visual.attnpool is not None:
                std = self.visual.attnpool.c_proj.in_features ** -0.5
                nn.init.normal_(self.visual.attnpool.q_proj.weight, std=std)
                nn.init.normal_(self.visual.attnpool.k_proj.weight, std=std)
                nn.init.normal_(self.visual.attnpool.v_proj.weight, std=std)
                nn.init.normal_(self.visual.attnpool.c_proj.weight, std=std)

            for resnet_block in [self.visual.layer1, self.visual.layer2, self.visual.layer3, self.visual.layer4]:
                for name, param in resnet_block.named_parameters():
                    if name.endswith("bn3.weight"):
                        nn.init.zeros_(param)

        proj_std = (self.transformer.width ** -0.5) * ((2 * self.transformer.layers) ** -0.5)
        attn_std = self.transformer.width ** -0.5
        fc_std = (2 * self.transformer.width) ** -0.5
        for block in self.transformer.resblocks:
            nn.init.normal_(block.attn.in_proj_weight, std=attn_std)
            nn.init.normal_(block.attn.out_proj.weight, std=proj_std)
            nn.init.normal_(block.mlp.c_fc.weight, std=fc_std)
            nn.init.normal_(block.mlp.c_proj.weight, std=proj_std)

        if self.text_projection is not None:
            nn.init.normal_(self.text_projection, std=self.transformer.width ** -0.5)

    def build_attention_mask(self):
        # lazily create causal attention mask, with full attention between the vision tokens
        # pytorch uses additive attention mask; fill with -inf
        mask = torch.empty(self.context_length, self.context_length)
        mask.fill_(float("-inf"))
        mask.triu_(1)  # zero out the lower diagonal
        return mask

    @property
    def dtype(self):
        if not hasattr(self.visual, "conv1"):
            return self.visual.module.conv1.weight.dtype
        return self.visual.conv1.weight.dtype

    def encode_image(self, image, alpha):
        assert alpha is not None
        return self.visual(image.type(self.dtype), alpha.type(self.dtype))

    def encode_text(self, text):
        x = self.token_embedding(text).type(self.dtype)  # [batch_size, n_ctx, d_model]

        x = x + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)] @ self.text_projection

        return x



    def forward(self, image, text, alpha):
        image_features = self.encode_image(image, alpha)
        text_features = self.encode_text(text)

        # normalized features
        image_features = image_features / image_features.norm(dim=1, keepdim=True)
        text_features = text_features / text_features.norm(dim=1, keepdim=True)

        # cosine similarity as logits
        logit_scale = self.logit_scale.exp()
        logits_per_image = logit_scale * image_features @ text_features.t()
        logits_per_text = logits_per_image.t()

        # shape = [global_batch_size, global_batch_size]
        return logits_per_image, logits_per_text


def convert_weights(model: nn.Module):
    """Convert applicable model parameters to fp16"""

    def _convert_weights_to_fp16(l):
        if isinstance(l, (nn.Conv1d, nn.Conv2d, nn.Linear)):
            l.weight.data = l.weight.data.half()
            if l.bias is not None:
                l.bias.data = l.bias.data.half()

        if isinstance(l, nn.MultiheadAttention):
            for attr in [*[f"{s}_proj_weight" for s in ["in", "q", "k", "v"]], "in_proj_bias", "bias_k", "bias_v"]:
                tensor = getattr(l, attr)
                if tensor is not None:
                    tensor.data = tensor.data.half()

        for name in ["text_projection", "proj"]:
            if hasattr(l, name):
                attr = getattr(l, name)
                if attr is not None:
                    attr.data = attr.data.half()

    model.apply(_convert_weights_to_fp16)


def build_model(state_dict: dict, lora_adapt=False, rank=16):
    vit = "visual.proj" in state_dict

    if vit:
        vision_width = state_dict["visual.conv1.weight"].shape[0]
        vision_layers = len([k for k in state_dict.keys() if k.startswith("visual.") and k.endswith(".attn.in_proj_weight")])
        vision_patch_size = state_dict["visual.conv1.weight"].shape[-1]
        grid_size = round((state_dict["visual.positional_embedding"].shape[0] - 1) ** 0.5)
        image_resolution = vision_patch_size * grid_size
    else:
        counts: list = [len(set(k.split(".")[2] for k in state_dict if k.startswith(f"visual.layer{b}"))) for b in [1, 2, 3, 4]]
        vision_layers = tuple(counts)
        vision_width = state_dict["visual.layer1.0.conv1.weight"].shape[0]
        output_width = round((state_dict["visual.attnpool.positional_embedding"].shape[0] - 1) ** 0.5)
        vision_patch_size = None
        assert output_width ** 2 + 1 == state_dict["visual.attnpool.positional_embedding"].shape[0]
        image_resolution = output_width * 32

    embed_dim = state_dict["text_projection"].shape[1]
    context_length = state_dict["positional_embedding"].shape[0]
    vocab_size = state_dict["token_embedding.weight"].shape[0]
    transformer_width = state_dict["ln_final.weight"].shape[0]
    transformer_heads = transformer_width // 64
    transformer_layers = len(set(k.split(".")[2] for k in state_dict if k.startswith("transformer.resblocks")))

    # always load lora version
    model = CLIP(
        embed_dim,
        image_resolution, vision_layers, vision_width, vision_patch_size,
        context_length, vocab_size, transformer_width, transformer_heads, transformer_layers,
        lora_adapt=lora_adapt, rank=rank,
    )

    for key in ["input_resolution", "context_length", "vocab_size"]:
        if key in state_dict:
            del state_dict[key]
    # para_wb to linear
    new_state_dict = collections.OrderedDict()
    for k, v in state_dict.items():
        if 'visual' in k:
            if 'in_proj_weight' in k:
                new_state_dict[k.replace('in_proj_weight', 'in_proj.weight')] = v
            elif 'in_proj_bias' in k:
                new_state_dict[k.replace('in_proj_bias', 'in_proj.bias')] = v
            else:
                new_state_dict[k] = v
        else:
            new_state_dict[k] = v
                
    state_dict = new_state_dict
    # add rgba_conv_weight
    if 'visual.conv1_alpha.weight' not in state_dict.keys(): # zero initialization on alpha channel
        rgb_weight = state_dict['visual.conv1.weight'].clone().detach()
        rgba_weigth = torch.zeros_like(rgb_weight)[:, 0:1, :, :]
        state_dict['visual.conv1_alpha.weight'] = rgba_weigth
    convert_weights(model)
    model.load_state_dict(state_dict, strict=False)
    return model.eval()
