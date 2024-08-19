import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
import logging
from nnunet.network_architecture.neural_network import SegmentationNetwork
import diynnu.network_architecture.MiT_encoder_eccv as MiT_encoder
from diynnu.network_architecture.utils import trunc_normal_
from scipy import ndimage

logger = logging.getLogger(__name__)


class Conv3d_wd(nn.Conv3d):

    def __init__(self, in_channels, out_channels, kernel_size, stride=(1,1,1), padding=(0,0,0), dilation=(1,1,1), groups=1, bias=False):
        super(Conv3d_wd, self).__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)

    def forward(self, x):
        w = self.weight
        v, m = torch.var_mean(w, dim=[1, 2, 3, 4], keepdim=True, unbiased=False)
        w = (w - m) / torch.sqrt(v + 1e-10)
        return F.conv3d(x, w, self.bias, self.stride, self.padding, self.dilation, self.groups)

def conv3x3x3(in_planes, out_planes, kernel_size, stride=(1, 1, 1), padding=(0, 0, 0), dilation=(1, 1, 1), bias=False, weight_std=False):
    "3x3x3 convolution with padding"
    if weight_std:
        return Conv3d_wd(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, bias=bias)
    else:
        return nn.Conv3d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, bias=bias)


def Norm_layer(norm_cfg, inplanes):
    if norm_cfg == 'BN':
        out = nn.BatchNorm3d(inplanes)
    elif norm_cfg == 'SyncBN':
        out = nn.SyncBatchNorm(inplanes)
    elif norm_cfg == 'GN':
        out = nn.GroupNorm(16, inplanes)
    elif norm_cfg == 'IN':
        out = nn.InstanceNorm3d(inplanes, affine=True)
    elif norm_cfg == 'LN':
        out = nn.LayerNorm(inplanes, eps=1e-6)

    return out

def Activation_layer(activation_cfg, inplace=True):
    if activation_cfg == 'ReLU':
        out = nn.ReLU(inplace=inplace)
    elif activation_cfg == 'LeakyReLU':
        out = nn.LeakyReLU(negative_slope=1e-2, inplace=inplace)

    return out


class Conv3dBlock(nn.Module):
    def __init__(self, in_channels, out_channels, norm_cfg, activation_cfg, kernel_size, stride=(1, 1, 1),
                 padding=(0, 0, 0), dilation=(1, 1, 1), bias=False, weight_std=False):
        super(Conv3dBlock, self).__init__()
        self.conv = conv3x3x3(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding,
                              dilation=dilation, bias=bias, weight_std=weight_std)
        self.norm = Norm_layer(norm_cfg, out_channels)
        self.nonlin = Activation_layer(activation_cfg, inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        x = self.nonlin(x)
        return x

def drop_path(x, drop_prob: float = 0., training: bool = False):
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 3D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """

    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x, dim):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., sr_ratio=1):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."

        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.sr_ratio = sr_ratio
        if sr_ratio > 1:
            self.sr = nn.Conv3d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio)
            self.norm = nn.LayerNorm(dim)

    def forward(self, x, dim):
        B, N, C = x.shape
        q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        if self.sr_ratio > 1:
            x_ = x.permute(0, 2, 1).reshape(B, C, dim[0], dim[1], dim[2])
            x_ = self.sr(x_).reshape(B, C, -1).permute(0, 2, 1)
            x_ = self.norm(x_)
            kv = self.kv(x_).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        else:
            kv = self.kv(x).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x


class Block(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, sr_ratio=1):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim,
            num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
            attn_drop=attn_drop, proj_drop=drop, sr_ratio=sr_ratio)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x, dim):
        x = x + self.drop_path(self.attn(self.norm1(x), dim))
        x = x + self.drop_path(self.mlp(self.norm2(x), dim))

        return x

class PatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """

    def __init__(self, img_size=[16, 96, 96], patch_size=[16, 16, 16], in_chans=1, embed_dim=768):
        super().__init__()
        num_patches = (img_size[0] * patch_size[0]) * (img_size[1] * patch_size[1]) * (img_size[2] * patch_size[2])
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches

        self.proj = nn.ConvTranspose3d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        B, C, D, H, W = x.shape
        x = self.proj(x).flatten(2).transpose(1, 2)

        return x, (D*self.patch_size[0], H*self.patch_size[1], W*self.patch_size[1])


class MiT(nn.Module):

    def __init__(self, norm_cfg='BN', activation_cfg='ReLU', weight_std=False, img_size=[48, 192, 192], num_classes=None, in_chans=1, 
                 embed_dims=[192,64,32], num_heads=[8,4,2], mlp_ratios=[4, 4, 4], qkv_bias=False, qk_scale=None, 
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0., norm_layer=nn.LayerNorm, depths=[2, 2, 2], 
                 sr_ratios=[2, 2, 4], encoder='tiny'):

        super().__init__()
        self.MODEL_NUM_CLASSES = num_classes
        self.embed_dims = embed_dims

        # Encoder
        if encoder=='small':
            self.transformer = MiT_encoder.model_small(norm_cfg=norm_cfg, activation_cfg=activation_cfg, weight_std=weight_std, img_size=img_size, in_chans=in_chans)
        elif encoder=='tiny':
            self.transformer = MiT_encoder.model_tiny(norm_cfg=norm_cfg, activation_cfg=activation_cfg, weight_std=weight_std, img_size=img_size, in_chans=in_chans)

        total = sum([param.nelement() for param in self.transformer.parameters()])
        print('  + Number of Transformer Params: %.2f(e6)' % (total / 1e6))

        # upsampling
        self.upsamplex122 = nn.Upsample(scale_factor=(1, 2, 2), mode='trilinear')

        self.DecEmbed0 = PatchEmbed(img_size=[img_size[0]//8, img_size[1]//16, img_size[2]//16], patch_size=[2, 2, 2], in_chans=self.transformer.embed_dims[-1], embed_dim=embed_dims[0])
        self.DecEmbed1 = PatchEmbed(img_size=[img_size[0]//4, img_size[1]//8, img_size[2]//8], patch_size=[2, 2, 2], in_chans=embed_dims[0], embed_dim=embed_dims[1])
        self.DecEmbed2 = PatchEmbed(img_size=[img_size[0]//2, img_size[1]//4, img_size[2]//4], patch_size=[2, 2, 2], in_chans=embed_dims[1], embed_dim=embed_dims[2])

        # Decoder patchEmbed
        self.DecPosEmbed0 = nn.Parameter(torch.zeros(1, self.transformer.patch_embed3.num_patches, embed_dims[0]))
        self.DecPosDrop0 = nn.Dropout(p=drop_rate)
        self.DecPosEmbed1 = nn.Parameter(torch.zeros(1, self.transformer.patch_embed2.num_patches, embed_dims[1]))
        self.DecPosDrop1 = nn.Dropout(p=drop_rate)
        self.DecPosEmbed2 = nn.Parameter(torch.zeros(1, self.transformer.patch_embed1.num_patches, embed_dims[2]))
        self.DecPosDrop2 = nn.Dropout(p=drop_rate)

        # Decoder transformer
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule
        cur = 0
        self.Decblock0 = nn.ModuleList([Block(
            dim=embed_dims[0], num_heads=num_heads[0], mlp_ratio=mlp_ratios[0], qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + i], norm_layer=norm_layer,
            sr_ratio=sr_ratios[0])
            for i in range(depths[0])])

        cur += depths[0]
        self.Decblock1 = nn.ModuleList([Block(
            dim=embed_dims[1], num_heads=num_heads[1], mlp_ratio=mlp_ratios[1], qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + i], norm_layer=norm_layer,
            sr_ratio=sr_ratios[1])
            for i in range(depths[1])])

        cur += depths[1]
        self.Decblock2 = nn.ModuleList([Block(
            dim=embed_dims[2], num_heads=num_heads[2], mlp_ratio=mlp_ratios[2], qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + i], norm_layer=norm_layer,
            sr_ratio=sr_ratios[2])
            for i in range(depths[2])])

        self.norm = norm_layer(embed_dims[2])

        self.transposeconv_stage3 = nn.ConvTranspose3d(embed_dims[2], embed_dims[3], kernel_size=2, stride=2)
        self.stage3_de = Conv3dBlock(embed_dims[3], embed_dims[3], norm_cfg=norm_cfg, activation_cfg=activation_cfg, weight_std=weight_std, kernel_size=3, stride=1, padding=1)

        # Seg head
        self.ds0_cls_conv = nn.Conv3d(embed_dims[0], self.MODEL_NUM_CLASSES, kernel_size=1)
        self.ds1_cls_conv = nn.Conv3d(embed_dims[1], self.MODEL_NUM_CLASSES, kernel_size=1)
        self.ds2_cls_conv = nn.Conv3d(embed_dims[2], self.MODEL_NUM_CLASSES, kernel_size=1)
        self.ds3_cls_conv = nn.Conv3d(embed_dims[3], self.MODEL_NUM_CLASSES, kernel_size=1)

        self.cls_conv = nn.Conv3d(embed_dims[3], self.MODEL_NUM_CLASSES, kernel_size=1)

        trunc_normal_(self.DecPosEmbed0, std=.02)
        trunc_normal_(self.DecPosEmbed1, std=.02)
        trunc_normal_(self.DecPosEmbed2, std=.02)
        
        self.apply(self._init_weights)


    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.Conv3d, Conv3d_wd)):
            nn.init.kaiming_normal_(m.weight, mode='fan_out')
        elif isinstance(m, (nn.LayerNorm, nn.BatchNorm3d, nn.GroupNorm, nn.InstanceNorm3d, nn.SyncBatchNorm)):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, inputs):
        B = inputs.shape[0]
        
        ####### encoder
        x_encoder, (D, H, W) = self.transformer(inputs)
        x_trans = x_encoder[-1].reshape(B, D, H, W, -1).permute(0, 4, 1, 2, 3).contiguous()

        ####### decoder
        # stage 0
        x, (D, H, W) = self.DecEmbed0(x_trans)
        skip0 = x_encoder[-2]
        x = x + skip0.flatten(2).transpose(1, 2)
        x = x + self.DecPosEmbed0   
        x = self.DecPosDrop0(x)
        for blk in self.Decblock0:
            x = blk(x, (D, H, W))
        x = x.reshape(B, D, H, W, -1).permute(0, 4, 1, 2, 3).contiguous()
        ds0 = self.ds0_cls_conv(x)
        
        # stage 1
        x, (D, H, W) = self.DecEmbed1(x)
        skip1 = x_encoder[-3]
        x = x + skip1.flatten(2).transpose(1, 2)
        x = x + self.DecPosEmbed1
        x = self.DecPosDrop1(x)
        for blk in self.Decblock1:
            x = blk(x, (D, H, W))
        x = x.reshape(B, D, H, W, -1).permute(0, 4, 1, 2, 3).contiguous()
        ds1 = self.ds1_cls_conv(x)

        # stage 2
        x, (D, H, W) = self.DecEmbed2(x)
        skip2 = x_encoder[-4]
        x = x + skip2.flatten(2).transpose(1, 2)
        x = x + self.DecPosEmbed2
        x = self.DecPosDrop2(x)
        for blk in self.Decblock2:
            x = blk(x, (D, H, W))
        x = self.norm(x)
        x = x.reshape(B, D, H, W, -1).permute(0, 4, 1, 2, 3).contiguous()
        ds2 = self.ds2_cls_conv(x)

        # stage 3
        x = self.transposeconv_stage3(x)
        skip3 = x_encoder[-5]
        x = x + skip3
        x = self.stage3_de(x)
        ds3 = self.ds3_cls_conv(x)

        x = self.upsamplex122(x)
        result = self.cls_conv(x)

        return [result, ds3, ds2, ds1, ds0]


class MiTnet(SegmentationNetwork):
    """
    MiTnet
    """

    def __init__(self, norm_cfg='BN', activation_cfg='ReLU', weight_std=False, img_size=None, num_classes=None, in_chans=1, 
                 pretrain=False, pretrain_path=None, deep_supervision=False):
        super().__init__()

        # used networks 
        # small  
        self.model = MiT(norm_cfg, activation_cfg, weight_std, img_size, num_classes, in_chans,
        embed_dims=[256,128,48,32], num_heads=[8, 4, 2], depths=[3, 4, 3], mlp_ratios=[4, 4, 4], sr_ratios=[2, 4, 6], encoder='small')  

        # # tiny
        # self.model = MiT(norm_cfg, activation_cfg, weight_std, img_size, num_classes, in_chans,
        # embed_dims=[256,128,48,32], num_heads=[8, 4, 2], depths=[1, 1, 1], mlp_ratios=[4, 4, 4], sr_ratios=[2, 4, 6], encoder='tiny')

        total = sum([param.nelement() for param in self.model.parameters()])
        print('  + Number of Network Params: %.2f(e6)' % (total / 1e6))

        if pretrain:
            pre_type = 'student'  #teacher student
            print('*********loading from checkpoint ssl: {}'.format(pretrain_path))

            if pre_type == 'teacher': 
                pre_dict_ori = torch.load(pretrain_path, map_location="cpu")[pre_type]
                pre_dict_ori = {k.replace("backbone.", ""): v for k, v in pre_dict_ori.items()}
                print('Teacher: length of pre-trained layers: %.f' % (len(pre_dict_ori)))
            else:
                pre_dict_ori = torch.load(pretrain_path, map_location="cpu")[pre_type]
                pre_dict_ori = {k.replace("module.backbone.", ""): v for k, v in pre_dict_ori.items()}
                print('Student: length of pre-trained layers: %.f' % (len(pre_dict_ori)))

            pre_dict_ori = {k.replace("3D", ""): v for k, v in pre_dict_ori.items()}

            model_dict = self.model.state_dict()
            print('length of new layers: %.f' % (len(model_dict)))
            print('before loading weights: %.12f' % (self.model.state_dict()['transformer.block1.0.mlp.fc1.weight'].mean()))

            # Patch_embeddings
            print('Patch_embeddings layer1 weights: %.12f' % (self.model.state_dict()['transformer.patch_embed1.proj.conv.weight'].mean()))
            print('Patch_embeddings layer2 weights: %.12f' % (self.model.state_dict()['transformer.patch_embed2.proj.conv.weight'].mean()))

            # Position_embeddings
            print('Position_embeddings weights: %.12f' % (self.model.transformer.pos_embed1.data.mean()))

            if model_dict['transformer.patch_embed0.conv.weight'].shape[1] != 1:
                pre_dict_ori['transformer.patch_embed0.conv.weight'] = pre_dict_ori['transformer.patch_embed0.conv.weight'].repeat_interleave(model_dict['transformer.patch_embed0.conv.weight'].shape[1],1)

            for k, v in pre_dict_ori.items():
                if '2D' not in k:
                    if (('transformer.pos_embed' in k) or ('DecPosEmbed' in k)): 
                        if ('DecPosEmbed' in k) or ('transformer.pos_embed4' in k):
                            posemb = pre_dict_ori[k][:,1::]
                            posemb_new = model_dict[k]
                        else:
                            posemb = pre_dict_ori[k]
                            posemb_new = model_dict[k]                        

                        if posemb.size() == posemb_new.size():
                            print(k+'layer is matched')
                            pre_dict_ori[k] = posemb
                        else:
                            ntok_new = posemb_new.size(1)
                            posemb_zoom = ndimage.zoom(posemb[0], (ntok_new / posemb.size(1), 1), order=1)
                            posemb_zoom = np.expand_dims(posemb_zoom, 0)
                            pre_dict_ori[k] = torch.from_numpy(posemb_zoom)

            pre_dict = {k: v for k, v in pre_dict_ori.items() if k in model_dict}
            print('length of matched layers: %.f' % (len(pre_dict)))

            # Update weigts
            model_dict.update(pre_dict)
            self.model.load_state_dict(model_dict)
            print('after loading weights: %.12f' % (self.model.state_dict()['transformer.block1.0.mlp.fc1.weight'].mean()))
            print('Patch_embeddings layer1 pretrained weights: %.12f' % (self.model.state_dict()['transformer.patch_embed1.proj.conv.weight'].mean()))
            print('Patch_embeddings layer2 pretrained weights: %.12f' % (self.model.state_dict()['transformer.patch_embed2.proj.conv.weight'].mean()))
            print('Position_embeddings pretrained weights: %.12f' % (self.model.transformer.pos_embed1.data.mean()))

        else:
            print('before loading weights: %.12f' % (self.model.state_dict()['transformer.block1.0.mlp.fc1.weight'].mean()))
            print('Patch_embeddings layer1 weights: %.12f' % (self.model.state_dict()['transformer.patch_embed1.proj.conv.weight'].mean()))
            print('Patch_embeddings layer2 weights: %.12f' % (self.model.state_dict()['transformer.patch_embed2.proj.conv.weight'].mean()))
            print('Position_embeddings weights: %.12f' % (self.model.transformer.pos_embed1.data.mean()))

        if weight_std==False:
            self.conv_op = nn.Conv3d
        else:
            self.conv_op = Conv3d_wd
        if norm_cfg == 'BN':
            self.norm_op = nn.BatchNorm3d
        if norm_cfg == 'SyncBN':
            self.norm_op = nn.SyncBatchNorm
        if norm_cfg == 'GN':
            self.norm_op = nn.GroupNorm
        if norm_cfg == 'IN':
            self.norm_op = nn.InstanceNorm3d
        self.dropout_op = nn.Dropout3d
        self.num_classes = num_classes
        self._deep_supervision = deep_supervision
        self.do_ds = deep_supervision

    def forward(self, x):
        seg_output = self.model(x)
        if self._deep_supervision and self.do_ds:
            return seg_output
        else:
            return seg_output[0]