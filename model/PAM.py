from dynamic_network_architectures.architectures.unet import PlainConvUNet
import torch.nn.functional as F
from typing import Union, Type, List, Tuple
from copy import deepcopy

import numpy as np
import os 
import torch
from dynamic_network_architectures.building_blocks.residual_encoders import ResidualEncoder
from dynamic_network_architectures.building_blocks.residual import BasicBlockD, BottleneckD
from torch import nn
from torch.nn.modules.conv import _ConvNd
from torch.nn.modules.dropout import _DropoutNd

from dynamic_network_architectures.building_blocks.plain_conv_encoder import PlainConvEncoder
from dynamic_network_architectures.building_blocks.unet_decoder import UNetDecoder
from dynamic_network_architectures.building_blocks.helper import convert_conv_op_to_dim 
from dynamic_network_architectures.building_blocks.helper import convert_dim_to_conv_op, get_matching_instancenorm

class CrossAttention(nn.Module):
    def __init__(self, embed_size, heads):
        super(CrossAttention, self).__init__()
        self.embed_size = embed_size
        self.heads = heads
        self.head_dim = embed_size // heads

        assert (
            self.head_dim * heads == embed_size
        ), "Embedding size needs to be divisible by heads"

        self.values = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.keys = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.queries = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.fc_out = nn.Linear(heads * self.head_dim, embed_size)

    def forward(self, query, key, value):
        N = query.shape[0]
        value_len, key_len, query_len = value.shape[1], key.shape[1], query.shape[1]

        # Split the embedding into self.heads different pieces
        values = value.reshape(N, value_len, self.heads, self.head_dim)
        keys = key.reshape(N, key_len, self.heads, self.head_dim)
        queries = query.reshape(N, query_len, self.heads, self.head_dim)

        values = self.values(values)
        keys = self.keys(keys)
        queries = self.queries(queries)

        # Scaled dot-product attention
        energy = torch.einsum("nqhd,nkhd->nhqk", [queries, keys])  # (N, heads, query_len, key_len)

        attention = F.softmax(energy / (self.embed_size ** (1 / 2)), dim=3)

        out = torch.einsum("nhql,nlhd->nqhd", [attention, values]).reshape(
            N, query_len, self.heads * self.head_dim
        )

        out = self.fc_out(out)
        return out

class ProposingStage(PlainConvUNet):
    def __init__(self,
                 input_channels: int,
                 n_stages: int,
                 features_per_stage: Union[int, List[int], Tuple[int, ...]],
                 conv_op: Type[_ConvNd],
                 kernel_sizes: Union[int, List[int], Tuple[int, ...]],
                 strides: Union[int, List[int], Tuple[int, ...]],
                 n_conv_per_stage: Union[int, List[int], Tuple[int, ...]],
                 num_classes: int,
                 n_conv_per_stage_decoder: Union[int, Tuple[int, ...], List[int]],
                 conv_bias: bool = False,
                 norm_op: Union[None, Type[nn.Module]] = None,
                 norm_op_kwargs: dict = None,
                 dropout_op: Union[None, Type[_DropoutNd]] = None,
                 dropout_op_kwargs: dict = None,
                 nonlin: Union[None, Type[torch.nn.Module]] = None,
                 nonlin_kwargs: dict = None,
                 deep_supervision: bool = False,
                 nonlin_first: bool = False,
                 attn_heads: int = 1,
                 n_attn_stage: int = 4,
                 from_scratch_ratio: float = 0.25,
                 ):
        super().__init__(input_channels, n_stages, features_per_stage, conv_op, kernel_sizes,
                 strides, n_conv_per_stage, num_classes, n_conv_per_stage_decoder,
                 conv_bias, norm_op, norm_op_kwargs, dropout_op, dropout_op_kwargs,
                 nonlin, nonlin_kwargs, deep_supervision, nonlin_first)
        self.n_stages = n_stages
        self.n_attn_stage = n_attn_stage
        self.mask_encoder = PlainConvEncoder(1, n_stages, features_per_stage, conv_op, kernel_sizes, strides,
                                        n_conv_per_stage, conv_bias, norm_op, norm_op_kwargs, dropout_op,
                                        dropout_op_kwargs, nonlin, nonlin_kwargs, return_skips=True,
                                        nonlin_first=nonlin_first)
        self.cross_attns = nn.ModuleList([CrossAttention(embed_size=f, heads=attn_heads) for f in features_per_stage[-self.n_attn_stage:]])
        self.from_scratch_ratio = from_scratch_ratio


    def forward_with_prompt(self, guiding_x, guiding_prp, adjacent_x):
        Nq = adjacent_x.shape[1]
        query_skips = self.encoder(adjacent_x.view(adjacent_x.shape[0]*adjacent_x.shape[1], *adjacent_x.shape[-3:]).float())
        guiding_skips = self.encoder(guiding_x.float())
        mask_skips = self.mask_encoder(guiding_prp.unsqueeze(dim=1).float())
        skips = []
        attn_i = 0
        for i in range(len(query_skips)):
            if i < self.n_stages - self.n_attn_stage:
                skips.append(query_skips[i])
            else:
                BNq, C, H, W = query_skips[i].shape
                query = query_skips[i].view(BNq, C, H*W).permute(0, 2, 1).contiguous()
                key = guiding_skips[i].unsqueeze(dim=1).repeat(1, Nq, 1, 1, 1).view(BNq, C, H*W).permute(0, 2, 1).contiguous()
                value = mask_skips[i].unsqueeze(dim=1).repeat(1, Nq, 1, 1, 1).view(BNq, C, H*W).permute(0, 2, 1).contiguous()
                attn = self.cross_attns[attn_i](query=query, key=key, value=value)
                attn = attn.permute(0, 2, 1).view(BNq, C, H, W).contiguous()
                attn_i += 1
                skips.append(query_skips[i] + attn)
        region_predictions, region_features = self.decoder(skips)
        return {
            "region_predictions": region_predictions,
            "region_features": region_features,
            "predictions": region_predictions,
        }

    def forward(self, guiding_x, guiding_prp, adjacent_x):
        return self.forward_with_prompt(guiding_x, guiding_prp, adjacent_x)


class Box2SegNet(PlainConvUNet):
    def __init__(self, args):
        conv_op = convert_dim_to_conv_op(args.conv_dim)
        super().__init__(
            args.input_channels, 
            args.n_stages, 
            [min(32 * 2 ** i, args.max_channels) for i in range(args.n_stages)],
            conv_op, 
            [[3,3]]*args.n_stages,
            [[1,1]] + [[2,2]]*(args.n_stages-1), 
            [2] * args.n_stages,
            args.num_classes,
            [2] * (args.n_stages - 1),
            True,
            get_matching_instancenorm(conv_op),
            {'eps': 1e-5, 'affine': True},
            None,
            None,
            nn.LeakyReLU,
            {'inplace': True},
            args.deep_supervision,
            False)
    
    def forward(self, batch):
        xs = batch["query_xs"]
        B, N, C, H, W = xs.shape
        xs = xs.view(B*N, C, H, W)
        skips = self.encoder(xs)
        predictions, features = self.decoder(skips)
        return {
            "region_predictions": predictions,
            "region_features": features,
            "predictions": predictions,
        }

class PAM(nn.Module):
    def __init__(self, conv_dim, input_channels, n_stages, max_channels, num_classes, deep_supervision, n_attn_stage, from_scratch_ratio):
        super().__init__()
        conv_op = convert_dim_to_conv_op(conv_dim)
        self.proposing_stage = ProposingStage(
            input_channels=input_channels, 
            n_stages=n_stages,
            features_per_stage=[min(32 * 2 ** i, max_channels) for i in range(n_stages)],
            conv_op=conv_op,
            kernel_sizes=[[3,3]]*n_stages,
            strides=[[1,1]] + [[2,2]]*(n_stages-1), 
            num_classes=num_classes,
            deep_supervision=deep_supervision,
            n_conv_per_stage=[2] * n_stages,
            n_conv_per_stage_decoder=[2] * (n_stages - 1),
            conv_bias=True,
            norm_op=get_matching_instancenorm(conv_op),
            norm_op_kwargs={'eps': 1e-5, 'affine': True},
            dropout_op=None,
            dropout_op_kwargs=None, 
            nonlin=nn.LeakyReLU,
            nonlin_kwargs={'inplace': True},
            n_attn_stage=n_attn_stage,
            from_scratch_ratio=from_scratch_ratio,
        )

    def forward(self, batch):
        return self.proposing_stage(guiding_x=batch["guiding_x"], guiding_prp=batch["guiding_prp"], adjacent_x=batch["adjacent_x"])
