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


    def forward_with_prompt(self, support_x, support_y, query_x):
        Nq = query_x.shape[1]
        query_skips = self.encoder(query_x.view(query_x.shape[0]*query_x.shape[1], *query_x.shape[-3:]).float())
        support_skips = self.encoder(support_x.float())
        mask_skips = self.mask_encoder(support_y.unsqueeze(dim=1).float())
        skips = []
        attn_i = 0
        for i in range(len(query_skips)):
            if i < self.n_stages - self.n_attn_stage:
                skips.append(query_skips[i])
            else:
                BNq, C, H, W = query_skips[i].shape
                query = query_skips[i].view(BNq, C, H*W).permute(0, 2, 1).contiguous()
                key = support_skips[i].unsqueeze(dim=1).repeat(1, Nq, 1, 1, 1).view(BNq, C, H*W).permute(0, 2, 1).contiguous()
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

    def forward_from_scratch(self, support_x, support_y, query_x):
        Nq = query_x.shape[1]
        query_skips = self.encoder(query_x.view(query_x.shape[0]*query_x.shape[1], *query_x.shape[-3:]).float())
        support_skips = self.encoder(support_x.float())
        mask_skips = self.mask_encoder(support_y.unsqueeze(dim=1).float())
        skips = []
        attn_i = 0
        for i in range(len(query_skips)):
            if i < self.n_stages - self.n_attn_stage:
                skips.append(query_skips[i])
            else:
                BNq, C, H, W = query_skips[i].shape
                query = query_skips[i].view(BNq, C, H*W).permute(0, 2, 1).contiguous()
                key = support_skips[i].unsqueeze(dim=1).repeat(1, Nq, 1, 1, 1).view(BNq, C, H*W).permute(0, 2, 1).contiguous()
                value = mask_skips[i].unsqueeze(dim=1).repeat(1, Nq, 1, 1, 1).view(BNq, C, H*W).permute(0, 2, 1).contiguous()
                attn = self.cross_attns[attn_i](query=query, key=key, value=value)
                attn = attn.permute(0, 2, 1).view(BNq, C, H, W).contiguous()
                attn_i += 1
                skips.append(query_skips[i] + attn * 0.0)
        region_predictions, region_features = self.decoder(skips)
        return {
            "region_predictions": region_predictions,
            "region_features": region_features,
            "predictions": region_predictions,
        }

    def forward(self, support_x, support_y, query_x):
        if self.training:
            if np.random.uniform(0, 1) <= self.from_scratch_ratio:
                return self.forward_from_scratch(support_x, support_y, query_x)
            else:
                return self.forward_with_prompt(support_x, support_y, query_x)
        else:
            return self.forward_with_prompt(support_x, support_y, query_x)
    
class RefiningStage(PlainConvUNet):
    def __init__(self,
                 proposing_stage,
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
                 ):
        super().__init__(input_channels, n_stages, features_per_stage, conv_op, kernel_sizes,
                 strides, n_conv_per_stage, num_classes, n_conv_per_stage_decoder,
                 conv_bias, norm_op, norm_op_kwargs, dropout_op, dropout_op_kwargs,
                 nonlin, nonlin_kwargs, deep_supervision, nonlin_first)
        self.proposing_stage = proposing_stage
        self.proposing_stage.eval()
        self.n_stages = n_stages
        self.seg_layers = nn.ModuleList([nn.Conv2d(f, num_classes, kernel_size=1, stride=1, padding=0, bias=True) for f in features_per_stage])

    def forward(self, support_x, support_y, query_x):
        with torch.no_grad():
            proposing_results = self.proposing_stage(support_x, support_y, query_x)
            proposing_region_predictions = proposing_results["region_predictions"]
            proposing_region_features = proposing_results["region_features"]
        # support_x: (B, 1, 224, 224)
        # support_y: (B, 1, 224, 224)
        # query_x: (B, Nq, 1, 224, 224)
        query_skips = self.encoder(query_x.view(query_x.shape[0]*query_x.shape[1], *query_x.shape[-3:]).float())
        _, region_features = self.decoder(query_skips)
        refining_region_features = [proposing_region_feature+region_feature for proposing_region_feature, region_feature in zip(proposing_region_features, region_features)]
        refining_region_predictions = [seg_layer(refining_region_feature) for seg_layer, refining_region_feature in zip(self.seg_layers, refining_region_features)]
        
        return {
            "proposing_region_predictions": proposing_region_predictions,
            "proposing_region_features": proposing_region_features,
            "refining_region_predictions": refining_region_predictions,
            "refining_region_features": refining_region_features,
            "predictions": refining_region_predictions,
        }

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

class PropNetV1(nn.Module):
    def __init__(self, args):
        super().__init__()
        conv_op = convert_dim_to_conv_op(args.conv_dim)
        self.proposing_stage = ProposingStage(
            input_channels=args.input_channels, 
            n_stages=args.n_stages,
            features_per_stage=[min(32 * 2 ** i, args.max_channels) for i in range(args.n_stages)],
            conv_op=conv_op,
            kernel_sizes=[[3,3]]*args.n_stages,
            strides=[[1,1]] + [[2,2]]*(args.n_stages-1), 
            num_classes=args.num_classes,
            deep_supervision=args.deep_supervision,
            n_conv_per_stage=[2] * args.n_stages,
            n_conv_per_stage_decoder=[2] * (args.n_stages - 1),
            conv_bias=True,
            norm_op=get_matching_instancenorm(conv_op),
            norm_op_kwargs={'eps': 1e-5, 'affine': True},
            dropout_op=None,
            dropout_op_kwargs=None, 
            nonlin=nn.LeakyReLU,
            nonlin_kwargs={'inplace': True},
            n_attn_stage=args.n_attn_stage,
            from_scratch_ratio=args.from_scratch_ratio,
        )

    def forward(self, batch):
        return self.proposing_stage(support_x=batch["support_x"], support_y=batch["support_y"], query_x=batch["query_xs"])
    