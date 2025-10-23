import torch.nn.functional as F
from typing import Union, Type, List, Tuple
from copy import deepcopy
import os 
import torch
from torch import nn
from torch.nn.modules.conv import _ConvNd
from torch.nn.modules.dropout import _DropoutNd
from segment_anything.modeling import ImageEncoderViT
from segment_anything.modeling.common import LayerNorm2d
from functools import partial
from pathlib import Path
import torch
from typing import List, Tuple, Type
import torch.nn as nn
import torch.nn.functional as F

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

class PropT(nn.Module):
    def __init__(self, args):
        super().__init__()
        img_encoder_embed_dim = args.img_encoder_embed_dim
        img_encoder_depth = args.img_encoder_depth
        img_encoder_num_heads = args.img_encoder_num_heads
        img_encoder_global_attn_indexes = args.img_encoder_global_attn_indexes
        mask_encoder_embed_dim = args.mask_encoder_embed_dim
        mask_encoder_depth = args.mask_encoder_depth
        mask_encoder_num_heads = args.mask_encoder_num_heads
        mask_encoder_global_attn_indexes = args.mask_encoder_global_attn_indexes
        img_size = args.target_size 
        vit_patch_size = args.vit_patch_size
        window_size = args.window_size 
        prompt_embed_dim = args.prompt_embed_dim
        num_classes = args.num_classes
        activation:  Type[nn.Module] = nn.GELU

        self.img_encoder = ImageEncoderViT(
            depth=img_encoder_depth,
            embed_dim=img_encoder_embed_dim,
            img_size=img_size,
            norm_layer=partial(torch.nn.LayerNorm, eps=1e-6),
            num_heads=img_encoder_num_heads,
            patch_size=vit_patch_size,
            qkv_bias=True,
            use_rel_pos=True,
            global_attn_indexes=img_encoder_global_attn_indexes,
            window_size=window_size,
            out_chans=prompt_embed_dim,
        )
        self.mask_encoder = ImageEncoderViT(
            depth=mask_encoder_depth,
            embed_dim=mask_encoder_embed_dim,
            img_size=img_size,
            norm_layer=partial(torch.nn.LayerNorm, eps=1e-6),
            num_heads=mask_encoder_num_heads,
            patch_size=vit_patch_size,
            qkv_bias=True,
            use_rel_pos=True,
            global_attn_indexes=mask_encoder_global_attn_indexes,
            window_size=window_size,
            out_chans=prompt_embed_dim
        )
        self.prompt_attention = CrossAttention(embed_size=prompt_embed_dim, heads=1)
        self.output_upscaling = nn.Sequential(
            nn.ConvTranspose2d(
                prompt_embed_dim, prompt_embed_dim // 4, kernel_size=2, stride=2
            ),
            LayerNorm2d(prompt_embed_dim // 4),
            activation(),
            nn.ConvTranspose2d(
                prompt_embed_dim // 4, prompt_embed_dim // 8, kernel_size=2, stride=2
            ),
            LayerNorm2d(prompt_embed_dim // 8),
            activation(),
            nn.ConvTranspose2d(
                prompt_embed_dim // 8, prompt_embed_dim // 8, kernel_size=2, stride=2
            ),
        )
        self.seg_layer = nn.Conv2d(prompt_embed_dim // 8, num_classes, 1, 1, 0, bias=True)

    def forward(self, batch):
        support_x = batch["support_x"] # (B, C, 224, 224)
        query_xs = batch["query_xs"] # (B, N, C, 224 ,224)
        xs = torch.cat([support_x.unsqueeze(dim=1), query_xs], dim=1)
        if xs.shape[2] == 1:
            xs = xs.repeat(1, 1, 3, 1, 1)
        B, N, C, H, W = xs.shape
        xs = xs.view(B*N, C, H, W)
        img_embeddings = self.img_encoder(xs) # (B*N, 256, H//8, W//8)
        img_embeddings = img_embeddings.view(B, N, *img_embeddings.shape[-3:])
        support_img_embeddings = img_embeddings[:, 0:1, ...] # (B, 1, 256, H//8, W//8)
        query_img_embeddings = img_embeddings[:, 1:, ...] # (B, N_query, 256, H//8, W//8)
        
        support_y = batch["support_y"] # (B, C, 224, 224)
        if len(support_y.shape) == 3:
            support_y = support_y.unsqueeze(dim=1)
        if support_y.shape[1] == 1:
            support_y = support_y.repeat(1, 3, 1, 1)
        support_mask_embeddings = self.mask_encoder(support_y).unsqueeze(dim=1) # (B, 1, 256, H//8, W//8)

        B, N_query, N_channel, H_feat, W_feat = query_img_embeddings.shape
        prompt_query = query_img_embeddings.contiguous().view(B*N_query, N_channel, H_feat*W_feat).permute(0, 2, 1).contiguous()
        prompt_key = support_img_embeddings.repeat(1, N_query, 1, 1, 1).view(B*N_query, N_channel, H_feat*W_feat).permute(0, 2, 1).contiguous()
        prompt_value = support_mask_embeddings.repeat(1, N_query, 1, 1, 1).view(B*N_query, N_channel, H_feat*W_feat).permute(0, 2, 1).contiguous()
        prompt_embeddings = self.prompt_attention(query=prompt_query, key=prompt_key, value=prompt_value)
        prompt_embeddings = prompt_embeddings.permute(0, 2, 1).contiguous().view(B*N_query, N_channel, H_feat, W_feat)
        
        region_features = self.output_upscaling(prompt_embeddings)
        region_predictions = self.seg_layer(region_features)

        return {
            "region_predictions": [region_predictions,],
            "region_features": [region_features,],
            "predictions": [region_predictions,],
        }

if __name__ == "__main__":
    device = "cuda:0"
    model = PropNet().to(device)
    batch = {
        "support_x": torch.Tensor(2, 1, 224, 224).to(device),
        "query_xs": torch.Tensor(2, 4, 1, 224, 224).to(device),
        "support_y": torch.ones(2, 1, 224, 224).to(device),
    }
    pred = model(batch)
    print(pred["region_predictions"].shape)
    print(pred["region_features"].shape)
    print(pred["predictions"].shape)
    print("")