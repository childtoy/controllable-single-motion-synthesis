from abc import abstractmethod

import math

import numpy as np
import torch
import torch as th
import torch.nn as nn
import torch.nn.functional as F

from diffusion.fp16_util import convert_module_to_f16, convert_module_to_f32
from diffusion.nn import (
    checkpoint,
    conv_nd,
    linear,
    avg_pool_nd,
    zero_module,
    normalization,
    timestep_embedding,
)
from models.qna import FusedQnA1d


class TimestepBlock(nn.Module):
    """
    Any module where forward() takes timestep embeddings as a second argument.
    """

    @abstractmethod
    def forward(self, x, emb, cond):
        """
        Apply the module to `x` given `emb` timestep embeddings.
        """


class TimestepEmbedSequential(nn.Sequential, TimestepBlock):
    """
    A sequential module that passes timestep embeddings to the children that
    support it as an extra input.
    """

    def forward(self, x, emb, cond=None):
        for layer in self:
            if isinstance(layer, TimestepBlock):
                x, cond = layer(x, emb, cond)
            else:
                x = layer(x)
        return x, cond


class Upsample(nn.Module):
    """
    An upsampling layer with an optional convolution.

    :param channels: channels in the inputs and outputs.
    :param use_conv: a bool determining if a convolution is applied.
    :param dims: determines if the signal is 1D, 2D, or 3D. If 3D, then
                 upsampling occurs in the inner-two dimensions.
    """

    def __init__(self, channels, use_conv, dims=2, out_channels=None, padding_mode='zeros', padding=1):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.dims = dims
        if use_conv:
            self.conv = conv_nd(dims, self.channels, self.out_channels, 3, padding=padding, padding_mode=padding_mode)

    def forward(self, x):
        assert x.shape[1] == self.channels
        if self.dims == 3:
            x = F.interpolate(
                x, (x.shape[2], x.shape[3] * 2, x.shape[4] * 2), mode="nearest"
            )
        else:
            x = F.interpolate(x, scale_factor=2, mode="nearest")
        if self.use_conv:
            x = self.conv(x)
        return x


class Downsample(nn.Module):
    """
    A downsampling layer with an optional convolution.

    :param channels: channels in the inputs and outputs.
    :param use_conv: a bool determining if a convolution is applied.
    :param dims: determines if the signal is 1D, 2D, or 3D. If 3D, then
                 downsampling occurs in the inner-two dimensions.
    """

    def __init__(self, channels, use_conv, dims=2, out_channels=None, padding_mode='zeros', padding=1):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.dims = dims
        stride = 2 if dims != 3 else (1, 2, 2)
        if use_conv:
            self.op = conv_nd(
                dims, self.channels, self.out_channels, 3, stride=stride, padding=padding, padding_mode=padding_mode
            )
        else:
            assert self.channels == self.out_channels
            self.op = avg_pool_nd(dims, kernel_size=stride, stride=stride)

    def forward(self, x):
        assert x.shape[1] == self.channels
        return self.op(x)


class ResBlock(TimestepBlock):
    """
    A residual block that can optionally change the number of channels.

    :param channels: the number of input channels.
    :param emb_channels: the number of timestep embedding channels.
    :param dropout: the rate of dropout.
    :param out_channels: if specified, the number of out channels.
    :param use_conv: if True and out_channels is specified, use a spatial
        convolution instead of a smaller 1x1 convolution to change the
        channels in the skip connection.
    :param dims: determines if the signal is 1D, 2D, or 3D.
    :param use_checkpoint: if True, use gradient checkpointing on this module.
    :param up: if True, use this block for upsampling.
    :param down: if True, use this block for downsampling.
    """

    def __init__(
        self,
        channels,
        emb_channels,
        dropout,
        out_channels=None,
        use_conv=False,
        use_scale_shift_norm=False,
        dims=2,
        use_checkpoint=False,
        up=False,
        down=False,
        padding_mode='zeros',
        padding=1
    ):
        super().__init__()
        self.channels = channels
        self.emb_channels = emb_channels
        self.dropout = dropout
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.use_checkpoint = use_checkpoint
        self.use_scale_shift_norm = use_scale_shift_norm

        self.in_layers = nn.Sequential(
            normalization(channels),
            nn.SiLU(),
            conv_nd(dims, channels, self.out_channels, 3, padding=padding, padding_mode=padding_mode),
        )

        self.in_layers_cond = nn.Sequential(
            normalization(channels),
            nn.SiLU(),
            conv_nd(dims, channels, self.out_channels, 3, padding=padding, padding_mode=padding_mode),
        )

        self.updown = up or down

        if up:
            self.h_upd = Upsample(channels, False, dims, padding_mode=padding_mode, padding=padding)
            self.h_cond_upd = Upsample(channels, False, dims, padding_mode=padding_mode, padding=padding)
            self.x_upd = Upsample(channels, False, dims, padding_mode=padding_mode, padding=padding)
            self.x_cond_upd = Upsample(channels, False, dims, padding_mode=padding_mode, padding=padding)
        elif down:
            self.h_upd = Downsample(channels, False, dims, padding_mode=padding_mode, padding=padding)
            self.h_cond_upd = Downsample(channels, False, dims, padding_mode=padding_mode, padding=padding)
            self.x_upd = Downsample(channels, False, dims, padding_mode=padding_mode, padding=padding)
            self.x_cond_upd = Downsample(channels, False, dims, padding_mode=padding_mode, padding=padding)
        else:
            self.h_upd = self.x_upd = nn.Identity()
            self.h_cond_upd = self.x_cond_upd = nn.Identity()

        self.emb_layers = nn.Sequential(
            nn.SiLU(),
            linear(
                emb_channels,
                2 * self.out_channels if use_scale_shift_norm else self.out_channels,
            ),
        )
        self.out_layers = nn.Sequential(
            normalization(self.out_channels),
            nn.SiLU(),
            nn.Dropout(p=dropout),
            zero_module(
                conv_nd(dims, self.out_channels, self.out_channels, 3, padding=padding, padding_mode=padding_mode)
            ),
        )

        if self.out_channels == channels:
            self.skip_connection = nn.Identity()
            self.skip_connection_cond = nn.Identity()
        elif use_conv:
            self.skip_connection = conv_nd(
                dims, channels, self.out_channels, 3, padding=padding, padding_mode=padding_mode
            )
            self.skip_connection_cond = conv_nd(
                dims, channels, self.out_channels, 3, padding=padding, padding_mode=padding_mode
            )
        else:
            self.skip_connection = conv_nd(dims, channels, self.out_channels, 1)
            self.skip_connection_cond = conv_nd(dims, channels, self.out_channels, 1)

    def forward(self, x, emb, cond):
        """
        Apply the block to a Tensor, conditioned on a timestep embedding.

        :param x: an [N x C x ...] Tensor of features.
        :param emb: an [N x emb_channels] Tensor of timestep embeddings.
        :return: an [N x C x ...] Tensor of outputs.
        """
        return checkpoint(
            self._forward, (x, emb, cond), self.parameters(), self.use_checkpoint
        )

    def _forward(self, x, emb, cond):
        if self.updown:
            in_rest, in_conv = self.in_layers[:-1], self.in_layers[-1]
            in_rest_cond, in_conv_cond = self.in_layers_cond[:-1], self.in_layers_cond[-1]
            
            h = in_rest(x)
            h = self.h_upd(h)
            x = self.x_upd(x)
            h = in_conv(h)
            
            h_cond = in_rest_cond(cond)
            h_cond = self.h_cond_upd(h_cond)
            cond = self.x_cond_upd(cond)
            h_cond = in_conv_cond(h_cond)
        else:
            h = self.in_layers(x)
            h_cond = self.in_layers_cond(cond)
        emb_out = self.emb_layers(emb).type(h.dtype)
        while len(emb_out.shape) < len(h.shape):
            emb_out = emb_out[..., None]
        if self.use_scale_shift_norm:
            out_norm, out_rest = self.out_layers[0], self.out_layers[1:]
            scale, shift = th.chunk(emb_out, 2, dim=1)
            h = out_norm(h) * (1 + scale) + shift + h_cond
            h = out_rest(h)
        else:
            h = h + emb_out + h_cond
            h = self.out_layers(h)
        h = self.skip_connection(x) + h
        h_cond = self.skip_connection(cond) + h_cond
        return h, h_cond


class AttentionBlock(nn.Module):
    """
    An attention block that allows spatial positions to attend to each other.

    Originally ported from here, but adapted to the N-d case.
    https://github.com/hojonathanho/diffusion/blob/1e0dceb3b3495bbe19116a5e1b3596cd0706c543/diffusion_tf/models/unet.py#L66.
    """

    def __init__(
            self,
            channels,
            num_heads=1,
            num_head_channels=-1,
            use_checkpoint=False,
            use_new_attention_order=False,
            use_qna=False,
            kernel_size=3,
    ):
        super().__init__()
        self.channels = channels
        self.use_qna = use_qna
        if num_head_channels == -1:
            self.num_heads = num_heads
        else:
            assert (
                    channels % num_head_channels == 0
            ), f"q,k,v channels {channels} is not divisible by num_head_channels {num_head_channels}"
            self.num_heads = channels // num_head_channels
        self.use_checkpoint = use_checkpoint
        self.norm = normalization(channels)
        if not use_qna:
            self.qkv = conv_nd(1, channels, channels * 3, 1)
        if use_qna:
            self.attention = FusedQnA1d(
                in_features=self.channels,
                timesteps_features=None,
                hidden_features=self.channels,
                heads=self.num_heads,
                kernel_size=kernel_size,
                stride=1,
                padding=(kernel_size - 1) // 2,
            )
        elif use_new_attention_order:
            # split qkv before split heads
            self.attention = QKVAttention(self.num_heads)
        else:
            # split heads before split qkv
            self.attention = QKVAttentionLegacy(self.num_heads)
        if not use_qna:
            self.proj_out = zero_module(conv_nd(1, channels, channels, 1))

    def forward(self, x):
        return checkpoint(self._forward, (x,), self.parameters(), self.use_checkpoint)

    def _forward(self, x):
        b, c, *spatial = x.shape
        x = x.reshape(b, c, -1)

        if self.use_qna:
            h = self.norm(x).reshape(b, c, 1, -1)
            h = self.attention(h)
            h = h.reshape(b, c, -1)
        else:
            qkv = self.qkv(self.norm(x))
            h = self.attention(qkv)
            h = self.proj_out(h)
        return (x + h).reshape(b, c, *spatial)


def count_flops_attn(model, _x, y):
    """
    A counter for the `thop` package to count the operations in an
    attention operation.
    Meant to be used like:
        macs, params = thop.profile(
            model,
            inputs=(inputs, timestamps),
            custom_ops={QKVAttention: QKVAttention.count_flops},
        )
    """
    b, c, *spatial = y[0].shape
    num_spatial = int(np.prod(spatial))
    # We perform two matmuls with the same number of ops.
    # The first computes the weight matrix, the second computes
    # the combination of the value vectors.
    matmul_ops = 2 * b * (num_spatial ** 2) * c
    model.total_ops += th.DoubleTensor([matmul_ops])


class QKVAttentionLegacy(nn.Module):
    """
    A module which performs QKV attention. Matches legacy QKVAttention + input/ouput heads shaping
    """

    def __init__(self, n_heads):
        super().__init__()
        self.n_heads = n_heads

    def forward(self, qkv):
        """
        Apply QKV attention.

        :param qkv: an [N x (H * 3 * C) x T] tensor of Qs, Ks, and Vs.
        :return: an [N x (H * C) x T] tensor after attention.
        """
        bs, width, length = qkv.shape
        assert width % (3 * self.n_heads) == 0
        ch = width // (3 * self.n_heads)
        q, k, v = qkv.reshape(bs * self.n_heads, ch * 3, length).split(ch, dim=1)
        scale = 1 / math.sqrt(math.sqrt(ch))
        weight = th.einsum(
            "bct,bcs->bts", q * scale, k * scale
        )  # More stable with f16 than dividing afterwards
        weight = th.softmax(weight.float(), dim=-1).type(weight.dtype)
        a = th.einsum("bts,bcs->bct", weight, v)
        return a.reshape(bs, -1, length)


class QKVAttention(nn.Module):
    """
    A module which performs QKV attention and splits in a different order.
    """

    def __init__(self, n_heads):
        super().__init__()
        self.n_heads = n_heads

    def forward(self, qkv):
        """
        Apply QKV attention.

        :param qkv: an [N x (3 * H * C) x T] tensor of Qs, Ks, and Vs.
        :return: an [N x (H * C) x T] tensor after attention.
        """
        bs, width, length = qkv.shape
        assert width % (3 * self.n_heads) == 0
        ch = width // (3 * self.n_heads)
        q, k, v = qkv.chunk(3, dim=1)
        scale = 1 / math.sqrt(math.sqrt(ch))
        weight = th.einsum(
            "bct,bcs->bts",
            (q * scale).view(bs * self.n_heads, ch, length),
            (k * scale).view(bs * self.n_heads, ch, length),
        )  # More stable with f16 than dividing afterwards
        weight = th.softmax(weight.float(), dim=-1).type(weight.dtype)
        a = th.einsum("bts,bcs->bct", weight, v.reshape(bs * self.n_heads, ch, length))
        return a.reshape(bs, -1, length)

class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.embedding_l = nn.Linear(3, dim)

    def forward(self, time):
        device = time.device
        time = torch.cat([time, torch.cos(time), torch.sin(time)], dim=-1)
        return self.embedding_l(time)
    
class TimestepEmbedder(nn.Module):
    def __init__(self, latent_dim, time_embed_dim, sequence_pos_encoder, time_resolution=1000):
        super().__init__()
        self.time_resolution = time_resolution
        self.latent_dim = latent_dim
        self.sequence_pos_encoder = sequence_pos_encoder

        time_embed_dim = time_embed_dim
        self.time_embed = nn.Sequential(
            nn.Linear(self.latent_dim, time_embed_dim),
            nn.SiLU(),
            nn.Linear(time_embed_dim, time_embed_dim),
        )

    def forward(self, timesteps):
        timesteps = (timesteps * self.time_resolution).long()
        return self.time_embed(self.sequence_pos_encoder.pe[timesteps]).squeeze()


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)

        self.register_buffer("pe", pe)

    def forward(self, x):
        # not used in the final model
        x = x + self.pe[: x.shape[0], :]
        return self.dropout(x)

    
class COSMOS_FlowModel(nn.Module):
    """
    The full UNet model with attention and timestep embedding.

    :param in_channels: channels in the input Tensor.
    :param model_channels: base channel count for the model.
    :param out_channels: channels in the output Tensor.
    :param num_res_blocks: number of residual blocks per downsample.
    :param attention_resolutions: a collection of downsample rates at which
        attention will take place. May be a set, list, or tuple.
        For example, if this contains 4, then at 4x downsampling, attention
        will be used.
    :param dropout: the dropout probability.
    :param channel_mult: channel multiplier for each level of the UNet.
    :param conv_resample: if True, use learned convolutions for upsampling and
        downsampling.
    :param dims: determines if the signal is 1D, 2D, or 3D.
    :param num_classes: if specified (as an int), then this model will be
        class-conditional with `num_classes` classes.
    :param use_checkpoint: use gradient checkpointing to reduce memory usage.
    :param num_heads: the number of attention heads in each attention layer.
    :param num_heads_channels: if specified, ignore num_heads and instead use
                               a fixed channel width per attention head.
    :param num_heads_upsample: works with num_heads to set a different number
                               of heads for upsampling. Deprecated.
    :param use_scale_shift_norm: use a FiLM-like conditioning mechanism.
    :param resblock_updown: use residual blocks for up/downsampling.
    :param use_new_attention_order: use a different attention pattern for potentially
                                    increased efficiency.
    """

    def __init__(
        self,
        motion_args,
        image_size,
        in_channels,
        model_channels,
        out_channels,
        num_res_blocks,
        attention_resolutions,
        dropout=0,
        channel_mult=(1, 2, 4, 8),
        conv_resample=True,
        dims=2,
        num_classes=None,
        num_densecond_dim=1,
        use_checkpoint=False,
        use_fp16=False,
        num_heads=1,
        num_head_channels=-1,
        num_heads_upsample=-1,
        use_scale_shift_norm=False,
        resblock_updown=False,
        use_new_attention_order=False,
        padding_mode='zeros',
        padding=1,
        use_attention=False,
        use_qna=True,
        kernel_size=3,
    ):
        super().__init__()

        if num_heads_upsample == -1:
            num_heads_upsample = num_heads

        self.motion_args = motion_args
        self.image_size = image_size
        self.in_channels = in_channels
        self.model_channels = model_channels
        self.out_channels = out_channels
        self.num_res_blocks = num_res_blocks
        self.attention_resolutions = attention_resolutions
        self.dropout = dropout
        self.channel_mult = channel_mult
        self.conv_resample = conv_resample
        self.num_classes = num_classes
        self.num_densecond_dim = num_densecond_dim
        self.use_checkpoint = use_checkpoint
        self.dtype = th.float16 if use_fp16 else th.float32
        self.num_heads = num_heads
        self.num_head_channels = num_head_channels
        self.num_heads_upsample = num_heads_upsample
        self.sub_sample_mult = np.power(2, len(self.channel_mult))
        self.dims = dims
        self.padding_mode = padding_mode
        self.padding = padding


        for k, v in motion_args.items():
            setattr(self, k, v)

        time_embed_dim = model_channels * 4
        # self.time_embed = nn.Sequential(
        #     linear(model_channels, time_embed_dim),
        #     nn.SiLU(),
        #     linear(time_embed_dim, time_embed_dim),
        # )
        self.sequence_pos_encoder = PositionalEncoding(model_channels, self.dropout)        
        self.time_embed = TimestepEmbedder(
            model_channels, time_embed_dim, self.sequence_pos_encoder
        )

        # self.time_embed = TimestepEmbedder(
        #     self.latent_dim, self.sequence_pos_encoder
        # )

        if self.num_classes is not None:
            self.label_emb = nn.Embedding(num_classes, time_embed_dim)


            
        ch = input_ch = int(channel_mult[0] * model_channels)
        
        if self.num_densecond_dim > 0 :
            self.densecond_embed = nn.Sequential(
                nn.Conv1d(in_channels=num_densecond_dim, out_channels=ch, kernel_size=1, stride=1),
                nn.SiLU(),
                nn.Conv1d(in_channels=ch, out_channels=ch, kernel_size=1, stride=1)
            )
        self.input_blocks = nn.ModuleList(
            [TimestepEmbedSequential(conv_nd(dims, in_channels, ch, 3, padding=padding, padding_mode=self.padding_mode))]
        )
        self._feature_size = ch
        input_block_chans = [ch]
        ds = 1
        for level, mult in enumerate(channel_mult):
            for _ in range(num_res_blocks):
                layers = [
                    ResBlock(
                        ch,
                        time_embed_dim,
                        dropout,
                        out_channels=int(mult * model_channels),
                        dims=dims,
                        use_checkpoint=use_checkpoint,
                        use_scale_shift_norm=use_scale_shift_norm,
                        padding_mode=padding_mode,
                        padding=padding,
                    )
                ]
                ch = int(mult * model_channels)
                if use_attention:
                    if ds in attention_resolutions:
                        print(f'added attention block for input block, level {np.log2(ds) + 1}')
                        layers.append(
                            AttentionBlock(
                                ch,
                                use_checkpoint=use_checkpoint,
                                num_heads=num_heads,
                                num_head_channels=num_head_channels,
                                use_new_attention_order=use_new_attention_order,
                                use_qna=use_qna,
                                kernel_size=kernel_size,
                            )
                        )
                self.input_blocks.append(TimestepEmbedSequential(*layers))
                self._feature_size += ch
                input_block_chans.append(ch)
            if level != len(channel_mult) - 1:
                out_ch = ch
                self.input_blocks.append(
                    TimestepEmbedSequential(
                        ResBlock(
                            ch,
                            time_embed_dim,
                            dropout,
                            out_channels=out_ch,
                            dims=dims,
                            use_checkpoint=use_checkpoint,
                            use_scale_shift_norm=use_scale_shift_norm,
                            down=True,
                            padding_mode=padding_mode,
                            padding=padding,
                        )
                        # if resblock_updown
                        # else Downsample(
                            # ch, conv_resample, dims=dims, out_channels=out_ch, padding_mode=padding_mode, padding=padding
                        # )
                    )
                )
                ch = out_ch
                input_block_chans.append(ch)
                # ds *= 2
                self._feature_size += ch

        self.output_blocks = nn.ModuleList([])
        for level, mult in list(enumerate(channel_mult))[::-1]:
            for i in range(num_res_blocks + 1):
                ich = input_block_chans.pop()
                if level == len(channel_mult) - 1 and i == 0:
                    ich = 0
                layers = [
                    ResBlock(
                        ch + ich,
                        time_embed_dim,
                        dropout,
                        out_channels=int(model_channels * mult),
                        dims=dims,
                        use_checkpoint=use_checkpoint,
                        use_scale_shift_norm=use_scale_shift_norm,
                        padding_mode=padding_mode,
                        padding=padding,
                    )
                ]
                ch = int(model_channels * mult)
                if use_attention:
                    if ds in attention_resolutions:
                        print(f'added attention block for output block, level {np.log2(ds) + 1}')
                        layers.append(
                            AttentionBlock(
                                ch,
                                use_checkpoint=use_checkpoint,
                                num_heads=num_heads_upsample,
                                num_head_channels=num_head_channels,
                                use_new_attention_order=use_new_attention_order,
                                use_qna=use_qna,
                            )
                        )
                if level and i == num_res_blocks:
                    out_ch = ch
                    layers.append(
                        ResBlock(
                            ch,
                            time_embed_dim,
                            dropout,
                            out_channels=out_ch,
                            dims=dims,
                            use_checkpoint=use_checkpoint,
                            use_scale_shift_norm=use_scale_shift_norm,
                            up=True,
                            padding_mode=padding_mode,
                            padding=padding,
                        )
                        # if resblock_updown
                        # else Upsample(ch, conv_resample, dims=dims, out_channels=out_ch, padding_mode=padding_mode, padding=padding)
                    )
                    # ds //= 2
                self.output_blocks.append(TimestepEmbedSequential(*layers))
                self._feature_size += ch

        self.out = nn.Sequential(
            normalization(ch),
            nn.SiLU(),
            zero_module(conv_nd(dims, input_ch, out_channels, 3, padding=padding, padding_mode=padding_mode)),
        )
        
        self.out_cond = nn.Sequential(
            normalization(ch),
            nn.SiLU(),
            zero_module(conv_nd(dims, input_ch, 3, 3, padding=padding, padding_mode=padding_mode)),
        )
        
    def convert_to_fp16(self):
        """
        Convert the torso of the model to float16.
        """
        self.input_blocks.apply(convert_module_to_f16)
        self.output_blocks.apply(convert_module_to_f16)

    def convert_to_fp32(self):
        """
        Convert the torso of the model to float32.
        """
        self.input_blocks.apply(convert_module_to_f32)
        self.output_blocks.apply(convert_module_to_f32)

    def mask_cond(self, cond):
        bs, d = cond.shape
        if self.training and self.motion_args['cond_mask_prob'] > 0.:
            mask = th.bernoulli(th.ones(bs, device=cond.device) * self.motion_args['cond_mask_prob']).view(bs, 1)  # 1-> use null_cond, 0-> use real cond
            return cond * (1. - mask)
        else:
            return cond

    def forward(self, x, timesteps, y=None):
        """
        Apply the model to an input batch.

        :param x: an [N x C x ...] Tensor of inputs.
        :param timesteps: a 1-D batch of timesteps.
        :param y: an [N] Tensor of labels, if class-conditional.
        :return: an [N x C x ...] Tensor of outputs.
        """
        hs = []
        hs_cond = []
        y_densecond = y['frame_labels']
        # timesteps = timesteps.unsqueeze(1)
        if len(timesteps.shape) == 0:  # mainly in ODE sampling
            timesteps = repeat(timesteps.unsqueeze(0), "1 -> b", b=len(x))
        emb = self.time_embed(timesteps)  # [1, bs, d]
        if 'text' in self.motion_args['cond_mode']:
            emb += self.embed_text(self.mask_cond(y['text']))

        if self.motion_args['dataset'] == 'humanml':
            if self.dims == 1:
                self.n_samples, self.n_joints, self.n_feats, self.n_frames = x.shape
                x = x.reshape(self.n_samples, -1, self.n_frames)
        elif self.motion_args['dataset'] in ['mixamo', 'babel', 'bvh_general']:
            self.n_samples, self.n_joints, self.n_feats, self.n_frames = x.shape
            if self.dims == 1:
                x = x.reshape(self.n_samples, -1, self.n_frames)
            else:
                x = x.reshape(self.n_samples, -1, 1, self.n_frames)
            assert x.shape[1] == self.n_joints * self.n_feats
        else:
            raise 'dataset not supported yet.'

        if self.dims == 1:
            self.resid_frames = ((self.sub_sample_mult - x.shape[2:] % self.sub_sample_mult) % self.sub_sample_mult)[0]
            self.resid_joints = 0
        else:
            self.resid_joints, self.resid_frames = (self.sub_sample_mult - x.shape[2:] % self.sub_sample_mult) % self.sub_sample_mult

        if self.dims == 1:
            # pad frame axis with zeros
            x = th.cat(
                [x, th.zeros((x.shape[0], x.shape[1], self.resid_frames), device=x.device, dtype=x.dtype)],
                dim=-1)  # [bs, 1, J, n_frames] -> [bs, 1, J, padded n_frames]
            y_densecond = th.cat([y_densecond, th.zeros((x.shape[0], self.num_densecond_dim, self.resid_frames), device=x.device, dtype=y_densecond.dtype)],
                dim=-1)
        else:
            # pad frame axis with zeros
            x = th.cat([x, th.zeros((x.shape[0], x.shape[1], x.shape[2], self.resid_frames), device=x.device, dtype=x.dtype)],
                       dim=-1)  # [bs, 1, J, n_frames] -> [bs, 1, J, padded n_frames]

            # pad joint axis with zeros
            x = th.cat([x, th.zeros((x.shape[0], x.shape[1], self.resid_joints, x.shape[3]), device=x.device, dtype=x.dtype)],
                       dim=-2)  # [bs, 1, J, n_frames] -> [bs, 1, padded J, n_frames]

        if self.num_densecond_dim > 0:
            h_densecond = self.densecond_embed(y_densecond)


        if self.num_classes is not None:
            assert y.shape == (x.shape[0],)
            emb = emb + self.label_emb(y)
        h = x.type(self.dtype)  # if y is None else th.cat([x, y], dim=1).type(self.dtype)
        for module in self.input_blocks:
            h, h_densecond = module(h, emb, h_densecond)
            hs.append(h)
            hs_cond.append(h_densecond)
        for level, module in enumerate(self.output_blocks):
            if level == 0:
                h = hs.pop()
                h_densecond = hs_cond.pop()
            else:
                h = th.cat([h, hs.pop()], dim=1)
                h_densecond = th.cat([h_densecond, hs_cond.pop()], dim=1)
            h, h_densecond = module(h, emb, h_densecond)
        h = h.type(x.dtype)
        _out = self.out(h)
        _out_cond = self.out_cond(h_densecond)
        
        if self.resid_frames > 0:
            if self.dims == 1:
                _out = _out[:, :, :-self.resid_frames]
                _out_cond = _out_cond[:,:,:-self.resid_frames]
            else:
                _out = _out[:, :, :, :-self.resid_frames]

        if self.resid_joints > 0:
            _out = _out[:, :, :-self.resid_joints, :]

        if self.motion_args['dataset'] == 'humanml':
            if self.dims == 1:
                _out = _out.reshape(self.n_samples, self.n_joints, self.n_feats, self.n_frames)
        elif self.motion_args['dataset'] in ['mixamo', 'babel', 'bvh_general']:
            _out = _out.reshape(self.n_samples, self.n_joints, self.n_feats, self.n_frames)
            _out_cond = _out_cond.reshape(self.n_samples, 3, self.n_frames)
        else:
            raise 'dataset not supported yet.'    
        _out_cond = F.softmax(_out_cond, dim=1)
        return _out, _out_cond

class COSMOS_UNetModel(nn.Module):
    """
    The full UNet model with attention and timestep embedding.

    :param in_channels: channels in the input Tensor.
    :param model_channels: base channel count for the model.
    :param out_channels: channels in the output Tensor.
    :param num_res_blocks: number of residual blocks per downsample.
    :param attention_resolutions: a collection of downsample rates at which
        attention will take place. May be a set, list, or tuple.
        For example, if this contains 4, then at 4x downsampling, attention
        will be used.
    :param dropout: the dropout probability.
    :param channel_mult: channel multiplier for each level of the UNet.
    :param conv_resample: if True, use learned convolutions for upsampling and
        downsampling.
    :param dims: determines if the signal is 1D, 2D, or 3D.
    :param num_classes: if specified (as an int), then this model will be
        class-conditional with `num_classes` classes.
    :param use_checkpoint: use gradient checkpointing to reduce memory usage.
    :param num_heads: the number of attention heads in each attention layer.
    :param num_heads_channels: if specified, ignore num_heads and instead use
                               a fixed channel width per attention head.
    :param num_heads_upsample: works with num_heads to set a different number
                               of heads for upsampling. Deprecated.
    :param use_scale_shift_norm: use a FiLM-like conditioning mechanism.
    :param resblock_updown: use residual blocks for up/downsampling.
    :param use_new_attention_order: use a different attention pattern for potentially
                                    increased efficiency.
    """

    def __init__(
        self,
        motion_args,
        image_size,
        in_channels,
        model_channels,
        out_channels,
        num_res_blocks,
        attention_resolutions,
        dropout=0,
        channel_mult=(1, 2, 4, 8),
        conv_resample=True,
        dims=2,
        num_classes=None,
        num_densecond_dim=1,
        use_checkpoint=False,
        use_fp16=False,
        num_heads=1,
        num_head_channels=-1,
        num_heads_upsample=-1,
        use_scale_shift_norm=False,
        resblock_updown=False,
        use_new_attention_order=False,
        padding_mode='zeros',
        padding=1,
        use_attention=False,
        use_qna=True,
        kernel_size=3,
    ):
        super().__init__()

        if num_heads_upsample == -1:
            num_heads_upsample = num_heads

        self.motion_args = motion_args
        self.image_size = image_size
        self.in_channels = in_channels
        self.model_channels = model_channels
        self.out_channels = out_channels
        self.num_res_blocks = num_res_blocks
        self.attention_resolutions = attention_resolutions
        self.dropout = dropout
        self.channel_mult = channel_mult
        self.conv_resample = conv_resample
        self.num_classes = num_classes
        self.num_densecond_dim = num_densecond_dim
        self.use_checkpoint = use_checkpoint
        self.dtype = th.float16 if use_fp16 else th.float32
        self.num_heads = num_heads
        self.num_head_channels = num_head_channels
        self.num_heads_upsample = num_heads_upsample
        self.sub_sample_mult = np.power(2, len(self.channel_mult))
        self.dims = dims
        self.padding_mode = padding_mode
        self.padding = padding

        for k, v in motion_args.items():
            setattr(self, k, v)

        time_embed_dim = model_channels * 4
        self.time_embed = nn.Sequential(
            linear(model_channels, time_embed_dim),
            nn.SiLU(),
            linear(time_embed_dim, time_embed_dim),
        )


        if self.num_classes is not None:
            self.label_emb = nn.Embedding(num_classes, time_embed_dim)


            
        ch = input_ch = int(channel_mult[0] * model_channels)
        
        if self.num_densecond_dim > 0 :
            self.densecond_embed = nn.Sequential(
                nn.Conv1d(in_channels=num_densecond_dim, out_channels=ch, kernel_size=1, stride=1),
                nn.SiLU(),
                nn.Conv1d(in_channels=ch, out_channels=ch, kernel_size=1, stride=1)
            )
        self.input_blocks = nn.ModuleList(
            [TimestepEmbedSequential(conv_nd(dims, in_channels, ch, 3, padding=padding, padding_mode=self.padding_mode))]
        )
        self._feature_size = ch
        input_block_chans = [ch]
        ds = 1
        for level, mult in enumerate(channel_mult):
            for _ in range(num_res_blocks):
                layers = [
                    ResBlock(
                        ch,
                        time_embed_dim,
                        dropout,
                        out_channels=int(mult * model_channels),
                        dims=dims,
                        use_checkpoint=use_checkpoint,
                        use_scale_shift_norm=use_scale_shift_norm,
                        padding_mode=padding_mode,
                        padding=padding,
                    )
                ]
                ch = int(mult * model_channels)
                if use_attention:
                    if ds in attention_resolutions:
                        print(f'added attention block for input block, level {np.log2(ds) + 1}')
                        layers.append(
                            AttentionBlock(
                                ch,
                                use_checkpoint=use_checkpoint,
                                num_heads=num_heads,
                                num_head_channels=num_head_channels,
                                use_new_attention_order=use_new_attention_order,
                                use_qna=use_qna,
                                kernel_size=kernel_size,
                            )
                        )
                self.input_blocks.append(TimestepEmbedSequential(*layers))
                self._feature_size += ch
                input_block_chans.append(ch)
            if level != len(channel_mult) - 1:
                out_ch = ch
                self.input_blocks.append(
                    TimestepEmbedSequential(
                        ResBlock(
                            ch,
                            time_embed_dim,
                            dropout,
                            out_channels=out_ch,
                            dims=dims,
                            use_checkpoint=use_checkpoint,
                            use_scale_shift_norm=use_scale_shift_norm,
                            down=True,
                            padding_mode=padding_mode,
                            padding=padding,
                        )
                        # if resblock_updown
                        # else Downsample(
                            # ch, conv_resample, dims=dims, out_channels=out_ch, padding_mode=padding_mode, padding=padding
                        # )
                    )
                )
                ch = out_ch
                input_block_chans.append(ch)
                # ds *= 2
                self._feature_size += ch

        self.output_blocks = nn.ModuleList([])
        for level, mult in list(enumerate(channel_mult))[::-1]:
            for i in range(num_res_blocks + 1):
                ich = input_block_chans.pop()
                if level == len(channel_mult) - 1 and i == 0:
                    ich = 0
                layers = [
                    ResBlock(
                        ch + ich,
                        time_embed_dim,
                        dropout,
                        out_channels=int(model_channels * mult),
                        dims=dims,
                        use_checkpoint=use_checkpoint,
                        use_scale_shift_norm=use_scale_shift_norm,
                        padding_mode=padding_mode,
                        padding=padding,
                    )
                ]
                ch = int(model_channels * mult)
                if use_attention:
                    if ds in attention_resolutions:
                        print(f'added attention block for output block, level {np.log2(ds) + 1}')
                        layers.append(
                            AttentionBlock(
                                ch,
                                use_checkpoint=use_checkpoint,
                                num_heads=num_heads_upsample,
                                num_head_channels=num_head_channels,
                                use_new_attention_order=use_new_attention_order,
                                use_qna=use_qna,
                            )
                        )
                if level and i == num_res_blocks:
                    out_ch = ch
                    layers.append(
                        ResBlock(
                            ch,
                            time_embed_dim,
                            dropout,
                            out_channels=out_ch,
                            dims=dims,
                            use_checkpoint=use_checkpoint,
                            use_scale_shift_norm=use_scale_shift_norm,
                            up=True,
                            padding_mode=padding_mode,
                            padding=padding,
                        )
                        # if resblock_updown
                        # else Upsample(ch, conv_resample, dims=dims, out_channels=out_ch, padding_mode=padding_mode, padding=padding)
                    )
                    # ds //= 2
                self.output_blocks.append(TimestepEmbedSequential(*layers))
                self._feature_size += ch

        self.out = nn.Sequential(
            normalization(ch),
            nn.SiLU(),
            zero_module(conv_nd(dims, input_ch, out_channels, 3, padding=padding, padding_mode=padding_mode)),
        )
        
        self.out_cond = nn.Sequential(
            normalization(ch),
            nn.SiLU(),
            zero_module(conv_nd(dims, input_ch, 3, 3, padding=padding, padding_mode=padding_mode)),
        )
        
    def convert_to_fp16(self):
        """
        Convert the torso of the model to float16.
        """
        self.input_blocks.apply(convert_module_to_f16)
        self.output_blocks.apply(convert_module_to_f16)

    def convert_to_fp32(self):
        """
        Convert the torso of the model to float32.
        """
        self.input_blocks.apply(convert_module_to_f32)
        self.output_blocks.apply(convert_module_to_f32)

    def mask_cond(self, cond):
        bs, d = cond.shape
        if self.training and self.motion_args['cond_mask_prob'] > 0.:
            mask = th.bernoulli(th.ones(bs, device=cond.device) * self.motion_args['cond_mask_prob']).view(bs, 1)  # 1-> use null_cond, 0-> use real cond
            return cond * (1. - mask)
        else:
            return cond

    def forward(self, x, timesteps, y=None):
        """
        Apply the model to an input batch.

        :param x: an [N x C x ...] Tensor of inputs.
        :param timesteps: a 1-D batch of timesteps.
        :param y: an [N] Tensor of labels, if class-conditional.
        :return: an [N x C x ...] Tensor of outputs.
        """
        hs = []
        hs_cond = []
        y_densecond = y['frame_labels']
        emb = self.time_embed(timesteps)

        if 'text' in self.motion_args['cond_mode']:
            emb += self.embed_text(self.mask_cond(y['text']))

        if self.motion_args['dataset'] == 'humanml':
            if self.dims == 1:
                self.n_samples, self.n_joints, self.n_feats, self.n_frames = x.shape
                x = x.reshape(self.n_samples, -1, self.n_frames)
        elif self.motion_args['dataset'] in ['mixamo', 'babel', 'bvh_general']:
            self.n_samples, self.n_joints, self.n_feats, self.n_frames = x.shape
            if self.dims == 1:
                x = x.reshape(self.n_samples, -1, self.n_frames)
            else:
                x = x.reshape(self.n_samples, -1, 1, self.n_frames)
            assert x.shape[1] == self.n_joints * self.n_feats
        else:
            raise 'dataset not supported yet.'

        if self.dims == 1:
            self.resid_frames = ((self.sub_sample_mult - x.shape[2:] % self.sub_sample_mult) % self.sub_sample_mult)[0]
            self.resid_joints = 0
        else:
            self.resid_joints, self.resid_frames = (self.sub_sample_mult - x.shape[2:] % self.sub_sample_mult) % self.sub_sample_mult

        if self.dims == 1:
            # pad frame axis with zeros
            x = th.cat(
                [x, th.zeros((x.shape[0], x.shape[1], self.resid_frames), device=x.device, dtype=x.dtype)],
                dim=-1)  # [bs, 1, J, n_frames] -> [bs, 1, J, padded n_frames]
            y_densecond = th.cat([y_densecond, th.zeros((x.shape[0], self.num_densecond_dim, self.resid_frames), device=x.device, dtype=y_densecond.dtype)],
                dim=-1)
        else:
            # pad frame axis with zeros
            x = th.cat([x, th.zeros((x.shape[0], x.shape[1], x.shape[2], self.resid_frames), device=x.device, dtype=x.dtype)],
                       dim=-1)  # [bs, 1, J, n_frames] -> [bs, 1, J, padded n_frames]

            # pad joint axis with zeros
            x = th.cat([x, th.zeros((x.shape[0], x.shape[1], self.resid_joints, x.shape[3]), device=x.device, dtype=x.dtype)],
                       dim=-2)  # [bs, 1, J, n_frames] -> [bs, 1, padded J, n_frames]

        if self.num_densecond_dim > 0:
            h_densecond = self.densecond_embed(y_densecond)


        if self.num_classes is not None:
            assert y.shape == (x.shape[0],)
            emb = emb + self.label_emb(y)
        h = x.type(self.dtype)  # if y is None else th.cat([x, y], dim=1).type(self.dtype)
        for module in self.input_blocks:
            h, h_densecond = module(h, emb, h_densecond)
            hs.append(h)
            hs_cond.append(h_densecond)
        for level, module in enumerate(self.output_blocks):
            if level == 0:
                h = hs.pop()
                h_densecond = hs_cond.pop()
            else:
                h = th.cat([h, hs.pop()], dim=1)
                h_densecond = th.cat([h_densecond, hs_cond.pop()], dim=1)
            h, h_densecond = module(h, emb, h_densecond)
        h = h.type(x.dtype)
        _out = self.out(h)
        _out_cond = self.out_cond(h_densecond)
        
        if self.resid_frames > 0:
            if self.dims == 1:
                _out = _out[:, :, :-self.resid_frames]
                _out_cond = _out_cond[:,:,:-self.resid_frames]
            else:
                _out = _out[:, :, :, :-self.resid_frames]

        if self.resid_joints > 0:
            _out = _out[:, :, :-self.resid_joints, :]

        if self.motion_args['dataset'] == 'humanml':
            if self.dims == 1:
                _out = _out.reshape(self.n_samples, self.n_joints, self.n_feats, self.n_frames)
        elif self.motion_args['dataset'] in ['mixamo', 'babel', 'bvh_general']:
            _out = _out.reshape(self.n_samples, self.n_joints, self.n_feats, self.n_frames)
            _out_cond = _out_cond.reshape(self.n_samples, 3, self.n_frames)
        else:
            raise 'dataset not supported yet.'    
        _out_cond = F.softmax(_out_cond, dim=1)
        return _out, _out_cond
