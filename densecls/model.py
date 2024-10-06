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
    avg_pool_nd,
    zero_module,
    normalization,
)
from models.qna import FusedQnA1d

import torch
import torch.nn as nn
import torch.nn.functional as F


class OperateBlock(nn.Module):
    """
    Any module where forward() takes timestep embeddings as a second argument.
    """

    @abstractmethod
    def forward(self, x):
        """
        Apply the module to `x` given `emb` timestep embeddings.
        """


class OperateSequential(nn.Sequential, OperateBlock):
    """
    A sequential module that passes timestep embeddings to the children that
    support it as an extra input.
    """

    def forward(self, x):
        for layer in self:
            x = layer(x)
        return x


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


class ResBlock(OperateSequential):
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
        self.dropout = dropout
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.use_scale_shift_norm = use_scale_shift_norm
        self.use_checkpoint = use_checkpoint

        self.in_layers = nn.Sequential(
            normalization(channels),
            nn.SiLU(),
            conv_nd(dims, channels, self.out_channels, 3, padding=padding, padding_mode=padding_mode),
        )
        
        self.updown = up or down

        if up:
            self.h_upd = Upsample(channels, False, dims, padding_mode=padding_mode, padding=padding)
            self.x_upd = Upsample(channels, False, dims, padding_mode=padding_mode, padding=padding)
        elif down:
            self.h_upd = Downsample(channels, False, dims, padding_mode=padding_mode, padding=padding)
            self.x_upd = Downsample(channels, False, dims, padding_mode=padding_mode, padding=padding)
        else:
            self.h_upd = self.x_upd = nn.Identity()

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
        elif use_conv:
            self.skip_connection = conv_nd(
                dims, channels, self.out_channels, 3, padding=padding, padding_mode=padding_mode
            )
        else:
            self.skip_connection = conv_nd(dims, channels, self.out_channels, 1)

    def forward(self, x):
        """
        Apply the block to a Tensor, conditioned on a timestep embedding.

        :param x: an [N x C x ...] Tensor of features.
        :param emb: an [N x emb_channels] Tensor of timestep embeddings.
        :return: an [N x C x ...] Tensor of outputs.
        """
        return checkpoint(
            self._forward, (x, ), self.parameters(), self.use_checkpoint
        )


    def _forward(self, x):
        if self.updown:
            in_rest, in_conv = self.in_layers[:-1], self.in_layers[-1]
            h = in_rest(x)
            h = self.h_upd(h)
            x = self.x_upd(x)
            h = in_conv(h)
        else:
            h = self.in_layers(x)
        if self.use_scale_shift_norm:
            out_norm, out_rest = self.out_layers[0], self.out_layers[1:]
            h = out_norm(h) 
            h = out_rest(h)
        else:
            h = self.out_layers(h)
        return (self.skip_connection(x) + h)

    
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
        """
        Apply the block to a Tensor, conditioned on a timestep embedding.

        :param x: an [N x C x ...] Tensor of features.
        :param emb: an [N x emb_channels] Tensor of timestep embeddings.
        :return: an [N x C x ...] Tensor of outputs.
        """
        return checkpoint(
            self._forward, (x, ), self.parameters(), self.use_checkpoint
        )

            
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


class DenseCLS_UNet(nn.Module):
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
        in_channels,
        model_channels=256,
        out_channels=3,
        num_res_blocks=1,
        attention_resolutions=(1,2),
        dropout=0,
        channel_mult=(1,),
        conv_resample=True,
        dims=1,
        num_classes=3,
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
        use_qna=False,
        kernel_size=3,
    ):
        super().__init__()

        if num_heads_upsample == -1:
            num_heads_upsample = num_heads

        self.in_channels = in_channels
        self.model_channels = model_channels
        self.out_channels = out_channels
        self.num_res_blocks = num_res_blocks
        self.attention_resolutions = attention_resolutions
        self.dropout = dropout
        self.channel_mult = channel_mult
        self.conv_resample = conv_resample
        self.num_classes = num_classes
        self.use_checkpoint = use_checkpoint
        self.dtype = th.float16 if use_fp16 else th.float32
        self.num_heads = num_heads
        self.num_head_channels = num_head_channels
        self.num_heads_upsample = num_heads_upsample
        self.sub_sample_mult = np.power(2, len(self.channel_mult))
        self.dims = dims
        self.padding_mode = padding_mode
        self.padding = padding
            
        ch = input_ch = int(channel_mult[0] * model_channels)
        self.input_blocks = nn.ModuleList(
            [OperateSequential(conv_nd(dims, in_channels, ch, 3, padding=padding, padding_mode=self.padding_mode))]
        )
        self._feature_size = ch
        input_block_chans = [ch]
        ds = 1
        for level, mult in enumerate(channel_mult):
            for _ in range(num_res_blocks):
                layers = [
                    ResBlock(
                        ch,
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
                self.input_blocks.append(OperateSequential(*layers))
                self._feature_size += ch
                input_block_chans.append(ch)
            if level != len(channel_mult) - 1:
                out_ch = ch
                self.input_blocks.append(
                    OperateSequential(
                        ResBlock(
                            ch,
                            dropout,
                            out_channels=out_ch,
                            dims=dims,
                            use_checkpoint=use_checkpoint,
                            use_scale_shift_norm=use_scale_shift_norm,
                            down=True,
                            padding_mode=padding_mode,
                            padding=padding,
                        )
                        if resblock_updown
                        else Downsample(
                            ch, conv_resample, dims=dims, out_channels=out_ch, padding_mode=padding_mode, padding=padding
                        )
                    )
                )
                ch = out_ch
                input_block_chans.append(ch)
                ds *= 2
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
                            dropout,
                            out_channels=out_ch,
                            dims=dims,
                            use_checkpoint=use_checkpoint,
                            use_scale_shift_norm=use_scale_shift_norm,
                            up=True,
                            padding_mode=padding_mode,
                            padding=padding,
                        )
                        if resblock_updown
                        else Upsample(ch, conv_resample, dims=dims, out_channels=out_ch, padding_mode=padding_mode, padding=padding)
                    )
                    ds //= 2
                self.output_blocks.append(OperateSequential(*layers))
                self._feature_size += ch

        self.out = nn.Sequential(
            normalization(ch),
            nn.SiLU(),
            zero_module(conv_nd(dims, input_ch, out_channels, 3, padding=padding, padding_mode=padding_mode)),
        )

        
    def convert_to_fp16(self):
        """x
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

    def forward(self, x, y=None):
        """
        Apply the model to an input batch.

        :param x: an [N x C x ...] Tensor of inputs.
        :param timesteps: a 1-D batch of timesteps.
        :param y: an [N] Tensor of labels, if class-conditional.
        :return: an [N x C x ...] Tensor of outputs.
        """
        hs = []

        self.n_samples, self.n_joints, self.n_feats, self.n_frames = x.shape
        if self.dims == 1:
            x = x.reshape(self.n_samples, -1, self.n_frames)
        else:
            x = x.reshape(self.n_samples, -1, 1, self.n_frames)
        assert x.shape[1] == self.n_joints * self.n_feats

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
        else:
            # pad frame axis with zeros
            x = th.cat([x, th.zeros((x.shape[0], x.shape[1], x.shape[2], self.resid_frames), device=x.device, dtype=x.dtype)],
                       dim=-1)  # [bs, 1, J, n_frames] -> [bs, 1, J, padded n_frames]

            # pad joint axis with zeros
            x = th.cat([x, th.zeros((x.shape[0], x.shape[1], self.resid_joints, x.shape[3]), device=x.device, dtype=x.dtype)],
                       dim=-2)  # [bs, 1, J, n_frames] -> [bs, 1, padded J, n_frames]
        h = x.type(self.dtype)  # if y is None else th.cat([x, y], dim=1).type(self.dtype)
        for module in self.input_blocks:
            h = module(h)
            hs.append(h)
        for level, module in enumerate(self.output_blocks):
            if level == 0:
                h = hs.pop()
            else:
                h = th.cat([h, hs.pop()], dim=1)
            h = module(h)
        h = h.type(x.dtype)

        _out = self.out(h)
        

        if self.resid_frames > 0:
            if self.dims == 1:
                _out = _out[:, :, :-self.resid_frames]
            else:
                _out = _out[:, :, :, :-self.resid_frames]

        if self.resid_joints > 0:
            _out = _out[:, :, :-self.resid_joints, :]

        if self.num_classes == 3 : 
            _out = F.softmax(_out, dim=1)
        elif self.num_classes == 1 : 
            _out = th.sigmoid(_out)
        else : 
            assert('something wrong')
        _out = _out.permute(0,2,1)
        _out = _out.reshape(-1,self.num_classes)
        return _out


