"""
Charger APPO Model

Key features:
1. ResnetEncoder for spatial features (same as follower)
2. Support for loading and freezing follower encoder weights
3. Scalar features (xy, target_xy, battery, charge_xy) handled separately
"""
from argparse import Namespace
from typing import Literal, Tuple, List, Optional
import os
import glob
import json

import torch
import torch.nn as nn
from pydantic import BaseModel
from sample_factory.model.encoder import Encoder
from sample_factory.utils.typing import Config, ObsSpace
from sample_factory.algo.utils.torch_utils import calc_num_elements
from sample_factory.utils.utils import log


class EncoderConfig(BaseModel):
    """
    Configuration for the ResNet encoder.
    
    Args:
        extra_fc_layers: Number of extra fully connected layers
        num_filters: Number of convolutional filters
        num_res_blocks: Number of residual blocks per stage
        activation_func: Activation function ('ReLU', 'ELU', 'Mish','GeLU')
        hidden_size: Hidden size for extra FC layers
    """
    extra_fc_layers: int = 0
    num_filters: int = 64
    num_res_blocks: int = 1
    activation_func: Literal['ReLU', 'ELU', 'Mish','GELU'] = 'GELU'
    hidden_size: int = 128


def activation_func(cfg: EncoderConfig) -> nn.Module:
    """Returns activation function based on config."""
    if cfg.activation_func == "ELU":
        return nn.ELU(inplace=True)
    elif cfg.activation_func == "ReLU":
        return nn.ReLU(inplace=True)
    elif cfg.activation_func == "Mish":
        return nn.Mish(inplace=True)
    elif cfg.activation_func == "GELU":
        return nn.GELU()
    else:
        raise Exception(f"Unknown activation_func: {cfg.activation_func}")


class ResBlock(nn.Module):
    """
    Residual block in the encoder.

    Args:
        cfg (EncoderConfig): Encoder configuration.
        input_ch (int): Input channel size.
        output_ch (int): Output channel size.
    """

    def __init__(self, cfg: EncoderConfig, input_ch, output_ch):
        super().__init__()

        layers = [
            activation_func(cfg),
            nn.Conv2d(input_ch, output_ch, kernel_size=3, stride=1, padding=1),
            activation_func(cfg),
            nn.Conv2d(output_ch, output_ch, kernel_size=3, stride=1, padding=1),
        ]

        self.res_block_core = nn.Sequential(*layers)

    def forward(self, x):
        identity = x
        out = self.res_block_core(x)
        out = out + identity
        return out


class ResnetEncoder(Encoder):
    """
    ResNet-based encoder.

    Args:
        cfg (Config): Configuration.
        obs_space (ObsSpace): Observation space.
    """

    def __init__(self, cfg: Config, obs_space: ObsSpace):
        super().__init__(cfg)
        self.encoder_cfg: EncoderConfig = EncoderConfig(**cfg.encoder)

        input_ch = obs_space['obs'].shape[0]
        resnet_conf = [[self.encoder_cfg.num_filters, self.encoder_cfg.num_res_blocks]]
        curr_input_channels = input_ch
        layers = []

        for out_channels, res_blocks in resnet_conf:
            layers.extend([
                nn.Conv2d(curr_input_channels, out_channels, kernel_size=3, stride=1, padding=1)
            ])
            layers.extend([
                ResBlock(self.encoder_cfg, out_channels, out_channels) 
                for _ in range(res_blocks)
            ])
            curr_input_channels = out_channels

        layers.append(activation_func(self.encoder_cfg))
        self.conv_head = nn.Sequential(*layers)
        self.conv_head_out_size = calc_num_elements(self.conv_head, obs_space['obs'].shape)
        self.encoder_out_size = self.conv_head_out_size

        if self.encoder_cfg.extra_fc_layers:
            self.extra_linear = nn.Sequential(
                nn.Linear(self.encoder_out_size, self.encoder_cfg.hidden_size),
                activation_func(self.encoder_cfg),
            )
            self.encoder_out_size = self.encoder_cfg.hidden_size

        log.debug(f'Convolutional layer output size: {self.conv_head_out_size}')

    def get_out_size(self) -> int:
        return self.encoder_out_size

    def forward(self, x: dict) -> torch.Tensor:
        """
        Forward pass through encoder.
        
        Args:
            x: Observation dict with 'obs' key
               obs shape: (batch, channels, H, W)
               
        Returns:
            Encoded features: (batch, encoder_out_size)
        """
        obs = x['obs']
        x = self.conv_head(obs)
        x = x.contiguous().view(-1, self.conv_head_out_size)

        if self.encoder_cfg.extra_fc_layers:
            x = self.extra_linear(x)

        return x


def main():
    """Test the encoder."""
    exp_cfg = {'encoder': EncoderConfig().dict()}
    r = 5
    
    # Test with charger observation (2 channels:(obstacles+path)+(agent+other agents))
    obs = torch.rand(1, 2, r * 2 + 1, r * 2 + 1)
    q_obs = {'obs': obs}
    
    re = ResnetEncoder(Namespace(**exp_cfg), dict(obs=obs[0]))
    output = re(q_obs)
    print(f"Encoder output shape: {output.shape}")


if __name__ == '__main__':
    main()
