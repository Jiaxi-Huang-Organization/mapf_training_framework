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
        activation_func: Activation function ('ReLU', 'ELU', 'Mish')
        hidden_size: Hidden size for extra FC layers
        follower_checkpoint: Path to pre-trained follower checkpoint
        freeze_follower_encoder: Whether to freeze follower encoder parameters
    """
    extra_fc_layers: int = 0
    num_filters: int = 64
    num_res_blocks: int = 1
    activation_func: Literal['ReLU', 'ELU', 'Mish'] = 'ReLU'
    hidden_size: int = 128
    follower_checkpoint: Optional[str] = None
    freeze_follower_encoder: bool = True


def activation_func(cfg: EncoderConfig) -> nn.Module:
    """Returns activation function based on config."""
    if cfg.activation_func == "ELU":
        return nn.ELU(inplace=True)
    elif cfg.activation_func == "ReLU":
        return nn.ReLU(inplace=True)
    elif cfg.activation_func == "Mish":
        return nn.Mish(inplace=True)
    else:
        raise Exception(f"Unknown activation_func: {cfg.activation_func}")


class ResBlock(nn.Module):
    """Residual block for encoder."""

    def __init__(self, cfg: EncoderConfig, input_ch: int, output_ch: int):
        super().__init__()
        layers = [
            activation_func(cfg),
            nn.Conv2d(input_ch, output_ch, kernel_size=3, stride=1, padding=1),
            activation_func(cfg),
            nn.Conv2d(output_ch, output_ch, kernel_size=3, stride=1, padding=1),
        ]
        self.res_block_core = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        out = self.res_block_core(x)
        return out + identity


class ResnetEncoder(Encoder):
    """
    ResNet-based encoder for spatial features.
    
    This encoder is identical to follower's encoder, allowing direct
    weight loading and freezing.
    
    Input: dict with 'obs' key containing spatial features
           shape: (batch, channels, H, W)
           channels: obstacles, agents, charges, target, battery (for charger)
    Output: encoded feature vector
            shape: (batch, encoder_out_size)
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
        
        # Load and freeze follower weights if checkpoint provided
        if self.encoder_cfg.follower_checkpoint:
            self._load_follower_weights(self.encoder_cfg.follower_checkpoint)
        
        if self.encoder_cfg.freeze_follower_encoder:
            self._freeze_encoder()

    def _find_follower_checkpoint(self, checkpoint_path: str) -> Tuple[str, dict]:
        """
        Find and load follower checkpoint, similar to follower inference.
        
        Args:
            checkpoint_path: Path to follower checkpoint directory or file
            
        Returns:
            Tuple of (checkpoint_path, checkpoint_dict)
        """
        # If path is a directory, find the latest checkpoint
        if os.path.isdir(checkpoint_path):
            # Find config.json to verify this is a valid checkpoint directory
            config_path = os.path.join(checkpoint_path, 'config.json')
            if not os.path.exists(config_path):
                raise FileNotFoundError(f"config.json not found in {checkpoint_path}")
            
            # Find latest checkpoint file
            checkpoints = glob.glob(os.path.join(checkpoint_path, 'checkpoint_p0', 'checkpoint_*'))
            if not checkpoints:
                # Try alternative pattern
                checkpoints = glob.glob(os.path.join(checkpoint_path, '*.pt'))
            
            if not checkpoints:
                raise FileNotFoundError(f"No checkpoint files found in {checkpoint_path}")
            
            checkpoints.sort()
            checkpoint_path = checkpoints[-1]
            log.info(f"Using latest follower checkpoint: {checkpoint_path}")
        
        # Load checkpoint (without weights_only for older PyTorch compatibility)
        checkpoint_dict = torch.load(checkpoint_path, map_location='cpu')
        return checkpoint_path, checkpoint_dict

    def _load_follower_weights(self, checkpoint_path: str):
        """
        Load weights from follower checkpoint into this encoder.
        
        This method:
        1. Loads the follower checkpoint
        2. Extracts encoder weights from the checkpoint
        3. Loads weights with strict=False to handle channel differences
        
        Args:
            checkpoint_path: Path to follower checkpoint directory or file
        """
        try:
            checkpoint_path, checkpoint_dict = self._find_follower_checkpoint(checkpoint_path)
            
            # Load training config from checkpoint
            config_path = os.path.join(os.path.dirname(checkpoint_path), 'config.json')
            if os.path.exists(config_path):
                with open(config_path, 'r') as f:
                    follower_config = json.load(f)
                log.info(f"Loaded follower config from {config_path}")
                
                # Check original encoder configuration
                encoder_cfg = follower_config.get('encoder', {})
                orig_num_filters = encoder_cfg.get('num_filters', 64)
                orig_num_res_blocks = encoder_cfg.get('num_res_blocks', 1)
                
                # Verify compatibility
                if orig_num_filters != self.encoder_cfg.num_filters:
                    log.warning(
                        f"Follower num_filters ({orig_num_filters}) differs from "
                        f"charger config ({self.encoder_cfg.num_filters})"
                    )
                if orig_num_res_blocks != self.encoder_cfg.num_res_blocks:
                    log.warning(
                        f"Follower num_res_blocks ({orig_num_res_blocks}) differs from "
                        f"charger config ({self.encoder_cfg.num_res_blocks})"
                    )
            
            # Extract model state dict
            if 'model' in checkpoint_dict:
                state_dict = checkpoint_dict['model']
            elif 'actor_critic' in checkpoint_dict:
                state_dict = checkpoint_dict['actor_critic']
            else:
                state_dict = checkpoint_dict
            
            # Extract encoder weights (keys starting with 'encoder.')
            encoder_state = {}
            for key, value in state_dict.items():
                if key.startswith('encoder.'):
                    new_key = key[8:]  # Remove 'encoder.' prefix
                    encoder_state[new_key] = value
            
            if not encoder_state:
                log.warning(f"No encoder weights found in {checkpoint_path}")
                return
            
            # Load weights with strict=False to allow partial loading
            load_result = self.load_state_dict(encoder_state, strict=False)
            
            if load_result.missing_keys:
                log.debug(f"Missing keys in encoder: {load_result.missing_keys[:5]}...")
            if load_result.unexpected_keys:
                log.debug(f"Unexpected keys in encoder: {load_result.unexpected_keys[:5]}...")
            
            log.info(f"Successfully loaded follower encoder weights from {checkpoint_path}")
            
        except Exception as e:
            log.warning(f"Failed to load follower weights: {e}")
            import traceback
            log.debug(traceback.format_exc())

    def _freeze_encoder(self):
        """Freeze encoder parameters."""
        for param in self.parameters():
            param.requires_grad = False
        self.eval()
        log.info("Encoder frozen (follower weights)")

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
    
    # Test with charger observation (5 channels)
    obs = torch.rand(2, 5, r * 2 + 1, r * 2 + 1)
    q_obs = {'obs': obs}
    
    re = ResnetEncoder(Namespace(**exp_cfg), dict(obs=obs[0]))
    output = re(q_obs)
    print(f"Encoder output shape: {output.shape}")


if __name__ == '__main__':
    main()
