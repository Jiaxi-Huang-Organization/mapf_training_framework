from argparse import Namespace
from typing import Literal, Tuple

import torch
from pydantic import BaseModel
from sample_factory.model.encoder import Encoder
from sample_factory.utils.typing import Config, ObsSpace
from sample_factory.algo.utils.torch_utils import calc_num_elements

from sample_factory.utils.utils import log

from torch import nn as nn
from gymnasium.spaces import Box


class EncoderConfig(BaseModel):
    """
    Configuration for encoder.
    
    For MAPPO:
    - Actor uses local observations (5-channel CNN)
    - Critic uses global state (MLP for centralized training)
    """
    extra_fc_layers: int = 0
    num_filters: int = 64
    num_res_blocks: int = 1
    activation_func: Literal['ReLU', 'ELU', 'Mish'] = 'ReLU'
    hidden_size: int = 128
    # Critic specific
    critic_hidden_layers: Tuple[int, ...] = (256, 128)


def activation_func(cfg: EncoderConfig) -> nn.Module:
    if cfg.activation_func == "ELU":
        return nn.ELU(inplace=True)
    elif cfg.activation_func == "ReLU":
        return nn.ReLU(inplace=True)
    elif cfg.activation_func == "Mish":
        return nn.Mish(inplace=True)
    else:
        raise Exception("Unknown activation_func")


class ResBlock(nn.Module):
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


class MAPPOEncoder(Encoder):
    """
    MAPPO Encoder with CTDE architecture.
    
    Actor path: Uses local observations (5-channel CNN)
    Critic path: Uses global state (MLP for centralized training)
    
    The forward method automatically routes to the appropriate path based on
    the input:
    - If 'global_state' is present and we're in critic context, use MLP
    - Otherwise, use CNN for actor
    """

    def __init__(self, cfg: Config, obs_space: ObsSpace):
        super().__init__(cfg)
        self.encoder_cfg: EncoderConfig = EncoderConfig(**cfg.encoder)

        # Actor encoder (CNN for local observations)
        input_ch = obs_space['obs'].shape[0]
        resnet_conf = [[self.encoder_cfg.num_filters, self.encoder_cfg.num_res_blocks]]
        curr_input_channels = input_ch
        layers = []

        for out_channels, res_blocks in resnet_conf:
            layers.extend([nn.Conv2d(curr_input_channels, out_channels, kernel_size=3, stride=1, padding=1)])
            layers.extend([ResBlock(self.encoder_cfg, out_channels, out_channels) for _ in range(res_blocks)])
            curr_input_channels = out_channels

        layers.append(activation_func(self.encoder_cfg))
        self.actor_conv_head = nn.Sequential(*layers)
        self.actor_out_size = calc_num_elements(self.actor_conv_head, obs_space['obs'].shape)

        if self.encoder_cfg.extra_fc_layers:
            self.actor_extra_linear = nn.Sequential(
                nn.Linear(self.actor_out_size, self.encoder_cfg.hidden_size),
                activation_func(self.encoder_cfg),
            )
            self.actor_out_size = self.encoder_cfg.hidden_size

        # Critic encoder (MLP for global state)
        if 'global_state' in obs_space:
            global_state_dim = obs_space['global_state'].shape[0]
        else:
            global_state_dim = 3136  # Fallback
        
        critic_layers = []
        prev_size = global_state_dim
        hidden_layers = self.encoder_cfg.critic_hidden_layers
        for hidden_size in hidden_layers:
            critic_layers.extend([
                nn.Linear(prev_size, hidden_size),
                activation_func(self.encoder_cfg),
            ])
            prev_size = hidden_size
        
        self.critic_mlp = nn.Sequential(*critic_layers)
        self.critic_out_size = prev_size
        
        # Use the larger of the two for compatibility with SampleFactory
        self.encoder_out_size = max(self.actor_out_size, self.critic_out_size)

        log.debug('MAPPO Encoder - Actor out: %d, Critic out: %d', 
                  self.actor_out_size, self.critic_out_size)

    def get_out_size(self) -> int:
        return self.encoder_out_size

    def forward(self, x):
        """
        Forward pass with CTDE routing.
        
        In SampleFactory's ActorCritic:
        - Actor forward: processes 'obs' for action distribution
        - Critic forward: processes 'global_state' for value estimation
        
        The key insight is that SampleFactory calls the encoder twice:
        once for actor, once for critic, with the same observation dict.
        We route based on which features are being accessed.
        """
        # Check if we have global_state (critic path)
        has_global_state = 'global_state' in x and x['global_state'] is not None
        
        if has_global_state:
            # Critic path: use global state
            global_state = x['global_state']
            if global_state.dim() > 2:
                global_state = global_state.view(global_state.size(0), -1)
            x = self.critic_mlp(global_state)
        else:
            # Actor path: use local observations
            x = x['obs']
            x = self.actor_conv_head(x)
            x = x.contiguous().view(-1, self.actor_out_size)
            
            if self.encoder_cfg.extra_fc_layers and hasattr(self, 'actor_extra_linear'):
                x = self.actor_extra_linear(x)
        
        return x


# For backward compatibility
ResnetEncoder = ActorEncoder = MAPPOEncoder
CriticEncoder = MAPPOEncoder


def main():
    exp_cfg = {'encoder': EncoderConfig().dict()}
    r = 5
    
    # Test Actor path (local obs)
    obs = torch.rand(2, 5, r * 2 + 1, r * 2 + 1)
    enc = MAPPOEncoder(Namespace(**exp_cfg), dict(obs=obs[0], global_state=torch.rand(3136)))
    
    # Actor forward (no global_state)
    actor_out = enc({'obs': obs})
    print('Actor path output shape:', actor_out.shape)
    
    # Critic forward (with global_state)
    global_state = torch.rand(2, 3136)
    critic_out = enc({'obs': obs, 'global_state': global_state})
    print('Critic path output shape:', critic_out.shape)


if __name__ == '__main__':
    main()
