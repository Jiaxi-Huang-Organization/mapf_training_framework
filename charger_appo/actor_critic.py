"""
Charger APPO Actor-Critic

Key features:
1. Uses ResnetEncoder for spatial features (same as follower)
2. Processes scalar features: xy, target_xy, battery, charge_xy
3. Supports freezing encoder while training actor/critic heads
4. Compatible with Sample Factory's PPO training
"""
from typing import Dict, Tuple, Optional, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from sample_factory.utils.typing import Config, ObsSpace, ActionSpace
from sample_factory.utils.utils import log

from charger_appo.model import ResnetEncoder, EncoderConfig


class ScalarFeaturesMLP(nn.Module):
    """
    MLP for processing scalar features.
    
    Input scalar features:
    - xy: (2,) - agent position
    - target_xy: (2,) - target position  
    - battery: (1,) - normalized battery level
    - charge_xy: (2,) or (2*num_charges,) - nearest charger position
    
    Total: 7+ dimensions depending on configuration
    """
    
    def __init__(
        self, 
        input_size: int, 
        hidden_size: int = 64,
        output_size: int = 64,
        num_layers: int = 2,
        activation: str = 'ReLU'
    ):
        super().__init__()
        
        # Build MLP
        layers = []
        curr_size = input_size
        
        for i in range(num_layers - 1):
            layers.extend([
                nn.Linear(curr_size, hidden_size),
                self._get_activation(activation),
            ])
            curr_size = hidden_size
        
        layers.append(nn.Linear(curr_size, output_size))
        
        self.mlp = nn.Sequential(*layers)
        self.output_size = output_size
    
    def _get_activation(self, name: str) -> nn.Module:
        if name == 'ReLU':
            return nn.ReLU(inplace=True)
        elif name == 'ELU':
            return nn.ELU(inplace=True)
        elif name == 'Mish':
            return nn.Mish(inplace=True)
        else:
            return nn.ReLU(inplace=True)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x)


class ChargerActorCritic(nn.Module):
    """
    Charger Actor-Critic for battery-aware navigation.

    Architecture:
    1. ResnetEncoder processes spatial features (obstacles, agents, etc.)
    2. ScalarFeaturesMLP processes scalar features (xy, target_xy, battery, charge_xy)
    3. Features are concatenated and passed to actor/critic heads

    The encoder can be frozen to preserve follower behavior while
    fine-tuning the actor/critic for battery-aware decisions.

    Note: This is a nn.Module that follows Sample Factory's ActorCritic interface.
    We need to implement model_to_device() for Sample Factory compatibility.
    """

    def __init__(
        self,
        cfg: Config,
        obs_space: ObsSpace,
        action_space: ActionSpace
    ):
        super().__init__()
        
        self.encoder_cfg: EncoderConfig = EncoderConfig(**cfg.encoder)
        
        # 1. Encoder for spatial features
        self.encoder = ResnetEncoder(cfg, obs_space)
        encoder_out_size = self.encoder.get_out_size()
        
        # 2. Determine scalar feature sizes
        self.use_scalar_features = getattr(cfg.encoder, 'use_scalar_features', True)
        self.use_charger_xy_input = getattr(cfg.encoder, 'use_charger_xy_input', True)
        
        # Calculate scalar input size
        scalar_input_size = 0
        if self.use_scalar_features:
            # xy (2) + target_xy (2) + battery (1) = 5
            scalar_input_size = 5
            if self.use_charger_xy_input:
                # Add charge_xy (2) for nearest charger
                scalar_input_size += 2
        
        # 3. Scalar features MLP
        self.scalar_mlp = None
        scalar_out_size = 0
        if self.use_scalar_features and scalar_input_size > 0:
            self.scalar_mlp = ScalarFeaturesMLP(
                input_size=scalar_input_size,
                hidden_size=64,
                output_size=64,
                num_layers=2,
                activation='ReLU'
            )
            scalar_out_size = self.scalar_mlp.output_size
        
        # 4. Combined feature size
        combined_size = encoder_out_size + scalar_out_size
        
        # 5. Actor head (outputs action logits)
        self.num_actions = action_space.n
        self.actor = nn.Sequential(
            nn.Linear(combined_size, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, self.num_actions)
        )
        
        # 6. Critic head (outputs value estimate)
        self.critic = nn.Sequential(
            nn.Linear(combined_size, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 1)
        )
        
        # Initialize weights
        self._initialize_weights()
        
        log.info(f"ChargerActorCritic initialized:")
        log.info(f"  Encoder output: {encoder_out_size}")
        log.info(f"  Scalar output: {scalar_out_size}")
        log.info(f"  Combined: {combined_size}")
        log.info(f"  Num actions: {self.num_actions}")

    def _initialize_weights(self):
        """Initialize actor and critic weights."""
        for module in [self.actor, self.critic]:
            for m in module.modules():
                if isinstance(m, nn.Linear):
                    nn.init.orthogonal_(m.weight, gain=1.0)
                    nn.init.constant_(m.bias, 0.0)
        
        if self.scalar_mlp is not None:
            for m in self.scalar_mlp.modules():
                if isinstance(m, nn.Linear):
                    nn.init.orthogonal_(m.weight, gain=1.0)
                    nn.init.constant_(m.bias, 0.0)

    def _extract_scalar_features(self, obs: Dict) -> Optional[torch.Tensor]:
        """
        Extract scalar features from observation dictionary.
        
        Expected keys in obs:
        - xy: (batch, 2) or list of (2,)
        - target_xy: (batch, 2) or list of (2,)
        - battery: (batch,) or (batch, 1) or list of scalar
        - charge_xy: (batch, 2) or list of (2,) [optional]
        
        Returns:
            Scalar features tensor: (batch, scalar_input_size)
            or None if no scalar features available
        """
        if not self.use_scalar_features:
            return None
        
        features = []
        
        # xy (2,)
        if 'xy' in obs:
            xy = obs['xy']
            if isinstance(xy, (list, tuple)):
                xy = torch.stack([torch.tensor(x) for x in xy])
            elif not isinstance(xy, torch.Tensor):
                xy = torch.tensor(xy)
            if xy.dim() == 1:
                xy = xy.unsqueeze(0)
            features.append(xy.float())
        
        # target_xy (2,)
        if 'target_xy' in obs:
            target_xy = obs['target_xy']
            if isinstance(target_xy, (list, tuple)):
                target_xy = torch.stack([torch.tensor(t) for t in target_xy])
            elif not isinstance(target_xy, torch.Tensor):
                target_xy = torch.tensor(target_xy)
            if target_xy.dim() == 1:
                target_xy = target_xy.unsqueeze(0)
            features.append(target_xy.float())
        
        # battery (1,)
        if 'battery' in obs:
            battery = obs['battery']
            if isinstance(battery, (list, tuple)):
                battery = torch.tensor(battery)
            elif not isinstance(battery, torch.Tensor):
                battery = torch.tensor(battery)
            if battery.dim() == 0:
                battery = battery.unsqueeze(0)
            if battery.dim() == 1:
                battery = battery.unsqueeze(-1)
            features.append(battery.float())
        
        # charge_xy (2,) - optional
        if self.use_charger_xy_input and 'charge_xy' in obs:
            charge_xy = obs['charge_xy']
            if isinstance(charge_xy, (list, tuple)):
                charge_xy = torch.stack([torch.tensor(c) for c in charge_xy])
            elif not isinstance(charge_xy, torch.Tensor):
                charge_xy = torch.tensor(charge_xy)
            if charge_xy.dim() == 1:
                charge_xy = charge_xy.unsqueeze(0)
            features.append(charge_xy.float())
        
        if not features:
            return None
        
        return torch.cat(features, dim=-1)

    def forward(self, obs: Dict, rnn_states=None) -> Dict:
        """
        Forward pass through actor-critic.
        
        Args:
            obs: Observation dictionary
                 - 'obs': spatial features (batch, channels, H, W)
                 - 'xy', 'target_xy', 'battery', 'charge_xy': scalar features
            rnn_states: RNN states (not used in feedforward model)
            
        Returns:
            Dictionary with:
            - action_logits: (batch, num_actions)
            - values: (batch, 1)
            - log_prob_actions: (batch,)
            - actions: (batch,)
            - new_rnn_states: same as input rnn_states
        """
        # 1. Encode spatial features
        spatial_features = self.encoder(obs)  # (batch, encoder_out_size)
        
        # 2. Process scalar features
        combined_features = spatial_features
        if self.scalar_mlp is not None:
            scalar_input = self._extract_scalar_features(obs)
            if scalar_input is not None:
                scalar_features = self.scalar_mlp(scalar_input)
                combined_features = torch.cat([spatial_features, scalar_features], dim=-1)
        
        # 3. Actor head
        action_logits = self.actor(combined_features)
        
        # 4. Critic head
        values = self.critic(combined_features)
        
        # 5. Sample actions (for training)
        action_probs = F.softmax(action_logits, dim=-1)
        actions = torch.multinomial(action_probs, 1).squeeze(-1)
        log_probs = torch.log(action_probs.gather(1, actions.unsqueeze(-1))).squeeze(-1)
        
        # 6. RNN states passthrough (not used)
        new_rnn_states = rnn_states if rnn_states is not None else torch.zeros(1, 1, device=combined_features.device)
        
        return {
            'action_logits': action_logits,
            'values': values,
            'log_prob_actions': log_probs,
            'actions': actions,
            'new_rnn_states': new_rnn_states,
        }

    def get_action_logits(self, obs: Dict) -> torch.Tensor:
        """Get action logits for inference."""
        spatial_features = self.encoder(obs)
        combined_features = spatial_features
        
        if self.scalar_mlp is not None:
            scalar_input = self._extract_scalar_features(obs)
            if scalar_input is not None:
                scalar_features = self.scalar_mlp(scalar_input)
                combined_features = torch.cat([spatial_features, scalar_features], dim=-1)
        
        return self.actor(combined_features)

    def get_value(self, obs: Dict) -> torch.Tensor:
        """Get value estimate for inference."""
        spatial_features = self.encoder(obs)
        combined_features = spatial_features

        if self.scalar_mlp is not None:
            scalar_input = self._extract_scalar_features(obs)
            if scalar_input is not None:
                scalar_features = self.scalar_mlp(scalar_input)
                combined_features = torch.cat([spatial_features, scalar_features], dim=-1)

        return self.critic(combined_features)

    def model_to_device(self, device):
        """
        Move model to specified device.
        
        This method is required by Sample Factory's ActorCritic interface.
        
        Args:
            device: torch.device or str (e.g., 'cpu', 'cuda', 'cuda:0')
        """
        self.to(device)
        self.device = device if isinstance(device, str) else device.type


def create_charger_actor_critic(
    cfg: Config, 
    obs_space: ObsSpace, 
    action_space: ActionSpace
) -> ChargerActorCritic:
    """Factory function to create charger actor-critic."""
    return ChargerActorCritic(cfg, obs_space, action_space)


def main():
    """Test the charger actor-critic."""
    from argparse import Namespace
    
    # Mock config
    cfg = Namespace(
        encoder=EncoderConfig(
            extra_fc_layers=0,
            num_filters=64,
            num_res_blocks=1,
            hidden_size=128,
            use_scalar_features=True,
            use_charger_xy_input=True,
        ).dict(),
    )
    
    # Mock observation and action spaces
    r = 5
    obs_space = {'obs': dict(shape=(5, r * 2 + 1, r * 2 + 1))}
    action_space = Namespace(n=5)  # 5 actions
    
    # Create model
    model = ChargerActorCritic(cfg, obs_space, action_space)
    
    # Test forward pass
    batch_size = 4
    obs = {
        'obs': torch.rand(batch_size, 5, r * 2 + 1, r * 2 + 1),
        'xy': torch.rand(batch_size, 2),
        'target_xy': torch.rand(batch_size, 2),
        'battery': torch.rand(batch_size, 1),
        'charge_xy': torch.rand(batch_size, 2),
    }
    
    output = model(obs)
    print(f"Action logits shape: {output['action_logits'].shape}")
    print(f"Values shape: {output['values'].shape}")
    print(f"Actions shape: {output['actions'].shape}")


if __name__ == '__main__':
    main()
