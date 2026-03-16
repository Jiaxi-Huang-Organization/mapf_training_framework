"""
Charger APPO Inference

Inference utilities for the charger policy.
Follows the same pattern as follower inference for consistency.
"""
from typing import Optional, List
from argparse import Namespace
from os.path import join
import os
import json

import torch
import numpy as np

from sample_factory.utils.utils import log
from sample_factory.algo.learning.learner import Learner
from sample_factory.model.actor_critic import create_actor_critic
from sample_factory.algo.utils.make_env import make_env_func_batched
from sample_factory.utils.attr_dict import AttrDict
from sample_factory.algo.utils.rl_utils import prepare_and_normalize_obs
from sample_factory.model.model_utils import get_rnn_size

from charger_appo.register_env import register_custom_components
from charger_appo.register_training_utils import register_custom_model
from charger_appo.training_config import Experiment


class ChargerAppoInferenceConfig:
    """Configuration for charger appo inference."""
    
    def __init__(
        self,
        path_to_weights: str = 'model/charger_appo',
        device: str = 'cpu',
        override_config: Optional[dict] = None,
        custom_path_to_weights: Optional[str] = None,
    ):
        self.path_to_weights = path_to_weights
        self.device = device
        self.override_config = override_config
        self.custom_path_to_weights = custom_path_to_weights
        self.training_config = None


class ChargerAppoInference:
    """
    Charger APPO inference engine.
    
    Loads trained charger policy and provides act() interface
    for getting actions from observations.
    """
    
    def __init__(self, cfg: ChargerAppoInferenceConfig):
        self.algo_cfg = cfg
        self.device = torch.device(cfg.device)
        
        # Register custom components
        register_custom_components('PogemaMazes-v0')
        register_custom_model()
        
        # Load model
        self._load_model()
    
    def _load_model(self):
        """Load the charger actor-critic model."""
        path = self.algo_cfg.path_to_weights
        
        # Load config.json from checkpoint directory
        config_path = join(path, 'config.json')
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"config.json not found in {path}")
        
        with open(config_path, 'r') as f:
            flat_config = json.load(f)
        
        # Apply override config if provided
        if self.algo_cfg.override_config:
            self._recursive_dict_update(flat_config, self.algo_cfg.override_config)
        
        self.exp = Experiment(**flat_config)
        flat_config_ns = Namespace(**flat_config)
        
        # Set num_envs for inference
        flat_config_ns.num_envs = 1
        
        # Create environment to get observation/action spaces
        env = make_env_func_batched(
            flat_config_ns, 
            env_config=AttrDict(worker_index=0, vector_index=0, env_id=0)
        )
        
        # Create actor-critic using sample_factory factory
        actor_critic = create_actor_critic(
            flat_config_ns, 
            env.observation_space, 
            env.action_space
        )
        actor_critic.eval()
        env.close()
        
        # Move to device
        if self.algo_cfg.device != 'cpu' and not torch.cuda.is_available():
            os.environ['OMP_NUM_THREADS'] = '1'
            os.environ['MKL_NUM_THREADS'] = '1'
            self.device = torch.device('cpu')
            torch.set_num_threads(1)
            torch.set_num_interop_threads(1)
            log.warning('CUDA is not available, using CPU. This might be slow.')
        
        actor_critic.model_to_device(self.device)
        
        # Load checkpoint
        name_prefix = 'checkpoint'
        policy_index = 0 if 'policy_index' not in flat_config else flat_config['policy_index']
        
        checkpoints = Learner.get_checkpoints(
            os.path.join(path, f"checkpoint_p{policy_index}"),
            f"{name_prefix}_*"
        )
        
        # Use custom checkpoint path if provided
        if self.algo_cfg.custom_path_to_weights:
            checkpoints = [self.algo_cfg.custom_path_to_weights]
        
        checkpoint_dict = Learner.load_checkpoint(checkpoints, self.device)
        actor_critic.load_state_dict(checkpoint_dict['model'])
        log.info(f'Loaded {str(checkpoints)}')
        
        self.net = actor_critic
        self.cfg = flat_config_ns
        self.rnn_states = None
    
    @classmethod
    def _recursive_dict_update(cls, original_dict: dict, update_dict: dict):
        """Recursively update dictionary."""
        for key, value in update_dict.items():
            if key in original_dict and isinstance(original_dict[key], dict) and isinstance(value, dict):
                cls._recursive_dict_update(original_dict[key], value)
            else:
                original_dict[key] = value
    
    def act(self, observations, deterministic: bool = True):
        """
        Get actions for observations.
        
        Args:
            observations: List of observation dictionaries
            deterministic: Whether to use deterministic actions
            
        Returns:
            Array of actions
        """
        self.rnn_states = (
            torch.zeros(
                [len(observations), get_rnn_size(self.cfg)], 
                dtype=torch.float32,
                device=self.device
            ) 
            if self.rnn_states is None 
            else self.rnn_states
        )
        
        obs = AttrDict(self.transform_dict_observations(observations))
        
        with torch.no_grad():
            policy_outputs = self.net(
                prepare_and_normalize_obs(self.net, obs), 
                self.rnn_states
            )
        
        self.rnn_states = policy_outputs['new_rnn_states']
        
        if deterministic:
            actions = torch.argmax(policy_outputs['action_logits'], dim=-1)
        else:
            probs = torch.softmax(policy_outputs['action_logits'], dim=-1)
            actions = torch.multinomial(probs, 1).squeeze(-1)
        
        return actions.cpu().numpy()
    
    def reset_states(self):
        """Reset RNN states and random seed."""
        torch.manual_seed(getattr(self.algo_cfg, 'seed', 42))
        self.rnn_states = None
    
    @staticmethod
    def count_parameters(model):
        """Count trainable parameters."""
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    def get_model_parameters(self):
        """Get number of trainable parameters."""
        return self.count_parameters(self.net)
    
    @staticmethod
    def transform_dict_observations(observations):
        """Transform list of dict observations into a dict of lists."""
        obs_dict = dict()
        if isinstance(observations[0], (dict,)):
            for key in observations[0].keys():
                if not isinstance(observations[0][key], str):
                    obs_dict[key] = [o[key] for o in observations]
        else:
            obs_dict['obs'] = observations

        for key, x in obs_dict.items():
            obs_dict[key] = np.stack(x)

        return obs_dict
