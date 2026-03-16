"""
Charger APPO Training Configuration

Key differences from follower:
- Adds battery and charger configuration
- Adds charge_threshold for target switching
- Adds option to freeze follower encoder weights
"""
from typing import Optional, Union, List

from charger_appo.model import EncoderConfig
from charger_appo.preprocessing import PreprocessorConfig

try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal

from pogema import GridConfig
from pydantic import BaseModel, Field


class DecMAPFConfig(GridConfig):
    """
    Decentralized MAPF configuration with battery and charger support.
    
    Args:
        initial_battery: Initial battery level for all agents (list or int)
        battery_decrement: Battery decrement per step
        charge_increment: Battery increment when charging
        num_charges: Number of chargers on the map
    """
    integration: Literal['SampleFactory'] = 'SampleFactory'
    collision_system: Literal['priority', 'block_both', 'soft'] = 'soft'
    observation_type: Literal['POMAPF'] = 'POMAPF'
    auto_reset: Literal[False] = False

    num_agents: int = 64
    obs_radius: int = 5
    max_episode_steps: int = 512
    map_name: str = (
        '(wc3-[A-P]|sc1-[A-S]|sc1-TaleofTwoCities|street-[A-P]|'
        r'mazes-s[0-9]_|mazes-s[1-3][0-9]_|random-s[0-9]_|random-s[1-3][0-9]_)'
    )
    
    # Battery and charger configuration
    # Note: initial_battery must be a list for pogema-charge compatibility
    initial_battery: Union[int, List[int]] = 100
    battery_decrement: int = 1
    charge_increment: int = 3
    num_charges: int = 4  # ~1 charger per 16 agents


class Environment(BaseModel):
    """Base environment configuration."""
    grid_config: DecMAPFConfig = DecMAPFConfig()
    env: Literal['PogemaMazes-v0'] = "PogemaMazes-v0"
    with_animation: bool = False
    worker_index: int = None
    vector_index: int = None
    env_id: int = None
    target_num_agents: Optional[int] = None
    agent_bins: Optional[list] = [64, 128, 256, 256]
    use_maps: bool = True
    every_step_metrics: bool = False


class EnvironmentMazes(Environment):
    """Mazes environment configuration."""
    env: Literal['PogemaMazes-v0'] = "PogemaMazes-v0"
    use_maps: bool = True
    target_num_agents: Optional[int] = 256
    agent_bins: Optional[list] = [128, 256, 256, 256]
    grid_config: DecMAPFConfig = DecMAPFConfig(
        on_target='restart', 
        max_episode_steps=512,
        map_name=r'mazes-.+'
    )


class PreprocessorConfigExt(PreprocessorConfig):
    """
    Extended preprocessor configuration for charger appo.
    
    Args:
        charge_threshold: Battery threshold (0.0-1.0) below which agent seeks charger
        charger_intrinsic_reward: Intrinsic reward for charger subgoals
        use_charger_xy_input: Whether to pass nearest charger xy as input to network
    """
    charge_threshold: float = 0.3
    charger_intrinsic_reward: float = 0.01
    use_charger_xy_input: bool = True


class EncoderConfigExt(EncoderConfig):
    """
    Extended encoder configuration for charger appo.
    
    Args:
        follower_checkpoint: Path to pre-trained follower checkpoint.
                            Default: 'model/follower' (relative to project root)
        freeze_follower_encoder: Whether to freeze follower encoder parameters
        use_scalar_features: Whether to use scalar features (xy, target_xy, battery, etc.)
        use_charger_xy_input: Whether to include charger_xy in scalar inputs
    """
    follower_checkpoint: Optional[str] = 'model/follower'
    freeze_follower_encoder: bool = True
    use_scalar_features: bool = True
    use_charger_xy_input: bool = True


class Experiment(BaseModel):
    """
    Full experiment configuration for charger APPO training.
    
    This configuration extends follower with:
    - Battery and charger support
    - Dynamic target switching based on battery level
    - Optional freezing of follower encoder weights
    """
    # Environment
    environment: EnvironmentMazes = EnvironmentMazes()
    encoder: EncoderConfigExt = EncoderConfigExt()
    preprocessing: PreprocessorConfigExt = PreprocessorConfigExt()

    # Training hyperparameters
    rollout: int = 8
    num_workers: int = 4
    recurrence: int = 8
    use_rnn: bool = False
    rnn_size: int = 256

    # PPO parameters
    ppo_clip_ratio: float = 0.1
    batch_size: int = 2048
    exploration_loss_coeff: float = 0.018
    num_envs_per_worker: int = 4
    worker_num_splits: int = 1
    max_policy_lag: int = 1

    # Optimization
    force_envs_single_thread: bool = True
    optimizer: Literal["adam", "lamb"] = 'adam'
    restart_behavior: str = "overwrite"  # ["resume", "restart", "overwrite"]
    normalize_returns: bool = False
    async_rl: bool = False
    num_batches_per_epoch: int = 16
    num_batches_to_accumulate: int = 1
    normalize_input: bool = False
    decoder_mlp_layers: List[int] = Field(default_factory=list)
    save_best_metric: str = "avg_throughput"
    value_bootstrap: bool = True
    save_milestones_sec: int = -1

    # Checkpointing and logging
    keep_checkpoints: int = 1
    stats_avg: int = 10
    learning_rate: float = 0.000146
    train_for_env_steps: int = 1_000_000
    gamma: float = 0.965
    lr_schedule: str = 'kl_adaptive_minibatch'

    # Experiment metadata
    experiment: str = 'charger_exp'
    train_dir: str = 'experiments/train_dir/charger_appo'
    seed: Optional[int] = 42
    use_wandb: bool = False  # Default to False, can be enabled for production runs
    device: str = 'cpu'
    env: Literal['PogemaMazes-v0'] = "PogemaMazes-v0"
    
    # Charger-specific settings
    follower_checkpoint: Optional[str] = 'model/follower'  # Path to frozen follower weights
    freeze_follower: bool = True  # Freeze follower encoder during training
    
    # Battery-aware settings
    charge_threshold: float = 0.3  # Battery threshold for charger seeking
    charger_intrinsic_reward: float = 0.01  # Reward for charger subgoals
