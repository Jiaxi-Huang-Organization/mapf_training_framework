"""
Charger APPO Training Configuration

Key differences from follower:
- Adds battery and charger configuration
- Adds charge_threshold for target switching
- Adds option to freeze follower encoder weights
"""
from typing import Optional, Union, List

from charger_appo.encoder import EncoderConfig
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
        initial_battery: Initial battery level for all agents (list or int).
                        If None, automatically set to (map_height + map_width).
        battery_decrement: Battery decrement per step
        charge_increment: Battery increment when charging
        num_charges: Number of chargers on the map
    """
    integration: Literal['SampleFactory'] = 'SampleFactory'
    collision_system: Literal['priority', 'block_both', 'soft'] = 'soft'
    observation_type: Literal['POMAPF'] = 'POMAPF'
    auto_reset: Literal[False] = False

    num_agents: int = 64
    num_charges: int = 16  # ~1 charger per 4 agents
    obs_radius: int = 5
    max_episode_steps: int = 512
    map_name: str = (
        '(wc3-[A-P]|sc1-[A-S]|sc1-TaleofTwoCities|street-[A-P]|'
        r'mazes-s[0-9]_|mazes-s[1-3][0-9]_|random-s[0-9]_|random-s[1-3][0-9]_)'
    )
    
    # Battery and charger configuration
    # Note: Set to None to auto-assign based on map size (height + width)
    initial_battery: Optional[Union[int, List[int]]] = None
    battery_decrement: int = 1
    charge_increment: int = 3



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
    agent_bins: Optional[list] = [64, 128, 256, 256]
    agent_per_charge: int = 4  # Number of agents per charger
    grid_config: DecMAPFConfig = DecMAPFConfig(
        on_target='restart', 
        max_episode_steps=512,
        map_name=r'mazes-.+'
    )



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
    encoder: EncoderConfig = EncoderConfig(num_res_blocks=4,
                                            extra_fc_layers=1,
                                            hidden_size=256,
                                            num_filters=32
                                            )
    preprocessing: PreprocessorConfig = PreprocessorConfig()

    # Training hyperparameters
    rollout: int = 8
    num_workers: int = 4
    recurrence: int = 8
    use_rnn: bool = False
    rnn_size: int = 256

    # PPO parameters
    ppo_clip_ratio: float = 0.2
    batch_size: int = 2048
    exploration_loss_coeff: float = 0.03
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
    save_best_metric: str = "avg_throughput"#"avg_goal_battery_relative"
    value_bootstrap: bool = True
    save_milestones_sec: int = -1

    # Checkpointing and logging
    keep_checkpoints: int = 1
    stats_avg: int = 10
    learning_rate: float = 0.00022
    train_for_env_steps: int = 1_000_000#1_000_000
    gamma: float = 0.9756
    lr_schedule: str = 'kl_adaptive_minibatch'

    # Experiment metadata
    experiment: str = 'charger_exp'
    train_dir: str = 'experiments/train_dir/charger_appo'
    seed: Optional[int] = 42
    use_wandb: bool = True  # Default to False, can be enabled for production runs
    device: str = 'cpu'
    env: Literal['PogemaMazes-v0'] = "PogemaMazes-v0"

