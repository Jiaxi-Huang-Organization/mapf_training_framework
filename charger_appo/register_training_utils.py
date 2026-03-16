"""
Charger APPO Training Utilities Registration

Registers the charger encoder and actor-critic with Sample Factory.
Also handles parameter freezing for fine-tuning.
"""
from charger_appo.model import ResnetEncoder
from charger_appo.actor_critic import ChargerActorCritic, create_charger_actor_critic

from sample_factory.algo.utils.context import global_model_factory
from sample_factory.utils.typing import ObsSpace, ActionSpace
from sample_factory.model.encoder import Encoder
from tensorboardX import SummaryWriter
from sample_factory.utils.typing import Config, PolicyID
from sample_factory.algo.runners.runner import AlgoObserver, Runner

import numpy as np
from sample_factory.utils.utils import log


def pogema_extra_episodic_stats_processing(*args, **kwargs):
    """Placeholder for episodic stats processing."""
    pass


def pogema_extra_summaries(
    runner: Runner, 
    policy_id: PolicyID, 
    summary_writer: SummaryWriter, 
    env_steps: int
):
    """
    Log extra summaries to TensorBoard.
    
    Logs metrics like:
    - Success rate
    - Path length
    - Battery usage
    - Charger visits
    """
    policy_avg_stats = runner.policy_avg_stats
    for key in policy_avg_stats:
        if key in ['reward', 'len', 'true_reward', 'Done']:
            continue

        avg = np.mean(np.array(policy_avg_stats[key][policy_id]))
        summary_writer.add_scalar(key, avg, env_steps)
        log.debug(f'{policy_id}-{key}: {round(float(avg), 3)}')


class CustomExtraSummariesObserver(AlgoObserver):
    """Observer for logging extra summaries during training."""
    
    def extra_summaries(
        self, 
        runner: Runner, 
        policy_id: PolicyID, 
        writer: SummaryWriter, 
        env_steps: int
    ) -> None:
        pogema_extra_summaries(runner, policy_id, writer, env_steps)


def register_msg_handlers(cfg: Config, runner: Runner):
    """Register message handlers for training statistics."""
    runner.register_episodic_stats_handler(pogema_extra_episodic_stats_processing)
    runner.register_observer(CustomExtraSummariesObserver())


def make_custom_encoder(cfg: Config, obs_space: ObsSpace) -> Encoder:
    """Factory function to create charger encoder."""
    return ResnetEncoder(cfg, obs_space)


def make_custom_actor_critic(
    cfg: Config,
    obs_space: ObsSpace,
    action_space: ActionSpace
):
    """
    Factory function to create charger actor-critic.
    
    Note: We're now using Sample Factory's default actor-critic.
    The custom ResnetEncoder will be used by SF's default AC.
    This is the same approach as follower - simpler and more robust.
    """
    # Return None to let Sample Factory use its default ActorCritic
    # The custom encoder will be automatically used
    return None


def register_custom_model():
    """
    Register custom encoder with Sample Factory.
    
    We only register the encoder factory, similar to follower.
    Sample Factory will use its default ActorCritic with our custom encoder.
    This is simpler and more robust than implementing a full custom ActorCritic.
    """
    global_model_factory().register_encoder_factory(make_custom_encoder)
    # Note: We don't register actor_critic factory - let SF use its default


def freeze_encoder_params(model):
    """
    Freeze encoder parameters in the actor-critic model.
    
    This is useful for fine-tuning where you want to:
    1. Keep the learned spatial features from follower
    2. Only train the actor/critic heads for battery-aware behavior
    
    Args:
        model: ChargerActorCritic model
    """
    if hasattr(model, 'encoder'):
        for param in model.encoder.parameters():
            param.requires_grad = False
        model.encoder.eval()
        log.info("Encoder parameters frozen")
    
    # Also freeze scalar_mlp if you want to only train actor/critic heads
    if hasattr(model, 'scalar_mlp'):
        for param in model.scalar_mlp.parameters():
            param.requires_grad = False
        model.scalar_mlp.eval()
        log.info("Scalar MLP parameters frozen")


def get_trainable_params(model) -> dict:
    """
    Get information about trainable parameters in the model.
    
    Args:
        model: ChargerActorCritic model
        
    Returns:
        Dictionary with parameter counts
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    encoder_params = sum(p.numel() for p in model.encoder.parameters()) if hasattr(model, 'encoder') else 0
    encoder_trainable = sum(p.numel() for p in model.encoder.parameters() if p.requires_grad) if hasattr(model, 'encoder') else 0
    
    actor_params = sum(p.numel() for p in model.actor.parameters()) if hasattr(model, 'actor') else 0
    critic_params = sum(p.numel() for p in model.critic.parameters()) if hasattr(model, 'critic') else 0
    
    scalar_mlp_params = sum(p.numel() for p in model.scalar_mlp.parameters()) if hasattr(model, 'scalar_mlp') else 0
    scalar_mlp_trainable = sum(p.numel() for p in model.scalar_mlp.parameters() if p.requires_grad) if hasattr(model, 'scalar_mlp') else 0
    
    return {
        'total_params': total_params,
        'trainable_params': trainable_params,
        'encoder_params': encoder_params,
        'encoder_trainable': encoder_trainable,
        'actor_params': actor_params,
        'critic_params': critic_params,
        'scalar_mlp_params': scalar_mlp_params,
        'scalar_mlp_trainable': scalar_mlp_trainable,
    }


def log_model_summary(model):
    """
    Log a summary of the model architecture and trainable parameters.
    
    Args:
        model: ChargerActorCritic model
    """
    param_info = get_trainable_params(model)
    
    log.info("=" * 60)
    log.info("Charger Actor-Critic Model Summary")
    log.info("=" * 60)
    log.info(f"Total parameters: {param_info['total_params']:,}")
    log.info(f"Trainable parameters: {param_info['trainable_params']:,}")
    log.info(f"  Encoder: {param_info['encoder_trainable']:,} / {param_info['encoder_params']:,}")
    log.info(f"  Scalar MLP: {param_info['scalar_mlp_trainable']:,} / {param_info['scalar_mlp_params']:,}")
    log.info(f"  Actor: {param_info['actor_params']:,}")
    log.info(f"  Critic: {param_info['critic_params']:,}")
    log.info("=" * 60)
