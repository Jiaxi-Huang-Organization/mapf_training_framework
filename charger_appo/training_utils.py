"""
Charger APPO Training Utilities

Provides functions for setting up and running charger APPO training.
Follows the same pattern as follower training_utils for consistency.
"""
import os
import json
import yaml
from argparse import Namespace
from os.path import join

from sample_factory.utils.utils import log

from charger_appo.register_training_utils import (
    register_custom_model, 
    register_msg_handlers,
    log_model_summary,
)
from charger_appo.register_env import register_custom_components
from charger_appo.training_config import Experiment


def create_sf_config(experiment_cfg: Experiment) -> Namespace:
    """
    Convert Experiment config to Sample Factory config.
    
    This function follows the same pattern as follower training_utils.
    
    Args:
        experiment_cfg: Pydantic Experiment configuration
        
    Returns:
        Namespace object compatible with Sample Factory
    """
    from sample_factory.cfg.arguments import parse_sf_args, parse_full_cfg
    
    # First create a minimal config to initialize Sample Factory parser
    custom_argv = [f'--env={experiment_cfg.env}']
    parser, partial_cfg = parse_sf_args(argv=custom_argv, evaluation=False)
    
    # Set defaults from experiment config
    parser.set_defaults(**experiment_cfg.dict())
    final_cfg = parse_full_cfg(parser, argv=custom_argv)
    
    # Register custom components
    register_custom_components(experiment_cfg.environment.env)
    register_custom_model()
    
    # Handle follower_checkpoint path - convert relative path to absolute
    if hasattr(final_cfg, 'follower_checkpoint') and final_cfg.follower_checkpoint:
        checkpoint_path = final_cfg.follower_checkpoint
        # If path is relative, make it absolute relative to project root
        if not os.path.isabs(checkpoint_path):
            project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            checkpoint_path = os.path.join(project_root, checkpoint_path)
            final_cfg.follower_checkpoint = checkpoint_path
        
        log.info(f"Using follower checkpoint: {final_cfg.follower_checkpoint}")
        
        # Validate checkpoint path
        if os.path.isdir(checkpoint_path):
            config_path = join(checkpoint_path, 'config.json')
            if not os.path.exists(config_path):
                log.warning(f"config.json not found in follower checkpoint: {checkpoint_path}")
            else:
                log.info(f"Found follower config at {config_path}")
        elif os.path.isfile(checkpoint_path):
            log.info(f"Using follower checkpoint file: {checkpoint_path}")
        else:
            log.warning(f"Follower checkpoint not found: {checkpoint_path}")
    
    if hasattr(final_cfg, 'freeze_follower_encoder') and final_cfg.freeze_follower_encoder:
        log.info("Follower encoder will be frozen")
    
    log.info(f"Charge threshold: {getattr(final_cfg, 'charge_threshold', 0.3)}")
    log.info(f"Charger intrinsic reward: {getattr(final_cfg, 'charger_intrinsic_reward', 0.01)}")
    
    return final_cfg


def run(config=None):
    """
    Run charger APPO training.
    
    This function follows the same pattern as follower training_utils.run()
    
    Args:
        config: Configuration dictionary or None (will load from argparse)
    """
    from sample_factory.algo.utils.misc import ExperimentStatus
    import wandb
    
    # Register custom model first
    register_custom_model()

    if config is None:
        import argparse

        parser = argparse.ArgumentParser(description='Process training config.')

        parser.add_argument('--config_path', type=str, action="store", default='train-debug.yaml',
                            help='path to yaml file with single run configuration', required=False)

        parser.add_argument('--raw_config', type=str, action='store',
                            help='raw json config', required=False)

        parser.add_argument('--wandb_thread_mode', type=bool, action='store', default=False,
                            help='Run wandb in thread mode.', required=False)

        params = parser.parse_args()
        if params.raw_config:
            params.raw_config = params.raw_config.replace("\'", "\"")
            config = json.loads(params.raw_config)
        else:
            if params.config_path is None:
                raise ValueError("You should specify --config_path or --raw_config argument!")
            with open(params.config_path, "r") as f:
                config = yaml.safe_load(f)
    else:
        params = Namespace(**config)
        params.wandb_thread_mode = False

    exp = Experiment(**config)
    flat_config = Namespace(**exp.dict())
    env_name = exp.environment.env
    log.debug(f'env_name = {env_name}')
    register_custom_components(env_name)

    log.info(flat_config)

    # Disable wandb for short runs or if not configured
    if exp.train_for_env_steps == 1_000_000 or exp.train_for_env_steps <= 1000:
        exp.use_wandb = False

    if exp.use_wandb:
        if params.wandb_thread_mode:
            os.environ["WANDB_START_METHOD"] = "thread"
        try:
            wandb.init(
                project='Charger-APPO', 
                config=exp.dict(), 
                save_code=False, 
                sync_tensorboard=True,
                anonymous="allow", 
                job_type=exp.environment.env, 
                group='train'
            )
        except Exception as e:
            log.warning(f"Failed to initialize wandb: {e}. Disabling wandb.")
            exp.use_wandb = False

    # Create runner using sample_factory's make_runner
    from sample_factory.train import make_runner
    flat_config, runner = make_runner(create_sf_config(exp))
    register_msg_handlers(flat_config, runner)
    
    # Log model summary after initialization
    if hasattr(runner, 'algo') and hasattr(runner.algo, 'actor_critic'):
        model = runner.algo.actor_critic
        log_model_summary(model)
        
        # Load follower weights if checkpoint is provided
        if hasattr(flat_config, 'follower_checkpoint') and flat_config.follower_checkpoint:
            log.info(f"Loading follower weights from: {flat_config.follower_checkpoint}")
            load_follower_weights_into_charger(model, flat_config.follower_checkpoint, flat_config.device)
            
            # Freeze encoder if requested
            if hasattr(flat_config, 'freeze_follower_encoder') and flat_config.freeze_follower_encoder:
                log.info("Freezing encoder parameters...")
                if hasattr(model, 'encoder'):
                    for param in model.encoder.parameters():
                        param.requires_grad = False
                    model.encoder.eval()
    
    status = runner.init()
    if status == ExperimentStatus.SUCCESS:
        status = runner.run()

    return status


def load_follower_weights_into_charger(
    charger_model, 
    follower_checkpoint_path: str,
    device='cpu'
):
    """
    Load follower weights into charger model.
    
    This function:
    1. Finds the latest checkpoint in the follower directory
    2. Extracts encoder weights from the checkpoint
    3. Loads weights into the charger model's encoder
    
    Args:
        charger_model: ChargerActorCritic model
        follower_checkpoint_path: Path to follower checkpoint directory
        device: Device to load checkpoint to
    """
    import torch
    import glob
    
    if not os.path.exists(follower_checkpoint_path):
        log.warning(f"Follower checkpoint not found: {follower_checkpoint_path}")
        return False
    
    # Find checkpoint file if path is a directory
    original_path = follower_checkpoint_path
    if os.path.isdir(follower_checkpoint_path):
        checkpoints = glob.glob(os.path.join(follower_checkpoint_path, 'checkpoint_p0', 'checkpoint_*'))
        if not checkpoints:
            checkpoints = glob.glob(os.path.join(follower_checkpoint_path, '*.pt'))
        if checkpoints:
            checkpoints.sort()
            follower_checkpoint_path = checkpoints[-1]
            log.info(f"Using latest follower checkpoint: {follower_checkpoint_path}")
        else:
            log.warning(f"No checkpoint files found in {original_path}")
            return False
    
    try:
        checkpoint = torch.load(follower_checkpoint_path, map_location=device, weights_only=False)
        
        # Extract model state dict
        if 'model' in checkpoint:
            state_dict = checkpoint['model']
        elif 'actor_critic' in checkpoint:
            state_dict = checkpoint['actor_critic']
        else:
            state_dict = checkpoint
        
        # Extract encoder weights (keys starting with 'encoder.')
        encoder_weights = {}
        for key, value in state_dict.items():
            if key.startswith('encoder.'):
                new_key = key[8:]  # Remove 'encoder.' prefix
                encoder_weights[new_key] = value
        
        if not encoder_weights:
            log.warning(f"No encoder weights found in {follower_checkpoint_path}")
            return False
        
        # Load into encoder with strict=False to handle channel differences
        if hasattr(charger_model, 'encoder'):
            result = charger_model.encoder.load_state_dict(encoder_weights, strict=False)
            log.info(f"Loaded follower encoder weights: {len(encoder_weights)} keys")
            if result.missing_keys:
                log.debug(f"Missing keys: {len(result.missing_keys)}")
            if result.unexpected_keys:
                log.debug(f"Unexpected keys: {len(result.unexpected_keys)}")
            return True
        else:
            log.warning("Model has no encoder attribute")
            return False
            
    except Exception as e:
        log.warning(f"Failed to load follower weights: {e}")
        import traceback
        log.debug(traceback.format_exc())
        return False
