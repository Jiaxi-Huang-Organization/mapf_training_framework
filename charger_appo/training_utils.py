"""
Charger APPO Training Utilities

Provides functions for setting up and running charger APPO training.
Follows the same pattern as follower training_utils for consistency.
"""
import os
from os.path import join
import json

from argparse import Namespace
import yaml
from sample_factory.cfg.arguments import parse_sf_args, parse_full_cfg

from sample_factory.train import make_runner
from sample_factory.algo.utils.misc import ExperimentStatus

import wandb

from charger_appo.training_config import Experiment

from sample_factory.utils.utils import log

from charger_appo.register_training_utils import (
    register_custom_model, 
    register_msg_handlers,
    log_model_summary,
)
from charger_appo.register_env import register_custom_components



def create_sf_config(experiment_cfg: Experiment) -> Namespace:
    """
    Convert Experiment config to Sample Factory config.
    
    This function follows the same pattern as follower training_utils.
    
    Args:
        experiment_cfg: Pydantic Experiment configuration
        
    Returns:
        Namespace object compatible with Sample Factory
    """    
    # First create a minimal config to initialize Sample Factory parser
    custom_argv = [f'--env={experiment_cfg.env}']
    parser, partial_cfg = parse_sf_args(argv=custom_argv, evaluation=False)
    
    # Set defaults from experiment config
    parser.set_defaults(**experiment_cfg.dict())
    final_cfg = parse_full_cfg(parser, argv=custom_argv)
    
    # Register custom components
    register_custom_components(experiment_cfg.environment.env)
    register_custom_model()
            
    log.info(f"Intrinsic target reward: {getattr(final_cfg, 'intrinsic_target_reward', 0.01)}")
    log.info(f"On charger reward: {getattr(final_cfg, 'on_chargers_reward', 0.02)}")
    log.info(f"Battery reward coeff: {getattr(final_cfg, 'battery_reward_coeff', 0.01)}")

    return final_cfg


def run(config=None):
    """
    Run charger APPO training.
    
    This function follows the same pattern as follower training_utils.run()
    
    Args:
        config: Configuration dictionary or None (will load from argparse)
    """
    
    register_custom_model()

    if config is None:
        import argparse

        parser = argparse.ArgumentParser(description='Process training config.')

        parser.add_argument('--config_path', type=str, action="store", default='train-debug.yaml',
                            help='path to yaml file with single run configuration', required=False)

        parser.add_argument('--raw_config', type=str, action='store',
                            help='raw json config', required=False)

        parser.add_argument('--wandb_thread_mode', type=bool, action='store', default=False,
                            help='Run wandb in thread mode. Usefull for some setups.', required=False)

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


    flat_config, runner = make_runner(create_sf_config(exp))
    register_msg_handlers(flat_config, runner)
                        
    status = runner.init()
    if status == ExperimentStatus.SUCCESS:
        status = runner.run()

    return status