"""
Charger APPO Example

Example script demonstrating how to use the charger policy for inference.

Usage:
    python charger_appo_example.py \
        --path_to_weights=experiments/train_dir/charger_appo \
        --num_agents=64 \
        --animation  # Enable animation
"""
import argparse

from env.create_env import create_env_base
from env.custom_maps import MAPS_REGISTRY
from utils.eval_utils import run_episode
from charger_appo.training_config import EnvironmentMazes
from charger_appo.inference import ChargerAppoInferenceConfig, ChargerAppoInference
from charger_appo.preprocessing import charger_appo_preprocessor


def create_custom_env(cfg):
    """
    Creates a custom environment for charger_appo with battery and charger support.
    
    Note: initial_battery is automatically set to (height + width) by pogema-charge
    if not specified. This ensures agents have enough battery to reach anywhere.

    Args:
        cfg: Command line arguments with animation, num_agents, etc.
    """
    env_cfg = EnvironmentMazes(
        with_animation=cfg.animation,
    )
    env_cfg.grid_config.num_agents = cfg.num_agents
    env_cfg.grid_config.num_charges = cfg.num_charges
    env_cfg.grid_config.map_name = cfg.map_name
    env_cfg.grid_config.seed = cfg.seed
    env_cfg.grid_config.max_episode_steps = cfg.max_episode_steps
    
    # Battery settings (initial_battery is auto-set to height+width)
    #env_cfg.grid_config.battery_decrement = 1
    #env_cfg.grid_config.charge_increment = 3

    env = create_env_base(env_cfg)
    return env


def run_charger_appo(env, path_to_weights='model/charger_appo'):
    """Run charger appo inference on environment."""
    charger_cfg = ChargerAppoInferenceConfig(
        path_to_weights=path_to_weights,
        device='cpu',
    )
    algo = ChargerAppoInference(charger_cfg)

    env = charger_appo_preprocessor(env, charger_cfg)

    return run_episode(env, algo)


def main():
    parser = argparse.ArgumentParser(description='Charger APPO Inference Script')
    parser.add_argument('--animation', action='store_false', 
                        help='Enable animation (default: True)')
    parser.add_argument('--num_agents', type=int, default=128, 
                        help='Number of agents (default: %(default)d)')
    parser.add_argument('--num_charges', type=int, default=32, 
                        help='Number of charges (default: %(default)d)')
    parser.add_argument('--seed', type=int, default=0, 
                        help='Random seed (default: %(default)d)')
    parser.add_argument('--map_name', type=str, default='wfi_warehouse', 
                        help='Map name (default: %(default)s)')
    parser.add_argument('--max_episode_steps', type=int, default=256,
                        help='Maximum episode steps (default: %(default)d)')
    parser.add_argument('--show_map_names', action='store_true', 
                        help='Shows names of all available maps')
    parser.add_argument('--path_to_weights', type=str, default='model/charger_appo',
                        help='Path to model weights (default: %(default)s)')

    args = parser.parse_args()

    if args.show_map_names:
        for map_ in MAPS_REGISTRY:
            print(map_)
        return

    # Check if weights path exists
    import os
    if not os.path.exists(args.path_to_weights):
        print(f"\nWarning: Weights path '{args.path_to_weights}' does not exist.")
        print("Please train a model first or specify a valid path:")
        print("  python train_charger_appo.py --train_for_env_steps=1000000")
        print("\nOr use the follower model as a starting point:")
        print("  python charger_appo_example.py --path_to_weights=model/follower")
        return

    print(f"Running charger_appo evaluation with {args.num_agents} agents {args.num_charges} charges ...")
    result = run_charger_appo(create_custom_env(args), args.path_to_weights)
    print(f"Result: {result}")


if __name__ == '__main__':
    main()
