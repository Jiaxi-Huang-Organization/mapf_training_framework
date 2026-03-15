import argparse

from env.create_env import create_env_base
from env.custom_maps import MAPS_REGISTRY
from utils.eval_utils import run_episode
from charger_mappo.training_config import EnvironmentMazes
from charger_mappo.inference import chargerMappoInferenceConfig, chargerMappoInference
from charger_mappo.preprocessing import charger_mappo_preprocessor


def create_custom_env(cfg):
    """
    Creates a custom environment for charger_mappo with battery and charger support.
    
    Args:
        cfg: Configuration object with environment parameters
        
    Returns:
        gymnasium.Env: Configured environment
    """
    env_cfg = EnvironmentMazes(with_animation=cfg.animation)
    env_cfg.grid_config.num_agents = cfg.num_agents
    env_cfg.grid_config.map_name = cfg.map_name
    env_cfg.grid_config.seed = cfg.seed
    env_cfg.grid_config.max_episode_steps = cfg.max_episode_steps
    # Enable battery and charger features
    env_cfg.grid_config.initial_battery = [100] * cfg.num_agents
    env_cfg.grid_config.battery_decrement = 1
    env_cfg.grid_config.charge_increment = 3
    env_cfg.grid_config.num_charges = max(1, cfg.num_agents // 16)  # ~1 charger per 16 agents
    return create_env_base(env_cfg)


def run_charger_mappo(env, path_to_weights='model/charger_mappo', device='cpu'):
    """
    Runs charger_mappo algorithm on the given environment.
    
    Args:
        env: The environment to run
        path_to_weights: Path to trained model weights
        device: Device to run inference on ('cpu' or 'cuda')
        
    Returns:
        ResultsHolder: Episode results
    """
    charger_cfg = chargerMappoInferenceConfig(path_to_weights=path_to_weights, device=device)
    algo = chargerMappoInference(charger_cfg)

    env = charger_mappo_preprocessor(env, algo)

    return run_episode(env, algo)


def main():
    parser = argparse.ArgumentParser(description='Charger MAPPO Inference Script')
    parser.add_argument('--animation', action='store_false', help='Enable animation (default: %(default)s)')
    parser.add_argument('--num_agents', type=int, default=64, help='Number of agents (default: %(default)d)')
    parser.add_argument('--seed', type=int, default=0, help='Random seed (default: %(default)d)')
    parser.add_argument('--map_name', type=str, default='mazes-s0', help='Map name (default: %(default)s)')
    parser.add_argument('--max_episode_steps', type=int, default=512,
                        help='Maximum episode steps (default: %(default)d)')
    parser.add_argument('--show_map_names', action='store_true', help='Shows names of all available maps')

    parser.add_argument('--path_to_weights', type=str, default='model/charger_mappo',
                        help='Path to trained model weights (default: %(default)s)')
    parser.add_argument('--device', type=str, default='cpu', choices=['cpu', 'cuda'],
                        help='Device to run inference on (default: %(default)s)')

    args = parser.parse_args()

    if args.show_map_names:
        print("Available maps:")
        for map_ in sorted(MAPS_REGISTRY.keys()):
            print(f"  {map_}")
        return

    print(f"Running charger_mappo evaluation...")
    print(f"  Map: {args.map_name}")
    print(f"  Agents: {args.num_agents}")
    print(f"  Seed: {args.seed}")
    print(f"  Max steps: {args.max_episode_steps}")
    print(f"  Device: {args.device}")
    print()

    env = create_custom_env(args)
    results = run_charger_mappo(env, path_to_weights=args.path_to_weights, device=args.device)

    print("Episode Results:")
    for key, value in results.items():
        print(f"  {key}: {value}")


if __name__ == '__main__':
    main()
