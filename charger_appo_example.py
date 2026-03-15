import argparse

from env.create_env import create_env_base
from env.custom_maps import MAPS_REGISTRY
from utils.eval_utils import run_episode
from charger_appo.training_config import EnvironmentMazes
from charger_appo.inference import chargerAppoInferenceConfig, chargerAppoInference
from charger_appo.preprocessing import charger_appo_preprocessor


def create_custom_env(cfg):
    """
    Creates a custom environment for charger_appo with battery and charger support.
    
    Args:
        cfg: Configuration object with environment parameters
        
    Returns:
        gymnasium.Env: Configured environment
    """
    env_cfg = EnvironmentMazes(with_animation=cfg.animation)
    env_cfg.grid_config.num_agents = cfg.num_agents
    env_cfg.grid_config.num_charges = cfg.num_charges
    env_cfg.grid_config.map_name = cfg.map_name
    env_cfg.grid_config.seed = cfg.seed
    env_cfg.grid_config.max_episode_steps = cfg.max_episode_steps
    return create_env_base(env_cfg)


def run_charger_appo(env, path_to_weights='model/charger_appo', device='cpu'):
    """
    Runs charger_appo algorithm on the given environment.
    
    Args:
        env: The environment to run
        path_to_weights: Path to trained model weights
        device: Device to run inference on ('cpu' or 'cuda')
        
    Returns:
        ResultsHolder: Episode results
    """
    charger_cfg = chargerAppoInferenceConfig(path_to_weights=path_to_weights, device=device)
    algo = chargerAppoInference(charger_cfg)

    env = charger_appo_preprocessor(env, charger_cfg)

    return run_episode(env, algo)


def main():
    parser = argparse.ArgumentParser(description='Charger APPO Inference Script')
    parser.add_argument('--animation', action='store_false', help='Enable animation (default: %(default)s)')
    parser.add_argument('--num_agents', type=int, default=64, help='Number of agents (default: %(default)d)')
    parser.add_argument('--num_charges', type=int, default=16, help='Number of charges (default: %(default)d')
    parser.add_argument('--seed', type=int, default=0, help='Random seed (default: %(default)d)')
    parser.add_argument('--map_name', type=str, default='wfi_warehouse', help='Map name (default: %(default)s)')
    parser.add_argument('--max_episode_steps', type=int, default=128,
                        help='Maximum episode steps (default: %(default)d)')
    parser.add_argument('--show_map_names', action='store_true', help='Shows names of all available maps')

    parser.add_argument('--path_to_weights', type=str, default='model/charger_appo',
                        help='Path to trained model weights (default: %(default)s)')
    parser.add_argument('--device', type=str, default='cpu', choices=['cpu', 'cuda'],
                        help='Device to run inference on (default: %(default)s)')
    parser.add_argument('--test', action='store_true',
                        help='Run in test mode without loading pre-trained weights')

    args = parser.parse_args()

    if args.show_map_names:
        print("Available maps:")
        for map_ in sorted(MAPS_REGISTRY.keys()):
            print(f"  {map_}")
        return

    print(f"Running charger_appo evaluation...")
    print(f"  Map: {args.map_name}")
    print(f"  Agents: {args.num_agents}")
    print(f"  Seed: {args.seed}")
    print(f"  Max steps: {args.max_episode_steps}")
    print(f"  Device: {args.device}")
    print()

    env = create_custom_env(args)
    
    if args.test:
        print("Test mode: Running with random actions...")
        from utils.eval_utils import ResultsHolder
        obs, _ = env.reset(seed=args.seed)
        results = ResultsHolder()
        for step in range(args.max_episode_steps):
            import numpy as np
            actions = [np.random.randint(0, 5) for _ in range(args.num_agents)]
            obs, rew, dones, tr, infos = env.step(actions)
            results.after_step(infos)
            if all(dones) or all(tr):
                break
        print("\nTest episode completed!")
        print("Results:", results.get_final())
    else:
        results = run_charger_appo(env, path_to_weights=args.path_to_weights, device=args.device)
        print("Episode Results:")
        for key, value in results.items():
            print(f"  {key}: {value}")


if __name__ == '__main__':
    main()
