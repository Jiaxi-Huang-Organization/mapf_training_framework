"""
Charger APPO Smart Output

Outputs agent trajectories in the format:
    Agent 0:(x0,y0,0)->(x1,y1,1)->(x2,y2,2)->...

Usage:
    python charger_smart_output.py \
        --path_to_weights=experiments/train_dir/charger_appo \
        --num_agents=64 \
        --output_dir=./export
"""
import argparse
import os
import math

from env.create_smart_env import create_env_base
from env.smart_maps import MAPS_REGISTRY
from charger_appo.training_config import EnvironmentMazes
from charger_appo.inference import ChargerAppoInferenceConfig, ChargerAppoInference
from charger_appo.preprocessing import charger_appo_preprocessor


def create_custom_env(cfg):
    """
    Creates a custom environment for charger_appo with battery and charger support.
    """
    print(cfg.animation)
    env_cfg = EnvironmentMazes(
        with_animation=cfg.animation,
    )
    env_cfg.grid_config.num_agents = cfg.num_agents
    env_cfg.grid_config.num_charges = cfg.num_charges
    env_cfg.grid_config.map_name = cfg.map_name
    env_cfg.grid_config.seed = cfg.seed
    env_cfg.grid_config.max_episode_steps = cfg.max_episode_steps
    env_cfg.grid_config.observation_type = 'MAPF'
    env = create_env_base(env_cfg)
    return env


def get_map_dims(env):
    """Get map dimensions from the environment."""
    try:
        grid_config = env.unwrapped.grid_config
    except AttributeError:
        grid_config = env.grid_config

    if grid_config.map is not None:
        height = len(grid_config.map)
        width = len(grid_config.map[0]) if height > 0 else 0
        return width, height
    # Fallback: infer from obstacles after reset
    try:
        obstacles = env.unwrapped.grid.get_obstacles()
        r = env.unwrapped.grid.config.obs_radius
        return obstacles.shape[1] - 2 * r, obstacles.shape[0] - 2 * r
    except:
        return None, None


def run_charger_appo_smart_output(env, path_to_weights='model/charger_appo'):
    """
    Run charger appo inference and track trajectories.

    Returns:
        tuple: (trajectories, starts, targets)
    """
    charger_cfg = ChargerAppoInferenceConfig(
        path_to_weights=path_to_weights,
        device='cpu',
    )
    algo = ChargerAppoInference(charger_cfg)

    env = charger_appo_preprocessor(env, charger_cfg)

    # Reset and get initial positions
    algo.reset_states()
    obs, _ = env.reset(seed=env.grid_config.seed)

    # Track trajectories: list of list of (x, y, t)
    num_agents = len(obs)
    trajectories = [[] for _ in range(num_agents)]
    targets = []

    # Record initial positions at t=0
    agents_xy = env.get_agents_xy(only_active=False, ignore_borders=True)
    for i, (x, y) in enumerate(agents_xy):
        trajectories[i].append((x, y, 0))

    # Get targets from global_target_xy
    targets = env.get_targets_xy(only_active=False, ignore_borders=True)

    step_count = 0
    while True:
        actions = algo.act(obs)
        obs, rew, dones, tr, infos = env.step(actions)
        step_count += 1
        agents_xy = env.get_agents_xy(only_active=False, ignore_borders=True)
        for i, (x, y) in enumerate(agents_xy):
            trajectories[i].append((x, y, step_count))

        if all(dones) or all(tr):
            break

    return trajectories, targets


def format_trajectory(trajectory):
    """Format a single trajectory as a string (YX format)."""
    parts = []
    for x, y, t in trajectory:
        parts.append(f"({int(y)},{int(x)},{t})")
    return "->".join(parts)


def format_trajectories(trajectories):
    """Format all trajectories as a multi-line string."""
    lines = []
    for i, traj in enumerate(trajectories):
        lines.append(f"Agent {i}:" + format_trajectory(traj))
    return "\n".join(lines)


def format_scen_file(trajectories, targets, map_name, map_width, map_height):
    """
    Format trajectories as a .scen file (MovingAI format, YX format).

    Format: version 1
    agent_id    map_name    width    height    start_x    start_y    goal_x    goal_y    shortest_distance
    """
    lines = ["version 1"]
    for i, (traj, target) in enumerate(zip(trajectories, targets)):
        if len(traj) < 2:
            continue
        start_y, start_x, _ = traj[0]
        goal_y, goal_x = target
        distance = math.sqrt((goal_x - start_x)**2 + (goal_y - start_y)**2)
        lines.append(f"{i}\t{map_name}\t{map_width}\t{map_height}\t{int(start_x)}\t{int(start_y)}\t{int(goal_x)}\t{int(goal_y)}\t{distance:.8f}")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description='Charger APPO Smart Output')
    parser.add_argument('--animation', action='store_true', 
                        help='Enable animation (default: True)')
    parser.add_argument('--num_agents', type=int, default=64,
                        help='Number of agents (default: %(default)d)')
    parser.add_argument('--num_charges', type=int, default=16,
                        help='Number of charges (default: %(default)d)')
    parser.add_argument('--seed', type=int, default=0,
                        help='Random seed (default: %(default)d)')
    parser.add_argument('--map_name', type=str, default='warehouse',
                        help='Map name (default: %(default)s)')
    parser.add_argument('--max_episode_steps', type=int, default=512,
                        help='Maximum episode steps (default: %(default)d)')
    parser.add_argument('--show_map_names', action='store_true',
                        help='Shows names of all available maps')
    parser.add_argument('--path_to_weights', type=str, default='model/charger_appo',
                        help='Path to model weights (default: %(default)s)')
    parser.add_argument('--output_dir', type=str, default='./export',
                        help='Output directory (default: %(default)s)')
    parser.add_argument('--output_filename', type=str, default=None,
                        help='Output filename (default: auto-generated based on seed and num_agents)')
    args = parser.parse_args()

    if args.show_map_names:
        for map_ in MAPS_REGISTRY:
            print(map_)
        return

    # Check if weights path exists
    if not os.path.exists(args.path_to_weights):
        print(f"\nError: Weights path '{args.path_to_weights}' does not exist.")
        print("Please train a model first or specify a valid path.")
        return

    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Running charger_appo evaluation with {args.num_agents} agents {args.num_charges} charges ...")
    env = create_custom_env(args)
    trajectories, targets = run_charger_appo_smart_output(env, args.path_to_weights)

    # Get map dimensions from environment
    map_width, map_height = get_map_dims(env)

    # Format and save trajectory output
    output = format_trajectories(trajectories)

    if args.output_filename is None:
        args.output_filename = f"charger_{args.map_name}_{args.num_agents}agents_seed{args.seed}.txt"

    output_path = os.path.join(args.output_dir, args.output_filename)
    with open(output_path, 'w') as f:
        f.write(output)

    print(f"Trajectory output saved to: {output_path}")

    # Format and save .scen file
    scen_filename = args.output_filename.replace('.txt', '.scen')
    scen_path = os.path.join(args.output_dir, scen_filename)
    scen_content = format_scen_file(trajectories, targets, f"{args.map_name}.map", map_width, map_height)
    with open(scen_path, 'w') as f:
        f.write(scen_content)

    print(f".scen file saved to: {scen_path}")


if __name__ == '__main__':
    main()