"""
Charger APPO Training Script

Train a charger policy with modular reward design:
1. Uses follower's encoder weights (frozen)
2. Modular reward system:
   - intrinsic_target_reward: Reward for reaching subgoals
   - on_chargers_reward: Reward for being at charger position
   - on_target_reward: Reward for reaching final goal
   - battery_reward: Per-step reward based on battery level
3. Processes scalar features (xy, target_xy, battery, charge_xy)

Usage:
    # Train from scratch
    python train_charger_appo.py --train_dir=experiments/train_dir/charger_appo

    # Fine-tune from follower checkpoint
    python train_charger_appo.py

    # Adjust reward coefficients
    python train_charger_appo.py \
        --preprocessing.on_chargers_reward=0.05 \
        --preprocessing.battery_reward_coeff=0.02 \
        --preprocessing.battery_reward_type=linear
"""
from sys import argv

from charger_appo.training_config import Experiment
from charger_appo.training_utils import run, create_sf_config


def recursive_update(experiment: dict, key, value):
    """Recursively update a nested dictionary."""
    if key in experiment:
        experiment[key] = value
        return True
    else:
        for k, v in experiment.items():
            if isinstance(v, dict):
                if recursive_update(v, key, value):
                    return True
        return False


def update_dict(target_dict, keys, values):
    """Update dictionary with provided keys and values."""
    for key, value in zip(keys, values):
        if recursive_update(target_dict, key, value):
            print(f'Updated {key} to {value}')
        else:
            raise KeyError(f'Could not find {key} in experiment')


def parse_args_to_items(argv_):
    """Parse command line arguments into key-value pairs."""
    keys = []
    values = []

    for arg in argv_[1:]:
        key, value = arg.split('=')
        key = key.replace('--', '')
        keys.append(key)
        values.append(value)

    return keys, values


def main():
    """Main training entry point."""
    experiment = Experiment()
    experiment = create_sf_config(experiment).__dict__
    keys, values = parse_args_to_items(list(argv))

    # Check all args and replace them in experiment recursively
    update_dict(experiment, keys, values)
    
    # Print training configuration
    print("\n" + "="*60)
    print("Charger APPO Training Configuration")
    print("="*60)
    print(f"Train dir: {experiment.get('train_dir', 'experiments/train_dir/charger_appo')}")
    print(f"Train steps: {experiment.get('train_for_env_steps', 1_000_000)}")
    print(f"\nReward Configuration:")
    print(f"  Intrinsic target reward: {experiment.get('intrinsic_target_reward', 0.01)}")
    print(f"  On charger reward: {experiment.get('on_chargers_reward', 0.02)}")
    print(f"  On target reward: {experiment.get('on_target_reward', 0.05)}")
    print(f"  Battery reward coeff: {experiment.get('battery_reward_coeff', 0.01)}")
    print(f"  Battery reward type: {experiment.get('battery_reward_type', 'linear')}")
    print("="*60 + "\n")
    
    run(config=experiment)


if __name__ == '__main__':
    main()
