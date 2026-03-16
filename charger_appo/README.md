# Charger APPO

Charger policy for MAPF with battery-aware navigation. Extends follower with:
1. **Battery-aware target switching**: When battery < threshold, target nearest charger
2. **Scalar feature processing**: xy, target_xy, battery, charge_xy
3. **Follower weight initialization**: Load and optionally freeze follower encoder

## Overview

The charger policy uses the same planner as follower, but dynamically switches the planning target:

```
if battery < charge_threshold:
    target = nearest_charger_xy
else:
    target = goal_xy
```

This creates smooth transition between goal-seeking and charging behavior.

## Architecture

```
Observations
├── Spatial (ResnetEncoder)
│   ├── obstacles (with path标注)
│   ├── agents
│   ├── charges
│   ├── target
│   └── battery (spatial map)
│
└── Scalar (ScalarFeaturesMLP)
    ├── xy (2,)
    ├── target_xy (2,)
    ├── battery (1,)
    └── charge_xy (2,) - nearest charger
```

## Training

### From Scratch

```bash
python train_charger_appo.py \
    --train_dir=experiments/train_dir/charger_appo \
    --charge_threshold=0.3 \
    --train_for_env_steps=1000000
```

### Fine-tune from Follower

```bash
python train_charger_appo.py \
    --follower_checkpoint=experiments/train_dir/follower/xxx \
    --freeze_follower_encoder=True \
    --charge_threshold=0.3 \
    --train_dir=experiments/train_dir/charger_appo_finetuned
```

### Key Parameters

| Parameter                  | Default | Description                                     |
| -------------------------- | ------- | ----------------------------------------------- |
| `follower_checkpoint`      | None    | Path to pre-trained follower checkpoint         |
| `freeze_follower_encoder`  | True    | Whether to freeze encoder parameters            |
| `charge_threshold`         | 0.3     | Battery threshold for charger seeking (0.0-1.0) |
| `charger_intrinsic_reward` | 0.01    | Intrinsic reward for charger subgoals           |
| `use_charger_xy_input`     | True    | Include charge_xy in network input              |

## Inference

```bash
python charger_appo_example.py \
    --path_to_weights=experiments/train_dir/charger_appo \
    --num_agents=64 \
    --charge_threshold=0.3
```

## Configuration

### Environment

```python
from charger_appo.training_config import EnvironmentMazes, DecMAPFConfig

cfg = EnvironmentMazes()
cfg.grid_config.initial_battery = 100
cfg.grid_config.battery_decrement = 1
cfg.grid_config.charge_increment = 3
cfg.grid_config.num_charges = 4  # ~1 per 16 agents
```

### Preprocessing

```python
from charger_appo.preprocessing import PreprocessorConfig

preprocessor_cfg = PreprocessorConfig(
    network_input_radius=5,
    intrinsic_target_reward=0.01,
    charge_threshold=0.3,           # Battery threshold
    charger_intrinsic_reward=0.01,  # Can be higher to prioritize charging
)
```

## Reward Design

Same as follower: **intrinsic reward only**

```python
# When agent reaches subgoal (next point on planned path)
if battery >= threshold:
    reward = intrinsic_target_reward    # 0.01
else:
    reward = charger_intrinsic_reward   # 0.01 (can be different)
```

## Comparison with Follower

| Feature         | Follower              | Charger                           |
| --------------- | --------------------- | --------------------------------- |
| Target          | Always goal_xy        | goal_xy or charger_xy             |
| Input channels  | 3 (obs, agents, path) | 5 (+ charges, battery)            |
| Scalar features | xy, target_xy         | xy, target_xy, battery, charge_xy |
| Encoder         | Trainable             | Can be frozen from follower       |
| Behavior        | Goal-conditioned      | Battery-aware                     |

## Files

- `model.py`: ResnetEncoder with follower weight loading
- `actor_critic.py`: Actor-critic with scalar feature processing
- `preprocessing.py`: Battery-aware target switching
- `planning.py`: Path planner (reused from follower)
- `training_config.py`: Configuration classes
- `register_env.py`: Environment registration
- `register_training_utils.py`: Model registration and parameter freezing
- `training_utils.py`: Training utilities
- `inference.py`: Inference engine

## Tips

1. **Start with frozen encoder**: Fine-tune from follower with frozen encoder first
2. **Adjust charge_threshold**: Higher threshold = more conservative (charge earlier)
3. **Charger reward**: Can increase `charger_intrinsic_reward` to prioritize charging
4. **Monitor battery**: Add metrics for average battery level and charger visits
