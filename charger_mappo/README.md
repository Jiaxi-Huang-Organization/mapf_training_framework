# Charger MAPPO Implementation

**MAPPO (Multi-Agent PPO)** with **CTDE (Centralized Training with Decentralized Execution)** architecture for charger-aware multi-agent navigation.

## Key Features

### CTDE Architecture

**Training (Centralized):**
- **Actor**: Uses local observations (5-channel CNN) → action
- **Critic**: Uses global state (MLP) → value
  - Global obstacles map
  - All agents' positions
  - All charger positions
  - All agents' battery levels
  
**Execution (Decentralized):**
- **Actor only**: Uses local observations
- **Critic not needed**

### Architecture Comparison

| Component        | charger_appo (APPO) | charger_mappo (MAPPO)     |
| ---------------- | ------------------- | ------------------------- |
| **Encoder**      | Single CNN          | **Dual-path (CNN + MLP)** |
| **Actor Input**  | Local obs           | Local obs (same)          |
| **Critic Input** | Local obs           | **Global state**          |
| **Training**     | Decentralized       | **Centralized**           |
| **Execution**    | Decentralized       | Decentralized             |

## Implementation

### MAPPOEncoder with CTDE Routing

```python
# charger_mappo/model.py
class MAPPOEncoder(Encoder):
    def __init__(self, cfg, obs_space):
        # Actor: CNN for local observations
        self.actor_conv_head = nn.Sequential(...)
        
        # Critic: MLP for global state
        self.critic_mlp = nn.Sequential(...)
    
    def forward(self, x):
        # CTDE routing: different paths for actor and critic
        if 'global_state' in x:
            # Critic path: use global state
            return self.critic_mlp(x['global_state'])
        else:
            # Actor path: use local observations
            return self.actor_conv_head(x['obs'])
```

### Global State Construction

```python
# charger_mappo/preprocessing.py
class GlobalStateWrapper(ObservationWrapper):
    def _get_global_state(self):
        # 1. Global obstacles
        global_obstacles = self.env.get_global_obstacles()
        
        # 2. All agents' positions (one-hot)
        agents_xy = self.env.get_global_agents_xy()
        agent_map = np.zeros_like(global_obstacles)
        for x, y in agents_xy:
            agent_map[x, y] = 1.0
        
        # 3. All chargers' positions
        charges_xy = self.env.get_global_charges_xy()
        charger_map = np.zeros_like(global_obstacles)
        for x, y in charges_xy:
            charger_map[x, y] = 1.0
        
        # 4. All agents' battery
        batteries = self.env.get_agents_battery()
        
        # Concatenate
        return np.concatenate([
            global_obstacles.flatten(),
            agent_map.flatten(),
            charger_map.flatten(),
            batteries / 100.0
        ])
```

## Model Architecture

### Actor Path (Local Observations)
```
Input: {'obs': (batch, 5, 11, 11)}
       └─ 5 channels: obstacles, agents, charges, target, battery

ResNet:
  Conv2d(5, 64) → ResBlock × 1 → ReLU
  
Output: (batch, 7744)
```

### Critic Path (Global State)
```
Input: {'global_state': (batch, 3136)}
       └─ 3136 = 32×32×3 (maps) + 64 (agents' battery)

MLP:
  Linear(3136, 256) → ReLU → Linear(256, 128) → ReLU
  
Output: (batch, 128)
```

## Usage

### Training

```bash
# MAPPO training with centralized critic
python train_charger_mappo.py \
    --num_agents=64 \
    --map_name=mazes-s0 \
    --train_for_env_steps=10000000
```

### Evaluation

```bash
# Decentralized execution (actor only)
python charger_mappo_example.py \
    --path_to_weights=model/charger_mappo \
    --num_agents=64 \
    --map_name=mazes-s0
```

## How CTDE Works in SampleFactory

SampleFactory's ActorCritic calls the encoder twice:

```python
# Simplified ActorCritic forward
def forward(self, obs_dict):
    # Actor path
    actor_features = self.encoder(obs_dict)  # Uses 'obs'
    actions = self.actor_head(actor_features)
    
    # Critic path  
    critic_features = self.encoder(obs_dict)  # Uses 'global_state'
    values = self.critic_head(critic_features)
```

Our `MAPPOEncoder.forward()` routes to the appropriate path based on which key is accessed:
- Actor context: accesses `obs_dict['obs']` → CNN path
- Critic context: accesses `obs_dict['global_state']` → MLP path

## Benefits of MAPPO CTDE

1. **Better Credit Assignment**: Critic sees which agents contributed to success
2. **Improved Coordination**: Global info helps learn cooperative charging strategies
3. **Scalable Execution**: Same decentralized execution as independent agents
4. **Charger Awareness**: Critic learns global charger usage patterns

## Files

```
charger_mappo/
├── model.py
│   └── MAPPOEncoder      # Dual-path encoder (CTDE routing)
├── preprocessing.py
│   └── GlobalStateWrapper  # Adds global_state
├── register_training_utils.py
│   └── make_custom_encoder # Registers MAPPOEncoder
└── ...
```

## Requirements

- Python 3.8+
- PyTorch
- Sample Factory
- Pogema (with battery/charger support)
