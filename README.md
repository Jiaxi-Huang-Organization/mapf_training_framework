## mapf_training_framework
This repository contains training code for pogema-charge environment and adapt for algorithms include
- **charger-appo**(based on **Follower**) in **Sample Factory**
- **charger-iql** in **PyMarl**
- **charger-qmix** in **PyMarl**
- **charger-mappo** in **RLlib** through **PettingZoo**


## Installation:

```bash
pip3 install -r docker/requirements.txt
```

## Inference Example:

To execute the algorithm and produce an animation using pre-trained weights, use the following command:
```bash
python3 example.py
```
```bash
python3 example.py --map_name wfi_warehouse --num_agents 128
python3 example.py --map_name pico_s00_od20_na32 --num_agents 32 --algorithm charger-appo
python3 example.py --map_name pico_s00_od20_na32 --num_agents 32 --algorithm charger-iql
python3 example.py --map_name pico_s00_od20_na32 --num_agents 32 --algorithm charger-mappo
python3 example.py --map_name pico_s00_od20_na32 --num_agents 32 --algorithm charger-qmix
```


## Training:

To train **Follower** from scratch, use the following command:

```bash
python3 main.py  --actor_critic_share_weights=True --batch_size=16384 --env=PogemaMazes-v0 --exploration_loss_coeff=0.023 --extra_fc_layers=1 --gamma=0.9756 --hidden_size=512 --intrinsic_target_reward=0.01 --learning_rate=0.00022 --lr_schedule=constant --network_input_radius=5 --num_filters=64 --num_res_blocks=8 --num_workers=8 --optimizer=adam --ppo_clip_ratio=0.2   --train_for_env_steps=1000000000 --use_rnn=True
```

To train **FollowerLite** from scratch, use the following command:
```bash
python3 main.py  --actor_critic_share_weights=True --batch_size=16384 --env=PogemaMazes-v0 --exploration_loss_coeff=0.0156 --extra_fc_layers=0 --gamma=0.9716 --hidden_size=16 --intrinsic_target_reward=0.01 --learning_rate=0.00013 --lr_schedule=kl_adaptive_minibatch --network_input_radius=3 --num_filters=8 --num_res_blocks=1 --num_workers=4 --optimizer=adam --ppo_clip_ratio=0.2     --train_for_env_steps=20000000 --use_rnn=False
```
The parameters are set to the values used in the paper.

### Testing and Results Visualization 
To eval the main results using [pogema-charge-benchmark](https://github.com/Jiaxi-Huang-Organization/pogema-charge-benchmark)

#### Example Configuration:

```yaml
environment:
  name: Pogema-v0
  on_target: restart
  max_episode_steps: 512
  observation_type: POMAPF
  collision_system: soft  
  map_name: wfi_warehouse
  num_agents:
    grid_search: [ 32, 64, 96, 128, 160, 192 ]
  seed:
    grid_search: [ 0, 1, 2, 3, 4, 5, 6, 7, 8, 9 ]

algorithms:
  Follower:
    name: Follower
    num_process: 4
    parallel_backend: 'balanced_dask'


  No dynamic cost:
    name: Follower
    num_process: 4
    parallel_backend: 'balanced_dask'
    
    override_config:
      preprocessing:
        use_dynamic_cost: False

  No static cost:
    name: Follower
    num_process: 4
    num_threads: 4
    parallel_backend: 'balanced_dask'
    
    override_config:
      preprocessing:
        use_static_cost: False

results_views:
  TabularResults:
    type: tabular
    drop_keys: [ seed ]
    print_results: True

  05-warehouse:
    type: plot
    x: num_agents
    y: avg_throughput
    name: Warehouse $46 \times 33$
```

#### Description of Configuration:

The configuration defines the environment settings and the algorithms used for the experiments. It specifies the following:
- **Environment**: Includes parameters of the POGEMA environment, behavior on target (restart, corresponding to LifeLong), maximum episode steps (512), observation type, collision system, etc. It also sets up grid searches for the number of agents and seed values. The `grid_search` can be used for any environment parameter.
- **Algorithms**: Details the algorithms to be tested. The primary algorithm is **Follower**. Variants include "No dynamic cost" and "No static cost," which override specific preprocessing configurations. All algorithms are configurable to use `4` processes and the `balanced_dask` backend for parallelization, enhancing computational efficiency.
- **Results Views**: Defines how the results will be presented, including tabular and plot views.

This example configuration demonstrates how to set up experiments for the Pogema-v0 environment, varying the number of agents and seeds, and comparing different versions of the Follower algorithm.
#### Raw Data

The raw data, comprising the results of our experiments for Follower and FollowerLite, can be downloaded from the following link:
[Download Raw Data](https://github.com/AIRI-Institute/learn-to-follow/releases/download/v0/learn-to-follow-raw-data.zip)


## Citation:

```bibtex
@inproceedings{skrynnik2024learn,
  title={Learn to Follow: Decentralized Lifelong Multi-Agent Pathfinding via Planning and Learning},
  author={Skrynnik, Alexey and Andreychuk, Anton and Nesterova, Maria and Yakovlev, Konstantin and Panov, Aleksandr},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  volume={38},
  number={16},
  pages={17541--17549},
  year={2024}
}
```

