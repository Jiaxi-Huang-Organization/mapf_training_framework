"""
Charger APPO Preprocessing with Modular Reward Design

Reward Components:
1. intrinsic_target_reward: Reward for reaching subgoals along the path
2. on_chargers_reward: Reward for being at charger position (encourages charging)
3. on_target_reward: Reward for reaching final goal
4. battery_reward: Per-step reward based on battery level (encourages high battery)
5. early_death_penalty: Penalty for dying early (encourages staying alive by charging)
"""
import numpy as np
import gymnasium
from gymnasium import ObservationWrapper
from gymnasium.spaces import Box, Dict

from charger_appo.planning import ResettablePlanner, PlannerConfig


class PreprocessorConfig(PlannerConfig):
    """
    Configuration for charger appo preprocessing with modular reward design.
    """
    network_input_radius: int = 5

    # Reward component 1: Subgoal achievement
    intrinsic_target_reward: float = 0.01
    # Reward component 2: Being at charger
    on_chargers_reward: float = 0.05  # Can be higher to prioritize charging
    on_target_reward: float = 0.05

    # Reward component 3: Battery level (per-step)
    battery_reward_coeff: float = 0.01
    battery_reward_type: str = 'linear'  # 'linear', 'squared', or 'threshold'
    battery_reward_threshold: float = 0.3  # For 'threshold' reward type

    # Early death penalty
    early_death_penalty_coeff: float = 0.05  # Coefficient for early death penalty


def charger_appo_preprocessor(env, algo_config):
    """Wrap environment with charger appo preprocessing."""
    if hasattr(algo_config, 'training_config') and algo_config.training_config is not None:
        config = algo_config.training_config.preprocessing
    elif hasattr(algo_config, 'preprocessing'):
        config = algo_config.preprocessing
    else:
        config = PreprocessorConfig()

    env = wrap_preprocessors(env, config=config, auto_reset=False)
    return env


def wrap_preprocessors(env, config: PreprocessorConfig, auto_reset=False):
    """Apply all preprocessing wrappers."""
    env = ChargerWrapper(env=env, config=config)
    env = CutObservationWrapper(env, target_observation_radius=config.network_input_radius)
    env = ConcatPositionalFeatures(env)
    if auto_reset:
        env = AutoResetWrapper(env)
    return env


class ChargerWrapper(ObservationWrapper):
    """
    Main wrapper with modular reward design.
    """

    def __init__(self, env, config: PreprocessorConfig):
        super().__init__(env)
        self._cfg: PreprocessorConfig = config
        self.re_plan = ResettablePlanner(self._cfg, planner_type='target')
        self.ch_plan = ResettablePlanner(self._cfg, planner_type='charger')
        self.prev_goals = None
        self.intrinsic_reward = None
        self.position_reward = None
        self.battery_reward = None
        self.avg_intrinsic_reward = []
        self.avg_position_reward = []
        self.avg_battery_reward = []
        self.avg_early_death_penalty = []
        self.battery_scale = None
        self.max_episode_length = None
        self.current_length = 0

    @staticmethod
    def get_relative_xy(x, y, tx, ty, obs_radius):
        """Convert global coordinates to relative observation coordinates."""
        dx, dy = x - tx, y - ty
        if dx > obs_radius or dx < -obs_radius or dy > obs_radius or dy < -obs_radius:
            return None, None
        return obs_radius - dx, obs_radius - dy

    def _find_nearest_charger(self, obs: dict) -> tuple:
        """Find the nearest charger from agent's current position."""
        charges_xy = obs.get('charges_xy', [])
        if not charges_xy:
            return None
        
        agent_x, agent_y = obs['xy']
        min_dist = float('inf')
        nearest_charger = None
        
        for cx, cy in charges_xy:
            dist = abs(agent_x - cx) + abs(agent_y - cy)
            if dist < min_dist:
                min_dist = dist
                nearest_charger = (cx, cy)
        
        return nearest_charger

    def _compute_intrinsic_reward(self, subgoal_achieved: bool) -> float:
        """Component 1: Intrinsic reward for reaching subgoals."""
        return self._cfg.intrinsic_target_reward if subgoal_achieved else 0.0

    def _compute_position_reward(self, obs: dict, reached_goal: bool = False) -> float:
        """Component 2: Position-based reward (charger/goal)."""
        reward = 0.0
        
        # Check if agent is at a charger
        agent_xy = obs['xy']
        charges_xy = obs.get('charges_xy', [])
        
        for charger_xy in charges_xy:
            if agent_xy[0] == charger_xy[0] and agent_xy[1] == charger_xy[1]:
                reward += self._cfg.on_chargers_reward
                break
        
        # Bonus for reaching final goal
        if reached_goal:
            reward += self._cfg.on_target_reward
        
        return reward

    def _compute_battery_reward(self, obs: dict) -> float:
        """Component 3: Battery-based per-step reward."""
        battery_level = obs.get('battery')

        if self._cfg.battery_reward_type == 'linear':
            return self._cfg.battery_reward_coeff * battery_level
        elif self._cfg.battery_reward_type == 'squared':
            return self._cfg.battery_reward_coeff * (battery_level ** 2)
        elif self._cfg.battery_reward_type == 'threshold':
            if battery_level > self._cfg.battery_reward_threshold:
                return self._cfg.battery_reward_coeff
            return 0.0
        else:
            return self._cfg.battery_reward_coeff * battery_level

    def observation(self, observations):
        # Set battery_scale from initial battery
        if self.battery_scale is None and len(observations) > 0:
            initial_battery = observations[0].get('battery')
            if isinstance(initial_battery, (int, float)):
                self.battery_scale = initial_battery
            elif hasattr(initial_battery, '__len__') and len(initial_battery) > 0:
                self.battery_scale = initial_battery[0]

        # Update planner
        self.re_plan.update(observations)
        paths = self.re_plan.get_path()

        new_goals = []
        intrinsic_rewards = []
        position_rewards = []
        battery_rewards = []
        nearest_chargers = []
        true_paths = []
        # Set charge_xy from stored nearest_charger_xy
        for agent_idx, obs in enumerate(observations):
            # Proprocess battery(Normalize)
            observations[agent_idx]['battery'] = observations[agent_idx]['battery'] / self.battery_scale  
            # Find and store nearest charger
            nearest_charger = self._find_nearest_charger(observations[agent_idx])
            observations[agent_idx]['charge_xy'] = nearest_charger
        # Update charger planner
        self.ch_plan.update(observations)
        charge_paths = self.ch_plan.get_path()
        # Check if battery is low - add charger path to observation
        for agent_idx, obs in enumerate(observations):
            battery_level = obs['battery']
            if battery_level <= self._cfg.battery_reward_threshold and charge_paths:
                true_paths.append(charge_paths[agent_idx])
            elif battery_level > self._cfg.battery_reward_threshold and paths:
                true_paths.append(paths[agent_idx])
            else:
                raise ValueError('No paths found')
        for k, path in enumerate(true_paths):
            obs = observations[k]
            # Preprocess obstacles
            obs['obstacles'][obs['obstacles'] > 0] *= -1          
            # Add path to observation (target path)
            r = obs['obstacles'].shape[0] // 2
            for idx, (gx, gy) in enumerate(path):
                x, y = self.get_relative_xy(*obs['xy'], gx, gy, r)
                if x is not None and y is not None:
                    obs['obstacles'][x, y] = 1.0
                else:
                    break
            if path is None or len(path) < 2:
                battery_level = obs['battery']
                if battery_level <= self._cfg.battery_reward_threshold:
                    new_goals.append(obs['charge_xy'])
                else:
                    new_goals.append(obs['target_xy'])

                path = []
            else:
                # Check subgoal achievement
                subgoal_achieved = self.prev_goals and obs['xy'] == self.prev_goals[k]

                # Component 1: Intrinsic reward
                intrinsic_rewards.append(self._compute_intrinsic_reward(subgoal_achieved))

                # Component 2: Position reward
                reached_goal = (obs['xy'] == obs['target_xy'])
                position_rewards.append(self._compute_position_reward(obs, reached_goal))

                # Component 3: Battery reward
                battery_rewards.append(self._compute_battery_reward(obs))

                new_goals.append(path[1])
        # Store state
        self.prev_goals = new_goals
        self.intrinsic_reward = intrinsic_rewards
        self.position_reward = position_rewards
        self.battery_reward = battery_rewards

        return observations

    def get_intrinsic_rewards(self, reward):
        for agent_idx in range(len(reward)):
            reward[agent_idx] = self.intrinsic_reward[agent_idx]
        return reward

    def get_position_rewards(self, reward):
        for agent_idx in range(len(reward)):
            reward[agent_idx] = self.position_reward[agent_idx]
        return reward

    def get_battery_rewards(self, reward):
        for agent_idx in range(len(reward)):
            reward[agent_idx] = self.battery_reward[agent_idx]
        return reward

    def step(self, action):
        observation, reward, done, tr, info = self.env.step(action)
        if self.max_episode_length is None:
            self.max_episode_length = self.env.grid_config.max_episode_steps
        # Increment episode length counter
        self.current_length += 1

        # 1. Update observation and compute rewards
        observation = self.observation(observation)
        # 2. Compute total reward for each agent including early death penalty
        total_reward = []
        early_death_penalties = []
        
        for agent_idx in range(len(reward)):
            # Compute early death penalty if agent died early
            early_death_penalty = 0.0
            if done[agent_idx] or tr[agent_idx]:
                episode_progress = self.current_length / self.max_episode_length
                # Penalty is higher when episode ends earlier
                early_death_penalty = -self._cfg.early_death_penalty_coeff * (1.0 - episode_progress)
            
            early_death_penalties.append(early_death_penalty)
            
            total = (
                self.intrinsic_reward[agent_idx] +
                self.position_reward[agent_idx] +
                self.battery_reward[agent_idx] +
                early_death_penalty
            )
            total_reward.append(total)
        # Track average rewards for episode statistics
        if self.intrinsic_reward:
            self.avg_intrinsic_reward.append(np.mean(self.intrinsic_reward))
        if self.position_reward:
            self.avg_position_reward.append(np.mean(self.position_reward))
        if self.battery_reward:
            self.avg_battery_reward.append(np.mean(self.battery_reward))
        if early_death_penalty:
            self.avg_early_death_penalty.append(np.mean(early_death_penalties))
        # Store episode statistics only when episode ends
        if all(done) or all(tr):
            if info and len(info) > 0:
                if 'episode_extra_stats' not in info[0]:
                    info[0]['episode_extra_stats'] = {}
                info[0]['episode_extra_stats']['avg_intrinsic_reward'] = np.mean(self.avg_intrinsic_reward) if self.avg_intrinsic_reward else 0.0
                info[0]['episode_extra_stats']['avg_position_reward'] = np.mean(self.avg_position_reward) if self.avg_position_reward else 0.0
                info[0]['episode_extra_stats']['avg_battery_reward'] = np.mean(self.avg_battery_reward) if self.avg_battery_reward else 0.0
                info[0]['episode_extra_stats']['avg_early_death_penalty'] = np.mean(self.avg_early_death_penalty) if self.avg_early_death_penalty else 0.0

        return observation, total_reward, done, tr, info

    def reset_state(self):
        self.re_plan.reset_states()
        self.ch_plan.reset_states()
        if hasattr(self, 'get_global_obstacles'):
            self.re_plan._agent.add_grid_obstacles(
                self.get_global_obstacles(),
                self.get_global_agents_xy()
            )
            self.ch_plan._agent.add_grid_obstacles(
                self.get_global_obstacles(),
                self.get_global_agents_xy()
            )

        self.prev_goals = None
        self.intrinsic_reward = None
        self.position_reward = None
        self.battery_reward = None
        self.avg_intrinsic_reward = []
        self.avg_position_reward = []
        self.avg_battery_reward = []
        self.avg_early_death_penalty = []
        self.current_length = 0

    def reset(self, **kwargs):
        observations, infos = self.env.reset(**kwargs)
        self.reset_state()
        return self.observation(observations), infos


class CutObservationWrapper(ObservationWrapper):
    """Crop observations to target radius."""

    def __init__(self, env, target_observation_radius):
        super().__init__(env)
        self._target_obs_radius = target_observation_radius
        self._initial_obs_radius = self.env.observation_space['obstacles'].shape[0] // 2

        for key, value in self.observation_space.items():
            d = self._initial_obs_radius * 2 + 1
            if value.shape == (d, d):
                r = self._target_obs_radius
                self.observation_space[key] = Box(0.0, 1.0, shape=(r * 2 + 1, r * 2 + 1))

    def observation(self, observations):
        tr = self._target_obs_radius
        ir = self._initial_obs_radius
        d = ir * 2 + 1

        for obs in observations:
            for key, value in obs.items():
                if hasattr(value, 'shape') and value.shape == (d, d):
                    obs[key] = value[ir - tr:ir + tr + 1, ir - tr:ir + tr + 1]

        return observations


class ConcatPositionalFeatures(ObservationWrapper):
    """
    Concatenate spatial features and preserve scalar features.
    
    Spatial: obstacles, agents, charges, battery
    Scalar: xy, target_xy, charge_xy (nearest), battery
    """

    def __init__(self, env):
        super().__init__(env)
        self.to_concat = []

        observation_space = Dict()
        full_size = self.env.observation_space['obstacles'].shape[0]

        for key, value in self.observation_space.items():
            if value.shape == (full_size, full_size):
                self.to_concat.append(key)
            else:
                if key == 'charges_xy':
                    observation_space['charge_xy'] = Box(-1024, 1024, (2,), np.int64)
                else:
                    observation_space[key] = value

        obs_shape = (len(self.to_concat), full_size, full_size)
        observation_space['obs'] = Box(0.0, 1.0, shape=obs_shape)
        self.to_concat.sort(key=self.key_comparator)
        self.observation_space = observation_space

    def observation(self, observations):
        for agent_idx, obs in enumerate(observations):
            main_obs = np.concatenate([obs[key][None] for key in self.to_concat])
            for key in self.to_concat:
                del obs[key]

            for key in obs:
                obs[key] = np.array(obs[key], dtype=np.float32)

            if 'charges_xy' in obs:
                del obs['charges_xy']
            observations[agent_idx]['obs'] = main_obs.astype(np.float32)
        return observations

    @staticmethod
    def key_comparator(x):
        if x == 'obstacles':
            return '0_' + x
        elif 'agents' in x:
            return '1_' + x
        elif x == 'target':
            return '2_' + x
        elif x == 'charges':
            return '3_' + x
        elif x == 'battery':
            return '4_' + x
        return '5_' + x


class AutoResetWrapper(gymnasium.Wrapper):
    """Automatically reset environment when all agents are done."""

    def step(self, action):
        observations, rewards, terminated, truncated, infos = self.env.step(action)
        if all(terminated) or all(truncated):
            observations, _ = self.env.reset()
        return observations, rewards, terminated, truncated, infos
