import numpy as np
import gymnasium
from gymnasium import ObservationWrapper
from gymnasium.spaces import Box, Dict

from charger_mappo.planning import ResettablePlanner, PlannerConfig


class PreprocessorConfig(PlannerConfig):
    network_input_radius: int = 5
    intrinsic_target_reward: float = 0.01


def charger_mappo_preprocessor(env, algo_config):
    env = wrap_preprocessors(env, algo_config.training_config.preprocessing)
    return env


def wrap_preprocessors(env, config: PreprocessorConfig, auto_reset=False):
    env = charger_mappoWrapper(env=env, config=config)
    env = CutObservationWrapper(env, target_observation_radius=config.network_input_radius)
    env = ConcatPositionalFeatures(env)
    # Add global state wrapper for MAPPO critic (centralized training)
    env = GlobalStateWrapper(env)
    if auto_reset:
        env = AutoResetWrapper(env)
    return env


class charger_mappoWrapper(ObservationWrapper):

    def __init__(self, env, config: PreprocessorConfig):
        super().__init__(env)
        self._cfg: PreprocessorConfig = config
        self.re_plan = ResettablePlanner(self._cfg)
        self.prev_goals = None
        self.intrinsic_reward = None
        self.battery_scale = None
        
        obs_space = self.observation_space
        full_size = obs_space['obstacles'].shape[0]
        obs_space['charges'] = Box(0.0, 1.0, shape=(full_size, full_size))
        obs_space['target'] = Box(0.0, 1.0, shape=(full_size, full_size))
        obs_space['battery'] = Box(0.0, 1.0, shape=(full_size, full_size))

    @staticmethod
    def get_relative_xy(x, y, tx, ty, obs_radius):
        dx, dy = x - tx, y - ty
        if dx > obs_radius or dx < -obs_radius or dy > obs_radius or dy < -obs_radius:
            return None, None
        return obs_radius - dx, obs_radius - dy

    def observation(self, observations):
        if self.battery_scale is None and len(observations) > 0:
            initial_battery = observations[0].get('battery', 100)
            self.battery_scale = initial_battery if isinstance(initial_battery, (int, float)) else initial_battery[0] if hasattr(initial_battery, '__len__') else 100
        
        self.re_plan.update(observations)
        paths = self.re_plan.get_path()

        new_goals = []
        intrinsic_rewards = [0.0] * len(observations)

        for k, obs in enumerate(observations):
            path = paths[k] if k < len(paths) else None

            if path is None:
                new_goals.append(obs['target_xy'])
                path = []
            else:
                subgoal_achieved = self.prev_goals and obs['xy'] == self.prev_goals[k]
                intrinsic_rewards[k] = self._cfg.intrinsic_target_reward if subgoal_achieved else 0.0
                new_goals.append(path[1])

            obs['obstacles'][obs['obstacles'] > 0] *= -1

            r = obs['obstacles'].shape[0] // 2
            for idx, (gx, gy) in enumerate(path):
                x, y = self.get_relative_xy(*obs['xy'], gx, gy, r)
                if x is not None and y is not None:
                    obs['obstacles'][x, y] = 1.0
                else:
                    break
            
            obs['charges'] = np.zeros_like(obs['obstacles'])
            for cx, cy in obs.get('charges_xy', []):
                x, y = self.get_relative_xy(*obs['xy'], cx, cy, r)
                if x is not None and y is not None:
                    obs['charges'][x, y] = 1.0
            
            obs['target'] = np.zeros_like(obs['obstacles'])
            tx, ty = obs['target_xy']
            x, y = self.get_relative_xy(*obs['xy'], tx, ty, r)
            if x is not None and y is not None:
                obs['target'][x, y] = 1.0
            
            battery = obs.get('battery', self.battery_scale)
            battery_val = battery if isinstance(battery, (int, float)) else battery[0] if hasattr(battery, '__len__') else self.battery_scale
            obs['battery'] = np.full_like(obs['obstacles'], battery_val / self.battery_scale, dtype=np.float32)
            
        self.prev_goals = new_goals
        self.intrinsic_reward = intrinsic_rewards

        return observations

    def get_intrinsic_rewards(self, reward):
        for agent_idx, r in enumerate(reward):
            reward[agent_idx] = self.intrinsic_reward[agent_idx]
        return reward

    def step(self, action):
        observation, reward, done, tr, info = self.env.step(action)
        return self.observation(observation), self.get_intrinsic_rewards(reward), done, tr, info

    def reset_state(self):
        self.re_plan.reset_states()
        self.battery_scale = None
        if hasattr(self, 'get_global_obstacles'):
            self.re_plan._agent.add_grid_obstacles(self.get_global_obstacles(), self.get_global_agents_xy())
        else:
            self.re_plan._agent.add_grid_obstacles(None, None)

    def reset(self, **kwargs):
        observations, infos = self.env.reset(**kwargs)
        self.reset_state()
        return self.observation(observations), infos


class CutObservationWrapper(ObservationWrapper):
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

    def __init__(self, env):
        super().__init__(env)
        self.to_concat = []

        observation_space = Dict()
        full_size = self.env.observation_space['obstacles'].shape[0]

        for key, value in self.observation_space.items():
            if value.shape == (full_size, full_size):
                self.to_concat.append(key)
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
            observations[agent_idx]['obs'] = main_obs.astype(np.float32)
        return observations

    @staticmethod
    def key_comparator(x):
        if x == 'obstacles':
            return '0_' + x
        elif 'agents' in x:
            return '1_' + x
        elif x == 'charges':
            return '2_' + x
        elif x == 'target':
            return '3_' + x
        elif x == 'battery':
            return '4_' + x
        return '5_' + x


class GlobalStateWrapper(ObservationWrapper):
    """
    MAPPO: Adds global state for centralized critic training.
    
    Critic receives global state:
    - Global obstacles map
    - All agents' positions
    - All chargers' positions  
    - All agents' battery levels
    
    Actor uses only local observations (5-channel).
    """
    
    def __init__(self, env):
        super().__init__(env)
        self._global_obs = None
        self._num_agents = 1
        
        # Try to get grid dimensions
        try:
            if hasattr(env, 'get_global_obstacles'):
                global_obs = env.get_global_obstacles()
                if global_obs is not None:
                    h, w = np.array(global_obs).shape
                else:
                    h, w = 32, 32
            else:
                h, w = 32, 32
        except:
            h, w = 32, 32
        
        self._grid_h, self._grid_w = h, w
        
        # Global state will be added after reset when we know num_agents
        self._global_state_shape = None

    def _get_global_state(self):
        """Build global state from environment."""
        try:
            if hasattr(self.env, 'get_global_obstacles'):
                global_obstacles = np.array(self.env.get_global_obstacles())
            else:
                global_obstacles = np.zeros((self._grid_h, self._grid_w))
            
            if hasattr(self.env, 'get_global_agents_xy'):
                agents_xy = self.env.get_global_agents_xy()
            else:
                agents_xy = []
            
            if hasattr(self.env, 'get_global_charges_xy'):
                charges_xy = self.env.get_global_charges_xy()
            else:
                charges_xy = []
            
            # Get battery - try multiple methods
            batteries = [100] * self._num_agents
            try:
                if hasattr(self.env, 'get_agents_battery'):
                    batteries = self.env.get_agents_battery()
                elif hasattr(self.env, 'grid') and hasattr(self.env.grid, 'get_battery'):
                    batteries = self.env.grid.get_battery()
            except:
                pass
            
            # Normalize
            global_obstacles_norm = global_obstacles.astype(np.float32)
            
            agent_map = np.zeros_like(global_obstacles_norm)
            for x, y in agents_xy:
                if 0 <= x < global_obstacles_norm.shape[0] and 0 <= y < global_obstacles_norm.shape[1]:
                    agent_map[x, y] = 1.0
            
            charger_map = np.zeros_like(global_obstacles_norm)
            for x, y in charges_xy:
                if 0 <= x < global_obstacles_norm.shape[0] and 0 <= y < global_obstacles_norm.shape[1]:
                    charger_map[x, y] = 1.0
            
            batteries_norm = np.array([b / 100.0 for b in batteries], dtype=np.float32)
            
            global_state = np.concatenate([
                global_obstacles_norm.flatten(),
                agent_map.flatten(),
                charger_map.flatten(),
                batteries_norm
            ]).astype(np.float32)
            
            return global_state
        except Exception as e:
            # Fallback: return zeros
            fallback_size = self._grid_h * self._grid_w * 3 + self._num_agents
            return np.zeros(fallback_size, dtype=np.float32)

    def observation(self, observations):
        # Update num_agents
        self._num_agents = len(observations)
        
        # Get global state
        global_state = self._get_global_state()
        
        # Update observation space if needed
        if self._global_state_shape is None:
            self._global_state_shape = global_state.shape
            self.observation_space.spaces['global_state'] = Box(
                low=-1.0, high=1.0, shape=self._global_state_shape, dtype=np.float32
            )
        
        # Add to each observation
        for obs in observations:
            obs['global_state'] = global_state.copy()
        
        return observations

    def reset(self, **kwargs):
        observations, infos = self.env.reset(**kwargs)
        self._num_agents = len(observations)
        observations = self.observation(observations)
        return observations, infos


class AutoResetWrapper(gymnasium.Wrapper):
    def step(self, action):
        observations, rewards, terminated, truncated, infos = self.env.step(action)
        if all(terminated) or all(truncated):
            observations, _ = self.env.reset()
        return observations, rewards, terminated, truncated, infos
