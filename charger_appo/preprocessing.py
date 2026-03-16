"""
Charger APPO Preprocessing

Key modification from follower:
- When battery < charge_threshold, planner targets nearest charger instead of goal
- This creates smooth transition between goal-seeking and charging behavior
"""
import numpy as np
import gymnasium
from gymnasium import ObservationWrapper
from gymnasium.spaces import Box, Dict

from charger_appo.planning import ResettablePlanner, PlannerConfig


class PreprocessorConfig(PlannerConfig):
    """
    Configuration for charger appo preprocessing.
    
    Args:
        network_input_radius: Radius for observation cropping
        intrinsic_target_reward: Reward for achieving subgoals
        charge_threshold: Battery threshold below which agent seeks charger (0.0-1.0)
        charger_intrinsic_reward: Reward for charger subgoals (can be different from target reward)
    """
    network_input_radius: int = 5
    intrinsic_target_reward: float = 0.01
    charge_threshold: float = 0.3  # When battery < 30%, seek charger
    charger_intrinsic_reward: float = 0.01  # Can be higher to prioritize charging


def charger_appo_preprocessor(env, algo_config):
    """Wrap environment with charger appo preprocessing."""
    # Handle both training config and inference config
    if hasattr(algo_config, 'training_config') and algo_config.training_config is not None:
        config = algo_config.training_config.preprocessing
    elif hasattr(algo_config, 'preprocessing'):
        config = algo_config.preprocessing
    else:
        # Use default config
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
    Main wrapper with battery-aware target switching.

    When battery < charge_threshold:
    - Planner targets nearest charger instead of goal
    - Intrinsic reward uses charger_intrinsic_reward

    When battery >= charge_threshold:
    - Planner targets goal (same as follower)
    - Intrinsic reward uses intrinsic_target_reward
    """

    def __init__(self, env, config: PreprocessorConfig):
        super().__init__(env)
        self._cfg: PreprocessorConfig = config
        self.re_plan = ResettablePlanner(self._cfg)
        self.prev_goals = None
        self.intrinsic_reward = None
        self.battery_scale = None  # Will be set from initial battery
        self.nearest_charger_xy = None  # Store nearest charger for each agent

    @staticmethod
    def get_relative_xy(x, y, tx, ty, obs_radius):
        """Convert global coordinates to relative observation coordinates."""
        dx, dy = x - tx, y - ty
        if dx > obs_radius or dx < -obs_radius or dy > obs_radius or dy < -obs_radius:
            return None, None
        return obs_radius - dx, obs_radius - dy

    def _find_nearest_charger(self, obs: dict) -> tuple:
        """
        Find the nearest charger from agent's current position.

        Args:
            obs: Agent observation dictionary

        Returns:
            (cx, cy) of nearest charger, or None if no charger found
        """
        charges_xy = obs.get('charges_xy', [])
        if not charges_xy:
            return None

        # Find nearest charger by Manhattan distance
        agent_x, agent_y = obs['xy']
        min_dist = float('inf')
        nearest_charger = None

        for cx, cy in charges_xy:
            dist = abs(agent_x - cx) + abs(agent_y - cy)
            if dist < min_dist:
                min_dist = dist
                nearest_charger = (cx, cy)

        return nearest_charger

    def _get_battery_level(self, obs: dict) -> float:
        """
        Get normalized battery level (0.0 - 1.0).
        
        Args:
            obs: Agent observation dictionary
            
        Returns:
            Normalized battery level
        """
        battery = obs.get('battery', self.battery_scale)
        if isinstance(battery, (int, float)):
            battery_val = battery
        elif hasattr(battery, '__len__') and len(battery) > 0:
            battery_val = battery[0]
        else:
            battery_val = self.battery_scale
        
        return battery_val / self.battery_scale if self.battery_scale else 1.0

    def _determine_target(self, obs: dict) -> tuple:
        """
        Determine planning target based on battery level.

        Args:
            obs: Agent observation dictionary

        Returns:
            (target_x, target_y, is_charger_target, nearest_charger)
        """
        battery_level = self._get_battery_level(obs)
        nearest_charger = self._find_nearest_charger(obs)

        if battery_level < self._cfg.charge_threshold and nearest_charger is not None:
            # Battery low: target nearest charger
            return nearest_charger[0], nearest_charger[1], True, nearest_charger

        # Battery OK or no charger found: target goal
        return obs['target_xy'][0], obs['target_xy'][1], False, nearest_charger

    def observation(self, observations):
        # Set battery_scale from initial battery value if not set
        if self.battery_scale is None and len(observations) > 0:
            initial_battery = observations[0].get('battery', 100)
            if isinstance(initial_battery, (int, float)):
                self.battery_scale = initial_battery
            elif hasattr(initial_battery, '__len__') and len(initial_battery) > 0:
                self.battery_scale = initial_battery[0]
            else:
                self.battery_scale = 100

        # Determine targets and nearest chargers for each agent
        targets = []
        target_is_charger = []
        nearest_chargers = []
        for obs in observations:
            tx, ty, is_charger, nearest_charger = self._determine_target(obs)
            targets.append((tx, ty))
            target_is_charger.append(is_charger)
            nearest_chargers.append(nearest_charger if nearest_charger else (0, 0))
        
        # Store nearest charger info for ConcatPositionalFeatures
        self.nearest_charger_xy = nearest_chargers

        # Temporarily modify observations for planner
        # Planner uses target_xy to compute paths
        original_targets = [obs['target_xy'] for obs in observations]
        for k, obs in enumerate(observations):
            obs['target_xy'] = targets[k]

        # Update cost penalties and compute paths
        self.re_plan.update(observations)
        paths = self.re_plan.get_path()

        # Restore original targets (for observation consistency)
        for k, obs in enumerate(observations):
            obs['target_xy'] = original_targets[k]

        new_goals = []
        intrinsic_rewards = []

        # Process each agent
        for k, path in enumerate(paths):
            obs = observations[k]
            is_charger_target = target_is_charger[k]

            if path is None or len(path) < 2:
                # No valid path: use current target as goal
                new_goals.append(targets[k])
                intrinsic_rewards.append(0.0)
                path = []
            else:
                # Check if agent reached subgoal
                subgoal_achieved = self.prev_goals and obs['xy'] == self.prev_goals[k]

                # Select reward based on target type
                if is_charger_target:
                    reward_val = self._cfg.charger_intrinsic_reward if subgoal_achieved else 0.0
                else:
                    reward_val = self._cfg.intrinsic_target_reward if subgoal_achieved else 0.0

                intrinsic_rewards.append(reward_val)
                new_goals.append(path[1])

            # Preprocess obstacles: set obstacle values to -1.0
            obs['obstacles'][obs['obstacles'] > 0] *= -1

            # Add path to observation (+1.0 for path cells)
            r = obs['obstacles'].shape[0] // 2
            for idx, (gx, gy) in enumerate(path):
                x, y = self.get_relative_xy(*obs['xy'], gx, gy, r)
                if x is not None and y is not None:
                    obs['obstacles'][x, y] = 1.0
                else:
                    break

        # Update state for next step
        self.prev_goals = new_goals
        self.intrinsic_reward = intrinsic_rewards

        return observations

    def get_intrinsic_rewards(self, reward):
        """Replace environment rewards with intrinsic rewards."""
        for agent_idx in range(len(reward)):
            reward[agent_idx] = self.intrinsic_reward[agent_idx]
        return reward

    def step(self, action):
        observation, reward, done, tr, info = self.env.step(action)
        return (
            self.observation(observation), 
            self.get_intrinsic_rewards(reward), 
            done, 
            tr, 
            info
        )

    def reset_state(self):
        """Reset planner state."""
        self.re_plan.reset_states()
        self.battery_scale = None
        
        if hasattr(self, 'get_global_obstacles'):
            self.re_plan._agent.add_grid_obstacles(
                self.get_global_obstacles(), 
                self.get_global_agents_xy()
            )
        else:
            self.re_plan._agent.add_grid_obstacles(None, None)

        self.prev_goals = None
        self.intrinsic_reward = None

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

    Spatial features (concatenated into 'obs'):
    - obstacles, agents, charges, battery

    Scalar features (preserved as-is):
    - xy: (2,) - agent position
    - target_xy: (2,) - target position
    - charge_xy: (2,) - nearest charger position (same as planner uses)
    - battery: (1,) - scalar battery level

    Note: charge_xy is set by ChargerWrapper to match the planner's target charger.
    """

    def __init__(self, env):
        super().__init__(env)
        self.to_concat = []

        observation_space = Dict()
        full_size = self.env.observation_space['obstacles'].shape[0]

        for key, value in self.env.observation_space.items():
            if value.shape == (full_size, full_size):
                self.to_concat.append(key)
            else:
                # Keep scalar features as-is, but convert charges_xy to charge_xy
                if key == 'charges_xy':
                    # Replace charges_xy (Box for list) with charge_xy (Box for single (2,))
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
            
            # Set charge_xy from ChargerWrapper's stored nearest_charger_xy
            if hasattr(self.env, 'nearest_charger_xy') and self.env.nearest_charger_xy is not None:
                charger = self.env.nearest_charger_xy[agent_idx]
                obs['charge_xy'] = np.array(charger, dtype=np.float32)
            else:
                # Fallback: find nearest charger ourselves
                charges_xy = obs.get('charges_xy', [])
                if charges_xy:
                    agent_x, agent_y = obs['xy']
                    min_dist = float('inf')
                    nearest = (0, 0)
                    for cx, cy in charges_xy:
                        dist = abs(agent_x - cx) + abs(agent_y - cy)
                        if dist < min_dist:
                            min_dist = dist
                            nearest = (cx, cy)
                    obs['charge_xy'] = np.array(nearest, dtype=np.float32)
                else:
                    obs['charge_xy'] = np.array([0, 0], dtype=np.float32)
            
            # Remove charges_xy after converting to charge_xy
            if 'charges_xy' in obs:
                del obs['charges_xy']
            
            observations[agent_idx]['obs'] = main_obs.astype(np.float32)
        return observations

    @staticmethod
    def key_comparator(x):
        """Sort channels in consistent order."""
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


class AutoResetWrapper(gymnasium.Wrapper):
    """Automatically reset environment when all agents are done."""
    
    def step(self, action):
        observations, rewards, terminated, truncated, infos = self.env.step(action)
        if all(terminated) or all(truncated):
            observations, _ = self.env.reset()
        return observations, rewards, terminated, truncated, infos
