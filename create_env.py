from gymnasium import Wrapper
from loguru import logger
from pogema_toolbox.create_env import MultiMapWrapper
from pogema.wrappers.metrics import RuntimeMetricWrapper

from pogema import AnimationConfig, AnimationMonitor
from copy import deepcopy
from pogema import pogema_v0
from pogema.generator import generate_new_target, generate_from_possible_targets

class ProvideGlobalObstacles(Wrapper):
    def get_global_obstacles(self):
        return self.grid.get_obstacles().astype(int).tolist()

    def get_global_agents_xy(self):
        return self.grid.get_agents_xy()

def create_env_base(config):
    env = pogema_v0(grid_config=config)
    env = ProvideGlobalObstacles(env)
    env = MultiMapWrapper(env)
    if config.with_animation:
        logger.debug('Wrapping environment with AnimationMonitor')
        env = AnimationMonitor(env, AnimationConfig(save_every_idx_episode=None))

    # Adding runtime metrics
    env = RuntimeMetricWrapper(env)

    return env