from charger_mappo.model import ActorEncoder, CriticEncoder, EncoderConfig

from sample_factory.algo.utils.context import global_model_factory
from sample_factory.utils.typing import ObsSpace, Config
from sample_factory.model.encoder import Encoder
from tensorboardX import SummaryWriter
from sample_factory.utils.typing import PolicyID
from sample_factory.algo.runners.runner import AlgoObserver, Runner

import numpy as np

from sample_factory.utils.utils import log


def pogema_extra_episodic_stats_processing(*args, **kwargs):
    pass


def pogema_extra_summaries(runner: Runner, policy_id: PolicyID, summary_writer: SummaryWriter, env_steps: int):
    policy_avg_stats = runner.policy_avg_stats
    for key in policy_avg_stats:
        if key in ['reward', 'len', 'true_reward', 'Done']:
            continue

        avg = np.mean(np.array(policy_avg_stats[key][policy_id]))
        summary_writer.add_scalar(key, avg, env_steps)
        log.debug(f'{policy_id}-{key}: {round(float(avg), 3)}')


class CustomExtraSummariesObserver(AlgoObserver):
    def extra_summaries(self, runner: Runner, policy_id: PolicyID, writer: SummaryWriter, env_steps: int) -> None:
        pogema_extra_summaries(runner, policy_id, writer, env_steps)


def register_msg_handlers(cfg: Config, runner: Runner):
    runner.register_episodic_stats_handler(pogema_extra_episodic_stats_processing)
    runner.register_observer(CustomExtraSummariesObserver())


def make_custom_encoder(cfg: Config, obs_space: ObsSpace) -> Encoder:
    """
    Factory function for MAPPO encoder.
    
    This encoder handles both:
    - Actor: uses 'obs' key (local observations)
    - Critic: uses 'global_state' key (centralized training)
    
    The encoder automatically routes to the appropriate sub-encoder based on
    the context (actor vs critic) determined by SampleFactory.
    """
    return ActorEncoder(cfg, obs_space)


def register_custom_model():
    """
    Register custom MAPPO encoder.
    
    MAPPO CTDE is implemented through:
    1. GlobalStateWrapper adds 'global_state' to observations
    2. ActorEncoder processes 'obs' for actor
    3. CriticEncoder processes 'global_state' for critic
    
    SampleFactory's ActorCritic will use the same encoder class, but
    the encoder's forward method receives different keys for actor vs critic.
    """
    global_model_factory().register_encoder_factory(make_custom_encoder)
