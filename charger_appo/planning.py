"""
Charger APPO Planning

Reuses the planner from follower. The key difference is in preprocessing
where the target is dynamically switched between goal and charger based
on battery level.
"""
from pogema import GridConfig

# noinspection PyUnresolvedReferences
import cppimport.import_hook
# noinspection PyUnresolvedReferences
from follower_cpp.planner import planner

from pydantic import BaseModel

try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal


class PlannerConfig(BaseModel):
    """Configuration for the planner."""
    use_static_cost: bool = True
    use_dynamic_cost: bool = True
    reset_dynamic_cost: bool = True


class Planner:
    """Path planner for multi-agent navigation."""
    
    def __init__(self, cfg: PlannerConfig, planner_type: str):
        self.planner = None
        self.obstacles = None
        self.starts = None
        self.cfg = cfg
        self.type = planner_type

    def add_grid_obstacles(self, obstacles, starts):
        """Set global obstacles and agent start positions."""
        self.obstacles = obstacles
        self.starts = starts
        self.planner = None

    def update(self, obs):
        """Update planner with current observations."""
        num_agents = len(obs)
        obs_radius = len(obs[0]['obstacles']) // 2
        
        if self.planner is None:
            self.planner = [
                planner(
                    self.obstacles, 
                    self.cfg.use_static_cost, 
                    self.cfg.use_dynamic_cost, 
                    self.cfg.reset_dynamic_cost
                ) 
                for _ in range(num_agents)
            ]
            for i, p in enumerate(self.planner):
                p.set_abs_start(self.starts[i])
            
            if self.cfg.use_static_cost:
                pen_calc = planner(
                    self.obstacles, 
                    self.cfg.use_static_cost, 
                    self.cfg.use_dynamic_cost, 
                    self.cfg.reset_dynamic_cost
                )
                penalties = pen_calc.precompute_penalty_matrix(obs_radius)
                for p in self.planner:
                    p.set_penalties(penalties)
        if self.type == 'target':
            for k in range(num_agents):
                if obs[k]['xy'] == obs[k]['target_xy']:
                    continue
                obs[k]['agents'][obs_radius][obs_radius] = 0
                self.planner[k].update_occupations(
                    obs[k]['agents'], 
                    (obs[k]['xy'][0] - obs_radius, obs[k]['xy'][1] - obs_radius), 
                    obs[k]['target_xy']
                )
                obs[k]['agents'][obs_radius][obs_radius] = 1
                self.planner[k].update_path(obs[k]['xy'], obs[k]['target_xy'])
        elif self.type == 'charger':
            for k in range(num_agents):
                if obs[k]['xy'] == obs[k]['charge_xy']:
                    continue
                obs[k]['agents'][obs_radius][obs_radius] = 0
                self.planner[k].update_occupations(
                    obs[k]['agents'], 
                    (obs[k]['xy'][0] - obs_radius, obs[k]['xy'][1] - obs_radius), 
                    obs[k]['charge_xy']
                )
                obs[k]['agents'][obs_radius][obs_radius] = 1
                self.planner[k].update_path(obs[k]['xy'], obs[k]['charge_xy'])
        else:
            raise ValueError(f"Unknown planner type: {self.type}")

    def get_path(self):
        """Get planned paths for all agents."""
        results = []
        for idx in range(len(self.planner)):
            results.append(self.planner[idx].get_path())
        return results


class ResettablePlanner:
    """Wrapper for planner that supports resetting."""
    
    def __init__(self, cfg: PlannerConfig, planner_type: Literal['target', 'charger']):
        self._cfg = cfg
        self._type = planner_type
        self._agent = None

    def update(self, observations):
        """Update internal planner with observations."""
        return self._agent.update(observations)

    def get_path(self):
        """Get paths from internal planner."""
        return self._agent.get_path()

    def reset_states(self):
        """Reset the internal planner."""
        self._agent = Planner(self._cfg, self._type)
