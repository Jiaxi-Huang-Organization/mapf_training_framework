from create_env import create_env_base
from pogema_toolbox.create_env import Environment
from pogema_toolbox.evaluator import evaluation
from pogema import BatchAStarAgent

from pathlib import Path
import wandb

import yaml

from pogema_toolbox.registry import ToolboxRegistry

from follower.inference import FollowerInference, FollowerInferenceConfig
from follower.preprocessing import follower_preprocessor
from follower_cpp.inference import FollowerConfigCPP, FollowerInferenceCPP
from follower_cpp.preprocessing import follower_cpp_preprocessor
from charger_appo.inference import ChargerAppoInference, ChargerAppoInferenceConfig
from charger_appo.preprocessing import charger_appo_preprocessor

PROJECT_NAME = 'pogema-charge-evaluation'
BASE_PATH = Path('experiments')


def main(disable_wandb=False):
    ToolboxRegistry.register_env('Environment', create_env_base, Environment)
    #ToolboxRegistry.register_algorithm('A*', BatchAStarAgent)
    ToolboxRegistry.register_algorithm('Follower', FollowerInference, FollowerInferenceConfig,
                                       follower_preprocessor)
    ToolboxRegistry.register_algorithm('FollowerLite', FollowerInferenceCPP, FollowerConfigCPP,
                                       follower_cpp_preprocessor)
    ToolboxRegistry.register_algorithm('ChargerAppo', ChargerAppoInference, ChargerAppoInferenceConfig, 
                                       charger_appo_preprocessor)


    folder_names = [
        #'01-random-20x20',
        #'02-mazes',
        #'03-dense',
        #'04-Paris_1',
        #'04-movingai',
        #'05-puzzles',
        '06-warehouse',
    ]
    for folder in folder_names:
        map_path = BASE_PATH / folder / "maps.yaml"
        with open(map_path, 'r') as f:
            maps_to_register = yaml.safe_load(f)
        ToolboxRegistry.register_maps(maps_to_register)        
        config_path = BASE_PATH / folder / f"{Path(folder).name}_charger.yaml"
        eval_dir = BASE_PATH / folder

        with open(config_path) as f:
            evaluation_config = yaml.safe_load(f)

        evaluation(evaluation_config, eval_dir=eval_dir)


if __name__ == '__main__':
    main()
