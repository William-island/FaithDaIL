from .metadrive_env import HumanInTheLoopEnv

import copy
from utils.print_dict_utils import RecorderEnv





def make_train_env(args):
    if args.env_name == 'metadrive':
        train_env_config=dict(
            use_render=True,  # Open the interface
            manual_control=True,  # Allow receiving control signal from external device

            start_seed = 100,
            num_scenarios = 100,
        )
        train_env = HumanInTheLoopEnv(config=train_env_config)
    elif args.env_name == 'carla':
        from .carla_env import HumanInTheLoopCARLAEnv
        train_env_config=dict(
            obs_mode="birdview",    # choices=["birdview", "first", "birdview42", "firststack"],
            force_fps=20,
            disable_vis=False,
            port=args.port,
            enable_takeover=True,
            env=dict(visualize=dict(location="center"))
        )
        train_env = HumanInTheLoopCARLAEnv(external_config=train_env_config)
    else:
        raise NotImplementedError(f"Environment {args.env_name} not implemented.")
    
    return train_env



def make_eval_env(args):
    if args.env_name == 'metadrive':
        # eval_config = {
        #     "start_seed": 1000,  # We will use the map 1000 ~ 1050 as the default testing environment.
        #     "num_scenarios": 50,

        #     "manual_control": False,
        #     "use_render": False,  # Use the render
        # }
        # eval_env = HumanInTheLoopEnv(config=eval_config)

        baseline_eval_config = dict(manual_control=False, use_render=False,  start_seed=1000, num_scenarios = 100) # main_exp=False
        config = copy.deepcopy(baseline_eval_config)

        env = HumanInTheLoopEnv(config)
        eval_env = RecorderEnv(env)
    elif args.env_name == 'carla':
        from .carla_env import HumanInTheLoopCARLAEnv
        from envs.vec_env import (
            DummyVecEnv,
            VecTransposeImage,
        )
        ## init training env
        test_env_config=dict(
            obs_mode=args.obs_mode,
            force_fps=0,
            disable_vis=True, # True for test
            port=args.port,
            enable_takeover=False, # False for test
            env=dict(visualize=dict(location="center"))
        )



        env = HumanInTheLoopCARLAEnv(external_config=test_env_config)
        env = DummyVecEnv([lambda: env])
        eval_env = VecTransposeImage(env)
    else:
        raise NotImplementedError(f"Environment {args.env_name} not implemented.")
    
    return eval_env