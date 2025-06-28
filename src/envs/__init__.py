from functools import partial
from .multiagentenv import MultiAgentEnv
from .starcraft2 import StarCraft2Env
# from smac.env import StarCraft2Env
import sys
import os

def env_fn(env, **kwargs) -> MultiAgentEnv:
    return env(**kwargs)

REGISTRY = {}
REGISTRY["sc2"] = partial(env_fn, env=StarCraft2Env)

if sys.platform == "linux":
    os.environ.setdefault("SC2PATH",
                          "/fs/nexus-scratch/peihong/3rdparty/StarCraftII_2410")
    print(os.environ['SC2PATH'])
    
    # os.environ.setdefault("SC2PATH",
    #                       os.path.join(os.getcwd(), "3rdparty", "StarCraftII"))