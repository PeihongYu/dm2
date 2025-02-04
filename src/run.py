import datetime
import os
import pprint
import time
import threading
import gc
import torch as th
import json

from types import SimpleNamespace as SN
from utils.logging import Logger
from utils.timehelper import time_left, time_str
from os.path import dirname, abspath
from os import makedirs

from learners import REGISTRY as le_REGISTRY
from runners import REGISTRY as r_REGISTRY
from controllers import REGISTRY as mac_REGISTRY
from components.episode_buffer import ReplayBuffer
from components.transforms import OneHot


def run(_run, _config, _log):

    # check args sanity
    _config = args_sanity_check(_config, _log)

    args = SN(**_config)
    args.device = "cuda" if args.use_cuda else "cpu"

    # Create the local results directory
    if args.local_results_path == "":
        args.local_results_path = dirname(dirname(abspath(__file__)))
    makedirs(args.local_results_path, exist_ok=True)

    # setup loggers
    logger = Logger(_log)

    _log.info("Experiment Parameters:")
    experiment_params = pprint.pformat(_config,
                                       indent=4,
                                       width=1)
    _log.info("\n\n" + experiment_params + "\n")

    # configure tensorboard logger
    date_time = datetime.datetime.now().strftime("%m-%d-%H-%M-%S")
    envargs_list = [""] # env args we want to appear in name2q1qqsdwq
    if args.env == "mod_act": 
        envargs_list += [f"env-act={args.env_args['action_mod']}"]
        if args.env_args["action_mod"] == "sticky":
            envargs_list += [f"sticky-prob={args.env_args['sticky_prob']}"]
        elif args.env_args["action_mod"] == "permute": 
            envargs_list += [f"permute={args.env_args['permutation_type']}"]


    algargs_list = [] # [f"act={args.action_selector}"] # alg args we want to appear in name

    namelist = [args.name, args.env, args.label, *envargs_list, *algargs_list, f"seed={args.seed}", date_time]
    namelist = [name.replace("_", "-") for name in namelist if name is not None]
    args.unique_token = "_".join(namelist) 

    try:
        map_name = _config["env_args"]["map_name"]
    except:
        map_name = _config["env_args"]["key"]

    if args.use_tensorboard:
        # tb_logs_direc = os.path.join(args.local_results_path, "tb_logs")
        tb_logs_direc = os.path.join(
            args.local_results_path, "tb_logs", args.env, map_name
        )
        tb_exp_direc = os.path.join(tb_logs_direc, "{}").format(args.unique_token)
        logger.setup_tb(tb_exp_direc)

        # write config file
        config_str = json.dumps(vars(args), indent=4, sort_keys=True)
        with open(os.path.join(tb_exp_direc, "config.json"), "w") as f:
            f.write(config_str)

    # sacred is on by default
    logger.setup_sacred(_run)

    # Run and train
    run_sequential(args=args, logger=logger)

    # Clean up after finishing
    print("Exiting Main")

    print("Stopping all threads")
    for t in threading.enumerate():
        if t.name != "MainThread":
            print("Thread {} is alive! Is daemon: {}".format(t.name, t.daemon))
            t.join(timeout=1)
            print("Thread joined")

    print("Exiting script")

    # Making sure framework really exits
    os._exit(os.EX_OK)

def insert_episode_batch_gail(gails, ep_batch):
    for i, gail in enumerate(gails):
        # shape: (1, ts, n_agents, n_feats)
        obses = ep_batch["obs"][:, :, i, :]
        actions = ep_batch["actions"][:, :, i, :]
        gail.add_agent_data(obses, actions)

def save_traj_data(gails, root_path):
    for i, gail in enumerate(gails):
        gail.save_agent_data("{}/agent_{}".format(root_path, i))
        gail.flush() 

def evaluate_sequential(args, runner):
    if args.save_eval_traj:
        from learners.gail_learner import GailDiscriminator

        assert args.buffer_size >= args.test_nepisode, "Cannot store all of test_nepisodes in replay buffer"
        # only use dataloader functionality of GAIL discrimanator class
        args.gail_mask_ally_feats = False
        gails = [GailDiscriminator(args, 
                                    input_dim=32, 
                                    hidden_dim=64,
                                    device=th.device("cuda"),
                                    max_buffer_eps=args.buffer_size,
                                    epath=None,
                                    agent_idx=i, 
                                    obs_info=None) for i in range(args.n_agents)
                        ]
        for _ in range(args.test_nepisode):
            episode_batch = runner.run(test_mode=False) # when saving traj, want a variety of expert data
            insert_episode_batch_gail(gails, episode_batch)

        save_path = os.path.join(args.local_results_path, "agents_batches", args.unique_token, "_eval")
        os.makedirs(save_path, exist_ok=True)
        save_traj_data(gails, save_path)

    else:
        for _ in range(args.test_nepisode):
            runner.run(test_mode=True)

    if args.save_replay:
        runner.save_replay()

    runner.close_env()

def run_sequential(args, logger):

    # Init runner so we can get env info
    runner = r_REGISTRY[args.runner](args=args, logger=logger)
    print("ENV IS : ", args.env)
    # Set up schemes and groups here
    env_info = runner.get_env_info()
    args.n_agents = env_info["n_agents"]
    args.n_actions = env_info["n_actions"]
    args.state_shape = env_info["state_shape"]
    args.episode_limit = env_info["episode_limit"]

    # Default/Base scheme
    scheme = {
        "state": {"vshape": env_info["state_shape"]},
        "obs": {"vshape": env_info["obs_shape"], "group": "agents"},
        "actions": {"vshape": (1,), "group": "agents", "dtype": th.long},
        "avail_actions": {"vshape": (env_info["n_actions"],), "group": "agents", "dtype": th.int},
        "reward": {"vshape": (1,)},
        "terminated": {"vshape": (1,), "dtype": th.uint8},
    }
    groups = {
        "agents": args.n_agents
    }
    preprocess = {
        "actions": ("actions_onehot", [OneHot(out_dim=args.n_actions)])
    }

    buffer = ReplayBuffer(scheme, groups, args.buffer_size, env_info["episode_limit"] + 1,
                          preprocess=preprocess,
                          device="cpu" if args.buffer_cpu_only else args.device)

    # Setup multiagent controller here
    mac = mac_REGISTRY[args.mac](buffer.scheme, groups, args)

    # Give runner the scheme
    runner.setup(scheme=scheme, groups=groups, preprocess=preprocess, mac=mac)

    # Learner
    learner = le_REGISTRY[args.learner](mac, buffer.scheme, logger, args)

    if args.use_cuda:
        learner.cuda()

    if args.checkpoint_path != "":

        timesteps = []
        timestep_to_load = 0

        if not os.path.isdir(args.checkpoint_path):
            logger.console_logger.info("Checkpoint directiory {} doesn't exist".format(args.checkpoint_path))
            return

        # Go through all files in args.checkpoint_path
        for name in os.listdir(args.checkpoint_path):
            full_name = os.path.join(args.checkpoint_path, name)
            # Check if they are dirs the names of which are numbers
            if os.path.isdir(full_name) and name.isdigit():
                timesteps.append(int(name))

        if args.load_step == 0:
            # choose the max timestep
            timestep_to_load = max(timesteps)
        else:
            # choose the timestep closest to load_step
            timestep_to_load = min(timesteps, key=lambda x: abs(x - args.load_step))
        model_path = os.path.join(args.checkpoint_path, str(timestep_to_load))

        logger.console_logger.info("Loading model from {}".format(model_path))
        learner.load_models(model_path)
        runner.t_env = timestep_to_load

        if args.evaluate or args.save_replay:
            print("EVALUATING")
            evaluate_sequential(args, runner)
            return

    # start training
    episode = 0
    last_test_T = -args.test_interval - 1
    last_log_T = 0
    model_save_time = 0

    start_time = time.time()
    last_time = start_time

    logger.console_logger.info("Beginning training for {} timesteps".format(args.t_max))

    while runner.t_env <= args.t_max:

        # Run for a whole episode at a time
        episode_batch = runner.run(test_mode=False)
        buffer.insert_episode_batch(episode_batch)


        if buffer.can_sample(args.batch_size):
            episode_sample = buffer.sample(args.batch_size)

            # Truncate batch to only filled timesteps
            max_ep_t = episode_sample.max_t_filled()
            episode_sample = episode_sample[:, :max_ep_t]

            if episode_sample.device != args.device:
                episode_sample.to(args.device)

            learner.train(episode_sample, runner.t_env, episode)

        # Execute test runs once in a while
        n_test_runs = max(1, args.test_nepisode // runner.batch_size)
        if (runner.t_env - last_test_T) / args.test_interval >= 1.0:

            logger.console_logger.info("t_env: {} / {}".format(runner.t_env, args.t_max))
            logger.console_logger.info("Estimated time left: {}. Time passed: {}".format(
                time_left(last_time, last_test_T, runner.t_env, args.t_max), time_str(time.time() - start_time)))
            last_time = time.time()

            last_test_T = runner.t_env
            for _ in range(n_test_runs):
                runner.run(test_mode=True)

        if args.save_model and (runner.t_env - model_save_time >= args.save_model_interval or model_save_time == 0):
            model_save_time = runner.t_env
            try:
                map_name = args.env_args["map_name"]
            except:
                map_name = args.env_args["key"]
            save_path = os.path.join(
                args.local_results_path, "models", args.env, map_name, args.unique_token, str(runner.t_env)
            )
            # save_path = os.path.join(args.local_results_path, "models", args.unique_token, str(runner.t_env))
            #"results/models/{}".format(unique_token)
            os.makedirs(save_path, exist_ok=True)
            logger.console_logger.info("Saving models to {}".format(save_path))

            # learner should handle saving/loading -- delegate actor save/load to mac,
            # use appropriate filenames to do critics, optimizer states
            learner.save_models(save_path)

        episode += args.batch_size_run

        if (runner.t_env - last_log_T) >= args.log_interval:
            logger.log_stat("episode", episode, runner.t_env)
            logger.print_recent_stats()
            last_log_T = runner.t_env

    runner.close_env()
    logger.console_logger.info("Finished Training")


def args_sanity_check(config, _log):

    # set CUDA flags
    # config["use_cuda"] = True # Use cuda whenever possible!
    if config["use_cuda"] and not th.cuda.is_available():
        config["use_cuda"] = False
        _log.warning("CUDA flag use_cuda was switched OFF automatically because no CUDA devices are available!")

    if config["test_nepisode"] < config["batch_size_run"]:
        config["test_nepisode"] = config["batch_size_run"]
    else:
        config["test_nepisode"] = (config["test_nepisode"]//config["batch_size_run"]) * config["batch_size_run"]

    return config
