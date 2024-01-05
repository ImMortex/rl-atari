# There already exists an environment generator
# that will make and wrap atari environments correctly.
# Here we are also multiprocessing training (num_env=4 => 4 processes)
import logging
import os
import socket
import time
import traceback
from datetime import timedelta

import coloredlogs
import torch
from dotenv import load_dotenv
from stable_baselines3 import A2C
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import VecFrameStack, VecVideoRecorder

import wandb
from config.train_config import set_train_config, get_train_config
from src.overrides.gurski_a2c import GurskiA2C
from src.overrides.observation.gurski_make_atari_env import gurski_make_atari_env

load_dotenv()
device = torch.device("cuda" if torch.cuda.is_available() else torch.device("cpu"))
logging.info("device")
logging.info(str(device))

config: dict = {}
try:
    coloredlogs.install(level='INFO')
    set_train_config()  # reads static env variables and creates config once
    config = get_train_config()
    logging.info("train_config")
    logging.info(config)

    if config["freeze_pretrained_vision_encoder_weights"] and config["unfreeze_pretrained_vision_encoder_weights"]:
        logging.info("Frozen layers will be unfrozen after " + str(config["total_timesteps"]/10) + " steps")

    WANDB_KEY = os.getenv("WANDB_KEY")
    logging.info("login to wandb... ")
    if WANDB_KEY is not None:
        try:
            wandb.login(key=str(WANDB_KEY))
            logging.info("login to wandb done")
        except Exception as e:
            logging.error(e)
            traceback.print_exc()

    monitor_gym = config["video_length"] > 0
    sync_tensorboard = config["use_tensorboard"]

    run = wandb.init(
        project=config["project"],
        group=config["group"],
        config=config,
        sync_tensorboard=sync_tensorboard,  # auto-upload sb3's tensorboard metrics
        monitor_gym=monitor_gym,  # auto-upload the videos of agents playing the game
        save_code=False,  # optional
        id=config["run_id"],
        resume=False,
    )

    model_path = config["output_root_dir"] + "/models/" + config["run_id"]
    video_path = config["output_root_dir"] + "/videos/" + config["run_id"]
    tensorboard_path = config["output_root_dir"] + "/tensorboard/" + config["run_id"]

    if not sync_tensorboard:
        tensorboard_path = None

    if config["in_channels"] == 1:
        env = make_atari_env(config["env_name"], n_envs=config["n_envs"],
                                    seed=config["env_seed"],
                                    env_kwargs={"render_mode": "rgb_array"})  # , env_kwargs={"render_fps": 30})
    elif config["in_channels"] == 3:
        env = gurski_make_atari_env(config["env_name"], n_envs=config["n_envs"],
                                    seed=config["env_seed"],
                                    env_kwargs={"render_mode": "rgb_array"})  # , env_kwargs={"render_fps": 30})
    if monitor_gym:
        env = VecVideoRecorder(
            env,
            video_path,
            record_video_trigger=lambda x: x % config["model_save_freq"] == 0,
            video_length=config["video_length"],
        )
    env = VecFrameStack(env, n_stack=config["n_envs"])

    if config["policy_type"] == "GurskiCnnPolicy":
        model = GurskiA2C(config["policy_type"], env=env, verbose=1,
                          ent_coef=config["entropy_coef"],
                          gamma=config["gamma"],
                          learning_rate=config["learning_rate"],
                          seed=config["env_seed"],
                          tensorboard_log=tensorboard_path,
                          policy_kwargs={"features_extractor_kwargs": {"train_config": config,
                                                                       "features_dim": config[
                                                                           "vision_encoder_out_dim"]}})
    else:
        # using unchanged default NatureCNN with default args
        model = A2C(config["policy_type"], env, verbose=1,
                    tensorboard_log=tensorboard_path)

    callback = None
    """
    callback = GurskiWandbCallback(model_save_path=model_path,
                                             verbose=2,
                                             model_save_freq=config["model_save_freq"], log="all")
    """
    model.learn(total_timesteps=config["total_timesteps"],
                callback=callback)
    try:
        wandb.alert(
            title='finished ' + str(socket.gethostname()[-50:]),
            text=str(socket.gethostname()) + " Config: " + str(config),
            level=wandb.AlertLevel.INFO,
            wait_duration=timedelta(minutes=1)
        )
    except Exception as e:
        logging.error(e)
        traceback.print_exc()

    run.finish()
    #if config["upload_transitions_during_training"]:
    #    upload_transitions_to_minio_bucket()
    logging.info("finished. Waiting")
    while True:
        time.sleep(120)
except Exception as e:
    logging.error(e)
    traceback.print_exc()
    try:
        wandb.alert(
            title='crashed' + str(socket.gethostname())[-50:],  # limit 64 chars
            text=str(socket.gethostname()) + " Config: " + str(config) + "\nError: " + str(e),
            level=wandb.AlertLevel.INFO,
            wait_duration=timedelta(minutes=1)
        )
    except Exception as e:
        logging.error(e)
        traceback.print_exc()
        logging.info("finished. Waiting")
        while True:
            time.sleep(120)

logging.info("finished. Waiting")
while True:
    time.sleep(120)