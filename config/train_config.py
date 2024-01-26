from datetime import datetime
import os

from dotenv import load_dotenv

from src.common.helpers.helpers import save_dict_as_json, load_from_json_file

load_dotenv()

config_is_set = False


def str2bool(v):
    return str(v).lower() in ("yes", "true", "t", "1")


def set_train_config():
    global config_is_set

    if config_is_set:
        return

    # Train config is equal on agent and global net side or optional for pretraining trainer. Ensures reproduction of experiments
    train_config: dict = {
        # wandb
        "group": str(os.getenv("GROUP")),
        "project": str(os.getenv("PROJECT")),
        "run_id": str(os.getenv("RUN_ID", datetime.today().strftime('%Y-%m-%d.%H_%M_%S'))),  # set only this varaible manually if specific name is needed

        "env_name": str(os.getenv("ENV_NAME")),
        "in_channels": 3,  # RGB, Hardcoded
        "input_depth": int(os.getenv("INPUT_DEPTH")),
        # depth of 3D image is equal to len state sequence. Default: 1 for simple state
        "ram_gb_limit": int(os.getenv("RAM_GB_LIMIT")),  # ram limit needed because caching is used

        "pretrained_vision_encoder_model_path": str(os.getenv("PRETRAINED_VISION_ENCODER_PATH")),
        "freeze_pretrained_vision_encoder_weights": str2bool(os.getenv("FREEZE_PRETRAINED_VISION_ENCODER_WEIGHTS")),
        "unfreeze_pretrained_vision_encoder_weights": str2bool(os.getenv("UNFREEZE_PRETRAINED_VISION_ENCODER_WEIGHTS")), # unfreeze all vision encoder layers after steps == total_steps/10
        "learning_rate": float(os.getenv("LEARNING_RATE")),  # initial lr
        "optimizer": str(os.getenv("OPTIMIZER")).lower(),  # optimizer name
        "dropout": float(os.getenv("DROPOUT")),  # dropouts for dense layers. Default: 0.
        "vision_encoder_out_dim": int(os.getenv("VISION_ENCODER_OUT_DIM")),
        "net_architecture": str(os.getenv("NET_ARCHITECTURE")).lower(),

        "env_seed": int(os.getenv("ENV_SEED")),
        "entropy_coef": float(os.getenv("ENTROPY_COEF")),  # factor for using entropy loss in loss calculation
        "gamma": float(os.getenv("GAMMA")),  # for reinforcement learning loss
        "upload_transitions_during_training": str2bool(os.getenv("UPLOAD_TRANSITIONS_DURING_TRAINING")),
        "persist_transitions": str2bool(os.getenv("PERSIST_TRANSITIONS")),  # if transitions should be saved on hard disk after epoch
        "persisted_memory_out_dir": str(os.getenv("PERSISTENT_MEMORY_OUT_DIR")),  # Output path if "persist_transitions"

        "policy_type": str(os.getenv("POLICY_TYPE")),
        "total_timesteps": int(float(os.getenv("TOTAL_TIMESTEPS"))),
        "model_save_freq": int(float(os.getenv("MODEL_SAVE_FREQ"))),
        "n_envs": int(os.getenv("N_ENVS")),
        "output_root_dir": str(os.getenv("OUTPUT_ROOT_DIR")),
        "video_length": int(os.getenv("VIDEO_LENGTH")),
        "use_tensorboard": str2bool(os.getenv("USE_TENSORBOARD")),
    }
    # add additional parameters depending on other parameters
    bucket_name_postfix = train_config["env_name"].lower().replace("/","-").replace("_","-")
    train_config["local_filesystem_store_root_dir"] = (str(os.getenv("AGENT_PERSISTENT_MEMORY_OUT_DIR")) + "/"
                                                       + str(os.getenv("MINIO_BUCKET_NAME_PREFIX")) + bucket_name_postfix + "/")
    if not os.path.exists(train_config["local_filesystem_store_root_dir"]):
        os.makedirs(train_config["local_filesystem_store_root_dir"])
    train_config["bucket_name"] = str(os.getenv("MINIO_BUCKET_NAME_PREFIX")) + bucket_name_postfix

    pretrain_mode = "no pretraining"
    pretrained = False
    unfrozen = False
    if 'pretrained_vision_encoder_model_path' in train_config:
        if train_config['pretrained_vision_encoder_model_path'] is not None and train_config['pretrained_vision_encoder_model_path']!= "None":
            pretrained = True
    if 'unfreeze_pretrained_vision_encoder_weights' in train_config:
        if train_config['unfreeze_pretrained_vision_encoder_weights']:
            unfrozen = True
    if pretrained and unfrozen:
        pretrain_mode = "finetuning"

    if pretrained and not unfrozen:
        pretrain_mode = "linear probing"
    train_config["pretrain_mode"] = pretrain_mode  # for wandb logging

    if not os.path.exists("./tmp"):
        os.makedirs("./tmp")
    if not os.path.exists("./wandb_tmp"):
        os.makedirs("./wandb_tmp")
    if not os.path.exists("./used-config"):
        os.makedirs("./used-config")
    temp_path = "./used-config/train_config.json"
    save_dict_as_json(train_config, None, temp_path, sort_keys=False)


    if not os.path.isfile(temp_path):
        raise Exception("Train config was not created.")

    config_is_set = True


def get_train_config():
    return load_from_json_file("./used-config/train_config.json")
