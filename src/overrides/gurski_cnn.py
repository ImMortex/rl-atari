import logging
import os
from collections import OrderedDict

import gymnasium as gym
import torch
import torch as th
import torch.nn.functional as F
from gymnasium import spaces
from stable_baselines3.common.preprocessing import is_image_space
from stable_baselines3.common.torch_layers import NatureCNN
from torch import nn
from torchvision import models

device = torch.device("cuda" if torch.cuda.is_available() else torch.device("cpu"))


class GurskiCNN(NatureCNN):  # custom cnn GurskiCNN must be subclass of NatureCNN
    """
    CNN from DQN Nature paper:
        Mnih, Volodymyr, et al.
        "Human-level control through deep reinforcement learning."
        Nature 518.7540 (2015): 529-533.

    :param observation_space:
    :param features_dim: Number of features extracted.
        This corresponds to the number of unit for the last layer.
    :param normalized_image: Whether to assume that the image is already normalized
        or not (this disables dtype and bounds checks): when True, it only checks that
        the space is a Box and has 3 dimensions.
        Otherwise, it checks that it has expected dtype (uint8) and bounds (values in [0, 255]).
    """

    def __init__(
            self,
            observation_space: gym.Space,
            features_dim: int = 512,
            normalized_image: bool = False,
            train_config: dict = {}
    ) -> None:
        assert isinstance(observation_space, spaces.Box), (
            "GurskiCNN must be used with a gym.spaces.Box ",
            f"observation space, not {observation_space}",
        )
        super().__init__(observation_space, features_dim)
        # We assume CxHxW images (channels first)
        # Re-ordering will be done by pre-preprocessing or wrapper
        assert is_image_space(observation_space, check_channels=False, normalized_image=normalized_image), (
            "You should use GurskiCNN "
            f"only with images not with {observation_space}\n"
            "(you are probably using `CnnPolicy` instead of `MlpPolicy` or `MultiInputPolicy`)\n"
            "If you are using a custom environment,\n"
            "please check it using our env checker:\n"
            "https://stable-baselines3.readthedocs.io/en/master/common/env_checker.html.\n"
            "If you are using `VecNormalize` or already normalized channel-first images "
            "you should pass `normalize_images=False`: \n"
            "https://stable-baselines3.readthedocs.io/en/master/guide/custom_env.html"
        )
        self.observation_space = observation_space
        self.train_config = train_config  # experiment config
        self.dropout_rate = self.train_config["dropout"]

        model_name: str = self.train_config["net_architecture"].lower()
        del self.linear  # remove from superclass init
        del self.cnn  # remove from superclass init
        self.cnn: nn.Module = None
        self.frozen_layers = 0
        self.unfreeze_layer_id = 0
        if model_name.startswith("res"):
            self.cnn = models.get_model(model_name, num_classes=features_dim)
        elif model_name.lower() == "naturecnn":
            self.cnn = nn.Sequential(
                nn.Conv2d(self.train_config["in_channels"], 32, kernel_size=8, stride=4, padding=0),
                nn.ReLU(),
                nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0),
                nn.ReLU(),
                nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0),
                nn.ReLU(),
                nn.Flatten(),
            )

            # Compute shape by doing one forward pass
            with th.no_grad():
                sample = torch.randn(8, 3, observation_space.shape[1],
                                     observation_space.shape[2] * train_config["input_depth"])
                n_flatten = self.cnn(sample.float()).shape[1]

            self.cnn.append(nn.Sequential(nn.Linear(n_flatten, features_dim), nn.ReLU()))

        logging.info("Model: " + model_name)
        logging.info(self.cnn.__class__.__name__)

        self.train_config = train_config

        pretrained_model_path = train_config["pretrained_vision_encoder_model_path"]
        use_pretrained_weights = (isinstance(pretrained_model_path, str) and
                                  os.path.isfile(pretrained_model_path))
        if use_pretrained_weights:
            logging.info("Loading pretrained vision encoder weights " + pretrained_model_path)
            self.model_weights_dict: OrderedDict = torch.load(pretrained_model_path, map_location=device)
            self.cnn.load_state_dict(self.model_weights_dict)
            logging.info("Pretrained vision encoder weights loaded")
        else:
            logging.info("No pretrained vision encoder weights loaded")

        if use_pretrained_weights and train_config["freeze_pretrained_vision_encoder_weights"]:
            for child in self.cnn.children():
                print("freeze " + str(child))
                self.frozen_layers += 1
                self.unfreeze_layer_id = self.frozen_layers - 1
                for param in child.parameters():
                    param.requires_grad = False
            # for p in self.cnn.parameters():
            #    p.requires_grad = False
            logging.info("Vision encoder weights frozen")

        logging.info(str(self.get_total_parameters()) + " total parameters")
        logging.info(str(self.get_total_trainable_parameters()) + " trainable parameters")

    def forward(self, observations: th.Tensor) -> th.Tensor:
        observations_0 = observations[:, :3, :, :]
        return F.dropout(F.relu(self.cnn(observations_0)), self.dropout_rate)

    def get_total_parameters(self):
        return sum(self.get_parameters())

    def get_parameters(self):
        return [param.nelement() for param in self.parameters()]

    def get_total_trainable_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def unfreeze_all_layers(self):
        if self.train_config["freeze_pretrained_vision_encoder_weights"] and self.train_config[
            "unfreeze_pretrained_vision_encoder_weights"] and self.frozen_layers > 0:
            for child in self.cnn.children():
                print("unfreeze " + str(child))
                self.frozen_layers -= 1
                for param in child.parameters():
                    param.requires_grad = True
            return True
        return False

    def try_unfreeze_next_layer(self):
        if self.frozen_layers <= 0:
            return
        i = 0
        for child in self.cnn.children():
            if i == self.unfreeze_layer_id:
                print("unfreeze " + str(child))
                for param in child.parameters():
                    param.requires_grad = True
                self.unfreeze_layer_id -= 1
                self.frozen_layers -= 1
            i += 1

        logging.info(str(self.get_total_parameters()) + " total parameters")
        logging.info(str(self.get_total_trainable_parameters()) + " trainable parameters")

        return self.frozen_layers
