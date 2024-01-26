import logging
import socket
import sys
import time
import traceback
from collections import deque
from datetime import timedelta
from typing import Any, Dict, List, Optional, Tuple, Type, TypeVar, Union

import cv2
import numpy as np
import torch as th
import wandb
from PIL import Image
from gymnasium import spaces

from gymnasium.spaces import Box

from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.buffers import DictRolloutBuffer, RolloutBuffer
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.type_aliases import GymEnv, MaybeCallback, Schedule
from stable_baselines3.common.utils import obs_as_tensor, safe_mean
from stable_baselines3.common.vec_env import VecEnv

from src.common.observation_keys import view_key
from src.common.persisted_memory import PersistedMemory
from src.common.resource_metrics import get_resource_metrics
from src.common.transition import Transition

SelfOnPolicyAlgorithm = TypeVar("SelfOnPolicyAlgorithm", bound="OnPolicyAlgorithm")


class GurskiOnPolicyAlgorithm(BaseAlgorithm):
    """
    The base for On-Policy algorithms (ex: A2C/PPO).

    :param policy: The policy model to use (MlpPolicy, CnnPolicy, ...)
    :param env: The environment to learn from (if registered in Gym, can be str)
    :param learning_rate: The learning rate, it can be a function
        of the current progress remaining (from 1 to 0)
    :param n_steps: The number of steps to run for each environment per update
        (i.e. batch size is n_steps * n_env where n_env is number of environment copies running in parallel)
    :param gamma: Discount factor
    :param gae_lambda: Factor for trade-off of bias vs variance for Generalized Advantage Estimator.
        Equivalent to classic advantage when set to 1.
    :param ent_coef: Entropy coefficient for the loss calculation
    :param vf_coef: Value function coefficient for the loss calculation
    :param max_grad_norm: The maximum value for the gradient clipping
    :param use_sde: Whether to use generalized State Dependent Exploration (gSDE)
        instead of action noise exploration (default: False)
    :param sde_sample_freq: Sample a new noise matrix every n steps when using gSDE
        Default: -1 (only sample at the beginning of the rollout)
    :param rollout_buffer_class: Rollout buffer class to use. If ``None``, it will be automatically selected.
    :param rollout_buffer_kwargs: Keyword arguments to pass to the rollout buffer on creation.
    :param stats_window_size: Window size for the rollout logging, specifying the number of episodes to average
        the reported success rate, mean episode length, and mean reward over
    :param tensorboard_log: the log location for tensorboard (if None, no logging)
    :param monitor_wrapper: When creating an environment, whether to wrap it
        or not in a Monitor wrapper.
    :param policy_kwargs: additional arguments to be passed to the policy on creation
    :param verbose: Verbosity level: 0 for no output, 1 for info messages (such as device or wrappers used), 2 for
        debug messages
    :param seed: Seed for the pseudo random generators
    :param device: Device (cpu, cuda, ...) on which the code should be run.
        Setting it to auto, the code will be run on the GPU if possible.
    :param _init_setup_model: Whether or not to build the network at the creation of the instance
    :param supported_action_spaces: The action spaces supported by the algorithm.
    """

    rollout_buffer: RolloutBuffer
    policy: ActorCriticPolicy

    def __init__(
        self,
        policy: Union[str, Type[ActorCriticPolicy]],
        env: Union[GymEnv, str],
        learning_rate: Union[float, Schedule],
        n_steps: int,
        gamma: float,
        gae_lambda: float,
        ent_coef: float,
        vf_coef: float,
        max_grad_norm: float,
        use_sde: bool,
        sde_sample_freq: int,
        rollout_buffer_class: Optional[Type[RolloutBuffer]] = None,
        rollout_buffer_kwargs: Optional[Dict[str, Any]] = None,
        stats_window_size: int = 100,
        tensorboard_log: Optional[str] = None,
        monitor_wrapper: bool = True,
        policy_kwargs: Optional[Dict[str, Any]] = None,
        verbose: int = 0,
        seed: Optional[int] = None,
        device: Union[th.device, str] = "auto",
        _init_setup_model: bool = True,
        supported_action_spaces: Optional[Tuple[Type[spaces.Space], ...]] = None,
    ):
        super().__init__(
            policy=policy,
            env=env,
            learning_rate=learning_rate,
            policy_kwargs=policy_kwargs,
            verbose=verbose,
            device=device,
            use_sde=use_sde,
            sde_sample_freq=sde_sample_freq,
            support_multi_env=True,
            seed=seed,
            stats_window_size=stats_window_size,
            tensorboard_log=tensorboard_log,
            supported_action_spaces=supported_action_spaces,
        )

        self.n_steps = n_steps
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.ent_coef = ent_coef
        self.vf_coef = vf_coef
        self.max_grad_norm = max_grad_norm
        self.rollout_buffer_class = rollout_buffer_class
        self.rollout_buffer_kwargs = rollout_buffer_kwargs or {}

        # override: add sequence length#########
        self.train_config = None

        if "features_extractor_kwargs" in policy_kwargs and "train_config" in policy_kwargs["features_extractor_kwargs"]:
            self.train_config = policy_kwargs["features_extractor_kwargs"]["train_config"]

        if self.train_config is None:
            self.seq_len = 1
        else:
            self.seq_len = self.train_config["input_depth"]

        self.previous_states = deque([], maxlen=self.seq_len)
        self.wandb_image_count = 0
        if self.train_config["persist_transitions"]:
            self.persist_step_counters: dict = {}
            img_height = self.observation_space.shape[1]
            img_width = self.observation_space.shape[2]
            session_id: str = self.train_config["run_id"]
            #self.generation_t = str(datetime.now().strftime("%Y_%m_%d__%H_%M_%S"))
            self.generation_ids: dict = {}
            self.persisted_memories = []
            for agent_n in range(self.train_config["n_envs"]):
                self.persist_step_counters[str(agent_n)] = 0
                self.generation_ids[str(agent_n)] = 0
                agent_id = str(socket.gethostname()) + "_" + str(agent_n)
                local_filesystem_store_root_dir = self.train_config["local_filesystem_store_root_dir"]
                self.persisted_memory: PersistedMemory = PersistedMemory(img_shape=(img_height, img_width, 3),
                                                                         session_id=session_id,
                                                                         agent_id=agent_id,
                                                                         generation_id=str(self.generation_ids[str(agent_n)]),
                                                                         persist_to_local_filesystem=True,
                                                                         local_filesystem_store_root_dir=local_filesystem_store_root_dir)
                self.persisted_memories.append(self.persisted_memory)

        ########################################

        if _init_setup_model:
            self._setup_model()

    def _setup_model(self) -> None:
        self._setup_lr_schedule()
        self.set_random_seed(self.seed)

        if self.rollout_buffer_class is None:
            if isinstance(self.observation_space, spaces.Dict):
                self.rollout_buffer_class = DictRolloutBuffer
            else:
                self.rollout_buffer_class = RolloutBuffer

        self.rollout_buffer = self.rollout_buffer_class(
            self.n_steps,
            self.observation_space,  # type: ignore[arg-type]
            self.action_space,
            device=self.device,
            gamma=self.gamma,
            gae_lambda=self.gae_lambda,
            n_envs=self.n_envs,
            **self.rollout_buffer_kwargs,
        )
        self.policy = self.policy_class(  # type: ignore[assignment]
            self.observation_space, self.action_space, self.lr_schedule, use_sde=self.use_sde, **self.policy_kwargs
        )
        self.policy = self.policy.to(self.device)

    def collect_rollouts(
        self,
        env: VecEnv,
        callback: BaseCallback,
        rollout_buffer: RolloutBuffer,
        n_rollout_steps: int,
    ) -> bool:
        """
        Collect experiences using the current policy and fill a ``RolloutBuffer``.
        The term rollout here refers to the model-free notion and should not
        be used with the concept of rollout used in model-based RL or planning.

        :param env: The training environment
        :param callback: Callback that will be called at each step
            (and at the beginning and end of the rollout)
        :param rollout_buffer: Buffer to fill with rollouts
        :param n_rollout_steps: Number of experiences to collect per environment
        :return: True if function returned with at least `n_rollout_steps`
            collected, False if callback terminated rollout prematurely.
        """
        assert self._last_obs is not None, "No previous observation was provided"
        # Switch to eval mode (this affects batch norm / dropout)
        self.policy.set_training_mode(False)

        n_steps = 0

        # override observation_shape
        observation_shape = self.observation_space.shape
        observation_shape = (observation_shape[0], observation_shape[1], observation_shape[2] * self.seq_len)
        rollout_buffer.observation_space = Box(low=0, high=255, shape=observation_shape, dtype=np.uint8)
        rollout_buffer.obs_shape = rollout_buffer.observation_space.shape
        rollout_buffer.reset()
        # Sample new weights for the state dependent exploration
        if self.use_sde:
            self.policy.reset_noise(env.num_envs)

        callback.on_rollout_start()

        while n_steps < n_rollout_steps:
            if self.use_sde and self.sde_sample_freq > 0 and n_steps % self.sde_sample_freq == 0:
                # Sample a new noise matrix
                self.policy.reset_noise(env.num_envs)

            with th.no_grad():
                # Convert to pytorch tensor or to TensorDict
                # set initial sequence
                while len(self.previous_states) < self.seq_len:
                    self.previous_states.append(self._last_obs)
                # override initial observation with initial sequence
                if self._last_obs.shape[-1] != rollout_buffer.obs_shape[-1]:
                    self._last_obs = self.get_state_seq()
                obs_tensor = obs_as_tensor(self._last_obs, self.device)

                actions, values, log_probs = self.policy(obs_tensor)
            actions = actions.cpu().numpy()

            # Rescale and perform action
            clipped_actions = actions

            if isinstance(self.action_space, spaces.Box):
                if self.policy.squash_output:
                    # Unscale the actions to match env bounds
                    # if they were previously squashed (scaled in [-1, 1])
                    clipped_actions = self.policy.unscale_action(clipped_actions)
                else:
                    # Otherwise, clip the actions to avoid out of bound error
                    # as we are sampling from an unbounded Gaussian distribution
                    clipped_actions = np.clip(actions, self.action_space.low, self.action_space.high)

            new_obs, rewards, dones, infos = env.step(clipped_actions)

            self.num_timesteps += env.num_envs

            # Give access to local variables
            callback.update_locals(locals())
            if not callback.on_step():
                return False

            self._update_info_buffer(infos)
            n_steps += 1

            if isinstance(self.action_space, spaces.Discrete):
                # Reshape in case of discrete action
                actions = actions.reshape(-1, 1)

            # Handle timeout by bootstraping with value function
            # see GitHub issue #633
            for idx, done in enumerate(dones):
                if (
                    done
                    and infos[idx].get("terminal_observation") is not None
                    and infos[idx].get("TimeLimit.truncated", False)
                ):
                    terminal_obs = self.policy.obs_to_tensor(infos[idx]["terminal_observation"])[0]
                    with th.no_grad():
                        terminal_value = self.policy.predict_values(terminal_obs)[0]  # type: ignore[arg-type]
                    rewards[idx] += self.gamma * terminal_value

            rollout_buffer.add(
                self._last_obs,  # type: ignore[arg-type]
                actions,
                rewards,
                self._last_episode_starts,  # type: ignore[arg-type]
                values,
                log_probs,
            )
            try:
                if self.train_config["persist_transitions"]:
                    for agent_n in range(self.train_config["n_envs"]):
                        self.persist_transition_for_agent(actions, dones, new_obs, rewards, agent_n)
            except Exception as e:
                logging.error(e)
                traceback.print_exc()
                wandb.alert(
                    title='Exception' + str(socket.gethostname())[-50:],
                    text="Config: " + str(self.train_config) + "\nError: " + str(e),
                    level=wandb.AlertLevel.INFO,
                    wait_duration=timedelta(minutes=1)
                )

            while len(self.previous_states) < self.seq_len:
                self.previous_states.append(new_obs)
            self.previous_states.append(new_obs)


            new_obs = self.get_state_seq()

            """
            # debug
            seq = np.transpose(seq_new_obs,(0,2,3,1))
            if self.train_config["n_envs"] >=2:
                # batch element 0
                parts = []
                for i in range(self.train_config["n_envs"]):
                    parts.append(seq[0,:,:,i*3:(i+1)*3])
                b1 = np.concatenate(parts, axis=1)

                # batch element 1
                parts2 = []
                for i in range(self.train_config["n_envs"]):
                    parts2.append(seq[1,:,:,i*3:(i+1)*3])
                b2 = np.concatenate(parts2, axis=1)
                res_img = np.concatenate((b1, b2), axis=0)
            else:
                res_img = seq[0, :, :, :3]
            #Image.fromarray(np.transpose(seq_new_obs,(0,2,3,1))[0,:,:,:3], 'RGB').show()
            #Image.fromarray(res_img, 'RGB').show()
            debug_dir = "./tmp/" + self.train_config["env_name"] + "/"
            if not os.path.exists(debug_dir):
                os.makedirs(debug_dir)
            if n_steps < 1080:
                Image.fromarray(res_img, 'RGB').save(debug_dir + str(n_steps) + ".png")
            if dones.any():
                Image.fromarray(res_img, 'RGB').save(debug_dir + str(n_steps) + "Done.png")
            if n_steps == 1:
                Image.fromarray(res_img, 'RGB').save(debug_dir + str(n_steps) + "Begin.png")
            
            try:
                if dones[0] == True and self.wandb_image_count < 2:
                    wandb.log({"agent_screenshot": wandb.Image(np.transpose(new_obs,(0,2,3,1))[0,:,:,:3])}, commit=False)
                    self.wandb_image_count+=1
            except Exception as e:
                logging.error(e)
                traceback.print_exc()
            """


            self._last_obs = new_obs  # type: ignore[assignment]
            self._last_episode_starts = dones

        with th.no_grad():
            # Compute value for the last timestep
            values = self.policy.predict_values(obs_as_tensor(new_obs, self.device))  # type: ignore[arg-type]

        rollout_buffer.compute_returns_and_advantage(last_values=values, dones=dones)

        callback.update_locals(locals())

        callback.on_rollout_end()

        return True

    def persist_transition_for_agent(self, actions, dones, new_obs, rewards, agent_n):
        #Image.fromarray(np.transpose(new_obs, (0, 2, 3, 1))[agent_n, :, :, :3], 'RGB').show()
        state = {}
        state[view_key] = cv2.cvtColor(np.transpose(new_obs, (0, 2, 3, 1))[agent_n, :, :, :3], cv2.COLOR_BGR2RGB)
        action_id = int(actions[agent_n][0])
        reward = float(rewards[agent_n])
        transition: Transition = Transition(t=self.persist_step_counters[str(agent_n)], state=state, action_id=action_id,
                                            action={}, reward=reward,
                                            terminal_state="unknown",
                                            timestamp=time.time())
        self.persisted_memories[agent_n].save_timestep_in_ram(transition=transition)
        self.persist_step_counters[str(agent_n)] += 1
        if dones[agent_n] == True:
            self.generation_ids[str(agent_n)] += 1
            generation_id = self.generation_ids[str(agent_n)]
            self.persisted_memories[agent_n].generation_id = str(generation_id)
            self.persisted_memories[agent_n].save_from_ram_to_persisted(only_delete=False,
                                                             generation_id=str(generation_id))
            self.persist_step_counters[str(agent_n)] = 0

    def get_state_seq(self):
        seq_new_obs = None
        for tmp_state in self.previous_states:
            if seq_new_obs is None:
                seq_new_obs = tmp_state
            else:
                seq_new_obs = np.concatenate((seq_new_obs, tmp_state), axis=3)
        return seq_new_obs

    def train(self) -> None:
        """
        Consume current rollout data and update policy parameters.
        Implemented by individual algorithms.
        """
        raise NotImplementedError

    def learn(
        self: SelfOnPolicyAlgorithm,
        total_timesteps: int,
        callback: MaybeCallback = None,
        log_interval: int = 1,
        tb_log_name: str = "OnPolicyAlgorithm",
        reset_num_timesteps: bool = True,
        progress_bar: bool = False,
    ) -> SelfOnPolicyAlgorithm:
        iteration = 0

        total_timesteps, callback = self._setup_learn(
            total_timesteps,
            callback,
            reset_num_timesteps,
            tb_log_name,
            progress_bar,
        )

        callback.on_training_start(locals(), globals())

        assert self.env is not None

        while self.num_timesteps < total_timesteps:
            continue_training = self.collect_rollouts(self.env, callback, self.rollout_buffer, n_rollout_steps=self.n_steps)

            if not continue_training:
                break

            iteration += 1
            self._update_current_progress_remaining(self.num_timesteps, total_timesteps)

            # Display training infos
            if log_interval is not None and iteration % log_interval == 0:
                assert self.ep_info_buffer is not None
                time_elapsed = max((time.time_ns() - self.start_time) / 1e9, sys.float_info.epsilon)
                fps = int((self.num_timesteps - self._num_timesteps_at_start) / time_elapsed)
                self.logger.record("time/iterations", iteration, exclude="tensorboard")
                if len(self.ep_info_buffer) > 0 and len(self.ep_info_buffer[0]) > 0:
                    self.logger.record("rollout/ep_rew_mean", safe_mean([ep_info["r"] for ep_info in self.ep_info_buffer]))
                    self.logger.record("rollout/ep_len_mean", safe_mean([ep_info["l"] for ep_info in self.ep_info_buffer]))
                self.logger.record("time/fps", fps)
                self.logger.record("time/time_elapsed", int(time_elapsed), exclude="tensorboard")
                self.logger.record("time/total_timesteps", self.num_timesteps, exclude="tensorboard")
                self.logger.dump(step=self.num_timesteps)

            if iteration % 100 == 0:
                logging.info(get_resource_metrics())
            try:
                time_elapsed = max((time.time_ns() - self.start_time) / 1e9, sys.float_info.epsilon)
                fps = int((self.num_timesteps - self._num_timesteps_at_start) / time_elapsed)
                wandb_metrics: dict = {}
                wandb_metrics["time/fps"] = fps
                wandb_metrics["time/time_elapsed"] = int(time_elapsed)
                wandb_metrics["time/total_timesteps"] = self.num_timesteps
                wandb_metrics["time/iterations"] = iteration
                if len(self.ep_info_buffer) > 0 and len(self.ep_info_buffer[0]) > 0:
                    ep_rew_mean = safe_mean([ep_info["r"] for ep_info in self.ep_info_buffer])
                    ep_len_mean = safe_mean([ep_info["l"] for ep_info in self.ep_info_buffer])

                    wandb_metrics["rollout/ep_rew_mean"] = ep_rew_mean
                    wandb_metrics["rollout/ep_len_mean"] = ep_len_mean
                try:
                    wandb_metrics.update(get_resource_metrics())
                except Exception as e:
                    logging.error(e)
                    traceback.print_exc()
                wandb.log(wandb_metrics)

                try:
                    if self.train_config["freeze_pretrained_vision_encoder_weights"] and self.train_config[
                        "unfreeze_pretrained_vision_encoder_weights"]:
                        if self.num_timesteps >= (self.train_config["total_timesteps"] / 10) - 1:
                            success = self.policy.features_extractor.unfreeze_all_layers()
                            if success:
                                print("Unfrozen all layers at step " + str(self.num_timesteps))
                except Exception as e:
                    logging.error(e)
                    traceback.print_exc()
            except Exception as e:
                logging.error(e)
                traceback.print_exc()

            self.train()

        callback.on_training_end()

        return self

    def _get_torch_save_params(self) -> Tuple[List[str], List[str]]:
        state_dicts = ["policy", "policy.optimizer"]

        return state_dicts, []
