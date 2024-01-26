import torch
import wandb
from wandb.integration.sb3 import WandbCallback

from src.common.helpers.helpers import save_dict_as_json
from src.common.resource_metrics import get_resource_metrics
import logging
import os
import sys
from typing import Optional
import logging
import os
import sys
from typing import Optional
import traceback

if sys.version_info >= (3, 8):
    from typing import Literal
else:
    from typing_extensions import Literal

from stable_baselines3.common.callbacks import BaseCallback  # type: ignore

import wandb

logger = logging.getLogger(__name__)

class GurskiWandbCallback(WandbCallback):
    """Callback for logging experiments to Weights and Biases.

    Log SB3 experiments to Weights and Biases
        - Added model tracking and uploading
        - Added complete hyperparameters recording
        - Added gradient logging
        - Note that `wandb.init(...)` must be called before the WandbCallback can be used.

    Args:
        verbose: The verbosity of sb3 output
        model_save_path: Path to the folder where the model will be saved, The default value is `None` so the model is not logged
        model_save_freq: Frequency to save the model
        gradient_save_freq: Frequency to log gradient. The default value is 0 so the gradients are not logged
        log: What to log. One of "gradients", "parameters", or "all".
        train_config: custom config for training experiments
    """
    def __init__(
        self,
        verbose: int = 0,
        model_save_path: Optional[str] = None,
        model_save_freq: int = 0,
        gradient_save_freq: int = 0,
        log: Optional[Literal["gradients", "parameters", "all"]] = "all",
    ) -> None:
        super().__init__(verbose, model_save_path, model_save_freq, gradient_save_freq, log)




    def _on_step(self) -> bool:
        try:
            #if self.model_save_freq > 0:
                #if self.model_save_path is not None:
                #    if self.n_calls % self.model_save_freq == 0:
                #        self.save_model()

            if self.n_calls % 100 == 0:
                logging.info(get_resource_metrics())
        except Exception as e:
            logging.error(e)
            traceback.print_exc()

        return True

    def save_model(self) -> None:
        try:
            self.model.save(self.path)
            wandb.save(self.path, base_path=self.model_save_path)
            if self.verbose > 1:
                logger.info(f"Saving model checkpoint to {self.path}")
        except Exception as e:
            logging.error(e)
            traceback.print_exc()

