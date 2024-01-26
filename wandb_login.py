import os
import wandb
from dotenv import load_dotenv

load_dotenv()
WANDB_KEY = os.getenv("WANDB_KEY")
wandb.login(key=str(WANDB_KEY))
