import datetime
import os

import dotenv
import torch

import wandb

torch.set_float32_matmul_precision("medium")
dotenv.load_dotenv(dotenv_path="../../.env")
wandb.login(key=os.environ["WANDB_API_KEY"])
CURRENT_TIME = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
