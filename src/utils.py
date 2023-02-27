# pip install - q - -upgrade fastai
import fastai
from fastai.vision.all import *

#!pip install -qqq wandb
import wandb
from fastai.callback.wandb import *
from fastai.callback.tracker import SaveModelCallback
