import os 
import time
import math
import pickle
import numpy as numpy
from contextlib import nullcontext
import torch
from torch.distributed import destroy_process_group, init_process_group
from torch.nn.parallel import DistributedDataParallel as DistributedDataParallel
import logging

from model import Transformer, ModelArgs
from dataset import PretrainDataset



if __name__ == "__main__":
    