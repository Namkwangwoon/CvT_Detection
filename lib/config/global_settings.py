import os
from datetime import datetime

PASCAL_TRAIN_MEAN = [0.457342265910642, 0.4387686270106377, 0.4073427106250871]
PASCAL_TRAIN_STD = [0.26753769276329037, 0.2638145880487105, 0.2776826934044154]

# MNIST_TRAIN_MEAN = 0.5
# MNIST_TRAIN_STD = 0.25

#directory to save weights file
CHECKPOINT_PATH = 'checkpoint'

#total training epoches
EPOCH = 200
MILESTONES = [60, 120, 160]

#initial learning rate
#INIT_LR = 0.1

DATE_FORMAT = '%A_%d_%B_%Y_%Hh_%Mm_%Ss'
#time of we run the script
TIME_NOW = datetime.now().strftime(DATE_FORMAT)

#tensorboard log dir
LOG_DIR = 'runs'

#save weights file per SAVE_EPOCH epoch
SAVE_EPOCH = 5