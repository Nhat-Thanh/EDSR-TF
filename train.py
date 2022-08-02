import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = '3'

from tensorflow.keras.losses import MeanSquaredError, MeanAbsoluteError
from tensorflow.keras.optimizers import Adam, schedules
from utils.dataset import dataset
from utils.common import PSNR
from model import EDSR
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--steps",          type=int, default=100000, help='-')
parser.add_argument("--scale",          type=int, default=2,      help='-')
parser.add_argument("--batch-size",     type=int, default=16,     help='-')
parser.add_argument("--save-every",     type=int, default=1000,   help='-')
parser.add_argument("--save-best-only", type=int, default=0,      help='-')
parser.add_argument("--save-log",       type=int, default=0,      help='-')
parser.add_argument("--ckpt-dir",       type=str, default="",     help='-')


# -----------------------------------------------------------
# My learning rate schedule
# -----------------------------------------------------------

class MyLRSchedule(schedules.LearningRateSchedule):
    def __init__(self, initial_learning_rate, decay_steps, decay_rate):
        self.learning_rate = initial_learning_rate
        self.decay_rate = decay_rate
        self.decay_steps = decay_steps

    def __call__(self, step):
        # if step %% decay_steps == 0 then learning_rate *= decay_rate^1 else learning_rate *= decay_rate^0
        dict_exp_factor = { 0 : 1 }
        key = step % self.decay_steps;
        self.learning_rate *= self.decay_rate ** dict_exp_factor.get(key.ref(), 0)
        return self.learning_rate


# -----------------------------------------------------------
# global variables
# -----------------------------------------------------------

FLAG, unparsed = parser.parse_known_args()
steps = FLAG.steps
batch_size = FLAG.batch_size
save_every = FLAG.save_every
save_log = (FLAG.save_log == 1)
save_best_only = (FLAG.save_best_only == 1)

scale = FLAG.scale
if scale not in [2, 3, 4]:
    raise ValueError("scale must be 2, 3 or 4")

ckpt_dir = FLAG.ckpt_dir
if (ckpt_dir == "") or (ckpt_dir == "default"):
    ckpt_dir = "checkpoint/x{}".format(scale)
model_path = os.path.join(ckpt_dir, "EDSR-x{}.h5".format(scale))


# -----------------------------------------------------------
#  Init datasets
# -----------------------------------------------------------

dataset_dir = "dataset"
hr_crop_size = 48
lr_crop_size = hr_crop_size // scale

train_set = dataset(dataset_dir, "train")
train_set.generate(lr_crop_size, hr_crop_size)
train_set.load_data()

valid_set = dataset(dataset_dir, "validation")
valid_set.generate(lr_crop_size, hr_crop_size)
valid_set.load_data()


# -----------------------------------------------------------
#  Train
# -----------------------------------------------------------

def main():
    os.makedirs(os.path.join(dataset_dir, "train"), exist_ok=True)
    os.makedirs(os.path.join(dataset_dir, "test"), exist_ok=True)
    os.makedirs(os.path.join(dataset_dir, "validation"), exist_ok=True)

    lr_schedule = MyLRSchedule(initial_learning_rate=1e-4,
                               decay_steps=200000, 
                               decay_rate=0.5)
    optimizer = Adam(learning_rate=lr_schedule, beta_1=0.9, beta_2=0.999, epsilon=1e-8)

    model = EDSR(scale)
    model.setup(optimizer=optimizer, loss=MeanAbsoluteError(), 
                model_path=model_path, metric=PSNR)

    model.load_checkpoint(ckpt_dir)

    model.train(train_set, valid_set, batch_size, steps, 
                save_every, save_best_only, save_log, ckpt_dir)

if __name__ == "__main__":
    main()

