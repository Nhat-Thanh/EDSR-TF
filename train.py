import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = '3'

from tensorflow.keras.losses import MeanAbsoluteError
from tensorflow.keras.optimizers import Adam
from utils.dataset import DatasetRandomCrop, DatasetSubsample
from utils.common import PSNR
from utils.my_functions import MyPreprocess, MyResizeMethod, MyLRSchedule
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

root_dataset_dir = "dataset"
train_dataset_dir = os.path.join(root_dataset_dir, "train")
crop_size = 48
hr_shape = (crop_size, crop_size, 3)
lr_shape = (crop_size // scale, crop_size // scale, 3)

train_set = DatasetRandomCrop(train_dataset_dir)
train_set.load_data(lr_shape, hr_shape, resize=MyResizeMethod, preprocess=MyPreprocess)

valid_set = DatasetSubsample(root_dataset_dir, "validation", limit_per_image=50)
valid_set.load_data(lr_shape, hr_shape, resize=MyResizeMethod, preprocess=MyPreprocess)


# -----------------------------------------------------------
#  Train
# -----------------------------------------------------------

def main():
    os.makedirs(os.path.join(root_dataset_dir, "train"), exist_ok=True)
    os.makedirs(os.path.join(root_dataset_dir, "test"), exist_ok=True)
    os.makedirs(os.path.join(root_dataset_dir, "validation"), exist_ok=True)

    lr_schedule = MyLRSchedule(initial_learning_rate=1e-4,
                               decay_steps=100000, 
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

