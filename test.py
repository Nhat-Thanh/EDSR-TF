import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = '3'

from utils.dataset import DatasetSingleImage
from utils.common import *
from utils.my_functions import MyPreprocess
from model import EDSR 
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--scale',     type=int, default=2,  help='-')
parser.add_argument('--ckpt-path', type=str, default="", help='-')

# -----------------------------------------------------------
# global variables
# -----------------------------------------------------------

FLAGS, unparsed = parser.parse_known_args()
scale = FLAGS.scale
if scale not in [2, 3, 4]:
    raise ValueError("scale must be 2, 3, or 4")

ckpt_path = FLAGS.ckpt_path
if (ckpt_path == "") or (ckpt_path == "default"):
    ckpt_path = "checkpoint/x{0}/EDSR-x{0}.h5".format(scale)

sigma = 0.3 if scale == 2 else 0.2


# -----------------------------------------------------------
# test 
# -----------------------------------------------------------

def main():
    model = EDSR(scale)
    model.load_weights(ckpt_path)

    data_dir = "dataset/test/x{}/data".format(scale)
    label_dir = "dataset/test/x{}/labels".format(scale)
    test_dataset = DatasetSingleImage(data_dir, label_dir)
    test_dataset.load_data(preprocess=MyPreprocess)

    sum_psnr = 0
    isEnd = False
    while isEnd == False:
        lr_image, hr_image, isEnd = test_dataset.get_images()
        sr_image = model.predict(lr_image)
        sum_psnr += PSNR(hr_image, sr_image, max_val=1).numpy()

    print(sum_psnr / test_dataset.length())

if __name__ == "__main__":
    main()

