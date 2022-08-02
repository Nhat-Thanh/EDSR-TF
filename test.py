import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = '3'

from utils.common import *
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

    ls_data = sorted_list("dataset/test/x{}/data".format(scale))
    ls_labels = sorted_list("dataset/test/x{}/labels".format(scale))

    sum_psnr = 0
    for i in range(0, len(ls_data)):
        lr_image = read_image(ls_data[i])
        lr_image = gaussian_blur(lr_image, sigma=sigma)
        hr_image = read_image(ls_labels[i])

        lr_image = norm01(lr_image)
        hr_image = norm01(hr_image)

        lr_image = tf.expand_dims(lr_image, axis=0)
        sr_image = model.predict(lr_image)[0]

        sum_psnr += PSNR(hr_image, sr_image, max_val=1).numpy()

    print(sum_psnr / len(ls_data))

if __name__ == "__main__":
    main()

