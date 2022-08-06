from tensorflow.keras.optimizers import schedules
from utils.common import gaussian_blur, resize_bicubic, norm01


# -----------------------------------------------------------
# Some custom things
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

def MyResizeMethod(src, shape):
    image = gaussian_blur(src, kernel_size=3, sigma=0.5)
    image = resize_bicubic(image, shape)
    return image

def MyPreprocess(data, label):
    return norm01(data), norm01(label)


