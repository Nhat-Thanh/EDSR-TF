from abc import abstractmethod
from utils.common import *
import tensorflow as tf
import numpy as np
import os
import random

def DefaultPreprocess(data, label):
    return data, label

class DatasetBase:
    def __init__(self, dataset_dir) -> None:
        h, w, c = 0, 0, 3
        self.lr_shape = (h, w, c)
        self.hr_shape = (h, w, c)
        if not exists(dataset_dir):
            raise FileNotFoundError("Can not find the path: \"{}\"".format(dataset_dir))
        self.dataset_dir = dataset_dir

    def load_data(self, lr_shape, hr_shape):
        self.lr_shape = lr_shape
        self.hr_shape = hr_shape

    @abstractmethod 
    def get_batch(self, batch_size, shuffle_each_epoch=True):
        pass


class DatasetRandomCrop(DatasetBase):
    def __init__(self, dataset_dir) -> None:
        super().__init__(dataset_dir)
        self.image_paths = [""]
        self.images = []
        self.cur_idx = 0
        self.preprocess = DefaultPreprocess

    def load_data(self, lr_shape, hr_shape, resize=resize_bicubic, preprocess=DefaultPreprocess):
        super().load_data(lr_shape, hr_shape)
        self.image_paths = sorted_list(self.dataset_dir)
        for path in self.image_paths:
            image = read_image(path)
            self.images.append(image)
        self.preprocess = preprocess
        self.resize = resize

    def get_batch(self, batch_size, shuffle_each_epoch=True):
        h, w, c = self.lr_shape
        data = np.zeros(shape=(batch_size, h, w, c), dtype=np.float32)
        h, w, c = self.hr_shape
        label = np.zeros(shape=(batch_size, h, w, c), dtype=np.float32)
        isEnd = False
        if (self.cur_idx + batch_size) >= len(self.image_paths):
            isEnd = True
            self.cur_idx = 0
            if shuffle_each_epoch:
                random.shuffle(self.image_paths)

        for i in range(batch_size):
            image = self.images[self.cur_idx + i]
            subim_label = random_crop(image, self.hr_shape)
            subim_label = random_transform(subim_label)
            subim_data = self.resize(subim_label, self.lr_shape)
            subim_data, subim_label = self.preprocess(subim_data, subim_label)

            data[i] = subim_data.numpy()
            label[i] = subim_label.numpy()

        self.cur_idx += batch_size

        return data, label, isEnd

class DatasetSubsample(DatasetBase):
    def __init__(self, root_dataset_dir, subset, limit_per_image=-1) -> None:
        super().__init__(root_dataset_dir)
        self.dataset_dir = os.path.join(root_dataset_dir, subset)
        self.data_file = os.path.join(root_dataset_dir, "data_{}.npy".format(subset))
        self.label_file = os.path.join(root_dataset_dir, "label_{}.npy".format(subset))
        self.list_image_path = [""]
        self.cur_idx = 0
        self.data = None
        self.label = None
        self.limit_per_image = limit_per_image

    def load_data(self, lr_shape, hr_shape, resize=resize_bicubic, preprocess=DefaultPreprocess):
        super().load_data(lr_shape, hr_shape)
        self.list_image_path = sorted_list(self.dataset_dir)
        self.generate_data(resize, preprocess)
        self.data = np.load(self.data_file)
        self.label = np.load(self.label_file)
        self.data = tf.convert_to_tensor(self.data, tf.float32) 
        self.label = tf.convert_to_tensor(self.label, tf.float32) 

    def generate_data(self, resize=resize_bicubic, preprocess=DefaultPreprocess):
        data = []
        label = []

        if exists(self.data_file) and exists(self.label_file):
            print("{} and {} HAVE ALREADY EXISTED\n".format(self.data_file, self.label_file))
            return

        for path in self.list_image_path:
            image = read_image(path)
            if len(image.shape) != 3:
                ValueError("The image has invalid format - Path: \"{}\" - Shape: {}\n".format(path, image.shape))

            count = 0
            h, w, _ = image.shape
            for x in np.arange(0, h - self.hr_shape[0], self.hr_shape[0]):
                for y in np.arange(0, w - self.hr_shape[1], self.hr_shape[1]):
                    subim_label = image[x : x + self.hr_shape[0], y : y + self.hr_shape[1]]
                    subim_label = random_transform(subim_label)
                    subim_data = resize(subim_label, self.lr_shape)
                    subim_data, subim_label = preprocess(subim_data, subim_label)

                    data.append(subim_data.numpy())
                    label.append(subim_label.numpy())

                    count += 1
                    if (self.limit_per_image != -1) and (count >= self.limit_per_image):
                        break

                if (self.limit_per_image != -1) and (count >= self.limit_per_image):
                    break

        np.save(self.data_file, np.array(data, np.float32))
        np.save(self.label_file, np.array(label, np.float32))

    def get_batch(self, batch_size, shuffle_each_epoch=True):
        # Ignore remaining dataset because of  
        # shape error when run tf.reduce_mean()
        isEnd = False
        if self.cur_idx + batch_size > self.data.shape[0]:
            isEnd = True
            self.cur_idx = 0
            if shuffle_each_epoch:
                self.data, self.label = shuffle(self.data, self.label)

        data = self.data[self.cur_idx : self.cur_idx + batch_size]
        label = self.label[self.cur_idx : self.cur_idx + batch_size]
        self.cur_idx += batch_size

        return data, label, isEnd

class DatasetSingleImage:
    def __init__(self, data_dir, label_dir):
        if not exists(data_dir):
            raise FileNotFoundError("\"{}\" does not exist".format(data_dir))
        if not exists(label_dir):
            raise FileNotFoundError("\"{}\" does not exist".format(label_dir))

        self.data_dir = data_dir
        self.label_dir = label_dir
        self.data_image_paths = [""]
        self.label_image_paths = [""]
        self.cur_idx = 0

    def load_data(self, preprocess=DefaultPreprocess):
        self.data_image_paths = sorted_list(self.data_dir)
        self.label_image_paths = sorted_list(self.label_dir)
        self.preprocess = preprocess

    def get_images(self):
        data = read_image(self.data_image_paths[self.cur_idx])
        if len(data.shape) != 3:
            ValueError("The image has invalid format - Path: \"{}\" - Shape: {}\n".format(self.data_image_paths[self.cur_idx], data.shape))

        label = read_image(self.label_image_paths[self.cur_idx])
        if len(label.shape) != 3:
            ValueError("The image has invalid format - Path: \"{}\" - Shape: {}\n".format(self.label_image_paths[self.cur_idx], label.shape))

        data, label = self.preprocess(data, label)

        data = np.expand_dims(data.numpy(), 0)
        label = np.expand_dims(label.numpy(), 0)
        self.cur_idx += 1

        isEnd = False
        if self.cur_idx == len(self.data_image_paths):
            self.cur_idx = 0
            isEnd = True

        return data, label, isEnd

    def length(self):
        return len(self.data_image_paths)

