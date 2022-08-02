from utils.common import *
import tensorflow as tf
import numpy as np
import os

class dataset:
    def __init__(self, dataset_dir, subset):
        self.dataset_dir = dataset_dir
        self.subset = subset
        self.data = tf.convert_to_tensor([])
        self.labels = tf.convert_to_tensor([])
        self.data_file = os.path.join(self.dataset_dir, "data_{}.npy".format(self.subset))
        self.labels_file = os.path.join(self.dataset_dir, "labels_{}.npy".format(self.subset))
        self.cur_idx = 0

    def generate(self, lr_crop_size, hr_crop_size, transform=False):
        if exists(self.data_file) and exists(self.labels_file):
            print("{} and {} HAVE ALREADY EXISTED\n".format(self.data_file, self.labels_file))
            return
        data = []
        labels = []

        subset_dir = os.path.join(self.dataset_dir, self.subset)
        ls_images = sorted_list(subset_dir)
        for image_path in ls_images:
            print(image_path)
            hr_image = read_image(image_path)

            for _ in range(50):
                subim_label = random_crop(hr_image, hr_crop_size, hr_crop_size)
                if transform:
                    subim_label = random_transform(subim_label)

                subim_data = gaussian_blur(subim_label, sigma=0.6)
                subim_data = resize_bicubic(subim_data, lr_crop_size, lr_crop_size)

                subim_label = norm01(subim_label.numpy())
                subim_data = norm01(subim_data.numpy())

                labels.append(subim_label)
                data.append(subim_data)

        data = np.array(data)
        labels = np.array(labels)
        data, labels = shuffle(data, labels)

        np.save(self.data_file, data)
        np.save(self.labels_file, labels)

    def load_data(self):
        if not exists(self.data_file):
            raise ValueError("\n{} and {} DO NOT EXIST\n".format(self.data_file, self.labels_file))
        self.data = np.load(self.data_file)
        self.data = tf.convert_to_tensor(self.data)
        self.labels = np.load(self.labels_file)
        self.labels = tf.convert_to_tensor(self.labels)

    def get_batch(self, batch_size, shuffle_each_epoch=True):
        # Ignore remaining dataset because of  
        # shape error when run tf.reduce_mean()
        isEnd = False
        if self.cur_idx + batch_size > self.data.shape[0]:
            isEnd = True
            self.cur_idx = 0
            if shuffle_each_epoch:
                self.data, self.labels = shuffle(self.data, self.labels)

        data = self.data[self.cur_idx : self.cur_idx + batch_size]
        labels = self.labels[self.cur_idx : self.cur_idx + batch_size]
        self.cur_idx += batch_size

        return data, labels, isEnd
