import numpy as np
import os
import glob
import cv2
from fingerprint_ids.features.engineer import extract_label, random_partial_crop
from sklearn.utils import shuffle
from tensorflow import keras
import random


class FingerprintLoader:
    @staticmethod
    def load_npy_data(path_x, path_y):
        x = np.load(path_x)
        y = np.load(path_y)
        return x, y

    @staticmethod
    def create_label_dict(y_real):
        label_real_dict = {}
        for i, y in enumerate(y_real):
            key = ''.join(y.astype(str)).zfill(6)
            label_real_dict[key] = i
        return label_real_dict

    @staticmethod
    def load_raw_dataset(path_pattern, target_size=(90, 90)):
        """Load ảnh BMP từ SOCOFing → numpy uint8 (N, H, W, 1)."""
        img_list = sorted(glob.glob(path_pattern))
        print(f"Loading {len(img_list)} images from {path_pattern}...")

        imgs = np.empty((len(img_list), target_size[0], target_size[1], 1), dtype=np.uint8)
        labels = np.empty((len(img_list), 4), dtype=np.uint16)

        for i, img_path in enumerate(img_list):
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            img = cv2.resize(img, target_size)
            imgs[i] = img.reshape(target_size[0], target_size[1], 1)
            labels[i] = extract_label(img_path)
        return imgs, labels

    @staticmethod
    def load_sensor_dataset(folder_path, target_size=(90, 90)):
        """
        Load ảnh BMP đã thu từ SEN0348 cho fine-tuning.
        Cấu trúc: folder_path/person_XX/*.bmp
        Trả về: (images uint8 NHWC, labels int32)
        """
        imgs = []
        labels = []
        person_dirs = sorted(glob.glob(os.path.join(folder_path, "person_*")))
        for person_dir in person_dirs:
            try:
                person_id = int(os.path.basename(person_dir).split("_")[1])
            except (ValueError, IndexError):
                continue
            bmp_files = sorted(glob.glob(os.path.join(person_dir, "*.bmp")))
            for bmp_file in bmp_files:
                img = cv2.imread(bmp_file, cv2.IMREAD_GRAYSCALE)
                if img is None:
                    continue
                img = cv2.resize(img, target_size)
                imgs.append(img.reshape(target_size[0], target_size[1], 1))
                labels.append(person_id)
        return np.array(imgs, dtype=np.uint8), np.array(labels, dtype=np.int32)


class FingerprintGenerator(keras.utils.Sequence):
    def __init__(self, x, label, x_real, label_real_dict, batch_size=32,
                 shuffle_data=True, augmentor=None, partial_crop=False):
        self.x = x
        self.label = label
        self.x_real = x_real
        self.label_real_dict = label_real_dict
        self.batch_size = batch_size
        self.shuffle = shuffle_data
        self.augmentor = augmentor
        self.partial_crop = partial_crop
        self.on_epoch_end()

    def __len__(self):
        return int(np.floor(len(self.x) / self.batch_size))

    def __getitem__(self, index):
        x1_batch = self.x[index*self.batch_size:(index+1)*self.batch_size].copy()
        label_batch = self.label[index*self.batch_size:(index+1)*self.batch_size]

        x2_batch = np.empty((self.batch_size, 90, 90, 1), dtype=np.float32)
        y_batch = np.zeros((self.batch_size, 1), dtype=np.float32)

        if self.augmentor:
            x1_batch = self.augmentor.augment_images(x1_batch)

        # Partial crop để mô phỏng ảnh từ SEN0348
        if self.partial_crop:
            for i in range(len(x1_batch)):
                x1_batch[i] = random_partial_crop(x1_batch[i])

        for i, l in enumerate(label_batch):
            match_key = ''.join(l.astype(str)).zfill(6)
            if random.random() > 0.5:
                x2_batch[i] = self.x_real[self.label_real_dict[match_key]]
                y_batch[i] = 1.
            else:
                if len(self.label_real_dict) <= 1:
                    x2_batch[i] = self.x_real[0]
                    y_batch[i] = 0.
                else:
                    while True:
                        unmatch_key, unmatch_idx = random.choice(list(self.label_real_dict.items()))
                        if unmatch_key != match_key:
                            break
                    x2_batch[i] = self.x_real[unmatch_idx]
                    y_batch[i] = 0.

        return (x1_batch.astype(np.float32) / 255., x2_batch.astype(np.float32) / 255.), y_batch

    def on_epoch_end(self):
        if self.shuffle:
            self.x, self.label = shuffle(self.x, self.label)


class SensorPairGenerator(keras.utils.Sequence):
    """
    Generator tạo pair (positive/negative) từ ảnh SEN0348 cho fine-tuning.

    Pairs được tạo NGẪU NHIÊN mỗi lần → steps_per_epoch không phụ thuộc
    vào số lượng ảnh. Mặc định 200 steps/epoch = 200×batch_size pairs/epoch.
    """
    def __init__(self, images, labels, batch_size=16, augmentor=None,
                 steps_per_epoch=200):
        self.images = images
        self.labels = labels
        self.batch_size = batch_size
        self.augmentor = augmentor
        self.steps_per_epoch = steps_per_epoch
        self.unique_labels = np.unique(labels)
        self.label_to_indices = {l: np.where(labels == l)[0] for l in self.unique_labels}

    def __len__(self):
        return self.steps_per_epoch

    def __getitem__(self, index):
        h, w = self.images.shape[1], self.images.shape[2]
        x1 = np.empty((self.batch_size, h, w, 1), dtype=np.float32)
        x2 = np.empty((self.batch_size, h, w, 1), dtype=np.float32)
        y = np.zeros((self.batch_size, 1), dtype=np.float32)

        for i in range(self.batch_size):
            label_a = random.choice(self.unique_labels)
            idx_a = random.choice(self.label_to_indices[label_a])
            x1[i] = self.images[idx_a]

            if random.random() > 0.5 or len(self.unique_labels) == 1:
                idx_b = random.choice(self.label_to_indices[label_a])
                x2[i] = self.images[idx_b]
                y[i] = 1.0
            else:
                other_labels = [l for l in self.unique_labels if l != label_a]
                label_b = random.choice(other_labels)
                idx_b = random.choice(self.label_to_indices[label_b])
                x2[i] = self.images[idx_b]
                y[i] = 0.0

        if self.augmentor:
            x1 = self.augmentor.augment_images(x1.astype(np.uint8)).astype(np.float32)

        return (x1 / 255., x2 / 255.), y

    def on_epoch_end(self):
        pass
