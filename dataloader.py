import cv2
import numpy as np
import torch
import torch.utils.data as data


class Dataset(data.Dataset):
    """ Digits dataset."""

    def __init__(self, name, mode, windows, step):
        if name is 'IIIT5K':
            assert (mode == 'train' or mode == 'test')

            from prepare_IIIT5K_dataset import load_data, prepare_images
            print('Loading %s data...' % mode)
            self.mode = mode
            self.windows = windows
            self.step = step
            self.img_root = 'dataset/IIIT5K'
            self.img_names, self.labels = load_data(mode + 'data')
            self.images = prepare_images(self.img_names)

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, idx):
        image = self.images[idx]
        image = self.slide_image(image)
        label = self.labels[idx]
        image = torch.FloatTensor(image)
        label = torch.IntTensor([int(i) for i in label])
        return image, label

    def slide_image(self, image):
        h, w = image.shape  # No channel for gray image.
        output_image = []
        half_of_max_window = max(self.windows) // 2  # 从最大窗口的中线开始滑动，每次移动step的距离
        for center_axis in range(half_of_max_window, w - half_of_max_window, self.step):
            slice_channel = []
            for window_size in self.windows:
                image_slice = image[:, center_axis - window_size // 2: center_axis + window_size // 2]
                image_slice = cv2.resize(image_slice, (32, 32))
                slice_channel.append(image_slice)
            output_image.append(np.asarray(slice_channel, dtype='float32'))
        return np.asarray(output_image, dtype='float32')


class TrainBatch:
    def __init__(self, batch):
        transposed_data = list(zip(*batch))
        self.images = torch.stack(transposed_data[0], 0)
        self.labels = torch.cat(transposed_data[1], 0)
        self.label_lengths = torch.IntTensor([len(i) for i in transposed_data[1]])


class TestBatch:
    def __init__(self, batch):
        transposed_data = list(zip(*batch))
        self.images = torch.stack(transposed_data[0], 0)
        self.labels = [i.tolist() for i in transposed_data[1]]


def train_fn(batch):
    return TrainBatch(batch)


def test_fn(batch):
    return TestBatch(batch)
