import os
import pickle

import cv2
import numpy as np
import scipy.io as sio


def char_to_label(char):
    if ord('A') <= ord(char) <= ord('Z'):
        return ord(char) - ord('A') + 1
    return 26 + ord(char) - ord('0') + 1


def load_data_mat(name):
    print('Loading %s ...' % name)
    mat = sio.loadmat('dataset/IIIT5K/' + name + '.mat')[name][0]
    count = mat.shape[0]
    labels = []
    images = []
    for i in range(0, count):
        word = mat[i]['GroundTruth'][0]
        image = mat[i]['ImgName'][0]
        images.append('dataset/IIIT5K/' + image)
        labels.append([])
        for j in range(0, len(word)):
            labels[i].append(char_to_label(word[j]))
        labels[i] = np.asarray(labels[i], dtype='int32')
    labels = np.asarray(labels)
    return images, labels


def prepare_images(images):
    decoded_images = []
    for img in images:
        img = cv2.imread(img, 0)
        scale = 32 / img.shape[0]
        img = cv2.resize(img, None, fx=scale, fy=scale)
        if img.shape[1] < 256:
            # Padding
            img = np.concatenate([np.array([[0] * ((256 - img.shape[1]) // 2)] * 32), img], axis=1)
            img = np.concatenate([img, np.array([[0] * (256 - img.shape[1])] * 32)], axis=1)
        else:
            img = cv2.resize(img, None, fx=256 / img.shape[1], fy=1)
        if img.shape[1] != 256:
            raise ValueError('shape = %d,%d' % img.shape)
        decoded_images.append(img)
    return np.asarray(decoded_images, np.float32) / 255


def convert_if_needed(name):
    if os.path.exists('dataset/IIIT5K/' + name + '.pickle'):
        return
    images, labels = load_data_mat(name)

    with open('dataset/IIIT5K/' + name + '.pickle', 'wb') as f:
        pickle.dump((images, labels), f)


def load_data(name):
    with open('dataset/IIIT5K/' + name + '.pickle', 'rb') as f:
        return pickle.load(f)


def main():
    convert_if_needed('traindata')
    convert_if_needed('testdata')

    images, labels = load_data('traindata')
    images = prepare_images(images)
    # assert not np.any(np.isnan(images))
    # assert not np.any(np.isnan(labels))

    eval_images, eval_labels = load_data('testdata')
    eval_images = prepare_images(eval_images)
    # assert not np.any(np.isnan(eval_images))
    # assert not np.any(np.isnan(eval_images))


if __name__ == '__main__':
    main()
