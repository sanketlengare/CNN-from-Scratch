import pickle
import numpy as np

def load_cifar_batch(filename):
    with open(filename, 'rb') as f:
        data_dict = pickle.load(f, encoding = 'bytes')

    images = data_dict[b'data'].reshape(10000, 3, 32, 32).astype(np.float32)
    labels = np.array(data_dict[b'labels'])
    return images, labels

def load_cifar10_train(data_dir = './data/cifar-10-batches-py/'):
    images_list, labels_list = [], []

    for i in range(1, 6):
        file_name = f'{data_dir}data_batch_{i}'
        images, labels = load_cifar_batch(file_name)
        images_list.append(images)
        labels_list.append(labels)

    images = np.concatenate(images_list, axis = 0)
    labels = np.concatenate(labels_list, axis = 0)

    return images, labels

def load_cifar10_test(data_dir = './data/cifar-10-batches-py/'):
    file_name = f'{data_dir}test_batch'
    return load_cifar_batch(file_name)

def normalize_images(images):
    return images / 255.0

def random_crop(images, crop_size = 32, pad = 4):
    Num, Col, Hei, Wid = images.shape
    padded = np.pad(images, ((0, 0), (0, 0), (pad, pad), (pad, pad)), mode = 'constant')
    cropped = np.zeros((Num, Col, crop_size, crop_size), dtype = images.dtype)

    h_starts = np.random.randint(0, 2 * pad + 1, size = Num)
    w_starts = np.random.randint(0, 2 * pad + 1, size = Num)

    for i in range(Num):
        h, w = h_starts[i], w_starts[i]
        cropped[i] = padded[i, :, h:h + crop_size, w:w + crop_size]
    
    return cropped

def random_horizontal_flip(images, p = 0.5):
    N = images.shape[0]
    mask = np.random.rand(N) < p
    images[mask] = images[mask, :, :, ::-1]
    return images

def color_jitter(images, strength = 0.1):
    noise = np.random.uniform(-strength, strength, size = images.shape)
    jittered = images + noise
    return np.clip(jittered, 0, 1)

def augement(images):
    images = random_crop(images)
    images = random_horizontal_flip(images)
    images = color_jitter(images)
    return images