import gensim
import pickle
import os
import pandas as pd
import numpy as np
from PIL import Image

def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

def assemble_cifar_img(img_flat, resolution):
    img_cnt = img_flat.shape[0]
    imgs = []
    for i in range(img_cnt):
        img_R = img_flat[i,0:1024].reshape(32, 32)
        img_G = img_flat[i,1024:2048].reshape(32, 32)
        img_B = img_flat[i,2048:].reshape(32, 32)
        img = np.zeros((32, 32, 3), dtype=np.uint8)
        img[:,:,0] = img_R
        img[:,:,1] = img_G
        img[:,:,2] = img_B
        img_object = Image.fromarray(img, 'RGB')
        img_object = img_object.resize(resolution)
        img_arr = np.array(img_object).astype(float) / 255. - .5
        imgs.append(img_arr)
    imgs = np.array(imgs)
    return imgs

def read_cifar_labelname(filename):
    label_data = unpickle(filename)
    raw_label_names = None
    if b'label_names' in label_data:
        raw_label_names = label_data[b'label_names']
    if b'fine_label_names' in label_data:
        raw_label_names = label_data[b'fine_label_names']
    label_names = []
    for rlabel in raw_label_names:
        label_names.append(rlabel.decode('utf-8'))
    return label_names

def convert_labels(label_names, label_ids, model):
    labels = []
    for label_id in label_ids:
        label = label_names[label_id]
        words = label.split('_')
        label_arr = np.zeros(300)
        for w in words:
            label_arr = label_arr + model.word_vec(w)
        label_arr = label_arr / len(words)
        labels.append(label_arr)
    output = np.array(labels)
    return output

def read_embedding_model(model_filename):
    model = gensim.models.KeyedVectors.load_word2vec_format(model_filename, binary=True)
    return model

def read_single_image(filename):
    img = Image.open(filename).resize((224, 224))
    img_arr = np.array(img).astype(float)
    img_arr = img_arr / 255. - .5
    img_arr = np.expand_dims(img_arr, axis=0)
    return img_arr

def convert_prediction(pred, model):
    classes = model.similar_by_vector(pred)
    return classes

def read_raw_cifar(filename, sample_size):
    data_batch = unpickle(filename)
    data_raw = data_batch[b'data']
    labels_raw = None
    if b'labels' in data_batch:
        labels_raw = data_batch[b'labels']
    if b'fine_labels' in data_batch:
        labels_raw = data_batch[b'fine_labels']
    idxs = np.random.randint(data_raw.shape[0], size=sample_size)
    data = []
    labels = []
    for idx in idxs:
        data.append(data_raw[idx])
        labels.append(labels_raw[idx])
    output = {'data': np.array(data), 'labels': labels}
    return output

def read_cifar_batch(root, data_filename, meta_filename, model, resolution, sample_size):
    data_batch = read_raw_cifar(root + '/' + data_filename, sample_size)
    img_flat = data_batch['data']
    imgs = assemble_cifar_img(img_flat, resolution)
    label_names = read_cifar_labelname(root + '/' + meta_filename)
    label_ids = data_batch['labels']
    labels = convert_labels(label_names, label_ids, model)
    return imgs, labels

def read_one_caltech_class(root, class_name, resolution, model):
    label_parts = class_name.split('.')
    label = class_name
    if len(label_parts) > 1:
        label = label_parts[1].replace('-', '_')
    words = label.split('_')
    label_arr = np.zeros(300)
    for w in words:
        label_arr = label_arr + model.word_vec(w)
    label_arr = label_arr / len(words)
    filenames = os.listdir(root + '/' + class_name)
    outputs = []
    for filename in filenames:
        img = Image.open(root + '/' + class_name + '/' + filename).resize(resolution)
        img_arr = np.array(img) / 255. - .5
        if len(img_arr.shape) == 3:
            elt = {'img': img_arr, 'label': label_arr}
            outputs.append(elt)
    return outputs

def exist_in_dict(model, class_name):
    words = class_name.split('_')
    for word in words:
        if word not in model.vocab:
            return False
    return True

def read_caltech(root, class_size, resolution, model):
    class_names = os.listdir(root)
    data_list = []
    for i in range(class_size):
        idx = np.random.randint(0, len(class_names))
        class_name = class_names[idx]
        if exist_in_dict(model, class_name):
            class_data = read_one_caltech_class(root, class_name, resolution, model)
            data_list.extend(class_data)
    np.random.shuffle(data_list)
    x = []
    y = []
    for data_elt in data_list:
        x.append(data_elt['img'])
        y.append(data_elt['label'])
    x = np.array(x)
    y = np.array(y)
    return x, y
