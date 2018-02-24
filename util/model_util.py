from keras.models import Sequential
from keras.applications.vgg16 import VGG16
from keras.layers import Dense
from .dataset_util import *

def create_model():
    model = Sequential()
    vgg_model = VGG16(include_top=False, weights='imagenet', pooling='max')
    vgg_model.trainable = False
    model.add(vgg_model)
    model.add(Dense(2048, activation='tanh'))
    model.add(Dense(1024, activation='tanh'))
    model.add(Dense(512, activation='tanh'))
    model.add(Dense(300))
    return model

def predict(img_filename, model, embedding_model):
    img = read_single_image(img_filename)
    pred = model.predict(x=img, verbose=1)
    classes = convert_prediction(pred[0], embedding_model)
    return classes
