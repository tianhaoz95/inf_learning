from keras.models import Sequential
from keras.applications.vgg16 import VGG16
from keras.layers import Dense
from keras.optimizers import SGD
from .dataset_util import *

def create_model():
    model = Sequential()
    vgg_model = VGG16(include_top=False, weights='imagenet', pooling='max')
    model.add(vgg_model)
    model.add(Dense(2048, activation='relu'))
    model.add(Dense(1024, activation='relu'))
    model.add(Dense(512, activation='relu'))
    model.add(Dense(300))
    vgg_model.trainable = False
    sgd = SGD(lr=1e-5, decay=1e-6, momentum=0.9)
    model.compile(loss='mean_squared_error', optimizer=sgd)
    return model

def predict(img_filename, model, embedding_model):
    img = read_single_image(img_filename)
    pred = model.predict(x=img, verbose=1)
    classes = convert_prediction(pred[0], embedding_model)
    return classes
