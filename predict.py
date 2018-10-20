from tensorflow import keras
from constant import INDEX_CHAR_DICT
import train
import cv2
import numpy as np


def fastdecode(pred):
    pred = pred[0,0,:,:]
    array = pred.argmax(axis=1)
    char = ''
    confidence = 0.0
    for i, index in enumerate(array):
        if index<62 and (i==0 or index != array[i-1]):
            char += INDEX_CHAR_DICT[index]
            confidence += pred[i, index]
    confidence = confidence/len(char)
    return char, confidence


def predict(img):
    inputs, pred = train.build_model()
    model = keras.Model(inputs, pred)
    model.load_weights('model/result.h5')
    y = model.predict(img)

    return y


if __name__ == '__main__':
    img = cv2.imread('MsQ2E.png', cv2.IMREAD_GRAYSCALE)
    img = np.expand_dims(img, -1)
    img = np.array([img])
    y = predict(img)
    print(fastdecode(y))

# img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
# img = np.reshape(img, (img.shape[0], img.shape[1], 1))
# y = model.predict(np.array([img]))
# print(y.shape)
# print(y[0,:,0,:].argmax(axis=1))
# pre = K.ctc_decode(y[0,:,0,:], 11)
# print(pre)