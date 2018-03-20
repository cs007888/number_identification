from keras.models import Model
from keras import backend as K
import train
import cv2
import numpy as np

CHARS = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U',
'V', 'W', 'X', 'Y', 'Z', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']
CHAR_DICT = {i:c for i, c in enumerate(CHARS)}

def fastdecode(pred):
    pred = pred[0,0,:,:]
    array = pred.argmax(axis=1)
    char = ''
    confidence = 0.0
    for i, index in enumerate(array):
        if index<62 and (i==0 or index != array[i-1]):
            char += CHAR_DICT[index]
            confidence += pred[i, index]
    confidence = confidence/len(char)
    return char, confidence


inputs, pred = train.build_model()
model = Model(inputs, pred)
model.load_weights('model/result.hdf5')
img = cv2.imread('MsQ2E.png', cv2.IMREAD_GRAYSCALE)
img = np.expand_dims(img, -1)
img = np.array([img])
y = model.predict(img)
print(fastdecode(y))
# img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
# img = np.reshape(img, (img.shape[0], img.shape[1], 1))
# y = model.predict(np.array([img]))
# print(y.shape)
# print(y[0,:,0,:].argmax(axis=1))
# pre = K.ctc_decode(y[0,:,0,:], 11)
# print(pre)