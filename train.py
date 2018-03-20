import cv2
import numpy as np
import generate
from keras.layers import *
from keras.models import Model
from keras.optimizers import SGD
from keras import backend as K
from keras.callbacks import ModelCheckpoint
from keras import metrics

CHARS = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U',
'V', 'W', 'X', 'Y', 'Z', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']
CHAR_DICT = {c:i for i, c in enumerate(CHARS)}
NUM_CHAR = len(CHARS)

def build_model():
    inputs = Input((40, None, 1), name='inputs')
    x = inputs
    base_conv = 32
    for i in range(3):
        x = Conv2D(base_conv * (2**i), 3, padding='same')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = MaxPooling2D()(x)
    x = Conv2D(256, 5)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(512, 1)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    pred = Conv2D(NUM_CHAR + 1, 1, activation='softmax')(x)
    return inputs, pred

def encode(max_char, text):
    t = np.zeros((max_char,))
    for i, c in enumerate(text):
        t[i] = CHAR_DICT[c]
    if len(text) < max_char:
        t[len(text):max_char] = NUM_CHAR
    return t

class ImageGenerator:
    def __init__(self, img_size, num_examples, batch_size, max_len=6):
        self.num_examples = num_examples
        self.batch_size = batch_size
        self.max_len = max_len
        self.img_w, self.img_h = img_size
        self.input_len = self.img_w // 8 - 4

    def next_batch(self):
        labels = np.zeros((self.batch_size, self.max_len))
        images = np.zeros((self.batch_size, self.img_h, self.img_w, 1))
        for i in range(self.batch_size):
            image, text = generate.draw((self.img_w, self.img_h))
            label = encode(self.max_len, text)
            labels[i] = label
            image = np.asarray(image)
            image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            image = np.expand_dims(image, -1)
            images[i, ...] = image
        input_length = np.zeros((self.batch_size, 1))
        label_length = np.zeros((self.batch_size, 1))
        input_length[:] = self.input_len
        label_length[:] = self.max_len
        inputs = {
            'inputs':images,
            'labels':labels,
            'input_length':input_length,
            'label_length':label_length
        }
        outputs = {'ctc':np.zeros([self.batch_size])}
        return inputs, outputs
    
    def get_data(self):
        while True:
            yield self.next_batch()

def ctc_lambda_func(args):
    y_pred, labels, input_length, label_length = args
    y_pred = y_pred[:, 0, :, :]
    return K.ctc_batch_cost(labels, y_pred, input_length, label_length)

if __name__ == '__main__':
    inputs, y_pred = build_model()
    labels = Input([6], dtype='float32', name='labels')
    input_len = Input([1], dtype='int64', name='input_length')
    label_len = Input([1], dtype='int64', name='label_length')
    ctc_loss = Lambda(ctc_lambda_func, (1,), name='ctc')([y_pred, labels, input_len, label_len])
    sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True, clipnorm=5)
    model = Model(inputs=[inputs, labels, input_len, label_len], outputs=[ctc_loss])
    model.compile(loss={'ctc':lambda y_true, y_pred: y_pred}, optimizer=sgd)
    train = ImageGenerator((120, 40), 12800, 128)
    test = ImageGenerator((120, 40), 1280, 128)
    cb = ModelCheckpoint('model/result.hdf5', save_best_only=True)
    model.fit_generator(
        generator=train.get_data(),
        steps_per_epoch=int(train.num_examples / train.batch_size),
        epochs=5,
        validation_data=test.get_data(),
        validation_steps=int(test.num_examples / test.batch_size),
        callbacks=[cb]
    )

