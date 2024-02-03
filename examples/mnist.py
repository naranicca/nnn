import sys
from os import path
sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))
import nnn
import numpy as np

print('[+] MNIST example')
from tensorflow import keras
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
x_train = x_train / 255.
x_test = x_test / 255.

mnist = nnn.Model((28, 28))
mnist = mnist.reshape((28, 28, 1))
mnist = mnist.conv2d(32, 5)
mnist = mnist.maxpool(2)
mnist = mnist.conv2d(64, 2)
mnist = mnist.maxpool(2)
mnist = mnist.dropout(0.25)
mnist = mnist.flatten()
mnist = mnist.dense(1000, activation='relu')
mnist = mnist.dropout(0.5)
mnist = mnist.dense(10, activation='softmax')
mnist.summary()

print('[+] training')
mnist.train(x_train, y_train, batch_size=128, loss='sparse_categorical_crossentropy', epochs=10)

print('[+] evaluating')
output = np.argmax(mnist(x_test), axis=-1)
print('accuracy:', np.mean(np.equal(output, y_test).astype(np.float32)))

print('[+] saving the model')
mnist.save('model')
print(mnist.get_weights())

print('[+] quantization')
mnist.save('quantized_model.tflite', quantization='float16')

print('[+] load quantized model')
mnist.load('quantized_model.tflite')
print(mnist.get_weights())


