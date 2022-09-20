
from nnn import *

print('-- mnist --')
from tensorflow import keras
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
x_train = x_train / 255.
x_test = x_test / 255.

mnist = Model(input_shape=(28, 28))
mnist = mnist.add(tf.expand_dims, mnist, axis=-1)
mnist = mnist.conv2d(5, 32)
mnist = mnist.max_pool(2)
mnist = mnist.conv2d(2, 64)
mnist = mnist.max_pool(2)
mnist = mnist.dropout(0.25)
mnist = mnist.flatten()
mnist = mnist.add(tf.layers.dense, mnist, 1000, activation=tf.nn.relu)
mnist = mnist.dropout(0.5)
mnist = mnist.add(tf.layers.dense, mnist, 10, activation=tf.nn.softmax)

mnist.train({'input': x_train, 'label': y_train, 'batch_size': 128}, loss='sparse_softmax_crossentropy', epochs=10, validset=(x_test, y_test, 100))

output = np.argmax(mnist(x_test), axis=-1)
print('accuracy:', np.mean(np.equal(output, y_test).astype(np.float32)))


