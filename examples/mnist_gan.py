from nnn import *

print('-- mnist GAN --')
from tensorflow import keras
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
x_train = (x_train / 127.5 - 1).reshape(-1, 28*28)

noise_dim = 100
G = Model('G').dense(256).dropout(0.1).dense(256).dropout(0.1).dense(28*28, activation=tf.nn.tanh)
D = Model('D').dense(256, activation=tf.nn.leaky_relu).dropout(0.1).dense(256, activation=tf.nn.leaky_relu).dropout(0.1).dense(1, activation=tf.nn.sigmoid)

batch_size = 24
def random_noise(i, l):
    i = tf.random.uniform([noise_dim], -1, 1)
    return i, l
dataset = Dataset({'input': y_train, 'label': x_train, 'batch_size': batch_size}, preprocess=random_noise)

fake_images, real_images = G(dataset.input), dataset.label
d_real, d_fake = D(real_images), D(fake_images)
g_loss = Loss('binary_crossentropy', d_fake, 1)
d_loss = Loss('binary_crossentropy', d_real, 1) + Loss('binary_crossentropy', d_fake, 0)
adam = Optimizer('Adam', 0.0002, beta1=0.5)
train_tensor(dataset, [d_loss, g_loss], namespaces=['D', 'G'], optimizers=adam, epochs=10)

import matplotlib.pyplot as plt
noise = np.random.uniform(-1, 1, size=[24, noise_dim])
g_out = G(noise).reshape(-1, 28, 28)
plt.figure(figsize=(8, 4))
for i in range(g_out.shape[0]):
    plt.subplot(4, 6, i+1)
    plt.imshow(g_out[i], interpolation='nearest', cmap='Blues')
    plt.axis('off')
plt.tight_layout()
plt.show()
