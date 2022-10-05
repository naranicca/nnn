from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
try:
    import tensorflow.compat.v1 as tf
    tf.disable_v2_behavior()
except:
    import tensorflow as tf
import re
import datetime, time
import numpy as np
import multiprocessing
import pydoc
pydoc.pager = pydoc.plainpager

class Model(object):
    def __init__(self, name=None, input_shape=None):
        self.__name = name
        self.func = []
        self.tensor = None
        self.head = self
        self.idx = 0
        if input_shape is not None:
            self.input_shape = [None] + list(input_shape)
        global _nnnets_
        if not self in _nnnets_:
            _nnnets_.append(self)

    def __call__(self, input):
        if tf.is_tensor(input):
            assert not input in tf.global_variables(), 'This tensor is not acceptable: Try to set trainable parameter of variable() to False'
            return self.__build_network(input)
        else:
            global _feed_dict_
            t = _feed_dict_.get(_training_)
            _feed_dict_.update({_training_: False})
            try:
                if not 'testnet' in self.__dict__:
                    if 'input_shape' in self.head.__dict__:
                        shape = self.head.input_shape
                    else:
                        shape = [None] * len(np.array(input).shape)
                    x = tf.placeholder(tf.float32, shape=shape)
                    self.testnet = self.__build_network(x, silent=True)
                    print('[+] The network for test was built successfully: input_shape =', shape)
                ret = Session().run(self.testnet, feed_dict=_feed_dict_)
            except Exception as e:
                x = tf.constant(input)
                y = self.__build_network(x, silent=True)
                ret = Session().run(y, feed_dict=_feed_dict_)
            _feed_dict_.update({_training_: t})
            return ret

    def __str__(self):
        if hasattr(self, 'summary'):
            return '{}'.format(self.show_summary())
        else:
            return self.__repr__()

    @staticmethod
    def calc(func_or_tensor, *args, **kwargs):
        if isinstance(func_or_tensor, tf.Tensor):
            return Session().run(func_or_tensor)
        else:
            global _magiccode_
            return Model(name=_magiccode_).add(func_or_tensor, *args, **kwargs)

    def name(self, name):
        global _nnnets_
        for n in _nnnets_:
            if n.__name == name:
                raise AssertionError('name {} already exists'.format(name))
        self.__name = name
        return self

    def add(self, func, *args, **kwargs):
        global _model_
        assert _model_ is None, 'Model was already loaded. Build the network before loading the model'
        if isinstance(func, self.__class__):
            assert func.head.__name is not None, 'Nested network must have the name to avoid variable name collision.'
            assert len(args) == 0 and len(kwargs) == 0
            name = 'NNN: {}'.format(func.head.__name)
            args = [self]
        else:
            self.head.idx = self.head.idx + 1
            if func is None:
                name, func, args = 'identity', lambda x: x, [self]
            else:
                name = '{}/{}'.format(self.head.idx, func.__name__ if func.__name__ != '<lambda>' else 'lambda')
                if self.head.__name:
                    name = '{}/{}'.format(self.head.__name, name)
        node = self.__class__(name=name)
        node.func.append(func)
        node.func.append(args)
        node.func.append(kwargs)
        node.head = self.head
        return node

    def addx(self, func, *args, **kwargs):
        return self.add(func, self, *args, **kwargs)

    def train(self, dataset, loss='mse', optimizer=None, epochs=1, validset=None, callback_epoch=None, callback_iter=None):
        dataset = Dataset(dataset)
        output, labels = self.__call__(dataset.input), dataset.label

        print('[+] Summary of the network')
        self.show_summary()

        train_loss = Loss(loss, output, labels)

        if validset:
            validset = Dataset(validset, shuffle=False)
            valid_output, valid_label = self.__call__(validset.input), validset.label
            valid_loss = Loss(loss, valid_output, valid_label)
            def validate(epoch):
                Session().run(validset.iterator.initializer)
                loss, cnt = 0, 0
                global _feed_dict_
                t = _feed_dict_.get(_training_)
                _feed_dict_.update({_training_: False})
                while True:
                    try:
                        loss = loss + Session().run(valid_loss, feed_dict=_feed_dict_)
                        cnt = cnt + 1
                    except:
                        break
                print('    validation loss:', loss / cnt)
                _feed_dict_.update({_training_: t})
                if callback_epoch:
                    return callback_epoch(epoch)
        else:
            validate = callback_epoch

        train_tensor(dataset.iterator, train_loss, optimizers=optimizer, epochs=epochs, callback_epoch=validate, callback_iter=callback_iter, show_var_list=False)

    #def load_weights(self, filename, trainable=True):
    #    self.head.__weights = [0]
    #    assert filename.endswith('.h5'), 'Only .h5 files are supported'
    #    import h5py
    #    with h5py.File(filename, 'r') as f:
    #        for key in f.keys():
    #            group = f[key]
    #            for g in list(group):
    #                v = group[g][()]
    #                if trainable:
    #                    self.head.__weights.append(tf.constant(v, tf.float32))
    #                else:
    #                    self.head.__weights.append(v)
    #    self.head.__weights.append(filename)

    def __build_network(self, input, silent=False):
        global _magiccode_, _nnnets_, _model_, _random_seed_
        if _random_seed_ is not None:
            tf.set_random_seed(_random_seed_)
            np.random.seed(_random_seed_)
        summary = []
        def compile_node(node, input):
            if node.tensor is None:
                if node.func:
                    func, args, kwargs = node.func[0], list(node.func[1]), node.func[2]
                    for i, arg in enumerate(args):
                        if isinstance(arg, node.__class__):
                            args[i] = compile_node(arg, input)
                        elif type(arg) == tuple or type(arg) == list:
                            l = []
                            for a in arg:
                                if isinstance(a, node.__class__):
                                    l.append(compile_node(a, input))
                                else:
                                    l.append(a)
                            args[i] = l
                    for kwarg in kwargs:
                        if isinstance(kwargs[kwarg], node.__class__):
                            kwargs[kwarg] = compile_node(kwargs[kwarg], input)
                        elif kwarg == 'activation':
                            if kwargs[kwarg] == 'relu':
                                kwargs[kwarg] = tf.nn.relu
                            elif kwargs[kwarg] == 'leaky_relu':
                                kwargs[kwarg] = tf.nn.leaky_relu
                            elif kwargs[kwarg] == 'tanh':
                                kwargs[kwarg] = tf.nn.tanh
                            elif kwargs[kwarg] == 'sigmoid':
                                kwargs[kwarg] = tf.nn.sigmoid
                            elif kwargs[kwarg] is not None and not callable(kwargs[kwarg]):
                                raise AssertionError('Unknown activation: {}'.format(kwargs[kwarg]))
                    if isinstance(func, self.__class__):
                        summary.append({'name': '-'})
                        clear_node(func, [])
                        node.tensor = compile_node(func, args[0])
                        summary.append({'name': '-'})
                        return node.tensor
                    else:
                        with tf.variable_scope(node.__name, reuse=tf.AUTO_REUSE):
                            try:
                                node.tensor = func(*args, **kwargs)
                            except Exception as e:
                                if not silent:
                                    self.show_summary(footer=False)
                                    print(node.__name)
                                    print(str(e))
                                raise
                            try:
                                kwa = kwargs.copy()
                                kwa['training'] = True
                                tensor_training = func(*args, **kwa)
                                kwa['training'] = False
                                tensor_inference = func(*args, **kwa)
                                node.tensor = tf.cond(_training_, lambda: tensor_training, lambda: tensor_inference)
                            except TypeError:
                                pass
                    param = [v for v in tf.trainable_variables() if node.__name and v.name.startswith(node.__name)]
                else:
                    node.tensor = tf.cast(input, tf.float32)
                    param = []
                num_param = int(sum(np.prod(v.get_shape().as_list()) for v in param))
                param = '-' if len(param) == 0 else '{:,} ({})'.format(num_param, ', '.join([v.name.replace(node.__name, '~', 1) for v in param]))
                for s in summary:
                    if num_param > 0 and s['name'] == node.__name:
                        param = 'reused: {}'.format(param)
                        num_param = 0
                        break
                if node.head != node or len(summary) == 0:
                    name = ' ' if node.__name is None else node.__name.split('/')[-1] if node.__name.startswith(_magiccode_) else node.__name
                    summary.append({'name': name, 'shape': '{}'.format(node.tensor.get_shape()), 'num_param': num_param, 'param': param})
            return node.tensor
        if _random_seed_ is not None and self.tensor is None:
            print('[+] Network is built with random seed =', _random_seed_)
        # clear Model objects
        for n in _nnnets_:
            n.tensor = None
        self.summary = summary
        ret = compile_node(self, input)
        if _model_ is not None:
            load(_model_)
        return ret

    def show_summary(self, footer=True):
        nlen, plen = 0, 0
        for s in self.summary:
            nlen, plen = max(len(s['name'])+1, nlen), max(len(s['param'])+1, plen)
        mlen = nlen + 22 + plen
        nf = '{:' + str(nlen) + '}'
        print('-'*mlen)
        print((nf+' {:20}').format('Name', 'Shape'), '# of Param')
        print('='*mlen)
        num_params = 0
        for s in self.summary:
            print('-'*mlen if s['name'] == '-' else (nf+' {:20} {}').format(s['name'], s['shape'], s['param']))
            num_params = num_params + s['num_param']
        if not footer:
            return
        print('='*mlen)
        print((nf+' {:20} {:,}').format('output', '{}'.format(self.tensor.get_shape()), num_params))
        total_params = int(sum(np.prod(v.get_shape().as_list()) for v in tf.trainable_variables()))
        print('-'*mlen)

    # operator overloading
    def __add__(self, o):
        return self.add(tf.add, self, o)
    def __radd__(self, o):
        return self.add(tf.add, self, o)
    def __sub__(self, o):
        return self.add(tf.subtract, self, o)
    def __rsub__(self, o):
        return self.add(tf.subtract, o, self)
    def __mul__(self, o):
        return self.add(tf.multiply, self, o)
    def __rmul__(self, o):
        return self.add(tf.multiply, self, o)
    def __truediv__(self, o):
        return self.add(tf.divide, self, o)
    def __rtruediv__(self, o):
        return self.add(tf.divide, o, self)
    def __pow__(self, o):
        return self.add(tf.pow, self, o)
    def __rpow__(self, o):
        return self.add(tf.pow, self, o)

    def reshape_nhwc(self, x):
        if len(x.get_shape().as_list()) == 2:
            x = tf.expand_dims(x, axis=-1)
        if len(x.get_shape().as_list()) == 3:
            x = tf.expand_dims(x, axis=0)
        return x

    def conv2d(self, kernel_size, out_channels, bias=True, stride=1, activation='relu'):
        def conv2d(x, activation=activation):
            x = self.reshape_nhwc(x)
            c = x.get_shape().as_list()[-1]
            w = Variable('W', shape=(kernel_size, kernel_size, c, out_channels))
            b = Variable('b', shape=(out_channels)) if bias else 0
            out = tf.nn.conv2d(x, w, strides=[1, stride, stride, 1], padding='SAME') + b
            if activation:
                return activation(out)
            return out
        return self.add(conv2d, self, activation=activation)

    def relu(self, alpha=0):
        if alpha == 0:
            return self.add(tf.nn.relu, self)
        else:
            return self.add(tf.nn.leaky_relu, self, alpha=alpha)

    def max_pool(self, size, padding='SAME'):
        return self.add(tf.nn.max_pool, self, [1, size, size, 1], [1, size, size, 1], padding=padding)

    def concat(self, x, axis=-1):
        return self.add(tf.concat, [self, x], axis=axis)

    def slice(self, start, stop=None):
        def slice(x, start, stop):
            return x[..., start:stop]
        return self.add(slice, self, start, stop if stop else start+1)

    def flatten(self):
        return self.add(tf.layers.flatten, self)

    def dense(self, size, activation=tf.nn.relu):
        return self.add(tf.layers.dense, self, size, activation=activation)

    def dropout(self, rate, training=True):
        def dropout(x, rate, training):
            if training:
                return tf.nn.dropout(x, rate=rate)
            else:
                return x
        return self.add(dropout, self, rate, training=training)

    def bn(self):
        def bn(x):
            mean, var = tf.nn.moments(x, axes=[0, 1, 2])
            return tf.nn.batch_normalization(x, mean, var, offset=None, scale=None, variance_epsilon=1e-7)
        return self.add(bn, self)

    def depth_to_space(self, size=2):
        return self.add(tf.nn.depth_to_space, self, size)

class Dataset():
    def __init__(self, data, shuffle=True, preprocess=None):
        self.size = None
        global _datasets_
        try:
            if issubclass(data.__class__, Dataset):
                self.iterator = data.iterator
                self.input = data.input
                self.label = data.label
                self.shuffle = data.shuffle
                _datasets_[self.iterator] = self
                return
        except:
            pass
        dataset = self.get_tf_dataset(data, shuffle=shuffle, preprocess=preprocess)
        self.iterator = dataset.make_initializable_iterator()
        self.input, self.label = self.iterator.get_next()
        _datasets_[self.iterator] = self
    def get_tf_dataset(self, data, shuffle=True, preprocess=None):
        if isinstance(data, tf.data.Dataset):
            self.shuffle = None
            return data
        elif _isiterable_(data):
            inputs, labels, batch_size = data[0], data[1], 1 if len(data) == 2 else data[2]
        elif type(data) == dict:
            inputs, labels = data['input'], data['label']
            batch_size = data['batch_size'] if 'batch_size' in data else 1
        else:
            raise AssertionError('Unknown type of dataset')
        if type(inputs) == int or type(inputs) == float:
            inputs = [inputs]
            labels = [labels]
        assert iter(inputs), 'Data must be iterable'
        x = tf.constant(inputs)
        y = tf.constant(labels)
        dataset = tf.data.Dataset.from_tensor_slices((x, y))
        self.size = len(inputs)
        self.shuffle = shuffle
        if shuffle:
            dataset = dataset.shuffle(len(inputs))
        if preprocess:
            print('[+] Creating dataset')
            for proc in preprocess if type(preprocess) == tuple or type(preprocess) == list else [preprocess]:
                if proc == 'unbatch':
                    dataset = dataset.unbatch()
                elif proc == 'cache':
                    dataset = dataset.cache()
                else:
                    try:
                        dataset = dataset.map(proc, num_parallel_calls=multiprocessing.cpu_count())
                    except:
                        dataset = dataset.map(lambda i, l: tf.py_func(proc, [i, l], [tf.float32, tf.float32]), num_parallel_calls=multiprocessing.cpu_count())
                print('{}: {}'.format(proc if type(proc) == str else proc.__name__, dataset))
        dataset = dataset.batch(batch_size)
        def reshaper(a, b):
            sa = [batch_size] + [s if s is not None else -1 for s in a.shape[1:].as_list()]
            sb = [batch_size] + [s if s is not None else -1 for s in b.shape[1:].as_list()]
            return tf.reshape(a, sa), tf.reshape(b, sb)
        dataset = dataset.map(reshaper)
        if preprocess:
            print('batch/reshape:', dataset)
        dataset = dataset.prefetch(batch_size)
        print('[+] Dataset was built: size={}, batch_size={}, shuffle={}'.format(len(inputs), batch_size, shuffle))
        dataset.size = len(inputs)
        return dataset

def Loss(loss, pred, label):
    if type(loss) == str:
        loss = loss.lower()
    if not tf.is_tensor(label):
        label = tf.constant(label, tf.float32)
    if loss == 'mse' or loss == 'l2':
        return tf.reduce_mean(tf.square(tf.cast(label, 'float32') - pred))
    elif loss == 'mae' or loss == 'l1':
        return tf.reduce_mean(tf.abs(tf.cast(label, 'float32') - pred))
    elif loss == 'sparse_softmax_crossentropy':
        return tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=pred, labels=tf.cast(label, 'int32')))
    elif loss == 'binary_crossentropy':
        eps = 1e-7
        pred = tf.clip_by_value(pred, eps, 1 - eps)
        pred = tf.log(pred / (1 - pred))
        if not tf.is_tensor(label):
            label = tf.constant(label, tf.float32)
        if len(label.get_shape().as_list()) == 0:
            label = tf.broadcast_to(label, pred.get_shape().as_list())
        return tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.cast(label, tf.float32), logits=pred))
    #elif loss == 'vgg' or loss == 'vgg19':
    #    mean = [103.939, 116.779, 123.68]
    #    pred = pred[..., ::-1] - mean
    #    label = label[..., ::-1] - mean
    #    if not os.path.exists('vgg19_weights_tf_dim_ordering_tf_kernels.h5'):
    #        url = 'https://github.com/fchollet/deep-learning-models/releases/download/v0.1/vgg19_weights_tf_dim_ordering_tf_kernels.h5'
    #        print('[*] VGG model was not found. Downloading from:', url)
    #        import requests
    #        r = requests.get(url, allow_redirects=True)
    #        open('vgg19_weights_tf_dim_ordering_tf_kernels.h5', 'wb').write(r.content)
    #    vgg = Model('vgg')
    #    vgg = vgg.conv2d(3, 64).conv2d(3, 64).max_pool(2)
    #    vgg = vgg.conv2d(3, 128).conv2d(3, 128).max_pool(2)
    #    vgg = vgg.conv2d(3, 256).conv2d(3, 256).conv2d(3, 256).conv2d(3, 256).max_pool(2)
    #    vgg = vgg.conv2d(3, 512).conv2d(3, 512).conv2d(3, 512).conv2d(3, 512).max_pool(2)
    #    vgg = vgg.conv2d(3, 512).conv2d(3, 512).conv2d(3, 512).conv2d(3, 512, activation=None)
    #    vgg.load_weights('vgg19_weights_tf_dim_ordering_tf_kernels.h5', trainable=False)
    #    pred = vgg(pred)
    #    label = vgg(label)
    #    return tf.reduce_mean(tf.square(label - pred))
    elif callable(loss):
        return loss(pred, label)
    elif isinstance(loss, tf.Tensor):
        return loss
    else:
        raise AssertionError('Unknown loss: {}'.format(loss))

def Optimizer(optimizer='adam', lr=0.0001, **kwargs):
    optimizer = optimizer.lower()
    if optimizer == 'adam':
        return tf.train.AdamOptimizer(lr, **kwargs)
    else:
        raise Exception('Unknown optimizer: {}'.format(optimizer))

def Variable(var, shape=None):
    """ Variable is a trainable variable whose value cannot be changed by the user
    To create a variable var:
    >>> var = Variable('name', shape=(3, 3))

    If 'name' exists, existing variable with the name is retrieved

    To get the current value of a variable var:
    >>> Variable(var)

    All trainable variables are retrieve if var is None
    >>> all_var = Variable(None)
    """
    if var is None:
        return tf.trainable_variables()
    elif isinstance(var, tf.Variable):
        return Session().run(var)
    elif type(var) == str:
        t = [v for v in tf.global_variables() if v.op.name == var]
        if len(t) > 0:
            return t[0]
        global _random_seed_
        if list(map(int, tf.__version__.split('.')))[0] == 1:
            initializer = tf.contrib.layers.variance_scaling_initializer('FAN_AVG', seed=_random_seed_)
        else:
            initializer = tf.truncated_normal_initializer(stddev=0.1, seed=_random_seed_)
        return tf.get_variable(name=var, shape=shape if shape else (), initializer=initializer)
    else:
        raise TypeError(Variable.__doc__)

def Parameter(param, new_value=None):
    """ Parameter is a non-trainable variable that the user can change on the fly
    To create a parameter foo:
    >>> foo = Parameter(1.0)

    To get the value of a parameter foo:
    >>> Parameter(foo)
    1.0

    To change the value of a parameter foo:
    >>> Parameter(foo, 2.0)
    2.0
    """
    global _feed_dict_
    if param in _feed_dict_:
        if new_value:
            _feed_dict_.update({param: new_value})
        return _feed_dict_[param]
    else:
        assert new_value is None, 'Cannot find the parameter to be set to the new value'
        p = tf.placeholder(tf.float32, shape=np.array(param).shape)
        _feed_dict_.update({p: param})
        return p

class Logger(object):
    def __init__(self, dir):
        global _logger_
        _logger_ = self
        self.writer = tf.summary.FileWriter(dir)
        print('[+] Tensorboard data will be saved:', dir)
    def add_scalar(self, tag, index, value):
        s = tf.Summary(value=[tf.Summary.Value(tag=tag, simple_value=value)])
        self.writer.add_summary(s, index)
    def add_histogram(self, tag, index, values, bins=1000):
        cnt, bins = np.histogram(values, bins=bins)
        hist = tf.HistogramProto()
        hist.min = float(np.min(values))
        hist.max = float(np.max(values))
        hist.num = int(np.prod(values.shape))
        hist.sum = float(np.sum(values))
        hist.sum_squares = float(np.sum(values**2))
        for edge in bins[1:]:
            hist.bucket_limit.append(edge)
        for c in cnt:
            hist.bucket.append(c)
        s = tf.Summary(value=[tf.Summary.Value(tag=tag, histo=hist)])
        self.writer.add_summary(s, index)
    def add_image(self, tag, index, rgb_images):
        images = rgb_images
        min, max = np.min(images), np.max(images)
        if min >= 0 and max <= 1:
            images = images * 255
        elif min >= -1 and max <= 1:
            images = (images + 1) * 127.5
        images = np.clip(np.round(images).astype(np.uint8), 0, 255)
        if len(images.shape) < 4 and images.shape[-1] <= 4:
            images = [images]
        summary = []
        for i, img in enumerate(images):
            try:
                from StringIO import StringIO
                s = StringIO()
            except:
                from io import BytesIO
                s = BytesIO()
            from PIL import Image
            Image.fromarray(img).save(s, format='png')
            s = tf.Summary.Image(encoded_image_string=s.getvalue(), height=img.shape[0], width=img.shape[1])
            summary.append(tf.Summary.Value(tag='{}{}'.format(tag, '/'+str(i) if len(images) > 1 else ''), image=s))
        self.writer.add_summary(tf.Summary(value=summary), index)

def train_tensor(dataset_or_iterator, losses, namespaces=None, optimizers=None, epochs=1, callback_epoch=None, callback_iter=None, show_var_list=True):
    iterator = dataset_or_iterator
    if isinstance(dataset_or_iterator, tf.data.Dataset):
        dataset_or_iterator = Dataset(dataset_or_iterator)
    if isinstance(dataset_or_iterator, Dataset):
        iterator = dataset_or_iterator.iterator
    try:
        global _datasets_
        dataset = _datasets_[iterator]
    except:
        raise AssertionError('Unknown dataset')

    if not _isiterable_(losses):
        losses = [losses]
    if namespaces is None:
        namespaces = [None] * len(losses)
    elif type(namespaces) != tuple and type(namespaces) != list:
        namespaces = [namespaces]
    assert len(losses) == len(namespaces), '# of losses and # of namespaces must be equal'

    sess = Session()
    sess.run(tf.global_variables_initializer())

    optimizers = [] if optimizers is None else optimizers
    if not _isiterable_(optimizers):
        optimizers = [optimizers]
    d = len(namespaces) - len(optimizers)
    if d > 0:
        optimizers = optimizers + [Optimizer('adam') if len(optimizers) == 0 else optimizers[-1]] * d
    var_lists = []
    for i, namespace in enumerate(namespaces):
        var_list = [v for v in tf.trainable_variables() if namespace is None or namespace in v.name]
        assert len(var_list) > 0, 'No variables to optimize {}'.format('' if namespace is None else 'for ' + namespace)
        var_lists.append(var_list)
        optimizer = optimizers[i]
        op = optimizer.minimize(losses[i], var_list=var_list)
        optimizers[i] = op
        sess.run(tf.variables_initializer(optimizer.variables()))

    if show_var_list:
        print('[+] Variables to train')
        vlen, tlen, num_params = [], 0, []
        ns = [namespace if namespace else 'All' for namespace in namespaces]
        for l, namespace in zip(var_lists, ns):
            ll = max([len(v.name.replace(namespace, '~', 1)) for v in l])
            vlen.append('{:' + str(ll) + '}')
            tlen = tlen + ll + 2
            num_params.append('{:,}'.format(int(sum(np.prod(v.get_shape().as_list()) for v in l))))
        print('-' * tlen)
        for namespace, vl in zip(ns, vlen):
            print(vl.format(namespace), end='  ')
        print('')
        print('=' * tlen)
        num = max([len(l) for l in var_lists])
        for i in range(num):
            if i < 5 or i >= num-5:
                for l, namespace, vl in zip(var_lists, ns, vlen):
                    v = (l[i].name if i < len(l) else ' ').replace(namespace, '~', 1)
                    print(vl.format(v), end='  ')
                print('')
            elif i == 5:
                for l, namespace, vl in zip(var_lists, ns, vlen):
                    print(vl.format('...'), end='  ')
                print('')
        print('=' * tlen)
        for num_param, vl in zip(num_params, vlen):
            print(vl.format(num_param + ' params'), end='  ')
        print('')
        print('-' * tlen)

    print('[+] Training started at', datetime.datetime.now())
    size = dataset.size
    iter = 0
    def show_progress(cur, total, msg, size=15, lmsg=None, cr=False):
        global _tprog_
        if os.fstat(0) == os.fstat(1) or cr:
            t = time.time()
            if total is None:
                if t - _tprog_ > 1 or cr:
                    cr = '\n\033[?7h' if cr else '\r'
                    print('\033[?7l\rstep: {:,} - {}\033[K'.format(cur, msg), end=cr)
                    _tprog_ = t
            else:
                if t - _tprog_ > 1 or cr:
                    if lmsg is None:
                        lmsg = '{:,}/{:,}'.format(cur, total)
                    left = min(int(cur * size / total), size)
                    cr = '\n\033[?7h' if cr else '\r'
                    print('\033[?7l\r{} [{}{}] {}\033[K'.format(lmsg, '#'*left, ' '*(size-left), msg), end=cr)
                    _tprog_ = t
    for epoch in range(epochs):
        tbeg = time.time()
        i, loss_list = 0, []
        total_losses = [0] * len(losses)
        ee = (tf.errors.OutOfRangeError)
        lmsg = '{}/{}'.format(epoch+1, epochs)
        global _feed_dict_
        t = _feed_dict_.get(_training_)
        _feed_dict_.update({_training_: True})
        for loss_idx, _ in enumerate(losses):
            sess.run(iterator.initializer)
            while True:
                try:
                    ll = []
                    for op, l in zip(optimizers, losses):
                        _, loss = sess.run([op, l], feed_dict=_feed_dict_)
                        ll.append(loss)
                    if len(ll) == len(losses):
                        loss_list = 'loss: {}'.format(ll[0] if len(ll) == 1 else tuple(ll))
                except ee:
                    break
                show_progress(i+1, size, '{:.1f}s, {}'.format(time.time() - tbeg, loss_list), lmsg=lmsg)
                i = i + 1
                iter = iter + 1
                total_losses[loss_idx] = total_losses[loss_idx] + loss
                if callback_iter:
                    if callback_iter(iter) == False:
                        show_progress(i, size, msg='{:.1f}s, {}'.format(time.time() - tbeg, loss_list), lmsg=lmsg, cr=True)
                        print('[-] Training was aborted at iteration = {}'.format(iter))
                        return
                ee = (tf.errors.OutOfRangeError, tf.errors.InvalidArgumentError)
        _feed_dict_.update({_training_: t})
        size = i
        assert size > 0, 'No data to train'
        telapsed = (time.time() - tbeg)
        lmsg = '{}/{}'.format(epoch+1, epochs)
        show_progress(i+1, i+1, msg='{:.1f}s'.format(telapsed), lmsg=lmsg, cr=True)
        lmsg = ' ' * len(lmsg)
        print(lmsg, '- Iterations: {:,} ({:.1f} steps/s)'.format(size, float(size) / telapsed))
        print(lmsg, '- Total iterations: {:,}'.format(iter))
        global _logger_
        for i, (namespace, loss) in enumerate(zip(namespaces, total_losses)):
            loss = loss / size
            print(lmsg, '- Average loss{}: {}'.format(' for '+namespace if namespace else '', loss))
            if _logger_:
                namespace = 'loss/train'
                _logger_.add_scalar('{}{}'.format(namespace, '/'+str(i) if len(total_losses) > 1 else ''), epoch, loss)
        if callback_epoch:
            if callback_epoch(epoch+1) == False:
                print('[-] Training was aborted at epoch = {}'.format(epoch+1))
                return
    print('[+] Training finished at', datetime.datetime.now())

def save(name):
    global _saver_
    if _saver_ is None:
        _saver_ = tf.train.Saver(max_to_keep=None)
    _saver_.save(Session(), name, write_meta_graph=False)
    print('[+] Model was saved:', name)

def load(name):
    global _sess_, _saver_, _model_
    if name.endswith('.index') or re.search('\.data-[0-9]+-of-[0-9]+$', name):
        name = os.path.splitext(name)[0]
    _model_ = name
    sess = Session()
    try:
        if _saver_ is None:
            _saver_ = tf.train.Saver(max_to_keep=None)
    except ValueError: # No variables to save
        print('[*] Network is not built yet. Model will be loaded after the network is built')
        if sess is None:
            _sess_.close()
            _sess_ = None
        return
    if os.path.isdir(name):
        name = tf.train.latest_checkpoint(name)
        if name is None:
            print('[-] Cannot find the checkpoint in {}. Try to specify the filename of the model.'.format(_model_))
            exit(1)
    _saver_.restore(_sess_, name)
    print('[+] Model was successfully loaded:', name)
    _model_ = None

def set_random_seed(seed):
    global _random_seed_
    _random_seed_ = seed
    tf.set_random_seed(seed)

def Session():
    global _sess_
    if _sess_ is None:
        print('[+] Starting a session')
        _sess_ = tf.Session()
    return _sess_

# colorize logs
try:
    import __builtin__
except ImportError:
    import builtins as __builtin__
def print(*args, **kwargs):
    if os.fstat(0) == os.fstat(1):
        if type(args[0]) == str and args[0].startswith('[+]'):
            args = list(args)
            args[0] = '\033[36m{}'.format(args[0])
            args.append('\033[0m')
        elif type(args[0]) == str and args[0].startswith('[-]'):
            args = list(args)
            args[0] = '\033[31m{}'.format(args[0])
            args.append('\033[0m')
    return __builtin__.print(*args, **kwargs)

###############################################################################
# global variables
_sess_ = None
_nnnets_ = []
_saver_ = None
_model_ = None
_logger_ = None
_training_ = tf.placeholder(tf.bool, shape=())
_feed_dict_ = {}
_datasets_ = {}
_tprog_ = 0
_random_seed_ = None
_isiterable_ = lambda v: (type(v) == list or type(v) == tuple)
_magiccode_ = 'qliuhgalkjsdfakudhgvlajbnesiuhd'

