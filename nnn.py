import sys
import tensorflow as tf
import numpy as np
from tensorflow.keras.layers import Input, Reshape, Conv2D, MaxPooling2D, Dense, Flatten, Dropout, Activation

class Model(object):
    def __init__(self, input_shape=None, model_path=None):
        self.input = None
        self.tensor = None
        if type(input_shape) is str and model_path is None:
            input_shape, model_path = None, input_shape
        if input_shape is not None:
            self.input= Input(shape=input_shape, name='input')
            self.tensor = self.input
        elif model_path is not None:
            self.load(model_path)

    def __call__(self, input):
        assert self.model is not None, 'Model is not built yet'
        self.__used = True
        if isinstance(self.model, tf.keras.Model):
            return self.model.predict(input, verbose=0)
        else:
            input_index = self.model.get_input_details()[0]['index']
            output_index = self.model.get_output_details()[0]['index']
            self.model.set_tensor(input_index, input)
            self.model.invoke()
            return self.model.get_tensor(output_index)[0][0]

    def reshape(self, shape, **kwargs):
        return self.add(Reshape(shape, **kwargs))

    def conv2d(self, out_channels, kernel, **kwargs):
        return self.add(Conv2D(out_channels, kernel, **kwargs))

    def maxpool(self, pool_size, strides=None, padding='valid', **kwargs):
        return self.add(MaxPooling2D(pool_size, strides=strides, padding=padding, **kwargs))

    def dense(self, units, **kwargs):
        return self.add(Dense(units, **kwargs))

    def flatten(self, **kwargs):
        return self.add(Flatten(**kwargs))

    def dropout(self, rate, **kwargs):
        return self.add(Dropout(rate, **kwargs))

    def activation(self, activation, **kwargs):
        return self.add(Activation(activation, **kwargs))

    def add(self, func, **kwargs):
        assert self.tensor is not None, 'Model is empty!'
        node = self.__class__()
        node.input = self.input
        node.tensor = func(self.tensor, **kwargs)
        return node

    def summary(self):
        self.__build()
        self.model.summary()

    def train(self, x, y=None, batch_size=None, loss='mse', optimizer='adam', epochs=1, validation_data=None):
        assert self.tensor is not None, 'Model is empty!'
        self.__build()
        self.model.compile(loss=loss, optimizer=optimizer)
        # disable my logger since it doesn't compatible with fit's output
        logger = sys.stdout
        sys.stdout = sys.stdout.stdout
        self.model.fit(x, y, epochs=epochs, batch_size=batch_size, validation_data=validation_data)
        sys.stdout = logger

    def get_weights(self):
        if hasattr(self, 'weights'):
            return self.weights
        elif hasattr(self, 'model'):
            weights = []
            if(isinstance(self.model, tf.keras.Model)):
                for layer in self.model.layers:
                    if len(layer.weights) > 0:
                        weights.append(layer.weights)
            else:
                if not hasattr(self, '__used'):
                    self.__call__(np.ones(tuple(self.model.get_input_details()[0]['shape']), dtype=np.float32))
                tensor_details = self.model.get_tensor_details()
                dq_weights = []
                for d in tensor_details:
                    if d['name'].endswith('_dequantize'):
                        dq_weights.append(d['name'])
                for d in tensor_details:
                    index = d['index']
                    name = d['name']
                    if name == 'input':
                        continue
                    if len(dq_weights) == 0 or name + '_dequantize' in dq_weights:
                        scale = d['quantization_parameters']['scales']
                        zp = d['quantization_parameters']['zero_points']
                        tensor = self.model.tensor(index)()
                        weights.append({'name': name, 'shape': tensor.shape, 'scale': scale, 'zero-points': zp, 'numpy': tensor})
            return weights
        else:
            return []

    def load(self, model_path):
        print('[+] loading ' + model_path)
        if model_path.endswith('.tflite'):
            tflite_model = open(model_path, 'rb').read()
            interpreter = tf.lite.Interpreter(model_content=tflite_model)
            interpreter.allocate_tensors()
            self.model = interpreter
        else:
            self.model = tf.keras.models.load_model(model_path)
            self.weights = list(self.model.__dict__['_serialized_attributes']['variables'])

    def save(self, model_path, type=None, quantization=None):
        print('[+] save model: ' + model_path)
        assert self.model, 'This does not contain a model to save'
        if type is None:
            if model_path.endswith('.tflite') or quantization is None:
                type = 'tflite'
                if quantization is not None and not model_path.endswith('.tflite'):
                    model_path = model_path + quantization + '.tflite'
                    print('[-] model_path was set to: ' + model_path)
            else:
                type = 'savedModel'
        assert type == 'savedModel' or type == 'tflite'
        if quantization is not None:
            self.__quantize(target=quantization)
        if isinstance(self.model, tf.keras.Model):
            if type == 'savedModel':
                self.model.save(model_path)
            elif type == 'tflite':
                if not hasattr(self, 'converter'):
                    self.converter = tf.lite.TFLiteConverter.from_keras_model(self.model)
                converted = self.converter.convert()
                open(model_path, 'wb').write(converted)
            else:
                raise Exception('Unknown type: ' + type)
        else:
            raise Exception('Not implemented yet')

    def __quantize(self, target='float16', x_train=None):
        self.converter = tf.lite.TFLiteConverter.from_keras_model(self.model)
        if target == 'float16':
            print('[+] quantizing to ' + target)
            self.converter.target_spec.supported_types = [tf.float16]
            self.converter.optimizations = [tf.lite.Optimize.DEFAULT]
        elif target == 'int8':
            print('[+] quantizing to ' + target)
            assert dataset is not None, 'x_train must be supplied to quantize to int8'
            self.converter.optimizations = [tf.lite.Optimize.DEFAULT]
            self.converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
            self.converter.inference_input_type = tf.int8
            self.converter.inference_output_type = tf.int8
        else:
            raise Exception('Unknown target: ' + target)

    def __build(self):
        if not hasattr(self, 'model') or self.model is None:
            self.model = tf.keras.Model(inputs=self.input, outputs=self.tensor)

def set_logger(filename=None):
    class Logger(object):
        def __init__(self, filename):
            self.str = ''
            self.stdout = sys.stdout if not isinstance(sys.stdout, Logger) else sys.stdout.stdout
            self.file = open(filename, 'w') if filename else None
        def write(self, str):
            if str == '\n':
                if self.str:
                    return
                str += '\033[0m'
            if str.startswith('\r'):
                sys.stderr.write('\b' * len(self.str) + str + '\033[0m\033[K')
                sys.stderr.flush()
                self.str = str
                return
            if self.str:
                sys.stderr.write('\n')
            if self.file:
                self.file.write(str + '\n')
                self.file.flush()
            if str.startswith('[+]'):
                str = '\033[36m' + str
            elif str.startswith('[-]'):
                str = '\033[31m' + str
            self.stdout.write(str)
            self.str = ''
        def flush(self):
            #self.write('')
            self.stdout.flush()
    sys.stdout = Logger(filename)
set_logger(None)


