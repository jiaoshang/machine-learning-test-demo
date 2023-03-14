import tensorflow as tf


class SubclassingNet(tf.keras.Model):
    def __init__(self, name='mnist_model'):
        super(SubclassingNet, self).__init__(name=name)
        self.flatten = tf.keras.layers.Flatten(input_shape=(28, 28))
        self.dense1 = tf.keras.layers.Dense(128, activation='relu')
        self.dense2 = tf.keras.layers.Dense(10, name='output', activation='softmax')

    def call(self, inputs):
        x = self.flatten(inputs)
        x = self.dense1(x)
        return self.dense2(x)


def create_model_with_functional_api():
    inputs = tf.keras.Input(shape=(28, 28), name='input')
    flatten = tf.keras.layers.Flatten(input_shape=(28, 28), name='flatten')(inputs)
    dense1 = tf.keras.layers.Dense(128, name='dense1', activation='relu')(flatten)
    dense2 = tf.keras.layers.Dense(10, name='output', activation='softmax')(dense1)
    return tf.keras.Model(inputs=inputs, outputs=dense2, name='mnist_model')


if __name__ == '__main__':
    # model = SubclassingNet()
    # temp_inputs = tf.keras.Input(shape=(28, 28, 1))
    # model(temp_inputs)
    # model.summary()
    model = create_model_with_functional_api()
    # model.summary()
    print(model.get_layer('input').input_shape)
    assert model.get_layer('input').input_shape == [(None, 28, 28)]
    assert model.get_layer('output').output_shape == (None, 10)
