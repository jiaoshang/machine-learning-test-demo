import pytest

import numpy as np
import tensorflow as tf

from mltest.model import create_model_with_functional_api
from mltest.model import SubclassingNet

# TODO setup abstract test class/file?
@pytest.fixture
def functional_api_model():
    return create_model_with_functional_api()


@pytest.fixture
def subclassing_model():
    return SubclassingNet()


def test_functional_api_with_wrong_input_shape_for_tensor_input(functional_api_model):
    with pytest.raises(ValueError,
                       # match=r"^(.*?)Input 0 of layer \"mnist_model\" is incompatible with the layer: expected shape=\(None, 28, 28\), found shape=\(.*\)$"
                       ):
        random_input = np.random.rand(1, 27, 27)

        functional_api_model.predict(random_input)


def test_functional_api_with_wrong_input_shape_for_numpy_array_input(functional_api_model):
    with pytest.raises(ValueError,
                       match=r"^Input 0 of layer \"mnist_model\" is incompatible with the layer: expected shape=\(None, 28, 28\), found shape=\(.*\)$"):
        random_input = tf.random.uniform(shape=(1, 27, 27))

        functional_api_model(random_input)


@pytest.mark.parametrize('batch_size', [1, 2])
def test_functional_api_output_shape_with_tensor_input(batch_size, functional_api_model):
    random_input = tf.random.uniform(shape=(batch_size, 28, 28))

    output = functional_api_model(random_input)
    assert output.shape == (batch_size, 10)


@pytest.mark.parametrize('batch_size', [1, 2])
def test_subclassing_output_shape_with_tensor_input(batch_size, subclassing_model):
    random_input = tf.random.uniform(shape=(batch_size, 28, 28))

    output = subclassing_model(random_input)
    assert output.shape == (batch_size, 10)


@pytest.mark.parametrize('batch_size', [1, 2])
def test_functional_api_output_shape_with_numpy_array_input(batch_size, functional_api_model):
    random_input = np.random.rand(batch_size, 28, 28)

    output = functional_api_model.predict(random_input)
    assert output.shape == (batch_size, 10)

    for i in range(output.shape[0]):
        assert np.sum(output[i]) == 1


# def test_functional_api_input_and_output_shape():
#     model = create_model_with_functional_api()
#     assert model.get_layer('input').input_shape == [(None, 28, 28)]
#     assert model.get_layer('output').output_shape == (None, 10)
