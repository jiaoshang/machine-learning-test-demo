import pytest
import tensorflow as tf
from tensorflow import keras

from mltest.input import get_infinite_dataset
from mltest.model import SubclassingNet, create_model_with_functional_api


@pytest.fixture
def subclassing_model():
    return SubclassingNet()


@pytest.fixture
def functional_api_model():
    return create_model_with_functional_api()

# TODO rename the test method name
def test_single_training_step_on_a_batch_of_data_should_yield_loss_decrease_for_subclassing_model(subclassing_model):
    train_dataset = get_infinite_dataset('../data/mnist/3.0.1/mnist-train.tfrecord-00000-of-00001')
    sample_train, sample_label = next(iter(train_dataset.take(1)))

    loss_fn = keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    optimizer = tf.keras.optimizers.Adam(0.001)

    # Compute the initial loss on the test data
    initial_logits = subclassing_model(sample_train)
    initial_loss = loss_fn(sample_label, initial_logits)

    # Perform a single gradient step on the test data
    with tf.GradientTape() as tape:
        logits = subclassing_model(sample_train, training=True)
        loss_value = loss_fn(sample_label, logits)
    gradients = tape.gradient(loss_value, subclassing_model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, subclassing_model.trainable_variables))

    # Compute the final loss on the test data
    final_logits = subclassing_model(sample_train)
    final_loss = loss_fn(sample_label, final_logits)

    assert final_loss < initial_loss, f"Final loss {final_loss} is less than initial loss {initial_loss}"


def test_single_training_step_on_a_batch_of_data_should_yield_loss_decrease_for_functional_api_model(functional_api_model):
    train_dataset = get_infinite_dataset('../data/mnist/3.0.1/mnist-train.tfrecord-00000-of-00001')
    sample_train, sample_label = next(iter(train_dataset.take(1)))

    loss_fn = keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    optimizer = tf.keras.optimizers.Adam(0.001)

    # Compute the initial loss on the test data
    initial_logits = functional_api_model(sample_train)
    initial_loss = loss_fn(sample_label, initial_logits)

    # Perform a single gradient step on the test data
    with tf.GradientTape() as tape:
        logits = functional_api_model(sample_train, training=True)
        loss_value = loss_fn(sample_label, logits)
    gradients = tape.gradient(loss_value, functional_api_model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, functional_api_model.trainable_variables))

    # Compute the final loss on the test data
    final_logits = functional_api_model(sample_train)
    final_loss = loss_fn(sample_label, final_logits)

    assert final_loss < initial_loss, f"Final loss {final_loss} is less than initial loss {initial_loss}"

