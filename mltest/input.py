import tensorflow as tf

# https://www.oreilly.com/library/view/ai-and-machine/9781492078180/ch04.html
# https://www.tensorflow.org/datasets/external_tfrecord#load_dataset_with_tfds
feature_description = {
    'image': tf.io.FixedLenFeature([], dtype=tf.string),
    'label': tf.io.FixedLenFeature([], dtype=tf.int64),
}


def _parse_function(example_proto):
    # Parse the input `tf.Example` proto using the dictionary above
    example = tf.io.parse_single_example(example_proto, feature_description)
    image = tf.io.decode_png(example['image'], channels=1)
    return image, example['label']


def normalize_img(image, label):
    """Normalizes images: `uint8` -> `float32`."""
    return tf.cast(image, tf.float32) / 255., label


def get_infinite_dataset(file_path):
    """
    Loads tf record files of a specific file_path
    and returns an infinite dataset.

    Args:
        file_path: Directory path where files are located
    """
    files = tf.data.Dataset.list_files(file_path)
    dataset = tf.data.TFRecordDataset(files, num_parallel_reads=tf.data.experimental.AUTOTUNE)
    dataset = dataset.map(_parse_function, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    dataset = dataset.batch(64)
    dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
    # dataset = dataset.repeat()
    return dataset


if __name__ == '__main__':
    dataset = get_infinite_dataset('../data/mnist/3.0.1/mnist-train.tfrecord-00000-of-00001')
    for image, label in dataset.take(1):
        print(image, label)

