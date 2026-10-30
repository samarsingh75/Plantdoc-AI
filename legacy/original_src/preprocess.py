import tensorflow as tf

IMAGE_SIZE = (128, 128)
BATCH_SIZE = 32
NUM_CLASSES = 38


def load_training_data(data_dir: str) -> tf.data.Dataset:
    dataset = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        color_mode='rgb',
        label_mode='categorical',
        labels='inferred',
        shuffle=True,
        interpolation='bilinear',
    )
    return dataset


def load_validation_data(data_dir: str) -> tf.data.Dataset:
    dataset = tf.keras.utils.image_dataset_from_directory(
        directory=data_dir,
        labels='inferred',
        label_mode='categorical',
        batch_size=BATCH_SIZE,
        image_size=IMAGE_SIZE,
        color_mode='rgb',
        shuffle=True,
        interpolation='bilinear',
        follow_links=False,
    )
    return dataset
