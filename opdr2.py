import pathlib

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from keras import layers
from keras.models import Sequential

b_size = 40
img_height = 128
img_width = 128
epochs = 20


def make_datasets():
    train_dir = pathlib.Path('Images/Train')
    valid_dir = pathlib.Path('Images/Validation')
    test_dir = pathlib.Path('Images/Test')

    train_ds = tf.keras.utils.image_dataset_from_directory(
        train_dir,
        seed=123,
        image_size=(img_height, img_width),
        batch_size=b_size,
    )

    val_ds = tf.keras.utils.image_dataset_from_directory(
        valid_dir,
        seed=123,
        image_size=(img_height, img_width),
        batch_size=b_size,
    )

    test_ds = tf.keras.utils.image_dataset_from_directory(
        test_dir,
        seed=123,
        image_size=(img_height, img_width),
        batch_size=b_size
    )
    return train_ds, val_ds, test_ds


def data_augmentation(train_ds, valid_ds):
    classnames = train_ds.class_names
    n_classes = len(classnames)
    AUTOTUNE = tf.data.AUTOTUNE
    train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
    valid_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
    normalization_layer = layers.Rescaling(1. / 255)
    normalized_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
    return train_ds, valid_ds, classnames, n_classes


def create_model():
    model = Sequential([
        layers.Rescaling(1. / 255, input_shape=(img_height, img_width, 3)),
        layers.RandomFlip("horizontal",
                          input_shape=(img_height,
                                       img_width,
                                       3)),
        layers.RandomRotation(0.1),
        layers.RandomZoom(0.1),
        layers.Conv2D(16, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Conv2D(32, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Conv2D(64, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Dropout(0.2),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dense(n_classes, activation="softmax")
    ])
    model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                  optimizer='adam',
                  metrics=['accuracy'])
    return model


def run_model(model, train_ds, val_ds):
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs
    )
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']

    loss = history.history['loss']
    val_loss = history.history['val_loss']
    return model, acc, val_acc, loss, val_loss


def predict_img(model, class_names):
    img = tf.keras.utils.load_img(
        "opdr2/testimage.png", target_size=(img_height, img_width)
    )
    img_array = tf.keras.utils.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)  # Create a batch

    predictions = model.predict(img_array)
    score = tf.nn.softmax(predictions[0])

    print(
        "This image most likely belongs to {} with a {:.2f} percent confidence.\n"
        .format(class_names[np.argmax(score)], 100 * np.max(score))
    )


def evaluate_and_plot(model, acc, val_acc, loss, val_loss):
    print("Evaluate on test data")
    results = model.evaluate(test_ds, batch_size=32)
    print("test loss, test acc:", results)
    print("\nGenerate predictions for 3 samples")
    predictions = model.predict(test_ds)
    print("\npredictions shape:", predictions.shape)

    epochs_range = range(epochs)
    plt.figure(figsize=(8, 8))
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Training Accuracy')
    plt.plot(epochs_range, val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy')

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Training Loss')
    plt.plot(epochs_range, val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')
    plt.show()


if __name__ == "__main__":
    train_ds, valid_ds, test_ds = make_datasets()
    train_ds, valid_ds, classnames, n_classes = data_augmentation(train_ds,
                                                                  valid_ds)
    model = create_model()
    model, acc, val_acc, loss, val_loss = run_model(model, train_ds, valid_ds)
    # predict_img(model,classnames)
    evaluate_and_plot(model, acc, val_acc, loss, val_loss)
