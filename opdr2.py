""" This is our machine learning model script
created for our school project BI10T-ApC
Date: 31/05/2023
Contributors: Stef van Breemen, Yorick Cleijsen, Ward Strik
"""

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras import layers
from keras.models import Sequential
import pathlib


#Basic inputs for training
b_size = 32
img_height = 128
img_width = 128
epochs = 10


def make_datasets():
    """This function uses the keras.utils.image_dataset_from_directory
    to create three datasets from the data in the directories.

    Returns:
        train_ds: dataset for training
        valid_ds: dataset for validation
        test_ds: dataset for testing
    """
    train_dir = pathlib.Path('opdr2/dataset2/train')
    valid_dir = pathlib.Path('opdr2/dataset2/valid')
    test_dir = pathlib.Path('opdr2/dataset2/test')

    # Creates training dataset
    train_ds = tf.keras.utils.image_dataset_from_directory(
        train_dir,
        seed=123,
        image_size=(img_height, img_width),
        batch_size=b_size,
    )

    # Creates validation dataset
    val_ds = tf.keras.utils.image_dataset_from_directory(
        valid_dir,
        seed=123,
        image_size=(img_height, img_width),
        batch_size=b_size,
    )
    # Creates training dataset
    test_ds = tf.keras.utils.image_dataset_from_directory(
        test_dir,
        seed=123,
        image_size=(img_height, img_width),
        batch_size=b_size
    )
    return train_ds, val_ds, test_ds


def data_augmentation(train_ds, valid_ds):
    """This functino uses existing training data form the training dataset
    and uses data augmentation to generate more data
    Args:
        train_ds : Training dataset created from images in the training dir
        valid_ds : Validation dataset created from images in the validation dir

    Returns:
        train_ds : augmented dataset for training
        valid_ds : augmented dataset for validation
        classnames : list of classnames presented in dataset
        n_classes : integer of number of classes present in dataset
    """

    classnames = train_ds.class_names
    n_classes = len(classnames)
    AUTOTUNE = tf.data.AUTOTUNE
    train_ds = train_ds.cache().shuffle(1000).prefetch(
        buffer_size=AUTOTUNE)
    valid_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
    # Data normalization
    normalization_layer = layers.Rescaling(1. / 255)
    normalized_ds = train_ds.map(
        lambda x, y: (normalization_layer(x), y))
    return train_ds, valid_ds, classnames, n_classes


def create_model():
    """This function creates a model that can later be fitted in the
    run_model() function. The function used is the relu and the softmax
    function

    Returns:
        model : The machine learning model
    """
    model = Sequential([
        # Layers for constructing a neural network
        layers.Rescaling(1. / 255,
                         input_shape=(img_height, img_width, 3)),
        layers.RandomFlip("horizontal_and_vertical",
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
    # Compiling model and keeping track of accuracy
    model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                  optimizer='adam',
                  metrics=['accuracy'])
    return model


def run_model(model, train_ds, val_ds):
    """This function runs the model, and keeps track of certain metrics,
    like Train/validation accuracy and loss. In this script it runs with
    10 epochs.
    Args:
        model : The machine learning model returned by create_model()
        train_ds : Training dataset created from images in the training dir
        val_ds : Validation dataset created from images in the validation dir

    Returns:
        model: The fitted machine learning model
        acc : The accuracy of the model on the training data
        val_acc : The accuracy of model on the validation data
        loss : The loss of the model on the training data
        val_loss : The loss of the model on the validation data
    """
    # Fitting the model
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs
    )
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']

    loss = history.history['loss']
    val_loss = history.history['val_loss']
    # Saving the model
    tf.saved_model.save(model, model_dir)
    return model, acc, val_acc, loss, val_loss


def predict_img(model, class_names):
    """This function loads images from the test set, these are then
    predicted by our model and based on the right answers the model gives
    a score is printed.
    Args:
        model : The machine learning model returned by run_model()
        class_names : Names of the classes in the train directory
    """
    img = tf.keras.utils.load_img(
        "opdr2/testimage.png", target_size=(img_height, img_width)
    )
    img_array = tf.keras.utils.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)  # Create a batch

    # Predicting images from array
    predictions = model.predict(img_array)
    # Score based on the model predictions
    score = tf.nn.softmax(predictions[0])

    print(
        "This image most likely belongs to {} with a {:.2f} percent confidence.\n"
        .format(class_names[np.argmax(score)], 100 * np.max(score))
    )


def evaluate_and_plot(model, acc, val_acc, loss, val_loss):
    """This function uses the test set to test the predictions of the
    model. It then plots the results.

    Args:
        model : The machine learning model returned by run_model()
        acc : The accuracy of the model on the training data
        val_acc : The accuracy of model on the validation data
        loss : The loss of the model on the training data
        val_loss : The loss of the model on the validation data
    """
    print("Evaluate on test data")
    results = model.evaluate(test_ds, batch_size=32)
    print("test loss, test acc:", results)
    print("\nGenerate predictions for 3 samples")
    predictions = model.predict(test_ds)
    print("\npredictions shape:", predictions.shape)

    epochs_range = range(epochs)
    # Plotting results
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
    """This is the main, everything is executed here
    """
    train_ds, valid_ds, test_ds = make_datasets()
    train_ds, valid_ds, classnames, n_classes = data_augmentation(
        train_ds, valid_ds)
    model = create_model()
    model, acc, val_acc, loss, val_loss = run_model(model, train_ds,
                                                    valid_ds)
    predict_img(model, classnames)
    evaluate_and_plot(model, acc, val_acc, loss, val_loss)
