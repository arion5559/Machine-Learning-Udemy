import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Conv2D, MaxPool2D, Flatten, Dense
from keras.preprocessing import image
from ann_visualizer.visualize import ann_viz


def convert_to_test(path):
    test_image = image.load_img(path, target_size=(150, 150))
    test_image = image.img_to_array(test_image)
    test_image = np.expand_dims(test_image, axis=0)
    return test_image


def print_result(result):
    if result[0][0] == 1:
        prediction = "It's a dog"
    else:
        prediction = "It's a cat"
    return prediction


train_datagen = ImageDataGenerator(rescale=1. / 255, shear_range=0.2, zoom_range=0.2, horizontal_flip=True)

train_set = train_datagen.flow_from_directory("dataset/training_set", target_size=(150, 150),
                                              batch_size=32, class_mode="binary")

test_datagen = ImageDataGenerator(rescale=1. / 255)

test_set = train_datagen.flow_from_directory("dataset/test_set", target_size=(150, 150),
                                             batch_size=32, class_mode="binary")

cnn = tf.keras.models.Sequential()

cnn.add(Conv2D(filters=32, kernel_size=3, activation="relu", input_shape=[150, 150, 3]))

cnn.add(MaxPool2D(pool_size=2, strides=2))

cnn.add(Conv2D(filters=32, kernel_size=3, activation="relu"))
cnn.add(MaxPool2D(pool_size=2, strides=2))

cnn.add(Conv2D(filters=32, kernel_size=3, activation="relu"))
cnn.add(MaxPool2D(pool_size=2, strides=2))

cnn.add(Flatten())

cnn.add(Dense(units=128, activation=tf.nn.relu))
cnn.add(Dense(units=128, activation=tf.nn.relu))
cnn.add(Dense(units=128, activation=tf.nn.relu))

cnn.add(Dense(units=1, activation=tf.nn.sigmoid))

cnn.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

cnn.fit(x=train_set, validation_data=test_set, epochs=100)

test_image_1 = convert_to_test("dataset/single_prediction/cat_or_dog_1.jpg")
test_image_2 = convert_to_test("dataset/single_prediction/cat_or_dog_2.jpg")
test_cuca = convert_to_test("dataset/single_prediction/cat_or_dog_cuca.jpeg")
test_tsuki = convert_to_test("dataset/single_prediction/cat_or_dog_tsuki.jpeg")

result_1 = cnn.predict(test_image_1)
result_2 = cnn.predict(test_image_2)
result_cuca = cnn.predict(test_cuca)
result_tsuki = cnn.predict(test_tsuki)

print(train_set.class_indices)

print(f"Test 1: {print_result(result_1)}")
print(f"Test 2: {print_result(result_2)}")
print(f"Test cuca: {print_result(result_cuca)}")
print(f"Test tsuki: {print_result(result_tsuki)}")

tf.keras.utils.plot_model(cnn, show_shapes=True, show_layer_names=True)

ann_viz(cnn, title="My first convolutinal neural network")
