import numpy as np
import csv
import cv2
import os.path
# TODO: import Keras layers you need here
from keras.models import Sequential
from keras.layers.core import Dense, Flatten, Lambda, Activation
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras import backend as K

def load_data(data_dir):
    driving_log = data_dir + "/driving_log.csv"
    image_dir = data_dir + "/IMG"
    print("Loading data(driving log file: ", driving_log, ", image directory: ", image_dir, ")")

    # Format: Center Image, Left Image, Right Image, Steering Angle, Throttle, Break, Speed
    lines = []
    with open(driving_log) as csvfile:
        has_header = csv.Sniffer().has_header(csvfile.read(1024))
        csvfile.seek(0)  # rewind
        reader = csv.reader(csvfile)
        if has_header:
            next(reader)  # skip header row
        for line in reader:
            lines.append(line)

    images = []
    steering_angles = []
    for line in lines:
        src_path = line[0]
        filename = os.path.split(src_path)[-1]
        file_path = image_dir + "/" + filename

        image = cv2.imread(file_path, cv2.IMREAD_COLOR)
        images.append(image)
        steering_angle = float(line[3])
        steering_angles.append(steering_angle)

        image_flipped = np.fliplr(image)
        images.append(image_flipped)
        steering_angle_flipped = -steering_angle
        steering_angles.append(steering_angle_flipped)
    X_train = np.array(images)
    y_train = np.array(steering_angles)
    return X_train, y_train

def LeNet(input_shape):
    model = Sequential()

    model.add(Lambda(preprocess, input_shape=input_shape))

    model.add(Conv2D(6, (5, 5)))
    model.add(MaxPooling2D((2, 2)))
    model.add(Activation("relu"))

    model.add(Conv2D(6, (5, 5)))
    model.add(MaxPooling2D((2, 2)))
    model.add(Activation("relu"))

    model.add(Flatten())
    model.add(Dense(120))
    model.add(Dense(84))
    model.add(Dense(1))

    return model

def preprocess(image):
    return image/255 - 0.5

if __name__ == '__main__':
    X_train, y_train = load_data("data")
    print("X_train shape: ", X_train.shape)
    print("y_train shape: ", y_train.shape)
    # Build model
    model = LeNet(input_shape=(160, 320, 3))
    model.compile(optimizer="adam", loss="mse")
    # Train model
    model.fit(x=X_train, y=y_train, batch_size=128, validation_split=0.3, shuffle=True, epochs=5)
    # Save model
    model_file = "model_v2.h5"
    model.save(model_file)
    print("Saved model to: ", model_file)

    # Temporary fix - AttributeError: 'NoneType' object has no attribute 'TF_NewStatus
    K.clear_session()