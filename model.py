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
    raw_lines = read_driving_log(driving_log)
    lines = map(lambda line: parse_driving_log_line(line=line, image_dir=image_dir), raw_lines)

    images = []
    steering_angles = []
    for line in lines:
        (center_image_path, left_image_path, right_image_path, center_steering_angle, throttle, _break, speed) = line

        center_image = cv2.imread(center_image_path)
        left_image = cv2.imread(left_image_path)
        right_image = cv2.imread(right_image_path)
        center_image_flipped = np.fliplr(center_image)

        steering_correction_factor = 0.2 # this is a parameter to tune
        left_steering_angle = center_steering_angle + steering_correction_factor
        right_steering_angle = center_steering_angle - steering_correction_factor
        center_steering_angle_flipped = -center_steering_angle

        images.extend([center_image, left_image, right_image, center_image_flipped])
        steering_angles.extend([center_steering_angle, left_steering_angle, right_steering_angle, center_steering_angle_flipped])

    X_train = np.array(images)
    y_train = np.array(steering_angles)
    return X_train, y_train

def parse_driving_log_line(line, image_dir):
    # Format: Center Image, Left Image, Right Image, Steering Angle, Throttle, Break, Speed
    [center_image_path, left_image_path, right_image_path, steering_angle, throttle, _break, speed] = line
    def map_path(path):
        filename = os.path.split(path)[-1]
        return image_dir + "/" + filename
    return (map_path(center_image_path), map_path(left_image_path), map_path(right_image_path), float(steering_angle), throttle, _break, speed)

def read_driving_log(filename):
    lines = []
    with open(filename) as csvfile:
        has_header = csv.Sniffer().has_header(csvfile.read(1024))
        csvfile.seek(0)  # rewind
        reader = csv.reader(csvfile)
        if has_header:
            next(reader)  # skip header row
        for line in reader:
            lines.append(line)
        return lines

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
    # model = LeNet(input_shape=(160, 320, 3))
    # model.compile(optimizer="adam", loss="mse")
    # # Train model
    # model.fit(x=X_train, y=y_train, batch_size=128, validation_split=0.3, shuffle=True, epochs=5)
    # # Save model
    # model_file = "model.h5"
    # model.save(model_file)
    # print("Saved model to: ", model_file)
    #
    # # Temporary fix - AttributeError: 'NoneType' object has no attribute 'TF_NewStatus
    # K.clear_session()