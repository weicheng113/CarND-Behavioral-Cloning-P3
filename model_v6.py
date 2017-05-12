import numpy as np
import csv
import cv2
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import tensorflow as tf
import matplotlib.pyplot as plt
from itertools import chain
from unittest import TestCase
# TODO: import Keras layers you need here
from keras.models import Sequential
from keras.layers.core import Dense, Flatten, Lambda, Activation, Dropout
from keras.layers.convolutional import Conv2D, MaxPooling2D, Cropping2D
from keras.optimizers import Adam
from keras import backend as K
from keras.callbacks import TensorBoard

flags = tf.app.flags
FLAGS = flags.FLAGS

# command line flags
flags.DEFINE_string('data_dirs', 'data', "Data directory list, separated by comma")
flags.DEFINE_integer('epochs', 5, "Training epochs")

def load_from_dirs(data_dirs):
    combined_lines = []
    for data_dir in data_dirs:
        lines = load_from_dir(data_dir)
        combined_lines.extend(lines)
    return combined_lines

def load_from_dir(data_dir):
    driving_log = data_dir + "/driving_log.csv"
    image_dir = data_dir + "/IMG"
    print("Loading data(driving log file: ", driving_log, ", image directory: ", image_dir, ")")

    # Format: Center Image, Left Image, Right Image, Steering Angle, Throttle, Break, Speed
    raw_lines = read_driving_log(driving_log)
    lines = map(lambda line: parse_driving_log_line(line=line, image_dir=image_dir), raw_lines)
    return lines

def parse_driving_log_line(line, image_dir):
    # Format: Center Image, Left Image, Right Image, Steering Angle, Throttle, Break, Speed
    [center_image_path, left_image_path, right_image_path, steering_angle, throttle, _break, speed] = line
    def map_path(path):
        filename = path.replace('\\', '/').split("/")[-1]
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

def generator(sample_lines, batch_size):
    num_samples = len(sample_lines)
    test = TestCase()
    while 1:
        lines = shuffle(sample_lines)
        for offset in range(0, num_samples, batch_size):
            batch_lines = lines[offset: offset + batch_size]

            images = []
            steering_angles = []
            for line in batch_lines:
                (line_images, line_steering_angles) = preprocess(line)
                test.assertAlmostEqual(sum(line_steering_angles), 0.0)

                images.extend(line_images)
                steering_angles.extend(line_steering_angles)
            X_train = np.array(images)
            y_train = np.array(steering_angles)
            yield X_train, y_train

def preprocess(line):
    (center_image_path, left_image_path, right_image_path, center_steering_angle, throttle, _break, speed) = line
    def preprocess_images():
        center = cv2.imread(center_image_path)
        left = cv2.imread(left_image_path)
        right = cv2.imread(right_image_path)

        center_flipped = np.fliplr(center)
        left_flipped = np.fliplr(left)
        right_flipped = np.fliplr(right)
        return [center, left, right,
                center_flipped, left_flipped, right_flipped]

    return (preprocess_images(), preprocess_steering_angles(center_steering_angle))

def preprocess_steering_angles(center):
    correction_factor = 0.1 # this is a parameter to tune
    left = center + correction_factor
    right = center - correction_factor

    center_flipped = -center
    left_flipped = -left
    right_flipped = -right
    return [center, left, right,
            center_flipped, left_flipped, right_flipped]

def NvidiaNet(input_shape):
    model = Sequential()

    def preprocess(image):
        return image/255 - 0.5
    model.add(Cropping2D(cropping=((50,0), (0,0)), input_shape=input_shape))
    model.add(Lambda(preprocess, input_shape=input_shape))

    model.add(Conv2D(3, (5, 5)))
    model.add(Dropout(0.5))
    model.add(MaxPooling2D((2, 2)))
    model.add(Activation("relu"))

    model.add(Conv2D(24, (5, 5)))
    model.add(MaxPooling2D((2, 2), padding="same"))
    model.add(Activation("relu"))

    model.add(Conv2D(36, (5, 5)))
    model.add(Dropout(0.5))
    model.add(MaxPooling2D((2, 2), padding="same"))
    model.add(Activation("relu"))

    model.add(Conv2D(48, (5, 3)))
    model.add(MaxPooling2D((2, 2), padding="same"))
    model.add(Activation("relu"))

    model.add(Conv2D(64, (3, 3)))
    model.add(Activation("relu"))

    model.add(Flatten())
    model.add(Dense(100))
    model.add(Dense(50))
    model.add(Dense(10))
    model.add(Dense(1))

    return model

def steps(samples, batch_size):
    #return (len(samples)+batch_size-1)/batch_size
    return len(samples)/batch_size

def parse_data_dirs():
    raw_dirs = FLAGS.data_dirs.split(",")
    dirs = []
    for dir in raw_dirs:
        dirs.append(dir.strip())
    return dirs

def parse_epochs():
    return FLAGS.epochs

def steering_angle_distribution(lines):
    test = TestCase()
    def steering_angles(line):
        (center_image_path, left_image_path, right_image_path, center_steering_angle, throttle, _break, speed) = line
        angles = preprocess_steering_angles(center_steering_angle)
        test.assertAlmostEqual(sum(angles), 0.0)
        #print("line_steering_angles: ", line_steering_angles)
        return angles

    steering_angles = list(flatmap(steering_angles, lines))
    plt.hist(steering_angles)
    plt.title("Steering Angle Distribution")
    plt.xlabel("Angle")
    plt.ylabel("Frequency")
    plt.show()

def flatmap(f, items):
    return chain.from_iterable(map(f, items))

def train(sample_lines, epochs, batch_size):
    train_sample_lines, validation_sample_lines = train_test_split(sample_lines, test_size=0.2)
    print("Samples(total: ", len(sample_lines), ", train: ", len(train_sample_lines), ", validation: ", len(validation_sample_lines), ")")
    train_generator = generator(train_sample_lines, batch_size=batch_size)
    validation_generator = generator(validation_sample_lines, batch_size=batch_size)
    # Build model
    model = NvidiaNet(input_shape=(160, 320, 3))
    optimizer = Adam(lr=0.001)
    model.compile(optimizer=optimizer, loss="mse")
    # Train model
    steps_per_epoch = steps(train_sample_lines, batch_size)
    validation_steps = steps(validation_sample_lines, batch_size)
    model.fit_generator(
        generator=train_generator,
        steps_per_epoch=steps_per_epoch,
        epochs=epochs,
        validation_data=validation_generator,
        validation_steps=validation_steps)
    # Save model
    model_file = "model_v6.h5"
    model.save(model_file)
    print("Saved model to: ", model_file)
    # Output model summary.
    print(model.summary())
    # Temporary fix - AttributeError: 'NoneType' object has no attribute 'TF_NewStatus
    K.clear_session()

def main(_):
    # pip install --upgrade tensorflow-gpu
    # https://d17h27t6h515a5.cloudfront.net/topher/2016/December/584f6edd_data/data.zip
    # Load data
    data_dirs = parse_data_dirs()
    epochs = parse_epochs()
    sample_lines = load_from_dirs(data_dirs)

    train(sample_lines = sample_lines, epochs=epochs, batch_size = 32)

    #steering_angle_distribution(lines = sample_lines)

# parses flags and calls the `main` function above
if __name__ == '__main__':
    tf.app.run()