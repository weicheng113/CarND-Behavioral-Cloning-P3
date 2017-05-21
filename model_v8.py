import numpy as np
import csv
import cv2
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import tensorflow as tf
import matplotlib.pyplot as plt
from itertools import chain
from functools import reduce
from unittest import TestCase
import pandas as pd
from random import randint
# TODO: import Keras layers you need here
from keras.models import Sequential
from keras.layers.core import Dense, Flatten, Lambda, Activation, Dropout
from keras.layers.convolutional import Conv2D, MaxPooling2D, Cropping2D
from keras.optimizers import Adam
from keras import backend as K
from keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping
from image_processing import random_shadow, contrast_image

flags = tf.app.flags
FLAGS = flags.FLAGS

# command line flags
flags.DEFINE_string('data_dirs', 'data', "Data directory list, separated by comma")
flags.DEFINE_integer('epochs', 50, "Training epochs")

class DrivingImage(object):
    def __init__(self, path):
        self.path = path
    def load(self):
        bgr = cv2.imread(self.path)
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        return rgb

class FlippedImage(DrivingImage):
    def __init__(self, path):
        DrivingImage.__init__(self, path)

    def load(self):
        image = super().load()
        flipped = np.fliplr(image)
        return flipped

class ContrastImage(DrivingImage):
    def __init__(self, driving_image):
        DrivingImage.__init__(self, driving_image.path)
        self.driving_image = driving_image

    def load(self):
        image = self.driving_image.load()
        return contrast_image(image)

class RandomShadowImage(DrivingImage):
    def __init__(self, driving_image):
        DrivingImage.__init__(self, driving_image.path)
        self.driving_image = driving_image

    def load(self):
        image = self.driving_image.load()
        return random_shadow(image)

def load_from_dirs(data_dirs):
    """
    Return a list of tuples. Each tuple represents a line in driving log csv under given directory set.
    
    Parse the driving log csv in the given directory set and combine the result.
    """
    combined_lines = []
    for data_dir in data_dirs:
        lines = load_from_dir(data_dir)
        combined_lines.extend(lines)
    return combined_lines

def load_from_dir(data_dir):
    """
    Return a list of tuples. Each tuple represents a line in driving log csv under given directory.
    
    Parse the driving log csv in the given directory.
    """
    driving_log = data_dir + "/driving_log.csv"
    image_dir = data_dir + "/IMG"
    print("Loading data(driving log file: ", driving_log, ", image directory: ", image_dir, ")")

    # Format: Center Image, Left Image, Right Image, Steering Angle, Throttle, Break, Speed
    raw_lines = read_driving_log(driving_log)
    lines = map(lambda line: parse_driving_log_line(line=line, image_dir=image_dir), raw_lines)
    return lines

def parse_driving_log_line(line, image_dir):
    """
    Return a tuple containing - Center Image, Left Image, Right Image, Steering Angle, Throttle, Break, Speed.
    
    Parse given line and replace the image directory.
    """
    # Format: Center Image, Left Image, Right Image, Steering Angle, Throttle, Break, Speed
    [center_image_path, left_image_path, right_image_path, steering_angle, throttle, _break, speed] = line
    def map_path(path):
        filename = path.replace('\\', '/').split("/")[-1]
        return image_dir + "/" + filename
    return (map_path(center_image_path), map_path(left_image_path), map_path(right_image_path), float(steering_angle), throttle, _break, speed)

def read_driving_log(filename):
    """
    Return a list of lines in given csv file.
    
    Read lines from the given csv file.
    """
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

def generator(samples, batch_size):
    """
    Return a generator and in each iteration, it returns a batch of images and their steering angles.
    
    Normal generator only load a batch of images at a time, which is to avoid out-of-memory issue.
    """
    num_samples = len(samples)
    test = TestCase()
    while True:
        shuffled_samples = shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch = shuffled_samples[offset: offset + batch_size]

            images = []
            steering_angles = []
            for pair in batch:
                (driving_image, steering_angle) = pair
                images.append(driving_image.load())
                steering_angles.append(steering_angle)
            X_train = np.array(images)
            y_train = np.array(steering_angles)
            yield X_train, y_train

def make_partitions(samples, max_num_partitions):
    """
    Return partitions.
    
    Partition samples by angles. Used by equally_distributed_generator function.
    """
    df = pd.DataFrame(samples, columns=['image', 'steering_angle'])
    max_angle = df['steering_angle'].max()
    smaller_than_min_angle = df['steering_angle'].min() - 0.000001

    boundaries = [np.linspace(smaller_than_min_angle, max_angle, max_num_partitions+1)[i: i + 2] for i in range(max_num_partitions)]
    partitions = []
    for boundary in boundaries:
        start, end = boundary
        partition = df[(df.steering_angle > start) & (df.steering_angle <= end)]
        size = partition.shape[0]
        if size > 0:
            partitions.append(partition)

    size = reduce(lambda acc, partition: acc + partition.shape[0], partitions, 0)
    print("total partition size: ", size, ", sample len: ", len(samples))
    assert(size == len(samples))

    return partitions

def equally_distributed_generator(samples, batch_size):
    """
    Return a generator and in each iteration, it returns a batch of images and their steering angles.
    
    Equally distributed generator tries to provide balanced sample data from different steering angles to overcome data bias.
    Normally, the data contains more steering angles near the center 0 and bigger angles are fewer, 
    which may result in biased mode.
    """
    shuffled_samples = shuffle(samples)
    max_num_partitions = batch_size
    partitions = make_partitions(shuffled_samples, max_num_partitions = max_num_partitions)
    num_partitions = len(partitions)
    assert(num_partitions <= max_num_partitions)

    while True:
        images = []
        steering_angles = []
        partition_index_increment = 2 # Any number greater than 0
        data_index_increment = 0
        for no_op in range(batch_size):
            partition_index = partition_index_increment % num_partitions
            partition = partitions[partition_index]
            row = data_index_increment % partition.shape[0]
            driving_image, steering_angle = partition.iloc[row]
            images.append(driving_image.load())
            steering_angles.append(steering_angle)
            partition_index_increment = partition_index_increment + 1
            if(partition_index == 0):
                data_index_increment = data_index_increment + 1

        X_train = np.array(images)
        y_train = np.array(steering_angles)
        yield X_train, y_train

def alternating_generator(samples, batch_size):
    """
    Return a generator and in each iteration, it returns a batch of images and their steering angles.
    
    The goal of alternating generator is to balance between sample data coverage and data equal distribution.
    It will alternate to pick patches from normal generator and equally distributed generator based on alternating factor.
    """
    alternating_factor = 1
    gen1 = generator(samples, batch_size)
    gen2 = equally_distributed_generator(samples, batch_size)
    while True:
        if (alternating_factor % 3) > 0:
            yield next(gen1)
        else:
            yield next(gen2)
        alternating_factor = alternating_factor + 1

def NvidiaNet(input_shape):
    """
    Return a model consists of 5 conv layers and 4 full connected layers.
    """
    model = Sequential()

    def preprocess(image):
        return image/255 - 0.5
    model.add(Cropping2D(cropping=((50,20), (0,0)), input_shape=input_shape))
    model.add(Lambda(preprocess))

    model.add(Conv2D(16, (5, 5), strides=(2, 2), activation='elu'))
    model.add(Conv2D(24, (5, 5), strides=(2, 2), activation='elu'))
    model.add(Conv2D(36, (5, 5), strides=(2, 2), activation='elu'))
    model.add(Conv2D(48, (3, 3), activation='elu'))
    model.add(Conv2D(64, (3, 3), activation='elu'))

    model.add(Dropout(0.5))

    model.add(Flatten())
    model.add(Dense(100, activation='elu'))
    model.add(Dense(50, activation='elu'))
    model.add(Dense(10, activation='elu'))
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

def transform(lines):
    """
    Return a new list of tuples. Each tuple contains image and its steering angle.
    
    It performs some data augmentation from original driving_log csv lines, 
    which includes left and right images with angle correction factor and image flipping.
    """
    def transform_line(line):
        (center_image_path, left_image_path, right_image_path, center_steering_angle, throttle, _break, speed) = line
        # correction_factor = 0.9
        # correction = center_steering_angle * correction_factor
        # left_steering_angle = center_steering_angle + correction
        # right_steering_angle = center_steering_angle - correction
        #
        # return [(center_image_path, center_steering_angle),
        #         (left_image_path, left_steering_angle),
        #         (right_image_path, right_steering_angle)]
        correction_factor_level1 = 0.06
        correction_factor_level2 = 0.2
        correction_factor_level3 = 0.6

        abs_center_steering_angle = abs(center_steering_angle)
        left_steering_angle = center_steering_angle
        right_steering_angle = center_steering_angle

        if abs_center_steering_angle < 0.05:
            left_steering_angle = center_steering_angle + correction_factor_level1
            right_steering_angle = center_steering_angle - correction_factor_level1
        elif abs_center_steering_angle < 0.12:
            left_steering_angle = center_steering_angle + correction_factor_level2
            right_steering_angle = center_steering_angle - correction_factor_level2
        else:
            left_steering_angle = center_steering_angle + correction_factor_level3
            right_steering_angle = center_steering_angle - correction_factor_level3

        return [(center_image_path, center_steering_angle),
                (left_image_path, left_steering_angle),
                (right_image_path, right_steering_angle)]

    def normal_and_flipped(pair):
        image_path, steering_angle = pair
        return [(RandomShadowImage(ContrastImage(DrivingImage(image_path))), steering_angle),
                (RandomShadowImage(ContrastImage(FlippedImage(image_path))), -steering_angle)]

    image_steering_angle_pairs = list(flatmap(transform_line, lines))
    pairs_with_flipped = list(flatmap(normal_and_flipped, image_steering_angle_pairs))
    return pairs_with_flipped

def steering_angle_distribution(samples):
    steering_angles = list(map(lambda pair: pair[1], samples))

    test = TestCase()
    test.assertAlmostEqual(sum(steering_angles), 0.0)
    show_steering_angle_distribution(steering_angles)

def show_steering_angle_distribution(steering_angles):
    plt.hist(steering_angles)
    plt.title("Steering Angle Distribution")
    plt.xlabel("Angle")
    plt.ylabel("Frequency")
    plt.show()

def flatmap(f, items):
    return chain.from_iterable(map(f, items))

def train(samples, epochs, batch_size):
    # Prepare samples.
    train_samples, validation_samples = train_test_split(shuffle(samples), test_size=0.2)
    train_generator = alternating_generator(train_samples, batch_size=batch_size)
    #train_generator = generator(train_samples, batch_size=batch_size)
    validation_generator = generator(validation_samples, batch_size=batch_size)

    # Build model
    model = NvidiaNet(input_shape=(160, 320, 3))
    optimizer = Adam() #Adam(lr=0.001)
    model.compile(optimizer=optimizer, loss="mse")

    # Output model summary.
    print(model.summary())

    # Callbacks:
    # a. Checkpoint to save model at each epochs.
    # b. TersorBoard to save TensorBoard logs.
    # c. EarlyStopping to stop the training when there is no improvement in certain number of epochs.
    model_file="model_v8-{epoch:02d}-{val_loss:.2f}.h5"
    cb_checkpoint = ModelCheckpoint(filepath=model_file, verbose=1)
    cb_tensor_board = TensorBoard(log_dir='./logs', histogram_freq=0, write_graph=True, write_images=True)
    cb_early_stopping = EarlyStopping(patience=2)

    # Train model
    steps_per_epoch = steps(train_samples, batch_size)
    validation_steps = steps(validation_samples, batch_size)

    print("Sample split(train: ", len(train_samples), ", validation: ", len(validation_samples), ")")
    model.fit_generator(
        generator=train_generator,
        steps_per_epoch=steps_per_epoch,
        epochs=epochs,
        validation_data=validation_generator,
        validation_steps=validation_steps,
        callbacks=[cb_checkpoint, cb_tensor_board, cb_early_stopping])

    # Temporary fix - AttributeError: 'NoneType' object has no attribute 'TF_NewStatus
    K.clear_session()

def test_equally_distributed_generator(samples, batch_size):
    gen = equally_distributed_generator(samples, batch_size)
    steering_angles_100_batches = []
    for i in range(100):
        images, steering_angles = next(gen)
        steering_angles_100_batches.extend(steering_angles)
    show_steering_angle_distribution(steering_angles_100_batches)

def main(_):
    # pip install --upgrade tensorflow-gpu
    # https://d17h27t6h515a5.cloudfront.net/topher/2016/December/584f6edd_data/data.zip
    # Load data
    data_dirs = parse_data_dirs()
    epochs = parse_epochs()
    lines = load_from_dirs(data_dirs)
    samples = transform(lines)
    print("Samples(total lines: ", len(lines), ", sample total: ", len(samples), ")")

    train(samples = samples, epochs=epochs, batch_size = 128)

    #steering_angle_distribution(samples = samples)
    #test_equally_distributed_generator(samples=samples, batch_size=128)
# parses flags and calls the `main` function above
if __name__ == '__main__':
    tf.app.run()

def clahe(image):
    print("image type", type(image))
    lab= cv2.cvtColor(np.array(image), cv2.COLOR_BGR2LAB)
    #-----Splitting the LAB image to different channels-------------------------
    l, a, b = cv2.split(lab)

    #-----Applying CLAHE to L-channel-------------------------------------------
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    cl = clahe.apply(l)

    #-----Merge the CLAHE enhanced L-channel with the a and b channel-----------
    limg = cv2.merge((cl,a,b))

    #-----Converting image from LAB Color model to RGB model--------------------
    final = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
    return final