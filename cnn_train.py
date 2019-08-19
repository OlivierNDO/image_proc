# Configuration
###############################################################################
# Import packages
import datetime
import gc
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pickle as pkl
from operator import itemgetter
import os
import random
from skimage.transform import resize
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.python.client import device_lib
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense, Activation, BatchNormalization
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img
import time
from tqdm import tqdm

# Input data file paths
folder_path = "D:/dogs_cats/train/train/"
cat_file_paths = [folder_path + f for f in os.listdir(folder_path) if 'cat' in f]
dog_file_paths = [folder_path + f for f in os.listdir(folder_path) if 'dog' in f]

# Processing config
config_test_percent = 0.15
config_validation_perc = 0.1
config_img_height = 180
config_img_width = 180

# Model save & callback config
config_model_timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M")
config_model_save_name = "D:/dogs_cats/model_save_dir/keras_dog_cat_img_model_{dt_tm}.hdf5".format(dt_tm = config_model_timestamp)
config_log_folder = 'D:/dogs_cats/log_files/'
config_max_worse_epochs = 3

# Model training config
config_batch_size = 10
config_learning_rate = 0.0001
config_epochs = 50
config_cnn_activation = 'relu'
config_dropout = 0.1


# Define Functions
###############################################################################
def seconds_to_time(sec):
    """convert seconds (integer or float) to time in 'hh:mm:ss' format"""
    import numpy as np
    if (sec // 3600) == 0:
        HH = '00'
    elif (sec // 3600) < 10:
        HH = '0' + str(int(sec // 3600))
    else:
        HH = str(int(sec // 3600))
    min_raw = (np.float64(sec) - (np.float64(sec // 3600) * 3600)) // 60
    if min_raw < 10:
        MM = '0' + str(int(min_raw))
    else:
        MM = str(int(min_raw))
    sec_raw = (sec - (np.float64(sec // 60) * 60))
    if sec_raw < 10:
        SS = '0' + str(int(sec_raw))
    else:
        SS = str(int(sec_raw))
    return HH + ':' + MM + ':' + SS + ' (hh:mm:ss)'



def sec_to_time_elapsed(end_tm, start_tm, return_time = False):
    """apply seconds_to_time function to start and end times
       * dependency on seconds_to_time() function *"""
    import numpy as np
    sec_elapsed = (np.float64(end_tm) - np.float64(start_tm))
    if return_time:
        return seconds_to_time(sec_elapsed)
    else:
        print('Execution Time: ' + seconds_to_time(sec_elapsed))



def img_add_flip(arr, flip_horiz = True, flip_vert = False):
    """
    Flip numpy array horizontally and/or vertically
    Args:
        arr: three dimensional numpy array
        flip_horiz: flip image horizontally
        flip_vert: flip image vertically
    """
    assert len(arr.shape) == 3, "'arr' input array must be three dimensional"
    arr_copy = arr.copy()
    if flip_horiz:
        arr_copy = np.fliplr(arr_copy)
    if flip_vert:
        arr_copy = np.flipud(arr_copy)
    return arr_copy
    


def img_add_random_noise(arr, noise_perc = 0.05):
    """
    Randomly change <noise_perc> % of pixels in an image
    Args:
        arr: three dimensional numpy array
        noise_perc: percentage of values to replace    
    """
    assert len(arr.shape) == 3, "'arr' input array must be three dimensional"
    arr_copy = arr.copy()
    noise = np.random.randint(0, 255, size = arr.shape)
    rand_numbers = random.sample(range(100), int(noise_perc * 100))
    for x in range(arr.shape[0]):
        for y in range(arr.shape[1]):
            for z in range(arr.shape[2]):
                if random.choice(range(100)) in rand_numbers:
                    arr_copy[x][y][z] += noise[x][y][z]
    return arr_copy
                


def img_add_stripe_noise(arr, stripe_perc = 0.05, stripe_direction = 'h'):
    """
    Add horizontal stripe of randomly colored pixels within a 3d numpy array
    Args:
        arr: three dimensional numpy array
        stripe_perc: height or width of noise stripe (percentage of array dimension)
        stripe_direction: the direction of the noise strip (horizonal or vertical)
    """
    assert len(arr.shape) == 3, "'arr' input array must be three dimensional"
    assert stripe_direction in ['h', 'v'], "'stripe_direction' parameter must be 'v' or 'h'"
    arr_copy = arr.copy()
    noise = np.random.randint(0, 255, size = arr.shape)
    if stripe_direction == 'h':
        use_dim = 0
    else:
        use_dim = 1
    
    # Stripe position
    pixel_length = int(stripe_perc * arr.shape[use_dim])
    stripe_start = random.choice(range(arr.shape[use_dim] - pixel_length))
    stripe_end = stripe_start + pixel_length
    stripe_range = range(stripe_start, stripe_end)
    
    # Value replacement
    for x in range(arr.shape[0]):
        for y in range(arr.shape[1]):
            for z in range(arr.shape[2]):
                if stripe_direction == 'h':
                    if x in stripe_range:
                        arr_copy[x][y][z] += noise[x][y][z]
                else:
                    if y in stripe_range:
                        arr_copy[x][y][z] += noise[x][y][z]
    return arr_copy



def describe_img_sizes(list_of_arrays):
    """
    Describe distribution of array shapes for a list of 3d arrays
    Args:
        list_of_arrays: list of 3d numpy arrays
    Dependencies:
        numpy (import as np)
        pandas (import as pd)
    Returns:
        pandas DataFrame
    """    
    # Dimensions of each array
    img_ht = [a.shape[0] for a in list_of_arrays]
    img_wd = [a.shape[1] for a in list_of_arrays]
    img_d = [a.shape[2] for a in list_of_arrays]
    img_hwd = [a.shape for a in list_of_arrays]
    
    # Mean, median, mode of each single measure
    mean_size = [np.mean(x) for x in [img_ht, img_wd, img_d]]
    median_size = [np.median(x) for x in [img_ht, img_wd, img_d]]
    modal_size = [max(x, key = x.count) for x in [img_ht, img_wd, img_d]]
    
    # Output summary statistics and print most common dimension
    output_df = pd.DataFrame({'dimension': ['height', 'width', 'depth'],
                              'mean' : mean_size,
                              'median' : median_size,
                              'mode' : modal_size})
    
    modal_arr_shape = max(img_hwd, key = img_hwd.count)
    modal_freq = img_hwd.count(modal_arr_shape)
    modal_perc_freq = round(modal_freq / len(img_hwd) * 100, 1)
    print_msg = "The most common array shape is {a} with a frequency of {b} ({c}%)".\
    format(a = str(modal_arr_shape),
           b = str(int(modal_freq)),
           c = str(modal_perc_freq))
    print(print_msg)
    return output_df



def shuffle_two_lists(list_a, list_b):
    """
    Randomly shuffle two lists with the same order, return numpy arrays
    Args:
        list_a: first list you want to shuffle
        list_b: second list you want to shuffle
    Dependencies:
        numpy (import as np)
        operator.itemgetter
        random
    """
    assert len(list_a) == len(list_b), "two input lists must have the same length"
    
    # Define inner function
    def index_slice_list(lst, indices):
    # Slice list by list of indices
        list_slice = itemgetter(*indices)(lst)
        if len(indices) == 1:
            return [list_slice]
        else:
            return list(list_slice)
    
    # Randomly shuffle positional indices
    shuffle_indices = [i for i in range(len(list_a))]
    random.shuffle(shuffle_indices)
    
    # Reorder and return lists
    a_shuffled = index_slice_list(lst = list_a, indices = shuffle_indices)
    b_shuffled = index_slice_list(lst = list_b, indices = shuffle_indices)
    return a_shuffled, b_shuffled



def read_proc_catdog(cat_file_paths, dog_file_paths, config_img_height, config_img_width,
                     augment_images = False, augment_perc = 1, flip_h = False,
                     flip_v = False, gauss_noise = 0, column_noise = 0, bar_noise = 0):
    """
    Read cat and dog image files. Resize, shuffle, and augment numpy 3d image arrays
    Args:
        cat_file_paths: list of file paths to cat pictures (list)
        dog_file_paths: list of file paths to dog pictures (list)
        config_img_height: resize height for pictures (int)
        config_img_width: resize width for pictures (int)
        augment_images: whether to duplicate and modify some *portion* of the images (bool)
        augment_perc: *portion* of images to augment (float)
        flip_h: flip augmented images horizontally (bool)
        flip_v: flip augmented images vertically (bool)
        gauss_noise: percentage of pixels in augmented images to randomly replace (float)
        column_noise: percentage of image width to replace with random vertical column
        bar_noise: percentage of image height to replace with random horizontal bar
    """
    start_tm = time.time()
    # Create binary response variable
    y_cat = [1 for c in range(len(cat_file_paths))]
    y_dog = [0 for d in range(len(dog_file_paths))]
    
    # Load and resize image arrays
    x_cat = [resize(np.array(load_img(c)), (config_img_height, config_img_width)) for c in cat_file_paths]
    x_dog = [resize(np.array(load_img(d)), (config_img_height, config_img_width)) for d in dog_file_paths]
    
    # Combine cat and dog image & label arrays, shuffle 
    y, x = shuffle_two_lists(list_a = y_cat + y_dog, list_b = x_cat + x_dog)
    
    # Augment images
    if augment_images:
        sample_indices = random.sample(range(len(x)), int(augment_perc * len(x)))
        aug_y = []
        aug_x = []
        for i, xi in enumerate(x):
            if i not in sample_indices:
                pass
            else:
                if gauss_noise > 0.0:
                    xi_aug = img_add_random_noise(xi, gauss_noise)
                if column_noise > 0.0:
                    xi_aug = img_add_stripe_noise(xi_aug, column_noise, stripe_direction = 'v')
                if bar_noise > 0.0:
                    xi_aug = img_add_stripe_noise(xi_aug, bar_noise, stripe_direction = 'h')
                if (gauss_noise + column_noise + bar_noise) == 0:
                    xi_aug = img_add_flip(xi, flip_h, flip_v)
                else:
                    xi_aug = img_add_flip(xi_aug, flip_h, flip_v)
                aug_x.append(xi_aug)
                aug_y.append(y[i])
        y, x = shuffle_two_lists(list_a = y + aug_y, list_b = x + aug_x)

    # Print execution time and return numpy arrays y and x
    end_tm = time.time()
    sec_to_time_elapsed(end_tm, start_tm)
    return np.array(y), np.array(x)
    


def get_number_gpu():
    """
    Return number of available GPUs
    Dependencies:
        from tensorflow.python.client import device_lib
    """
    n_gpu = len([x.name for x in device_lib.list_local_devices() if x.device_type == 'GPU'])
    return n_gpu





# Read & Transform Source Data
###############################################################################
# Split files of each class into train and test
train_dog_paths, test_dog_paths = train_test_split(dog_file_paths, test_size = config_test_percent)
train_cat_paths, test_cat_paths = train_test_split(cat_file_paths, test_size = config_test_percent)



# Process training set
train_y, train_x = read_proc_catdog(cat_file_paths = train_cat_paths,
                                    dog_file_paths = train_dog_paths,
                                    config_img_height = config_img_height,
                                    config_img_width = config_img_width,
                                    augment_images = True,
                                    augment_perc = 0.3,
                                    flip_h = True,
                                    flip_v = True,
                                    gauss_noise = 0,
                                    column_noise = 0,
                                    bar_noise = 0)



# Process test set
test_y, test_x = read_proc_catdog(cat_file_paths = test_cat_paths,
                                  dog_file_paths = test_dog_paths,
                                  config_img_height = config_img_height,
                                  config_img_width = config_img_width,
                                  augment_images = False)

# Fit CNN on Training Set
###############################################################################
# Start timer and clear session
train_start_time = time.time()
keras.backend.clear_session()
    
# Checkpoint and logging
check_point = keras.callbacks.ModelCheckpoint(config_model_save_name, monitor = 'val_loss', verbose = 1, save_best_only = True, mode = 'min')
early_stop = keras.callbacks.EarlyStopping(monitor = 'val_loss', mode = 'min', patience = config_max_worse_epochs)
csv_logger = keras.callbacks.CSVLogger("{fold}log_{ts}.csv".format(fold = config_log_folder, ts = config_model_timestamp))

# Define network structure
model = Sequential()
model.add(Conv2D(30, (3,3), activation = config_cnn_activation, input_shape = train_x.shape[1:]))
model.add(MaxPooling2D(pool_size = (2,2)))
model.add(Conv2D(75, (3,3), activation = config_cnn_activation, input_shape = train_x.shape[1:]))
model.add(MaxPooling2D(pool_size = (2,2)))
model.add(Conv2D(100, (3,3), activation = config_cnn_activation))
model.add(MaxPooling2D(pool_size = (2,2)))
model.add(Conv2D(200, (5,5), activation = config_cnn_activation))
model.add(MaxPooling2D(pool_size = (2,2)))
model.add(Conv2D(100, (3,3), activation = config_cnn_activation))
model.add(MaxPooling2D(pool_size = (2,2)))
model.add(Flatten())
model.add(Dropout(config_dropout))
model.add(Dense(100, activation = config_cnn_activation))
model.add(Dense(1, activation = 'sigmoid'))



# Scale to multiple GPUs
parallel_model = keras.utils.multi_gpu_model(model, gpus = get_number_gpu())
parallel_model.compile(loss='binary_crossentropy',
                       optimizer = keras.optimizers.Adam(lr = config_learning_rate),
                       metrics = ['accuracy'],
                       weighted_metrics = ['accuracy'])

# Fit model
parallel_model.fit(train_x, train_y,
                   validation_split = config_validation_perc,
                   epochs = config_epochs,
                   batch_size = config_batch_size,
                   callbacks = [check_point, early_stop, csv_logger])
    
train_end_time = time.time()
sec_to_time_elapsed(train_end_time, train_start_time)

# Evaluate CNN on Test Set
###############################################################################
# Load model and predict on test set
cnn_model = keras.models.load_model(config_model_save_name)
pred_probs = [pp[0] for pp in cnn_model.predict(test_x)]

# Evaluate
pred_bin = [int(round(p,0)) for p in pred_probs]
residuals = [pred_probs[i] - test_y[i] for i in range(len(test_y))]
n_correct = np.sum([pred_bin[i] == test_y[i] for i in range(len(test_y))])
test_acc = n_correct / len(test_y)

# Plot residuaaals
plt.hist(residuals, bins = 60)
plt.show()

print(test_acc)


# Notes
###############################################################################
"""
Test set accuracy 08/18/2019: 0.9064
"""

