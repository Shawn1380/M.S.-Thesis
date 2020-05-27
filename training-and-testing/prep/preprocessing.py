""" 
This module provides several preprocessing functions to change raw feature vectors 
into a representation that is more suitable for the downstream estimators. 
"""

from prep import constants
import numpy as np

def split_dataset(input_data, target_data, normalized_input_data = None, proportion = constants.data_proportion, shuffle = True):
    """ Return training set and testing set in tuple.

    # Arguments:

    input_data: numpy array
        The numpy array which is used as the input of the model. 
    target_data: numpy array
        The numpy array which is used as the target of the model's output.
    normalized_input_data: numpy array, optional
        The numpy array which is normalized input data.
    proportion: float, optional
        Float between 0 and 1. Fraction of the data to be used as training data.
    shuffle: boolean, optional
        whether to shuffle the training data before splitting.

    # Return:

    Tuple of Numpy arrays: (x_train, y_train, z_train), (x_test, y_test, z_test) if 'normalized_input_data' is not None
        x_train and x_test are numpy arrays used for training, 
        y_train and y_test are numpy arrays used for testing,
        z_train and z_test are numpy arrays used for simulation.

    Tuple of Numpy arrays: (x_train, y_train), (x_test, y_test) if 'normalized_input_data' is None
        x_train and x_test are numpy arrays used for training, 
        y_train and y_test are numpy arrays used for testing.
    """

    # Insert debugging assertions
    assert type(input_data) is np.ndarray, "The 'input_data' must be numpy array."
    assert type(target_data) is np.ndarray, "The 'target_data' must be numpy array."
    assert type(normalized_input_data) is np.ndarray or normalized_input_data is None, "The 'normalized_input_data' must be numpy array or None."
    assert len(input_data) == len(target_data), "The 'input_data' and 'target_data' must have same size in the first axis (batch size)."
    assert 0 <= proportion <= 1, "The 'proportion' must be float between 0 and 1."
    assert type(shuffle) is bool, "The 'shuffle' must be boolean."

    # Initialization of variables
    num_of_samples = len(input_data)

    if normalized_input_data is not None:
        # Shuffle the input numpy array along the first axis. The order is changed but their contents remains the same 
        if shuffle:
            randomize = np.arange(len(normalized_input_data))
            np.random.shuffle(randomize)
            normalized_input_data = normalized_input_data[randomize]
            input_data = input_data[randomize]
            target_data = target_data[randomize]

        # Calculate the slicing index
        slice_index = int(num_of_samples * proportion)

        # Split the normalized input data into training part and testing part
        x_train = normalized_input_data[:slice_index]
        x_test = normalized_input_data[slice_index:]

        # Split the target data into training part and testing part
        y_train = target_data[:slice_index]
        y_test = target_data[slice_index:]

        # Split the input data into training part and testing part
        z_train = input_data[:slice_index]
        z_test = input_data[slice_index:]

        # Return traning set and testing set 
        return (x_train, y_train, z_train), (x_test, y_test, z_test)

    else:
        # Shuffle the input numpy array along the first axis. The order is changed but their contents remains the same 
        if shuffle:
            randomize = np.arange(len(input_data))
            np.random.shuffle(randomize)
            input_data = input_data[randomize]
            target_data = target_data[randomize]

        # Calculate the slicing index
        slice_index = int(num_of_samples * proportion)

        # Split the input data into training part and testing part
        x_train = input_data[:slice_index]
        x_test = input_data[slice_index:]

        # Split the target data into training part and testing part
        y_train = target_data[:slice_index]
        y_test = target_data[slice_index:]

        # Return traning set and testing set 
        return (x_train, y_train), (x_test, y_test)

def get_input_shape(input_data):
    """ Return input shape (does not include the batch axis) of the given input numpy array.

    # Argument:

    input_data: numpy array
        The numpy array which is used as the input of the model. 

    # Return:

    input_shape: tuple
         tuple of interger, does not include the batch axis.
         e.g. input_shape = (3, 128, 128) for 128 x 128 RGB image if data_format = 'channels_first', 
         or (128, 128, 3) for 128 x 128 RGB image if data_format = 'channels_last'.
    """

    # Insert debugging assertions
    assert type(input_data) is np.ndarray, "The 'input_data' must be numpy array."

    # Calculate the shape of input data and exclude the batch axis
    input_shape = input_data.shape[1:]

    # Return input shape
    return input_shape

def get_target_shape(target_data):
    """ Return target shape (does not include the batch axis) of the given target numpy array.

    # Argument:

    target_data: numpy array
        The numpy array which is used as the target of the model's output. 

    # Return:

    target_shape: int
         dimensionality of the output space.
    """

    # Insert debugging assertions
    assert type(target_data) is np.ndarray, "The 'target_data' must be numpy array."

    # Calculate the shape of target data and exclude the batch axis
    target_shape = target_data.shape[1]

    # Return target shape
    return target_shape

def reshape_input_data_3D(input_data, image_data_format, rows, cols, channels):
    """ Gives a new shape (3D) to input numpy array without changing its data.

    # Arguments:

    input_data: numpy array
        The numpy array which is used as the input of the model.
    image_data_format: string
        Either 'channels_first' or 'channels_last'.
        It specifics which data format convention Keras will follow. (keras.backend.image_data_format() returns it)
    rows: int
        The first dimension of the new shape (does not include the batch axis).
    cols: int
        The second dimension of the new shape (does not include the batch axis).
    channels: int
        The third dimension of the new shape (does not include the batch axis).

    # Return 

    reshaped_input_data: numpy array
        This will be a new view object if possible; otherwise, the ValueError will be raised.
    """

    # Insert debugging assertions
    assert type(input_data) is np.ndarray, "The 'input_data' must be numpy array."
    assert type(image_data_format) is str, "The 'image_data_format' must be string."
    assert type(rows) is int, "The 'rows' must be integer."
    assert type(cols) is int, "The 'cols' must be integer."
    assert type(channels) is int, "The 'channels' must be integer."

    # Initialization of variables
    batch_size = len(input_data)

    # Reshape the input data
    if image_data_format == 'channels_first':
        reshaped_input_data = np.reshape(input_data, (batch_size, channels, rows, cols))
    elif image_data_format == 'channels_last':
        reshaped_input_data = np.reshape(input_data, (batch_size, rows, cols, channels))
    else:
        raise ValueError("'image_data_format' must be 'channels_first' or 'channels_last'.") 

    # Return reshaped input data
    return reshaped_input_data

def reshape_input_data_2D(input_data, steps, channels):
    """ Gives a new shape (2D) to input numpy array without changing its data.

    # Arguments:

    input_data: numpy array
        The numpy array which is used as the input of the model.
    steps: int
        The first dimension of the new shape (does not include the batch axis).
    channels: int
        The second dimension of the new shape (does not include the batch axis).

    # Return 

    reshaped_input_data: numpy array
        This will be a new view object if possible; otherwise, the ValueError will be raised.
    """

    # Insert debugging assertions
    assert type(input_data) is np.ndarray, "The 'input_data' must be numpy array."
    assert type(steps) is int, "The 'steps' must be integer."
    assert type(channels) is int, "The 'channels' must be integer."

    # Initialization of variables
    batch_size = len(input_data)

    # Reshape the input data
    reshaped_input_data = np.reshape(input_data, (batch_size, steps, channels))

    # Return reshaped input data
    return reshaped_input_data

def reshape_input_data_1D(input_data):
    """ Gives a new shape (1D) to input numpy array without changing its data.

    # Arguments:

    input_data: numpy array
        The numpy array which is used as the input of the model.
    
    # Return 

    reshaped_input_data: numpy array
        This will be a new view object if possible; otherwise, the ValueError will be raised.
    """

    # Insert debugging assertions
    assert type(input_data) is np.ndarray, "The 'input_data' must be numpy array."

    # Initialization of variables
    batch_size = len(input_data)
    length = 1

    # Loop over all dimensions
    for element in input_data.shape[1:]:
        length *= element

    # Reshape the input data
    reshaped_input_data = np.reshape(input_data, (batch_size, length))

    # Return reshaped input data
    return reshaped_input_data

def get_max_length(target_data_list):
    """ Return maximum length of the target data in the target data list.

    # Arguments:

    target_data_list: list of numpy arrays
        List of target data in different parameters setting (e.g., number of cells, number of CUEs, and number of D2Ds),
        each element in the list corresponds to a target data.

    # Return:

    max_length: int
        Maximum length of all target data in the target data list.
    """

    # Insert debugging assertions
    assert type(target_data_list) is list, "The 'target_data_list' must be list."

    # Initialization of variable
    max_length = 0

    # Get the maximum length of all target data in the target data list
    for target_data in target_data_list:
        if target_data.shape[1] > max_length:
            max_length = target_data.shape[1]

    # Return maximum length
    return max_length

def zero_padding(target_data, max_length):
    """ Add zeros to end of a target data to increases its length.

    # Arguments:

    target_data: numpy array
        The numpy array which is used as the target of the model's output.
    max_length: int
        Maximum length of all target data in the target data list.

    # Return:

    padded_target_data: numpy array
        The padded numpy array which is used as the target of the model's output.
    """

    # Insert debugging assertions
    assert type(target_data) is np.ndarray, "The 'target_data' must be numpy array."
    assert type(max_length) is int, "The 'max_length' must be integer."

    # Add zeros to end of a target data if its length is less than or equal to max length
    if target_data.shape[1] < max_length:
        padded_target_data = np.pad(target_data, ((0, 0), (0, max_length - target_data.shape[1])), 'constant')
    elif target_data.shape[1] == max_length:
        padded_target_data = target_data
    else:
        raise ValueError("The length of 'target_data' along the second axis must be less than or equal to 'max_length'.")

    # Return padded target data
    return padded_target_data

def remove_redundant_zeros(padded_target_data, num_of_cells, num_of_CUEs, num_of_D2Ds):
    """ Remove redundant zeros in the end of a padded target data to decreases its length.

    # Arguments:

    padded_target_data: numpy array
        The padded numpy array which is used as the target of the model's output.
    num_of_cells: int
        Number of the cells in the cellular system.
    num_of_CUEs: int
        Number of the CUEs in each cell.
    num_of_D2Ds: int
        Number of the D2D pairs in each cell.

    # Return:

    target_data: numpy array
        The numpy array which is used as the target of the model's output.
    """

    # Insert debugging assertions
    assert type(padded_target_data) is np.ndarray, "The 'padded_target_data' must be numpy array."
    assert num_of_cells in constants.cell_range, f"The 'num_of_cells' must be element in {constants.cell_range}."
    assert num_of_CUEs in constants.CUE_range, f"The 'num_of_CUEs' must be element in {constants.CUE_range}."
    assert num_of_D2Ds in constants.D2D_range, f"The 'num_of_D2Ds' must be element in {constants.D2D_range}."

    # Initialization of variable
    output_dims = num_of_cells * num_of_CUEs * (1 + num_of_D2Ds)

    # Remove redundant zeros in the end of a padded target data
    if padded_target_data.shape[1] > output_dims:
        target_data = padded_target_data[:, :output_dims]
    elif padded_target_data.shape[1] == output_dims:
        target_data = padded_target_data
    else:
        raise ValueError("The length of 'padded_target_data' along the second axis must be greater than or equal to output dimensions.")

    # Return target data
    return target_data