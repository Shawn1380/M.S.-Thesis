from keras import backend as K
from preprocessing import Constant
import scipy.io as sio
import numpy as np
import pathlib
import sys

def GetImageDataFormat():
    """ Return default data format convention.

    # Return:

    A string, either 'channels_first' or 'channels_last'.
    It specifics which data format convention Keras will follow. (keras.backend.image_data_format() returns it)
    """

    # Return image data format
    return K.image_data_format()

def GetInputData(num_of_cells, num_of_CUEs, num_of_D2Ds, num_of_samples, image_data_format):
    """ Return input data (channel gain matrix) in numpy array.

    # Aruguments:

    num_of_cells: int
        Number of the cells in the cellular system.
    num_of_CUEs: int
        Number of the CUEs in each cell.
    num_of_D2Ds: int
        Number of the D2D pairs in each cell.
    num_of_samples: int or tuple
        Number of the random channel realizations according to the above parameters setting.
    image_data_format: string
        Either 'channels_first' or 'channels_last'.
        It specifics which data format convention Keras will follow. (keras.backend.image_data_format() returns it)

    # Return:

    input_data: 4-D numpy array with shape (batch_size, rows, cols, channels) or (batch_size, channels, rows, cols)
        Input data in given .mat file. Each element in input_data stands for channel gain matrix, 
        which is the 3-D numpy array with shape (channels, rows, cols) if data_format is "channels_first",
        or 3-D numpy array with shape (rows, cols, channels) if data_format is "channels_last".
    """

    # Insert debugging assertions
    assert num_of_cells in Constant.cell_range, f"The 'num_of_cells' must be element in {Constant.cell_range}."
    assert num_of_CUEs in Constant.CUE_range, f"The 'num_of_CUEs' must be element in {Constant.CUE_range}."
    assert num_of_D2Ds in Constant.D2D_range, f"The 'num_of_D2Ds' must be element in {Constant.D2D_range}."

    # Define inner function
    def inner(num_of_samples):

        # Initialization of variables
        batch_size = num_of_samples
        rows = num_of_cells * (num_of_CUEs + num_of_D2Ds)
        cols = 1 + num_of_D2Ds
        channels = num_of_cells

        # Get the filname of the desired .mat file from the directory
        dataset_dir = pathlib.Path.cwd().joinpath('dataset')
        cell_dir = '{} cell'.format(num_of_cells)
        dataset_dir = dataset_dir.joinpath(cell_dir)
        filename = 'data_Cell_{}_CUE_{}_D2D_{}_{}.mat'.format(num_of_cells, num_of_CUEs, num_of_D2Ds, num_of_samples)
        mat_fname = dataset_dir.joinpath(filename)

        # Load the .mat file contents
        mat_content = sio.loadmat(mat_fname)
        input_data = mat_content['input_data']

        # Flatten the 2-D numpy array to 1-D numpy array
        input_data = np.ndarray.flatten(input_data)

        # Each element in the flattened 1-D numpy array is a Python list
        # Convert the numpy array of lists to the numpy array
        if image_data_format == 'channels_first':
            input_data = np.vstack(input_data)
            input_data = np.reshape(input_data, (batch_size, channels, rows, cols))
        elif image_data_format == 'channels_last':
            input_data = np.vstack(input_data)
            input_data = np.reshape(input_data, (batch_size, rows, cols, channels))
        else:
            raise ValueError("'image_data_format' must be 'channels_first' or 'channels_last'.")

        # Return input data
        return input_data

    if type(num_of_samples) is int:    
        return inner(num_of_samples)
    elif type(num_of_samples) is tuple:
        return np.concatenate(list(map(inner, num_of_samples)), axis = 0) 
    else:
        raise TypeError("'num_of_samples' must be integer or tuple.")

def GetTargetData(num_of_cells, num_of_CUEs, num_of_D2Ds, num_of_samples):
    """ Return target data (power allocation vector) in numpy array.

    # Arguments:

    num_of_cells: int
        Number of the cells in the cellular system.
    num_of_CUEs: int
        Number of the CUEs in each cell.
    num_of_D2Ds: int
        Number of the D2D pairs in each cell.
    num_of_samples: int or tuple
        Number of the random channel realizations according to the above parameters setting.

    # Return:

    target_data: 2-D numpy array with shape (batch_size, CUE_output_dim + D2D_output_dim)
        Target data in given .mat file. Each element in target_data stands for the power allocation vector, 
        which is the 1-D numpy array with shape (CUE_output_dim + D2D_output_dim, ).
    """

    # Insert debugging assertions
    assert num_of_cells in Constant.cell_range, f"The 'num_of_cells' must be element in {Constant.cell_range}."
    assert num_of_CUEs in Constant.CUE_range, f"The 'num_of_CUEs' must be element in {Constant.CUE_range}."
    assert num_of_D2Ds in Constant.D2D_range, f"The 'num_of_D2Ds' must be element in {Constant.D2D_range}."

    # Define inner function
    def inner(num_of_samples):
        
        # Initialization of variables
        batch_size = num_of_samples
        CUE_output_dim = num_of_CUEs * num_of_cells
        D2D_output_dim = num_of_D2Ds * num_of_CUEs * num_of_cells

        # Get the filname of the desired .mat file from the directory  
        dataset_dir = pathlib.Path.cwd().joinpath('dataset')
        cell_dir = '{} cell'.format(num_of_cells)
        dataset_dir = dataset_dir.joinpath(cell_dir)
        filename = 'data_Cell_{}_CUE_{}_D2D_{}_{}.mat'.format(num_of_cells, num_of_CUEs, num_of_D2Ds, num_of_samples)
        mat_fname = dataset_dir.joinpath(filename)

        # Load the .mat file contents
        mat_content = sio.loadmat(mat_fname)
        target_data = mat_content['target_data']
        optimal_CUE_power = target_data[0]
        optimal_D2D_power = target_data[1]

        # Each element in the numpy array is a Python list
        # Convert the numpy array of lists to the numpy array
        optimal_CUE_power = np.vstack(optimal_CUE_power)
        optimal_CUE_power = np.reshape(optimal_CUE_power, (batch_size, CUE_output_dim))
        optimal_D2D_power = np.vstack(optimal_D2D_power)
        optimal_D2D_power = np.reshape(optimal_D2D_power, (batch_size, D2D_output_dim))

        # Return target data
        target_data = np.hstack((optimal_CUE_power, optimal_D2D_power))
        return target_data

    if type(num_of_samples) is int:
        return inner(num_of_samples)
    elif type(num_of_samples) is tuple:
        return np.concatenate(list(map(inner, num_of_samples)), axis = 0) 
    else:
        raise TypeError("'num_of_samples' must be integer or tuple.")

def SplitDataset(input_data, target_data, normalized_input_data = None, proportion = Constant.data_proportion, shuffle = True):
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
        y_train and y_test are numpy arrays used for testing.
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

def GetInputShape(input_data):
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

def GetTargetShape(target_data):
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

def ReshapeInputData3D(input_data, image_data_format, rows, cols, channels):
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

def ReshapeInputData2D(input_data, steps, channels):
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

def ReshapeInputData1D(input_data):
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

def SimpleScaling(input_data):
    """ Scale the values of the input numpy array to the range [0, 1] by simple scaling.

    # Arguments:

    input_data: numpy array
        The numpy array which is used as the input of the model. 

    # Return:

    scaled_input_data: numpy array
        Scaled input data. Each feature is normalized individually such that it is in the range [0, 1].  
    """

    # Insert debugging assertions
    assert type(input_data) is np.ndarray, "The 'input_data' must be numpy array."

    # Get the minimum values of the input numpy array along the axis  
    Max = np.max(input_data, axis = 0)

    # Simple sclaing 
    scaled_input_data = input_data / (Max + sys.float_info.min)

    # Return scaled input data
    return scaled_input_data

def MinMaxNormalization(input_data):
    """ Scale the values of the input numpy array to the range [0, 1] by min-max normalization.

    # Arguments:

    input_data: numpy array
        The numpy array which is used as the input of the model. 

    # Return:

    normalized_input_data: numpy array
        Normalized input data. Each feature is normalized individually such that it is in the range [0, 1].  
    """

    # Insert debugging assertions
    assert type(input_data) is np.ndarray, "The 'input_data' must be numpy array."

    # Get the minimum and maximun values of the input numpy array along the axis  
    Max = np.max(input_data, axis = 0)
    Min = np.min(input_data, axis = 0)

    # Min-max normalization 
    normalized_input_data = (input_data - Min) / (Max - Min + sys.float_info.min)

    # Return normalized input data
    return normalized_input_data

def Standardization(input_data):
    """ Subtract the mean value and divide by standard deviation from the values of the input numpy array.

    # Arguments:

    input_data: numpy array
        The numpy array which is used as the input of the model.  

    # Return:

    standardized_input_data: numpy array
        Standardized input data. Each feature is standardized individually such that
        it follows the Gaussian distribution with zero mean and unit variance.
    """

    # Insert debugging assertions
    assert type(input_data) is np.ndarray, "The 'input_data must' be numpy array."

    # Get the mean values and the standard deviation of the input numpy array along the axis  
    Mean = np.mean(input_data, axis = 0)
    Std = np.std(input_data, axis = 0)

    # Standardization 
    standardized_input_data = (input_data - Mean) / (Std + sys.float_info.min)

    # Return standardized input data
    return standardized_input_data