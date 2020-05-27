""" 
This module provides several normalization functions to change raw feature vectors 
into a representation that is more suitable for the downstream estimators. 
"""

import numpy as np
import sys

def simple_scaling(input_data):
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

def min_max_normalization(input_data):
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

def standardization(input_data):
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
    assert type(input_data) is np.ndarray, "The 'input_data' must be numpy array."

    # Get the mean values and the standard deviation of the input numpy array along the axis  
    Mean = np.mean(input_data, axis = 0)
    Std = np.std(input_data, axis = 0)

    # Standardization 
    standardized_input_data = (input_data - Mean) / (Std + sys.float_info.min)

    # Return standardized input data
    return standardized_input_data