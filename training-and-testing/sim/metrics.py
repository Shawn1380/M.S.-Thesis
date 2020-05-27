""" This module implements several utility functions to measure regression performance. """

from sim import constants
import numpy as np

def get_sum_rate(CUE_rate, D2D_rate):
    """ Return system sum rate, CUE sum rate, and D2D sum rate in numpy arrays.

    # Arguments:

    CUE_rate: numpy array with shape (batch_size, num_of_CUEs, 1, num_of_cells)
        The data rate of all CUEs.
    D2D_rate: numpy array with shape (batch_size, num_of_D2Ds, num_of_CUEs, num_of_cells)
        The data rate of all D2D pairs.

    # Return:

    Tuple of numpy arrays: (system_sum_rate, CUE_sum_rate, D2D_sum_rate)
        system_sum_rate: numpy array with shape (batch_size, ), which stands for sum rate of the multi-cell system.
        CUE_sum_rate: numpy array with shape (batch_size, ), which stands for sum rate of all CUEs.
        D2D_sum_rate: numpy array with shape (batch_size, ), which stands for sum rate of all D2D pairs.
    """

    # Insert debugging assertions
    assert type(CUE_rate) is np.ndarray, "The 'CUE_rate' must be numpy array."
    assert type(D2D_rate) is np.ndarray, "The 'D2D_rate' must be numpy array."

    # Get the size of batch dimension
    batch_size = len(CUE_rate)

    # Define inner function
    def inner(CUE_rate, D2D_rate):
        
        # Calculate system sum rate, CUE sum rate, and D2D sum rate
        CUE_sum_rate = np.sum(CUE_rate)
        D2D_sum_rate = np.sum(D2D_rate)
        system_sum_rate = CUE_sum_rate + D2D_sum_rate

        # Return system sum rate, CUE sum rate, and D2D sum rate
        return system_sum_rate, CUE_sum_rate, D2D_sum_rate

    # Initialization of numpy arrays
    system_sum_rate, CUE_sum_rate, D2D_sum_rate = (np.zeros(batch_size) for _ in range(3))

    # Loop over the realizations in the batch
    for index, (CUE_rate, D2D_rate) in enumerate(zip(CUE_rate, D2D_rate)):
        system_sum_rate[index], CUE_sum_rate[index], D2D_sum_rate[index] = inner(CUE_rate, D2D_rate)

    # Return system sum rate, CUE sum rate, and D2D sum rate in batch
    return system_sum_rate, CUE_sum_rate, D2D_sum_rate

def get_power_consumption(CUE_power, D2D_power):
    """ Return system power consumption, CUE power consumption, and D2D power consumption in numpy arrays.

    # Arguments:

    CUE_power: numpy array with shape (batch_size, num_of_CUEs, 1, num_of_cells)
        The transmit power of all CUEs.
    D2D_power: numpy array with shape (batch_size, num_of_D2Ds, num_of_CUEs, num_of_cells)
        The transmit power of all D2D pairs.

    # Return:

    Tuple of numpy arrays: (system_power_consumption, CUE_power_consumption, D2D_power_consumption)
        system_power_consumption: numpy array with shape (batch_size, ), which stands for power consumption of the multi-cell system.
        CUE_power_consumption: numpy array with shape (batch_size, ), which stands for power consumption of all CUEs.
        D2D_power_consumption: numpy array with shape (batch_size, ), which stands for power consumption of all D2D pairs.
    """

    # Insert debugging assertions
    assert type(CUE_power) is np.ndarray, "The 'CUE_power' must be numpy array."
    assert type(D2D_power) is np.ndarray, "The 'D2D_power' must be numpy array."

    # Get the size of each dimension
    batch_size, num_of_D2Ds, num_of_CUEs, num_of_cells = (i for i in D2D_power.shape)

    # Define inner function
    def inner(CUE_power, D2D_power):
        
        # Calculate system power consumption, CUE power consumption, and D2D power consumption
        CUE_power_consumption = np.sum(CUE_power) / constants.PA_inefficiency_factor + constants.circuit_power * num_of_cells * num_of_CUEs
        D2D_power_consumption = np.sum(D2D_power) / constants.PA_inefficiency_factor + constants.circuit_power * num_of_cells * num_of_D2Ds * 2
        system_power_consumption = CUE_power_consumption + D2D_power_consumption

        # Return system power consumption, CUE power consumption, and D2D power consumption
        return system_power_consumption, CUE_power_consumption, D2D_power_consumption

    # Initialization of numpy arrays
    system_power_consumption, CUE_power_consumption, D2D_power_consumption = (np.zeros(batch_size) for _ in range(3))

    # Loop over the realizations in the batch
    for index, (CUE_power, D2D_power) in enumerate(zip(CUE_power, D2D_power)):
        system_power_consumption[index], CUE_power_consumption[index], D2D_power_consumption[index] = inner(CUE_power, D2D_power)

    # Return system power consumption, CUE power consumption, and D2D power consumption in batch
    return system_power_consumption, CUE_power_consumption, D2D_power_consumption

def get_EE(system_sum_rate, CUE_sum_rate, D2D_sum_rate, system_power_consumption, CUE_power_consumption, D2D_power_consumption):
    """ Return system energy effciency, CUE energy effciency, and D2D energy effciency in numpy arrays.

    # Arguments:

    system_sum_rate: numpy array with shape (batch_size, )
        The sum rate of multi-cell system.
    CUE_sum_rate: numpy array with shape (batch_size, )
        The sum rate of all CUEs.
    D2D_sum_rate: numpy array with shape (batch_size, )
        The sum rate of all D2D pairs.
    system_power_consumption: numpy array with shape (batch_size, )
        The power consumption of multi-cell system.
    CUE_power_consumption: numpy array with shape (batch_size, )
        The power consumption of all CUEs.
    D2D_power_consumption: numpy array with shape (batch_size, )
        The power consumption of all D2D pairs.

    # Return:

    Tuple of numpy arrays: (system_EE, CUE_EE, D2D_EE)
        system_EE: numpy array with shape (batch_size, ), which stands for energy efficiency of the multi-cell system.
        CUE_EE: numpy array with shape (batch_size, ), which stands for energy efficiency of all CUEs.
        D2D_EE: numpy array with shape (batch_size, ), which stands for energy efficiency of all D2D pairs.
    """

    # Insert debugging assertions
    assert type(system_sum_rate) is np.ndarray, "The 'system_sum_rate' must be numpy array."
    assert type(CUE_sum_rate) is np.ndarray, "The 'CUE_sum_rate' must be numpy array."
    assert type(D2D_sum_rate) is np.ndarray, "The 'D2D_sum_rate' must be numpy array."
    assert type(system_power_consumption) is np.ndarray, "The 'system_power_consumption' must be numpy array."
    assert type(CUE_power_consumption) is np.ndarray, "The 'CUE_power_consumption' must be numpy array."
    assert type(D2D_power_consumption) is np.ndarray, "The 'D2D_power_consumption' must be numpy array."

    # Calculate system energy effciency, CUE energy effciency, and D2D energy effciency 
    system_EE = np.divide(system_sum_rate, system_power_consumption)
    CUE_EE = np.divide(CUE_sum_rate, CUE_power_consumption)
    D2D_EE = np.divide(D2D_sum_rate, D2D_power_consumption)

    # Return system energy effciency, CUE energy effciency, and D2D energy effciency in batch
    return system_EE, CUE_EE, D2D_EE

def get_UIR(CUE_rate, D2D_rate, CUE_power, D2D_power, QoS_of_CUE):
    """ Return system infeasibility rate (per user), CUE infeasibility rate (per user), and D2D infeasibility rate (per user) in numpy arrays.

    # Arguments:

    CUE_rate: numpy array with shape (batch_size, num_of_CUEs, 1, num_of_cells)
        The data rate of all CUEs.
    D2D_rate: numpy array with shape (batch_size, num_of_D2Ds, num_of_CUEs, num_of_cells)
        The data rate of all D2D pairs.
    CUE_power: numpy array with shape (batch_size, num_of_CUEs, 1, num_of_cells)
        The transmit power of all CUEs.
    D2D_power: numpy array with shape (batch_size, num_of_D2Ds, num_of_CUEs, num_of_cells)
        The transmit power of all D2D pairs.
    QoS_of_CUE: numpy array with shape (batch_size, num_of_CUEs, 1, num_of_cells)
        The minimum rate requirement of all CUEs (bps/Hz).

    # Return: 
    
    Tuple of numpy arrays: (system_UIR, CUE_UIR, D2D_UIR)
        system_UIR: numpy array with shape (batch_size, ), which stands for infeasibility rate (per user) of the multi-cell system.
        CUE_UIR: numpy array with shape (batch_size, ), which stands for infeasibility rate (per user) of all CUEs.
        D2D_UIR: numpy array with shape (batch_size, ), which stands for infeasibility rate (per user) of all D2D pairs.

        The UIR is the number of users with unmet needs (minimum rate requirement & power budget) divided by the total number of users.
    """

    # Insert debugging assertions
    assert type(CUE_rate) is np.ndarray, "The 'CUE_rate' must be numpy array."
    assert type(D2D_rate) is np.ndarray, "The 'D2D_rate' must be numpy array."
    assert type(CUE_power) is np.ndarray, "The 'CUE_power' must be numpy array."
    assert type(D2D_power) is np.ndarray, "The 'D2D_power' must be numpy array."
    assert type(QoS_of_CUE) is np.ndarray, "The 'QoS_of_CUE' must be numpy array."

    # Get the size of each dimension
    batch_size, num_of_D2Ds, num_of_CUEs, num_of_cells = (i for i in D2D_rate.shape)
    total_users = (num_of_CUEs + num_of_D2Ds) * num_of_cells
    total_CUEs = num_of_CUEs * num_of_cells
    total_D2Ds = num_of_D2Ds * num_of_cells

    # Define inner function
    def inner(CUE_rate, D2D_rate, CUE_power, D2D_power, QoS_of_CUE):
            
        # Initialization of boolean numpy arrays
        CUE_feasible = np.ones((num_of_CUEs, 1, num_of_cells), dtype = bool)
        D2D_feasible = np.ones((num_of_D2Ds, 1, num_of_cells), dtype = bool)

        # CUE's power budget limitation
        CUE_feasible = np.logical_and(CUE_feasible, CUE_power <= constants.Pmax)
        CUE_feasible = np.logical_and(CUE_feasible, CUE_power >= 0)
        
        # CUE's minimum rate requirement
        CUE_feasible = np.logical_and(CUE_feasible, CUE_rate >= QoS_of_CUE - 1e-4)

        # D2D pair's power budget limitation
        D2D_feasible = np.logical_and(D2D_feasible, np.sum(D2D_power, axis = 1, keepdims = True) <= constants.Pmax)
        for index in range(num_of_CUEs):
            D2D_feasible = np.logical_and(D2D_feasible, D2D_power[:, [index], :] >= 0)
            
        # D2D pair's minimum rate requirement
        D2D_feasible = np.logical_and(D2D_feasible, np.sum(D2D_rate, axis = 1, keepdims = True) >= constants.QoS_of_D2D - 1e-4)

        # Calculate the number of infeasible CUEs and infeasible D2D pairs
        infeasible_CUE = np.count_nonzero(CUE_feasible == False)
        infeasible_D2D = np.count_nonzero(D2D_feasible == False)

        # Return system infeasibility rate (per user), CUE infeasibility rate (per user), and D2D infeasibility rate (per user)
        return (infeasible_CUE + infeasible_D2D) / total_users, infeasible_CUE / total_CUEs, infeasible_D2D / total_D2Ds 

    # Initialization of numpy arrays
    system_UIR, CUE_UIR, D2D_UIR = [np.zeros(batch_size) for _ in range(3)]

    # Loop over the realizations in the batch
    for index, (CUE_rate, D2D_rate, CUE_power, D2D_power, QoS_of_CUE) in enumerate(zip(CUE_rate, D2D_rate, CUE_power, D2D_power, QoS_of_CUE)):
        system_UIR[index], CUE_UIR[index], D2D_UIR[index] = inner(CUE_rate, D2D_rate, CUE_power, D2D_power, QoS_of_CUE)

    # Return system infeasibility rate (per user), CUE infeasibility rate (per user), and D2D infeasibility rate (per user) in batch
    return system_UIR, CUE_UIR, D2D_UIR

def get_RIR(CUE_rate, D2D_rate, CUE_power, D2D_power, QoS_of_CUE):
    """ Return system infeasibility rate (per realization), CUE infeasibility rate (per realization), and D2D infeasibility rate (per realization) in numpy arrays.

    # Arguments:

    CUE_rate: numpy array with shape (batch_size, num_of_CUEs, 1, num_of_cells)
        The data rate of all CUEs.
    D2D_rate: numpy array with shape (batch_size, num_of_D2Ds, num_of_CUEs, num_of_cells)
        The data rate of all D2D pairs.
    CUE_power: numpy array with shape (batch_size, num_of_CUEs, 1, num_of_cells)
        The transmit power of all CUEs.
    D2D_power: numpy array with shape (batch_size, num_of_D2Ds, num_of_CUEs, num_of_cells)
        The transmit power of all D2D pairs.
    QoS_of_CUE: numpy array with shape (batch_size, num_of_CUEs, 1, num_of_cells)
        The minimum rate requirement of all CUEs (bps/Hz).
    rate_format: string
        Determine what kind of format should be used. Either 'per_user' or 'per_realization'.

    # Return: 
     
    Tuple of numpy arrays: (system_RIR, CUE_RIR, D2D_RIR)
        system_RIR: numpy array with shape (batch_size, ), which stands for infeasibility rate (per realization) of the multi-cell system.
        CUE_RIR: numpy array with shape (batch_size, ), which stands for infeasibility rate (per realization) of all CUEs.
        D2D_RIR: numpy array with shape (batch_size, ), which stands for infeasibility rate (per realization) of all D2D pairs.

        If there is a user's need are not met, the RIR is 0; otherwise, the RIR is 1.
    """

    # Insert debugging assertions
    assert type(CUE_rate) is np.ndarray, "The 'CUE_rate' must be numpy array."
    assert type(D2D_rate) is np.ndarray, "The 'D2D_rate' must be numpy array."
    assert type(CUE_power) is np.ndarray, "The 'CUE_power' must be numpy array."
    assert type(D2D_power) is np.ndarray, "The 'D2D_power' must be numpy array."
    assert type(QoS_of_CUE) is np.ndarray, "The 'QoS_of_CUE' must be numpy array."

    # Get the size of each dimension
    batch_size, num_of_D2Ds, num_of_CUEs, num_of_cells = (i for i in D2D_rate.shape)

    # Define inner function
    def inner(CUE_rate, D2D_rate, CUE_power, D2D_power, QoS_of_CUE):

        # Initialization of boolean numpy arrays
        CUE_feasible = np.ones((num_of_CUEs, 1, num_of_cells), dtype = bool)
        D2D_feasible = np.ones((num_of_D2Ds, 1, num_of_cells), dtype = bool)

        # CUE's power budget limitation
        CUE_feasible = np.logical_and(CUE_feasible, CUE_power <= constants.Pmax)
        CUE_feasible = np.logical_and(CUE_feasible, CUE_power >= 0)
        
        # CUE's minimum rate requirement
        CUE_feasible = np.logical_and(CUE_feasible, CUE_rate >= QoS_of_CUE - 1e-4)

        # D2D pair's power budget limitation
        D2D_feasible = np.logical_and(D2D_feasible, np.sum(D2D_power, axis = 1, keepdims = True) <= constants.Pmax)
        for index in range(num_of_CUEs):
            D2D_feasible = np.logical_and(D2D_feasible, D2D_power[:, [index], :] >= 0)
            
        # D2D pair's minimum rate requirement
        D2D_feasible = np.logical_and(D2D_feasible, np.sum(D2D_rate, axis = 1, keepdims = True) >= constants.QoS_of_D2D - 1e-4)

        # Calculate the number of infeasible CUEs and infeasible D2D pairs
        infeasible_CUE = np.count_nonzero(CUE_feasible == False)
        infeasible_D2D = np.count_nonzero(D2D_feasible == False)

        # Calculate system infeasibility rate (per realization), CUE infeasibility rate (per realization), and D2D infeasibility rate (per realization)
        system_RIR = 1 if (infeasible_CUE + infeasible_D2D) > 0 else 0
        CUE_RIR = 1 if infeasible_CUE > 0 else 0
        D2D_RIR = 1 if infeasible_D2D > 0 else 0

        # Return system infeasibility rate (per realization), CUE infeasibility rate (per realization), and D2D infeasibility rate (per realization)
        return system_RIR, CUE_RIR, D2D_RIR 

    # Initialization of numpy arrays
    system_RIR, CUE_RIR, D2D_RIR = [np.zeros(batch_size) for _ in range(3)]

    # Loop over the realizations in the batch
    for index, (CUE_rate, D2D_rate, CUE_power, D2D_power, QoS_of_CUE) in enumerate(zip(CUE_rate, D2D_rate, CUE_power, D2D_power, QoS_of_CUE)):
        system_RIR[index], CUE_RIR[index], D2D_RIR[index] = inner(CUE_rate, D2D_rate, CUE_power, D2D_power, QoS_of_CUE)

    # Return system infeasibility rate (per realization), CUE infeasibility rate (per realization), and D2D infeasibility rate (per realization) in batch
    return system_RIR, CUE_RIR, D2D_RIR

def get_avg_sum_rate(system_sum_rate, CUE_sum_rate, D2D_sum_rate):
    """ Return average system sum rate, average CUE sum rate, and average D2D sum rate.

    # Arguments:

    system_sum_rate: numpy array with shape (batch_size, )
        The sum rate of the multi-cell system.
    CUE_sum_rate: numpy array with shape (batch_size, )
        The sum rate of all CUEs.
    D2D_sum_rate: numpy array with shape (batch_size, )
        The sum rate of all D2D pairs.

    # Return:

    Tuple of floats: (avg_system_sum_rate, avg_CUE_sum_rate, avg_D2D_sum_rate)
        avg_system_sum_rate: Summation over all system sum rate in batch divided by the number of realizations.
        avg_CUE_sum_rate: Summation over all CUE sum rate in batch divided by the number of realizations.
        avg_D2D_sum_rate: Summation over all D2D sum rate in batch divided by the number of realizations.
    """

    # Insert debugging assertions
    assert type(system_sum_rate) is np.ndarray, "The 'system_sum_rate' must be numpy array."
    assert type(CUE_sum_rate) is np.ndarray, "The 'CUE_sum_rate' must be numpy array."
    assert type(D2D_sum_rate) is np.ndarray, "The 'D2D_sum_rate' must be numpy array."

    # Calculate average system sum rate, average CUE sum rate, and average D2D sum rate
    avg_system_sum_rate = np.mean(system_sum_rate)
    avg_CUE_sum_rate = np.mean(CUE_sum_rate)
    avg_D2D_sum_rate = np.mean(D2D_sum_rate)

    # Return average system sum rate, average CUE sum rate, and average D2D sum rate
    return avg_system_sum_rate, avg_CUE_sum_rate, avg_D2D_sum_rate

def get_avg_power_consumption(system_power_consumption, CUE_power_consumption, D2D_power_consumption):
    """ Return average system power consumption, average CUE power consumption, and average D2D power consumption.

    # Arguments:

    system_power_consumption: numpy array with shape (batch_size, )
        The power consumption of the multi-cell system.
    CUE_power_consumption: numpy array with shape (batch_size, )
        The power consumption of all CUEs.
    D2D_power_consumption: numpy array with shape (batch_size, )
        The power consumption of all D2D pairs.

    # Return:

    Tuple of floats: (avg_system_power_consumption, avg_CUE_power_consumption, avg_D2D_power_consumption)
        avg_system_power_consumption: Summation over all system power consumption in batch divided by the number of realizations.
        avg_CUE_power_consumption: Summation over all CUE power consumption in batch divided by the number of realizations.
        avg_D2D_power_consumption: Summation over all D2D power consumption in batch divided by the number of realizations.
    """

    # Insert debugging assertions
    assert type(system_power_consumption) is np.ndarray, "The 'system_sum_rate' must be numpy array."
    assert type(CUE_power_consumption) is np.ndarray, "The 'CUE_sum_rate' must be numpy array."
    assert type(D2D_power_consumption) is np.ndarray, "The 'D2D_sum_rate' must be numpy array."

    # Calculate average system power consumption, average CUE power consumption, and average D2D power consumption
    avg_system_power_consumption = np.mean(system_power_consumption)
    avg_CUE_power_consumption = np.mean(CUE_power_consumption)
    avg_D2D_power_consumption = np.mean(D2D_power_consumption)

    # Return average system power consumption, average CUE power consumption, and average D2D power consumption
    return avg_system_power_consumption, avg_CUE_power_consumption, avg_D2D_power_consumption

def get_avg_EE(system_EE, CUE_EE, D2D_EE):
    """ Return average system energy effciency, average CUE energy effciency, and average D2D energy effciency.

    # Arguments:

    
    system_EE: numpy array with shape (batch_size, )
        The energy efficiency of the multi-cell system.
    CUE_EE: numpy array with shape (batch_size, )
        The energy efficiency of all CUEs.
    D2D_EE: numpy array with shape (batch_size, )
        The energy efficiency of all D2D pairs.

    # Return:

    Tuple of floats: (avg_system_EE, avg_CUE_EE, avg_D2D_EE)
        avg_system_EE: Summation over all system energy effciency in batch divided by the number of the realizations.
        avg_CUE_EE: Summation over all CUE energy effciency in batch divided by the number of the realizations.
        avg_D2D_EE: Summation over all D2D energy effciency in batch divided by the number of the realizations.
    """

    # Insert debugging assertions
    assert type(system_EE) is np.ndarray, "The 'system_EE' must be numpy array."
    assert type(CUE_EE) is np.ndarray, "The 'CUE_EE' must be numpy array."
    assert type(D2D_EE) is np.ndarray, "The 'D2D_EE' must be numpy array."

    # Calculate average system energy efficiency, average CUE energy efficiency, and average D2D energy efficiency
    avg_system_EE = np.mean(system_EE)
    avg_CUE_EE = np.mean(CUE_EE)
    avg_D2D_EE = np.mean(D2D_EE)

    # Return average system energy efficiency, average CUE energy efficiency, and average D2D energy efficiency
    return avg_system_EE, avg_CUE_EE, avg_D2D_EE

def get_avg_UIR(system_UIR, CUE_UIR, D2D_UIR):
    """ Return average system infeasibility rate (per user), average CUE infeasibility rate (per user), and average D2D infeasibility rate (per user) in numpy arrays.

    # Arguments:

    system_UIR: numpy array with shape (batch_size, )
        The infeasibility rate (per user) of the multi-cell system.
    CUE_UIR: numpy array with shape (batch_size, )
        The infeasibility rate (per user) of all CUEs.
    D2D_UIR: numpy array with shape (batch_size, )
        The infeasibility rate (per user) of all D2D pairs.

    # Return:

    Tuple of floats: (avg_system_UIR, avg_CUE_UIR, avg_D2D_UIR)
        avg_system_UIR: Summation over all system infeasibility rate (per user) in batch divided by the number of the realizations.
        avg_CUE_UIR: Summation over all CUE infeasibility rate (per user) in batch divided by the number of the realizations.
        avg_D2D_UIR: Summation over all D2D infeasibility rate (per user) in batch divided by the number of the realizations.
    """

    # Insert debugging assertions
    assert type(system_UIR) is np.ndarray, "The 'system_UIR' must be numpy array."
    assert type(CUE_UIR) is np.ndarray, "The 'CUE_UIR' must be numpy array."
    assert type(D2D_UIR) is np.ndarray, "The 'D2D_UIR' must be numpy array."

    # Calculate average system infeasibility rate (per user), average CUE infeasibility rate (per user), and average D2D infeasibility rate (per user)
    avg_system_UIR = np.mean(system_UIR)
    avg_CUE_UIR = np.mean(CUE_UIR)
    avg_D2D_UIR = np.mean(D2D_UIR)

    # Return average system infeasibility rate (per user), average CUE infeasibility rate (per user), and average D2D infeasibility rate (per user)
    return avg_system_UIR, avg_CUE_UIR, avg_D2D_UIR

def get_avg_RIR(system_RIR, CUE_RIR, D2D_RIR):
    """ Return average system infeasibility rate (per realization), average CUE infeasibility rate (per realization), and average D2D infeasibility rate (per realization) in numpy arrays.

    # Arguments:

    system_RIR: numpy array with shape (batch_size, )
        The infeasibility rate (per realization) of the multi-cell system.
    CUE_RIR: numpy array with shape (batch_size, )
        The infeasibility rate (per realization) of all CUEs.
    D2D_RIR: numpy array with shape (batch_size, )
        The infeasibility rate (per realization) of all D2D pairs.

    # Return:

    Tuple of floats: (avg_system_RIR, avg_CUE_RIR, avg_D2D_RIR)
        avg_system_RIR: Summation over all system infeasibility rate (per realization) in batch divided by the number of the realizations.
        avg_CUE_RIR: Summation over all CUE infeasibility rate (per realization) in batch divided by the number of the realizations.
        avg_D2D_RIR: Summation over all D2D infeasibility rate (per realization) in batch divided by the number of the realizations.
    """

    # Insert debugging assertions
    assert type(system_RIR) is np.ndarray, "The 'system_RIR' must be numpy array."
    assert type(CUE_RIR) is np.ndarray, "The 'CUE_RIR' must be numpy array."
    assert type(D2D_RIR) is np.ndarray, "The 'D2D_RIR' must be numpy array."

    # Calculate average system infeasibility rate (per realization), average CUE infeasibility rate (per realization), and average D2D infeasibility rate (per realization)
    avg_system_RIR = np.mean(system_RIR)
    avg_CUE_RIR = np.mean(CUE_RIR)
    avg_D2D_RIR = np.mean(D2D_RIR)

    # Return average system infeasibility rate (per realization), average CUE infeasibility rate (per realization), and average D2D infeasibility rate (per realization)
    return avg_system_RIR, avg_CUE_RIR, avg_D2D_RIR