import constant
import numpy as np

def GetChannelGainMatrix(input_data, num_of_cells, num_of_CUEs, num_of_D2Ds):
    """ Reshape input numpy array into channel gain matrix and return it.

    # Aruguments:

    input_data: numpy array 
        The numpy array which is passed into the input layer of the model.
        The computation of prediction is done in batches, so the firxt axis stands for batch size.
    num_of_cells: int
        Number of the cells in the cellular system.
    num_of_CUEs: int
        Number of the CUEs in each cell.
    num_of_D2Ds: int
        Number of the D2D pairs in each cell.

    # Return:

    channel_gain_matrix: numpy array with shape (batch_size, rows, cols, channels)
        A matrix which stands for channel gains of all links in the entire network.
        rows: num_of_cells * (num_of_CUEs + num_of_D2Ds).
        cols: 1 + num_of_D2Ds.
        channels: num_of_cells.
    """

    # Insert debugging assertions
    assert type(input_data) is np.ndarray, "The 'input_data' must be numpy array."
    assert num_of_cells in constant.cell_range, f"The 'num_of_cells' must be element in {constant.cell_range}."
    assert num_of_CUEs in constant.CUE_range, f"The 'num_of_CUEs' must be element in {constant.CUE_range}."
    assert num_of_D2Ds in constant.D2D_range, f"The 'num_of_D2Ds' must be element in {constant.D2D_range}."

    # Get the size of each dimension
    batch_size = len(input_data)
    rows = num_of_cells * (num_of_CUEs + num_of_D2Ds)
    cols = 1 + num_of_D2Ds
    channels = num_of_cells

    # Reshape input numpy array into channel gain matrix 
    channel_gain_matrix = np.reshape(input_data, (batch_size, rows, cols, channels))

    # Return channel gain matrix
    return channel_gain_matrix

def GetPowerAllocation(output_data, num_of_cells, num_of_CUEs, num_of_D2Ds):
    """ Split output numpy array into two numpy array (CUE_power and D2D_power) and return it.

    # Aruguments:

    output_data: numpy array
        Target data or prediction which is obtained from the output layer of the model.
        The computation of prediction is done in batches, so the firxt axis stands for batch size.
    num_of_cells: int
        Number of the cells in the cellular system.
    num_of_CUEs: int
        Number of the CUEs in each cell.
    num_of_D2Ds: int
        Number of the D2D pairs in each cell.

    # Return:

    Tuple of numpy arrays: (CUE_power, D2D_power)
        CUE_power: The numpy array with shape (batch_size, num_of_CUEs, 1, num_of_cells), which stands for power allocation of all CUEs.
        D2D_power: The numpy array with shape (batch_size, num_of_D2Ds, num_of_CUEs, num_of_cells), which stands for power allocation of all D2D pairs.
    """

    # Insert debugging assertions
    assert type(output_data) is np.ndarray, "The 'output_data' must be numpy array."
    assert num_of_cells in constant.cell_range, f"The 'num_of_cells' must be element in {constant.cell_range}."
    assert num_of_CUEs in constant.CUE_range, f"The 'num_of_CUEs' must be element in {constant.CUE_range}."
    assert num_of_D2Ds in constant.D2D_range, f"The 'num_of_D2Ds' must be element in {constant.D2D_range}."

    # Get the size of batch dimension
    batch_size = len(output_data)

    # Calculate the splitting index
    split_index = num_of_CUEs * num_of_cells

    # Split the output data into two numpy array (CUE_power and D2D_power)
    CUE_power = output_data[:, :split_index]
    D2D_power = output_data[:, split_index:]

    # Gives a new shape (3D) to these two numpy array without changing its data
    CUE_power = np.reshape(CUE_power, (batch_size, num_of_CUEs, 1, num_of_cells))
    D2D_power = np.reshape(D2D_power, (batch_size, num_of_D2Ds, num_of_CUEs, num_of_cells))

    # Return power allocation of CUEs and D2D pairs
    return CUE_power, D2D_power

def GetDataRate(channel_gain_matrix, CUE_power, D2D_power):
    """ Return data rate of all CUEs and all D2D pairs in numpy arrays.

    # Arguments:

    channel_gain_matrix: numpy array with shape (batch_size, rows, cols, channels)
        A matrix which stands for channel gains of all links in the entire network.
        rows: num_of_cells * (num_of_CUEs + num_of_D2Ds).
        cols: 1 + num_of_D2Ds.
        channels: num_of_cells.
    CUE_power: numpy array with shape (batch_size, num_of_CUEs, 1, num_of_cells)
        The transmit power of all CUEs.
    D2D_power: numpy array with shape (batch_size, num_of_D2Ds, num_of_CUEs, num_of_cells)
        The transmit power of all D2D pairs.

    # Return:
    
    Tuple of numpy arrays: (CUE_rate, D2D_rate)
        CUE_rate: numpy array with shape (batch_size, num_of_CUEs, 1, num_of_cells), which stands for data rate of all CUEs.
        D2D_rate: numpy array with shape (batch_size, num_of_D2Ds, num_of_CUEs, num_of_cells), which stands for data rate of all D2D pairs.
    """

    # Insert debugging assertions
    assert type(channel_gain_matrix) is np.ndarray, "The 'channel_gain_matrix' must be numpy array."
    assert type(CUE_power) is np.ndarray, "The 'CUE_power' must be numpy array."
    assert type(D2D_power) is np.ndarray, "The 'D2D_power' must be numpy array."

    # Get the size of each dimension
    batch_size, num_of_D2Ds, num_of_CUEs, num_of_cells = (i for i in D2D_power.shape)

    # Define inner function
    def inner(channel_gain_matrix, CUE_power, D2D_power):

        # Initialization of numpy arrays
        CUE_rate = np.zeros((num_of_CUEs, 1, num_of_cells))
        D2D_rate = np.zeros((num_of_D2Ds, num_of_CUEs, num_of_cells))

        # Loop over all cells
        for k in range(num_of_cells):

            # Loop over all CUEs
            for i in range(num_of_CUEs):

                # Calculate the power of desired signal
                desired_signal = CUE_power[i, 0, k] * channel_gain_matrix[i, 0, k]

                # Calculate the power of intra-cell interference from D2D pairs
                intra_cell_interference = np.sum(D2D_power[:, i, k] * channel_gain_matrix[num_of_CUEs : num_of_CUEs + num_of_D2Ds, 0, k])

                # Calculate the power of inter-cell interference
                inter_cell_interference_from_CUE = 0
                inter_cell_interference_from_D2D = 0
                for j in range(num_of_cells):
                    if j < k:
                        # CUE part
                        interference_from_CUE = CUE_power[i, 0, j] * channel_gain_matrix[num_of_CUEs + num_of_D2Ds + j * num_of_CUEs + i, 0, k] 
                        # D2D part
                        interference_from_D2D = np.sum(D2D_power[:, i, j] * channel_gain_matrix[num_of_cells * num_of_CUEs + num_of_D2Ds + j * num_of_D2Ds : num_of_cells * num_of_CUEs + num_of_D2Ds + j * num_of_D2Ds + num_of_D2Ds, 0, k]) 
                    elif j > k:
                        # CUE part
                        interference_from_CUE = CUE_power[i, 0, j] * channel_gain_matrix[num_of_CUEs + num_of_D2Ds + (j - 1) * num_of_CUEs + i, 0, k] 
                        # D2D part
                        interference_from_D2D = np.sum(D2D_power[:, i, j] * channel_gain_matrix[num_of_cells * num_of_CUEs + num_of_D2Ds + (j - 1) * num_of_D2Ds : num_of_cells * num_of_CUEs + num_of_D2Ds + (j - 1) * num_of_D2Ds + num_of_D2Ds, 0, k])
                    else:
                        continue

                    inter_cell_interference_from_CUE = inter_cell_interference_from_CUE + interference_from_CUE
                    inter_cell_interference_from_D2D = inter_cell_interference_from_D2D + interference_from_D2D

                # Calculate the data rate of CUE
                SINR = desired_signal / (constant.noise + intra_cell_interference + inter_cell_interference_from_CUE + inter_cell_interference_from_D2D)
                rate_of_CUE = np.log2(1 + SINR)
                CUE_rate[i, 0, k] = rate_of_CUE 

            # Loop over all D2D pairs
            for i in range(num_of_D2Ds):
                
                # Summation over all resource blocks
                for j in range(num_of_CUEs):
                        
                    # Calculate the power of desired signal
                    desired_signal = D2D_power[i, j, k] * channel_gain_matrix[num_of_CUEs + i, 1 + i, k]

                    # Calculate the power of intra-cell interference from CUE and other D2D pairs
                    interference_from_CUE = CUE_power[j, 0, k] * channel_gain_matrix[j, 1 + i, k]
                    interference_from_D2D = np.sum(D2D_power[:, j, k] * channel_gain_matrix[num_of_CUEs : num_of_CUEs + num_of_D2Ds, 1 + i, k]) - desired_signal
                    intra_cell_interference = interference_from_CUE + interference_from_D2D

                    # Calculate the power of inter-cell interference
                    inter_cell_interference_from_CUE = 0
                    inter_cell_interference_from_D2D = 0
                    for l in range(num_of_cells):
                        if l < k:
                            # CUE part
                            interference_from_CUE = CUE_power[j, 0, l] * channel_gain_matrix[num_of_CUEs + num_of_D2Ds + l * num_of_CUEs + j, 1 + i, k]
                            # D2D part
                            interference_from_D2D = np.sum(D2D_power[:, j, l] * channel_gain_matrix[num_of_cells * num_of_CUEs + num_of_D2Ds + l * num_of_D2Ds : num_of_cells * num_of_CUEs + num_of_D2Ds + l * num_of_D2Ds + num_of_D2Ds, 1 + i, k])
                        elif l > k:
                            # CUE part
                            interference_from_CUE = CUE_power[j, 0, l] * channel_gain_matrix[num_of_CUEs + num_of_D2Ds + (l - 1) * num_of_CUEs + j, 1 + i, k]
                            # D2D part
                            interference_from_D2D = np.sum(D2D_power[:, j, l] * channel_gain_matrix[num_of_cells * num_of_CUEs + num_of_D2Ds + (l - 1) * num_of_D2Ds : num_of_cells * num_of_CUEs + num_of_D2Ds + (l - 1) * num_of_D2Ds + num_of_D2Ds, 1 + i, k])
                        else:
                            continue

                        inter_cell_interference_from_CUE = inter_cell_interference_from_CUE + interference_from_CUE
                        inter_cell_interference_from_D2D = inter_cell_interference_from_D2D + interference_from_D2D

                    # Calculate the data rate of D2D pair on each resource block
                    SINR = desired_signal / (constant.noise + intra_cell_interference + inter_cell_interference_from_CUE + inter_cell_interference_from_D2D)
                    rate_of_D2D = np.log2(1 + SINR)
                    D2D_rate[i, j, k] = rate_of_D2D 

        # Return data rate of all CUEs and all D2D pairs
        return CUE_rate, D2D_rate

    # Initialization of numpy arrays
    CUE_rate = np.zeros((batch_size, num_of_CUEs, 1, num_of_cells))
    D2D_rate = np.zeros((batch_size, num_of_D2Ds, num_of_CUEs, num_of_cells))

    # Loop over the realizations in the batch
    for index, (channel_gain_matrix, CUE_power, D2D_power) in enumerate(zip(channel_gain_matrix, CUE_power, D2D_power)):
        CUE_rate[index], D2D_rate[index] = inner(channel_gain_matrix, CUE_power, D2D_power)

    # Return data rate of all CUEs and all D2D pairs in batch
    return CUE_rate, D2D_rate

def GetQoSofCUE(channel_gain_matrix, num_of_cells, num_of_CUEs):
    """ Return QoS (minimum rate requirement) of all CUEs.

    # Arguments:

    channel_gain_matrix: numpy array with shape (batch_size, rows, cols, channels)
        A matrix which stands for channel gains of all links in the entire network.
        rows: num_of_cells * (num_of_CUEs + num_of_D2Ds).
        cols: 1 + num_of_D2Ds.
        channels: num_of_cells.
    num_of_cells: int
        Number of the cells in the cellular system.
    num_of_CUEs: int
        Number of the CUEs in each cell.

    # Return:

    QoS_of_CUE: numpy array with shape (batch_size, num_of_CUEs, 1, num_of_cells)
        The minimum rate requirement of all CUEs (bps/Hz).
    """

    # Insert debugging assertions
    assert type(channel_gain_matrix) is np.ndarray, "The 'channel_gain_matrix' must be numpy array."
    assert num_of_cells in constant.cell_range, f"The 'num_of_cells' must be element in {constant.cell_range}."
    assert num_of_CUEs in constant.CUE_range, f"The 'num_of_CUEs' must be element in {constant.CUE_range}."

    # Get the size of batch dimension
    batch_size = len(channel_gain_matrix)

    # Define inner function
    def inner(channel_gain_matrix):
        
        # Initialization of numpy array
        QoS_of_CUE = np.zeros((num_of_CUEs, 1, num_of_cells))

        # Loop over all cells
        for k in range(num_of_cells):

            # Loop over all CUEs
            for i in range(num_of_CUEs):

                # Calculate the power of desired signal
                desired_signal = constant.Pmax * channel_gain_matrix[i, 0, k]

                # Calculate CUE's maximum data rate
                SINR = desired_signal / constant.noise
                QoS_of_CUE[i, 0, k] = np.log2(1 + SINR) * constant.rate_proportion 

        # Return QoS (minimum rate requirement) of all CUEs
        return QoS_of_CUE

    # Initialization of numpy array
    QoS_of_CUE = np.zeros((batch_size, num_of_CUEs, 1, num_of_cells))

    # Loop over the realizations in the batch
    for index, channel_gain_matrix in enumerate(channel_gain_matrix):
        QoS_of_CUE[index] = inner(channel_gain_matrix)

    # Return QoS (minimum rate requirement) of all CUEs in batch
    return QoS_of_CUE

def GetSumRate(CUE_rate, D2D_rate):
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

def GetPowerConsumption(CUE_power, D2D_power):
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
        CUE_power_consumption = np.sum(CUE_power) / constant.PA_inefficiency_factor + constant.circuit_power * num_of_cells * num_of_CUEs
        D2D_power_consumption = np.sum(D2D_power) / constant.PA_inefficiency_factor + constant.circuit_power * num_of_cells * num_of_D2Ds * 2
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

def GetEnergyEfficiency(system_sum_rate, CUE_sum_rate, D2D_sum_rate, system_power_consumption, CUE_power_consumption, D2D_power_consumption):
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

def GetUIR(CUE_rate, D2D_rate, CUE_power, D2D_power, QoS_of_CUE):
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
        CUE_feasible = np.logical_and(CUE_feasible, CUE_power <= constant.Pmax)
        CUE_feasible = np.logical_and(CUE_feasible, CUE_power >= 0)
        
        # CUE's minimum rate requirement
        CUE_feasible = np.logical_and(CUE_feasible, CUE_rate >= QoS_of_CUE)

        # D2D pair's power budget limitation
        D2D_feasible = np.logical_and(D2D_feasible, np.sum(D2D_power, axis = 1, keepdims = True) <= constant.Pmax)
        for index in range(num_of_CUEs):
            D2D_feasible = np.logical_and(D2D_feasible, D2D_power[:, [index], :] >= 0)
            
        # D2D pair's minimum rate requirement
        D2D_feasible = np.logical_and(D2D_feasible, np.sum(D2D_rate, axis = 1, keepdims = True) >= constant.QoS_of_D2D)

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

    # Return system infeasibility rate (per user), CUE infeasibility rate (per user), and D2D infeasibility rate (per user)
    return system_UIR, CUE_UIR, D2D_UIR

def GetRIR(CUE_rate, D2D_rate, CUE_power, D2D_power, QoS_of_CUE):
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
        CUE_feasible = np.logical_and(CUE_feasible, CUE_power <= constant.Pmax)
        CUE_feasible = np.logical_and(CUE_feasible, CUE_power >= 0)
        
        # CUE's minimum rate requirement
        CUE_feasible = np.logical_and(CUE_feasible, CUE_rate >= QoS_of_CUE)

        # D2D pair's power budget limitation
        D2D_feasible = np.logical_and(D2D_feasible, np.sum(D2D_power, axis = 1, keepdims = True) <= constant.Pmax)
        for index in range(num_of_CUEs):
            D2D_feasible = np.logical_and(D2D_feasible, D2D_power[:, [index], :] >= 0)
            
        # D2D pair's minimum rate requirement
        D2D_feasible = np.logical_and(D2D_feasible, np.sum(D2D_rate, axis = 1, keepdims = True) >= constant.QoS_of_D2D)

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

    # Return system infeasibility rate (per realization), CUE infeasibility rate (per realization), and D2D infeasibility rate (per realization)
    return system_RIR, CUE_RIR, D2D_RIR

def GetAvgSumRate(system_sum_rate, CUE_sum_rate, D2D_sum_rate):
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

def GetAvgPowerConsumption(system_power_consumption, CUE_power_consumption, D2D_power_consumption):
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

def GetAvgEnergyEfficiency(system_EE, CUE_EE, D2D_EE):
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

def GetAvgUIR(system_UIR, CUE_UIR, D2D_UIR):
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

def PrintDataRate(CUE_rate, D2D_rate, QoS_of_CUE, header, realization_index):
    """ Given the specific realization, print the data rate of all CUEs and D2D pairs.

    # Arguments:

    CUE_rate: numpy array with shape (batch_size, num_of_CUEs, 1, num_of_cells)
        The data rate of all CUEs.
    D2D_rate: numpy array with shape (batch_size, num_of_D2Ds, num_of_CUEs, num_of_cells)
        The data rate of all D2D pair.
    QoS_of_CUE: numpy array with shape (batch_size, num_of_CUEs, 1, num_of_cells)
        The minimum rate requirement of all CUEs (bps/Hz).
    header: string
        Determine what kind of string should be printed.
    realization_index: int
        Determine which realization should be considered.

    # Return:

    None
    """
    
    # Insert debugging assertions
    assert type(CUE_rate) is np.ndarray, "The 'CUE_rate' must be numpy array."
    assert type(D2D_rate) is np.ndarray, "The 'D2D_rate' must be numpy array."
    assert type(QoS_of_CUE) is np.ndarray, "The 'QoS_of_CUE' must be numpy array."
    assert type(header) is str, "The 'header' must be string."
    assert type(realization_index) is int, "The 'realization_index' must be integer."

    # Get the size of each dimension
    _, num_of_D2Ds, num_of_CUEs, num_of_cells = (i for i in D2D_rate.shape)

    # Numpy array indexing
    CUE_rate, D2D_rate, QoS_of_CUE = CUE_rate[realization_index], D2D_rate[realization_index], QoS_of_CUE[realization_index]

    # Loop over all cells
    for k in range(num_of_cells):

        # Print header information
        print(f"\nCell {k + 1}: {header} data rate (CUE)\n")
        print(" " * 8 + "RB".ljust(12, " ") + "Requirement")

        # Print data rate of all CUEs
        for i in range(num_of_CUEs):
            print(f"CUE {i + 1}".ljust(8, " ") + f"{CUE_rate[i, 0, k]:.6f}".ljust(12, " ") + f"{QoS_of_CUE[i, 0, k]:.6f}")

    # Loop over all cells
    for k in range(num_of_cells):

        # Print header information
        print(f"\nCell {k + 1}: {header} data rate (D2D)\n")

        print(" " * 8, end = "")
        for j in range(num_of_CUEs):
            print(f"RB {j + 1}".ljust(12, " "), end = "")
        print("Total".ljust(12, " ") + "Requirement")

        # Print data rate of all D2D pairs
        for i in range(num_of_D2Ds):
            print(f"D2D {i + 1}".ljust(8, " "), end = "")

            for j in range(num_of_CUEs):
                print(f"{D2D_rate[i, j, k]:.6f}".ljust(12, " "), end = "")

            print(f"{np.sum(D2D_rate, axis = 1)[i, k]:.6f}".ljust(12, " ") + f"{constant.QoS_of_D2D:.6f}")

def PrintTransmitPower(CUE_power, D2D_power, header, realization_index):
    """ Given the specific realization, print the transmit power of all CUEs and D2D pairs.

    # Arguments:

    CUE_power: numpy array with shape (batch_size, num_of_CUEs, 1, num_of_cells)
        The transmit power of all CUEs.
    D2D_power: numpy array with shape (batch_size, num_of_D2Ds, num_of_CUEs, num_of_cells)
        The transmit power of all D2D pairs.
    header: string
        Determine what kind of string should be printed.
    realization_index: int
        Determine which realization should be considered.

    # Return:

    None
    """

    # Insert debugging assertions
    assert type(CUE_power) is np.ndarray, "The 'CUE_power' must be numpy array."
    assert type(D2D_power) is np.ndarray, "The 'D2D_power' must be numpy array."
    assert type(header) is str, "The 'header' must be string."
    assert type(realization_index) is int, "The 'realization_index' must be integer."

    # Get the size of each dimension
    _, num_of_D2Ds, num_of_CUEs, num_of_cells = (i for i in D2D_power.shape)

    # Numpy array indexing
    CUE_power, D2D_power = CUE_power[realization_index], D2D_power[realization_index]

    # Loop over all cells
    for k in range(num_of_cells):

        # Print header information
        print(f"\nCell {k + 1}: {header} transmit power (CUE)\n")
        print(" " * 8 + "RB".ljust(12, " ") + "Limitation")

        # Print transmit power of all CUEs
        for i in range(num_of_CUEs):
            print(f"CUE {i + 1}".ljust(8, " ") + f"{CUE_power[i, 0, k]:.6f}".ljust(12, " ") + f"{constant.Pmax:.6f}")

    # Loop over all cells
    for k in range(num_of_cells):

        # Print header information
        print(f"\nCell {k + 1}: {header} transmit power(D2D)\n")

        print(" " * 8, end = "")
        for j in range(num_of_CUEs):
            print(f"RB {j + 1}".ljust(12, " "), end = "")
        print("Total".ljust(12, " ") + "Limitation")

        # Print transmit power of all D2D pairs
        for i in range(num_of_D2Ds):
            print(f"D2D {i + 1}".ljust(8, " "), end = "")

            for j in range(num_of_CUEs):
                print(f"{D2D_power[i, j, k]:.6f}".ljust(12, " "), end = "")

            print(f"{np.sum(D2D_power, axis = 1)[i, k]:.6f}".ljust(12, " ") + f"{constant.Pmax:.6f}")

def FeasibilityCheck(CUE_rate, D2D_rate, CUE_power, D2D_power, QoS_of_CUE, realization_index):
    """ Given the specific realization, check the feasibility of power allocation strategy.

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
    realization_index: int
        Determine which realization should be considered.

    # Return: 
     
    None
    """

    # Insert debugging assertions
    assert type(CUE_rate) is np.ndarray, "The 'CUE_rate' must be numpy array."
    assert type(D2D_rate) is np.ndarray, "The 'D2D_rate' must be numpy array."
    assert type(CUE_power) is np.ndarray, "The 'CUE_power' must be numpy array."
    assert type(D2D_power) is np.ndarray, "The 'D2D_power' must be numpy array."
    assert type(QoS_of_CUE) is np.ndarray, "The 'QoS_of_CUE' must be numpy array."
    assert type(realization_index) is int, "The 'realization_index' must be integer."

    # Numpy array indexing
    CUE_rate, D2D_rate, QoS_of_CUE = CUE_rate[realization_index], D2D_rate[realization_index], QoS_of_CUE[realization_index]
    CUE_power, D2D_power = CUE_power[realization_index], D2D_power[realization_index]

    # CUE's power budget limitation
    if np.all(CUE_power <= constant.Pmax) and np.all(CUE_power >= 0):
        print("\nCUE's power budget limitation: Satisfied\n")
    else:
        print("\nCUE's power budget limitation: Violated\n")

    # CUE's minimum rate requirement
    if np.all(CUE_rate >= QoS_of_CUE):
        print("CUE's minimum rate requirement: Satisfied\n")
    else:
        print("CUE's minimum rate requirement: Violated\n")

    # D2D pair's power budget limitation
    if np.all(np.sum(D2D_power, axis = 1) <= constant.Pmax) and np.all(D2D_power >= 0):
        print("D2D pair's power budget limitation: Satisfied\n")
    else:
        print("D2D pair's power budget limitation: Violated\n")

    # D2D pair's minimum rate requirement
    if np.all(np.sum(D2D_rate, axis = 1) >= constant.QoS_of_D2D):
        print("D2D pair's minimum rate requirement: Satisfied\n")
    else:
        print("D2D pair's minimum rate requirement: Violated\n")