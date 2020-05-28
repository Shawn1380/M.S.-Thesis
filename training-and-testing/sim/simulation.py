""" This module implements several functions to handle simulation of device-to-device communication. """

from sim import constants
import numpy as np

def get_channel_gain_matrix(input_data, num_of_cells, num_of_CUEs, num_of_D2Ds):
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
    assert num_of_cells in constants.cell_range, f"The 'num_of_cells' must be element in {constants.cell_range}."
    assert num_of_CUEs in constants.CUE_range, f"The 'num_of_CUEs' must be element in {constants.CUE_range}."
    assert num_of_D2Ds in constants.D2D_range, f"The 'num_of_D2Ds' must be element in {constants.D2D_range}."

    # Get the size of each dimension
    batch_size = len(input_data)
    rows = num_of_cells * (num_of_CUEs + num_of_D2Ds)
    cols = 1 + num_of_D2Ds
    channels = num_of_cells

    # Reshape input numpy array into channel gain matrix 
    channel_gain_matrix = np.reshape(input_data, (batch_size, rows, cols, channels))

    # Return channel gain matrix
    return channel_gain_matrix

def get_power_allocation(output_data, num_of_cells, num_of_CUEs, num_of_D2Ds):
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
    assert num_of_cells in constants.cell_range, f"The 'num_of_cells' must be element in {constants.cell_range}."
    assert num_of_CUEs in constants.CUE_range, f"The 'num_of_CUEs' must be element in {constants.CUE_range}."
    assert num_of_D2Ds in constants.D2D_range, f"The 'num_of_D2Ds' must be element in {constants.D2D_range}."

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

def get_data_rate(channel_gain_matrix, CUE_power, D2D_power):
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
                SINR = desired_signal / (constants.noise + intra_cell_interference + inter_cell_interference_from_CUE + inter_cell_interference_from_D2D)
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
                    SINR = desired_signal / (constants.noise + intra_cell_interference + inter_cell_interference_from_CUE + inter_cell_interference_from_D2D)
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

def get_QoS_of_CUE(channel_gain_matrix, num_of_cells, num_of_CUEs, rate_proportion = constants.rate_proportion):
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
        rate_proportion: float, optional
            The ratio of CUE's minimum rate requirement to CUE's maximum data rate.

    # Return:

        QoS_of_CUE: numpy array with shape (batch_size, num_of_CUEs, 1, num_of_cells)
            The minimum rate requirement of all CUEs (bps/Hz).
    """

    # Insert debugging assertions
    assert type(channel_gain_matrix) is np.ndarray, "The 'channel_gain_matrix' must be numpy array."
    assert num_of_cells in constants.cell_range, f"The 'num_of_cells' must be element in {constants.cell_range}."
    assert num_of_CUEs in constants.CUE_range, f"The 'num_of_CUEs' must be element in {constants.CUE_range}."

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
                desired_signal = constants.Pmax * channel_gain_matrix[i, 0, k]

                # Calculate CUE's maximum data rate
                SINR = desired_signal / constants.noise
                QoS_of_CUE[i, 0, k] = np.log2(1 + SINR) * rate_proportion 

        # Return QoS (minimum rate requirement) of all CUEs
        return QoS_of_CUE

    # Initialization of numpy array
    QoS_of_CUE = np.zeros((batch_size, num_of_CUEs, 1, num_of_cells))

    # Loop over the realizations in the batch
    for index, channel_gain_matrix in enumerate(channel_gain_matrix):
        QoS_of_CUE[index] = inner(channel_gain_matrix)

    # Return QoS (minimum rate requirement) of all CUEs in batch
    return QoS_of_CUE

def print_data_rate(CUE_rate, D2D_rate, QoS_of_CUE, header, realization_index):
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

            print(f"{np.sum(D2D_rate, axis = 1)[i, k]:.6f}".ljust(12, " ") + f"{constants.QoS_of_D2D:.6f}")

def print_power_consumption(CUE_power, D2D_power, header, realization_index):
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
            print(f"CUE {i + 1}".ljust(8, " ") + f"{CUE_power[i, 0, k]:.6f}".ljust(12, " ") + f"{constants.Pmax:.6f}")

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

            print(f"{np.sum(D2D_power, axis = 1)[i, k]:.6f}".ljust(12, " ") + f"{constants.Pmax:.6f}")

def feasibility_check(CUE_rate, D2D_rate, CUE_power, D2D_power, QoS_of_CUE, realization_index):
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
    if np.all(CUE_power <= constants.Pmax) and np.all(CUE_power >= 0):
        print("\nCUE's power budget limitation: Satisfied\n")
    else:
        print("\nCUE's power budget limitation: Violated\n")

    # CUE's minimum rate requirement
    if np.all(CUE_rate >= QoS_of_CUE):
        print("CUE's minimum rate requirement: Satisfied\n")
    else:
        print("CUE's minimum rate requirement: Violated\n")

    # D2D pair's power budget limitation
    if np.all(np.sum(D2D_power, axis = 1) <= constants.Pmax) and np.all(D2D_power >= 0):
        print("D2D pair's power budget limitation: Satisfied\n")
    else:
        print("D2D pair's power budget limitation: Violated\n")

    # D2D pair's minimum rate requirement
    if np.all(np.sum(D2D_rate, axis = 1) >= constants.QoS_of_D2D):
        print("D2D pair's minimum rate requirement: Satisfied\n")
    else:
        print("D2D pair's minimum rate requirement: Violated\n")