""" This module implements several utility functions to visualize the performance. """

from itertools import product
from sim import constants
import matplotlib.pyplot as plt
import numpy as np

def plot_sum_rate(mode, num_of_CUEs, num_of_D2Ds, CA_list, CNN_SPP_list, CNN_list, FCN_list):
    """ Create a grouped bar chart to show the performance of sum rate.

    # Aruguments:

        mode: string 
            A string, either 'system', 'CUE', or 'D2D'.
            It specifics which mode should be used for making grouped bar chart.
        num_of_CUEs: set
            Number of the CUEs in each cell.
        num_of_D2Ds: set
            Number of the D2D pairs in each cell.
        CA_list: list
            The heights of the bars, which stands for sum rate obtained by CVX solver.
        CNN_SPP_list: list
            The heights of the bars, which stands for sum rate obtained by CNN with spatial pyramid pooling layer.
        CNN_list: list
            The heights of the bars, which stands for sum rate obtained by CNN.
        FCN_list: list
            The heights of the bars, which stands for sum rate obtained by FCN.

    # Return:

        None
    """

    # Insert debugging assertions
    assert type(mode) is str, f"The 'mode' must be string."
    assert num_of_CUEs.issubset(constants.CUE_range), f"The 'num_of_CUEs' must be subset of {constants.CUE_range}."
    assert num_of_D2Ds.issubset(constants.D2D_range), f"The 'num_of_D2Ds' must be subset of {constants.D2D_range}."
    assert type(CA_list) is list, f"The 'CA_list' must be list."
    assert type(CNN_SPP_list) is list, f"The 'CNN_SPP_list' must be list."
    assert type(CNN_list) is list, f"The 'CNN_list' must be list."
    assert type(FCN_list) is list, f"The 'FCN_list' must be list."

    # Add custom x-axis tick labels
    x_labels = [f'({i}, {j})' for (i, j) in product(num_of_CUEs, num_of_D2Ds)]

    # Set the label locations and width of the bar
    x_index = np.arange(len(x_labels))
    width = 0.15

    # Set the size of figure
    plt.figure(figsize = (8.53, 4.8))

    # Each bar will be shifted 'width' units from the previous one
    plt.bar(x_index, CA_list, width, color = '#1f77b4', label = 'CA')
    plt.bar(x_index + 1 * width, CNN_SPP_list, width, color = '#ff7f0e', label = 'CNN-SPP')
    plt.bar(x_index + 2 * width, CNN_list, width, color = '#2ca02c', label = 'CNN')
    plt.bar(x_index + 3 * width, FCN_list, width, color = '#d62728', label = 'FCN')

    # Set the figure title and label for the y-axis
    if mode == 'system':
        plt.title('System sum rate')
        plt.ylabel('Sum rate (bps/Hz)')
    elif mode == 'CUE':
        plt.title('Sum rate of CUEs')
        plt.ylabel('Sum rate (bps/Hz)')
    elif mode == 'D2D':
        plt.title('Sum rate of D2D pairs')
        plt.ylabel('Sum rate (bps/Hz)')
    else:
        raise ValueError("The 'mode' must be 'system', 'CUE', or 'D2D'.")

    # Set the label for the x-axis
    plt.xlabel('(Number of CUEs, Number of D2D pairs)')
    
    # Show the major grid lines with dark grey lines
    plt.grid(axis = 'y', color = '#666666', linestyle = '-')

    # Set the tick locations and labels of the x-axis
    plt.xticks(x_index + 1.5 * width, x_labels)
    plt.legend(loc = 'best')
    plt.show()

def plot_power_consumption(mode, num_of_CUEs, num_of_D2Ds, CA_list, CNN_SPP_list, CNN_list, FCN_list):
    """ Create a grouped bar chart to show the performance of power consumption.

    # Aruguments:

        mode: string 
            A string, either 'system', 'CUE', or 'D2D'.
            It specifics which mode should be used for making grouped bar chart.
        num_of_CUEs: set
            Number of the CUEs in each cell.
        num_of_D2Ds: set
            Number of the D2D pairs in each cell.
        CA_list: list
            The heights of the bars, which stands for power consumption obtained by CVX solver.
        CNN_SPP_list: list
            The heights of the bars, which stands for power consumption obtained by CNN with spatial pyramid pooling layer.
        CNN_list: list
            The heights of the bars, which stands for power consumption obtained by CNN.
        FCN_list: list
            The heights of the bars, which stands for power consumption obtained by FCN.

    # Return:

        None
    """

    # Insert debugging assertions
    assert type(mode) is str, f"The 'mode' must be string."
    assert num_of_CUEs.issubset(constants.CUE_range), f"The 'num_of_CUEs' must be subset of {constants.CUE_range}."
    assert num_of_D2Ds.issubset(constants.D2D_range), f"The 'num_of_D2Ds' must be subset of {constants.D2D_range}."
    assert type(CA_list) is list, f"The 'CA_list' must be list."
    assert type(CNN_SPP_list) is list, f"The 'CNN_SPP_list' must be list."
    assert type(CNN_list) is list, f"The 'CNN_list' must be list."
    assert type(FCN_list) is list, f"The 'FCN_list' must be list."

    # Add custom x-axis tick labels
    x_labels = [f'({i}, {j})' for (i, j) in product(num_of_CUEs, num_of_D2Ds)]

    # Set the label locations and width of the bar
    x_index = np.arange(len(x_labels))
    width = 0.15

    # Set the size of figure
    plt.figure(figsize = (8.53, 4.8))

    # Each bar will be shifted 'width' units from the previous one
    plt.bar(x_index, CA_list, width, color = '#1f77b4', label = 'CA')
    plt.bar(x_index + 1 * width, CNN_SPP_list, width, color = '#ff7f0e', label = 'CNN-SPP')
    plt.bar(x_index + 2 * width, CNN_list, width, color = '#2ca02c', label = 'CNN')
    plt.bar(x_index + 3 * width, FCN_list, width, color = '#d62728', label = 'FCN')

    # Set the figure title and label for the y-axis
    if mode == 'system':
        plt.title('System power consumption')
        plt.ylabel('Power consumption (Watt)')
    elif mode == 'CUE':
        plt.title('Power consumption of CUEs')
        plt.ylabel('Power consumption (Watt)')
    elif mode == 'D2D':
        plt.title('Power consumption of D2D pairs')
        plt.ylabel('Power consumption (Watt)')
    else:
        raise ValueError("The 'mode' must be 'system', 'CUE', or 'D2D'.")

    # Set the label for the x-axis
    plt.xlabel('(Number of CUEs, Number of D2D pairs)')
    
    # Show the major grid lines with dark grey lines
    plt.grid(axis = 'y', color = '#666666', linestyle = '-')

    # Set the tick locations and labels of the x-axis
    plt.xticks(x_index + 1.5 * width, x_labels)
    plt.legend(loc = 'best')
    plt.show()

def plot_EE(mode, num_of_CUEs, num_of_D2Ds, CA_list, CNN_SPP_list, CNN_list, FCN_list):
    """ Create a grouped bar chart to show the performance of energy efficiency.

    # Aruguments:

        mode: string 
            A string, either 'system', 'CUE', or 'D2D'.
            It specifics which mode should be used for making grouped bar chart.
        num_of_CUEs: set
            Number of the CUEs in each cell.
        num_of_D2Ds: set
            Number of the D2D pairs in each cell.
        CA_list: list
            The heights of the bars, which stands for energy efficiency obtained by CVX solver.
        CNN_SPP_list: list
            The heights of the bars, which stands for energy efficiency obtained by CNN with spatial pyramid pooling layer.
        CNN_list: list
            The heights of the bars, which stands for energy efficiency obtained by CNN.
        FCN_list: list
            The heights of the bars, which stands for energy efficiency obtained by FCN.

    # Return:

        None
    """

    # Insert debugging assertions
    assert type(mode) is str, f"The 'mode' must be string."
    assert num_of_CUEs.issubset(constants.CUE_range), f"The 'num_of_CUEs' must be subset of {constants.CUE_range}."
    assert num_of_D2Ds.issubset(constants.D2D_range), f"The 'num_of_D2Ds' must be subset of {constants.D2D_range}."
    assert type(CA_list) is list, f"The 'CA_list' must be list."
    assert type(CNN_SPP_list) is list, f"The 'CNN_SPP_list' must be list."
    assert type(CNN_list) is list, f"The 'CNN_list' must be list."
    assert type(FCN_list) is list, f"The 'FCN_list' must be list."

    # Add custom x-axis tick labels
    x_labels = [f'({i}, {j})' for (i, j) in product(num_of_CUEs, num_of_D2Ds)]

    # Set the label locations and width of the bar
    x_index = np.arange(len(x_labels))
    width = 0.15

    # Set the size of figure
    plt.figure(figsize = (8.53, 4.8))

    # Each bar will be shifted 'width' units from the previous one
    plt.bar(x_index, CA_list, width, color = '#1f77b4', label = 'CA')
    plt.bar(x_index + 1 * width, CNN_SPP_list, width, color = '#ff7f0e', label = 'CNN-SPP')
    plt.bar(x_index + 2 * width, CNN_list, width, color = '#2ca02c', label = 'CNN')
    plt.bar(x_index + 3 * width, FCN_list, width, color = '#d62728', label = 'FCN')

    # Set the figure title and label for the y-axis
    if mode == 'system':
        plt.title('System energy efficiency')
        plt.ylabel('Energy efficiency (bps/Hz/Watt)')
    elif mode == 'CUE':
        plt.title('Energy efficiency of CUEs')
        plt.ylabel('Energy efficiency (bps/Hz/Watt)')
    elif mode == 'D2D':
        plt.title('Energy efficiency of D2D pairs')
        plt.ylabel('Energy efficiency (bps/Hz/Watt)')
    else:
        raise ValueError("The 'mode' must be 'system', 'CUE', or 'D2D'.")

    # Set the label for the x-axis
    plt.xlabel('(Number of CUEs, Number of D2D pairs)')
    
    # Show the major grid lines with dark grey lines
    plt.grid(axis = 'y', color = '#666666', linestyle = '-')

    # Set the tick locations and labels of the x-axis
    plt.xticks(x_index + 1.5 * width, x_labels)
    plt.legend(loc = 'best')
    plt.show()

def plot_UIR(mode, num_of_CUEs, num_of_D2Ds, CA_list, CNN_SPP_list, CNN_list, FCN_list):
    """ Create a grouped bar chart to show the performance of infeasibility rate (per user).

    # Aruguments:

        mode: string 
            A string, either 'system', 'CUE', or 'D2D'.
            It specifics which mode should be used for making grouped bar chart.
        num_of_CUEs: set
            Number of the CUEs in each cell.
        num_of_D2Ds: set
            Number of the D2D pairs in each cell.
        CA_list: list
            The heights of the bars, which stands for infeasibility rate (per user) obtained by CVX solver.
        CNN_SPP_list: list
            The heights of the bars, which stands for infeasibility rate (per user) obtained by CNN with spatial pyramid pooling layer.
        CNN_list: list
            The heights of the bars, which stands for infeasibility rate (per user) obtained by CNN.
        FCN_list: list
            The heights of the bars, which stands for infeasibility rate (per user) obtained by FCN.

    # Return:

        None
    """

    # Insert debugging assertions
    assert type(mode) is str, f"The 'mode' must be string."
    assert num_of_CUEs.issubset(constants.CUE_range), f"The 'num_of_CUEs' must be subset of {constants.CUE_range}."
    assert num_of_D2Ds.issubset(constants.D2D_range), f"The 'num_of_D2Ds' must be subset of {constants.D2D_range}."
    assert type(CA_list) is list, f"The 'CA_list' must be list."
    assert type(CNN_SPP_list) is list, f"The 'CNN_SPP_list' must be list."
    assert type(CNN_list) is list, f"The 'CNN_list' must be list."
    assert type(FCN_list) is list, f"The 'FCN_list' must be list."

    # Add custom x-axis and y-axis tick labels 
    x_labels = [f'({i}, {j})' for (i, j) in product(num_of_CUEs, num_of_D2Ds)]
    y_labels = [f'{i}%' for i in range(0, 110, 10)]

    # Set the label locations and width of the bar
    x_index = np.arange(len(x_labels))
    y_index = np.arange(0, 1.1, 0.1)
    width = 0.15

    # Set the size of figure
    plt.figure(figsize = (8.53, 4.8))

    # Each bar will be shifted 'width' units from the previous one
    plt.bar(x_index, CA_list, width, color = '#1f77b4', label = 'CA')
    plt.bar(x_index + 1 * width, CNN_SPP_list, width, color = '#ff7f0e', label = 'CNN-SPP')
    plt.bar(x_index + 2 * width, CNN_list, width, color = '#2ca02c', label = 'CNN')
    plt.bar(x_index + 3 * width, FCN_list, width, color = '#d62728', label = 'FCN')

    # Set the figure title and label for the y-axis
    if mode == 'system':
        plt.title('System infeasibility rate (per user)')
        plt.ylabel('Infeasibility rate (%)')
    elif mode == 'CUE':
        plt.title('Infeasibility rate of CUEs (per user)')
        plt.ylabel('Infeasibility rate (%)')
    elif mode == 'D2D':
        plt.title('Infeasibility rate of D2D pairs (per user)')
        plt.ylabel('Infeasibility Rrte (%)')
    else:
        raise ValueError("The 'mode' must be 'system', 'CUE', or 'D2D'.")

    # Set the label for the x-axis
    plt.xlabel('(Number of CUEs, Number of D2D pairs)')

    # Show the major grid lines with dark grey lines
    plt.grid(axis = 'y', color = '#666666', linestyle = '-')

    # Set the tick locations and labels of the x-axis and y-axis
    plt.xticks(x_index + 1.5 * width, x_labels)
    plt.yticks(y_index, y_labels)
    plt.legend(loc = 'best')
    plt.show()

def plot_RIR(mode, num_of_CUEs, num_of_D2Ds, CA_list, CNN_SPP_list, CNN_list, FCN_list):
    """ Create a grouped bar chart to show the performance of infeasibility rate (per realization).

    # Aruguments:

        mode: string 
            A string, either 'system', 'CUE', or 'D2D'.
            It specifics which mode should be used for making grouped bar chart.
        num_of_CUEs: set
            Number of the CUEs in each cell.
        num_of_D2Ds: set
            Number of the D2D pairs in each cell.
        CA_list: list
            The heights of the bars, which stands for infeasibility rate (per realization) obtained by CVX solver.
        CNN_SPP_list: list
            The heights of the bars, which stands for infeasibility rate (per realization) obtained by CNN with spatial pyramid pooling layer.
        CNN_list: list
            The heights of the bars, which stands for infeasibility rate (per realization) obtained by CNN.
        FCN_list: list
            The heights of the bars, which stands for infeasibility rate (per realization) obtained by FCN.

    # Return:

        None
    """

    # Insert debugging assertions
    assert type(mode) is str, f"The 'mode' must be string."
    assert num_of_CUEs.issubset(constants.CUE_range), f"The 'num_of_CUEs' must be subset of {constants.CUE_range}."
    assert num_of_D2Ds.issubset(constants.D2D_range), f"The 'num_of_D2Ds' must be subset of {constants.D2D_range}."
    assert type(CA_list) is list, f"The 'CA_list' must be list."
    assert type(CNN_SPP_list) is list, f"The 'CNN_SPP_list' must be list."
    assert type(CNN_list) is list, f"The 'CNN_list' must be list."
    assert type(FCN_list) is list, f"The 'FCN_list' must be list."

    # Add custom x-axis and y-axis tick labels  
    x_labels = [f'({i}, {j})' for (i, j) in product(num_of_CUEs, num_of_D2Ds)]
    y_labels = [f'{i}%' for i in range(0, 110, 10)]

    # Set the label locations and width of the bar
    x_index = np.arange(len(x_labels))
    y_index = np.arange(0, 1.1, 0.1)
    width = 0.15

    # Set the size of figure
    plt.figure(figsize = (8.53, 4.8))

    # Each bar will be shifted 'width' units from the previous one
    plt.bar(x_index, CA_list, width, color = '#1f77b4', label = 'CA')
    plt.bar(x_index + 1 * width, CNN_SPP_list, width, color = '#ff7f0e', label = 'CNN-SPP')
    plt.bar(x_index + 2 * width, CNN_list, width, color = '#2ca02c', label = 'CNN')
    plt.bar(x_index + 3 * width, FCN_list, width, color = '#d62728', label = 'FCN')

    # Set the figure title and label for the y-axis
    if mode == 'system':
        plt.title('System infeasibility rate (per realization)')
        plt.ylabel('Infeasibility rate (%)')
    elif mode == 'CUE':
        plt.title('Infeasibility rate of CUEs (per realization)')
        plt.ylabel('Infeasibility rate (%)')
    elif mode == 'D2D':
        plt.title('Infeasibility rate of D2D pairs (per realization)')
        plt.ylabel('Infeasibility rate (%)')
    else:
        raise ValueError("The 'mode' must be 'system', 'CUE', or 'D2D'.")

    # Set the label for the x-axis
    plt.xlabel('(Number of CUEs, Number of D2D pairs)')

    # Show the major grid lines with dark grey lines
    plt.grid(axis = 'y', color = '#666666', linestyle = '-')

    # Set the tick locations and labels of the x-axis and y-axis
    plt.xticks(x_index + 1.5 * width, x_labels)
    plt.yticks(y_index, y_labels)
    plt.legend(loc = 'best')
    plt.show()

def plot_NN_computational_time(num_of_CUEs, num_of_D2Ds, CNN_SPP_list, CNN_list, FCN_list):
    """ Create a grouped bar chart to show the different neural network based algorithms' performance of computational time.

    # Aruguments:

        num_of_CUEs: set
            Number of the CUEs in each cell.
        num_of_D2Ds: set
            Number of the D2D pairs in each cell.
        CNN_SPP_list: list
            The heights of the bars, which stands for computational time obtained by CNN with spatial pyramid pooling layer.
        CNN_list: list
            The heights of the bars, which stands for computational time obtained by CNN.
        FCN_list: list
            The heights of the bars, which stands for computational time obtained by FCN.

    # Return:

        None
    """

    # Insert debugging assertions
    assert num_of_CUEs.issubset(constants.CUE_range), f"The 'num_of_CUEs' must be subset of {constants.CUE_range}."
    assert num_of_D2Ds.issubset(constants.D2D_range), f"The 'num_of_D2Ds' must be subset of {constants.D2D_range}."
    assert type(CNN_SPP_list) is list, f"The 'CNN_SPP_list' must be list."
    assert type(CNN_list) is list, f"The 'CNN_list' must be list."
    assert type(FCN_list) is list, f"The 'FCN_list' must be list."

    # Add custom x-axis tick labels
    x_labels = [f'({i}, {j})' for (i, j) in product(num_of_CUEs, num_of_D2Ds)]

    # Set the label locations and width of the bar
    x_index = np.arange(len(x_labels))
    width = 0.15

    # Set the size of figure
    plt.figure(figsize = (8.53, 4.8))

    # Convert seconds to milliseconds
    CNN_SPP_list = [time * 1000 for time in CNN_SPP_list]
    CNN_list = [time * 1000 for time in CNN_list]
    FCN_list = [time * 1000 for time in FCN_list]

    # Each bar will be shifted 'width' units from the previous one
    plt.bar(x_index, CNN_SPP_list, width, color = '#ff7f0e',label = 'CNN-SPP')
    plt.bar(x_index + 1 * width, CNN_list, width, color = '#2ca02c', label = 'CNN')
    plt.bar(x_index + 2 * width, FCN_list, width, color = '#d62728', label = 'FCN')

    # Set the figure title and label for the y-axis
    plt.title('Computational time')
    plt.ylabel('Computational time (ms)')
 
    # Set the label for the x-axis
    plt.xlabel('(Number of CUEs, Number of D2D pairs)')
    
    # Show the major grid lines with dark grey lines
    plt.grid(axis = 'y', color = '#666666', linestyle = '-')

    # Set the tick locations and labels of the x-axis
    plt.xticks(x_index + 1 * width, x_labels)
    plt.legend(loc = 'best')
    plt.show()

def plot_CA_computational_time(num_of_CUEs, num_of_D2Ds, CA_list):
    """ Create a grouped bar chart to show the convex approximation based algorithm's performance of computational time.

    # Aruguments:

        num_of_CUEs: set
            Number of the CUEs in each cell.
        num_of_D2Ds: set
            Number of the D2D pairs in each cell.
        CA_list: list
            The heights of the bars, which stands for computational time obtained by CVX solver.

    # Return:

        None
    """

    # Insert debugging assertions
    assert num_of_CUEs.issubset(constants.CUE_range), f"The 'num_of_CUEs' must be subset of {constants.CUE_range}."
    assert num_of_D2Ds.issubset(constants.D2D_range), f"The 'num_of_D2Ds' must be subset of {constants.D2D_range}."
    assert type(CA_list) is list, f"The 'CA_list' must be list."

    # Add custom x-axis tick labels
    x_labels = [f'({i}, {j})' for (i, j) in product(num_of_CUEs, num_of_D2Ds)]

    # Set the label locations and width of the bar
    x_index = np.arange(len(x_labels))
    width = 0.2

    # Set the size of figure
    plt.figure(figsize = (8.53, 4.8))

    # Plot the bar
    plt.bar(x_index, CA_list, width, color = '#1f77b4', label = 'CA')

    # Set the figure title and label for the y-axis
    plt.title('Computational time')
    plt.ylabel('Computational time (sec)')
 
    # Set the label for the x-axis
    plt.xlabel('(Number of CUEs, Number of D2D pairs)')
    
    # Show the major grid lines with dark grey lines
    plt.grid(axis = 'y', color = '#666666', linestyle = '-')

    # Set the tick locations and labels of the x-axis
    plt.xticks(x_index, x_labels)
    plt.legend(loc = 'best')
    plt.show()