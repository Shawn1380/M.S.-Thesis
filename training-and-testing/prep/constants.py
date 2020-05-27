"""
This module includes a small number of constants.

The following variables are constants and their values cannot be changed.
The values of these constants are determined according to the simulation parameter settings. 

# Constants:

cell_range: dict
    Limitation on the number of cells.
    Default value: {2, 3}
CUE_range: dict
    Limitation on the number of CUEs.
    Default value: {2, 3, 4, 5}
D2D_range: dict
    Limitation on the number of D2D pairs.
    Default value: {2, 3, 4, 5}
data_proportion: float
    Float between 0 and 1. Fraction of the data to be used as training data.
    Default value: 0.8
"""
 
cell_range = {2, 3}
CUE_range = {2, 3, 4, 5}
D2D_range = {2, 3, 4, 5}
data_proportion = 0.8