"""
The variables below are constants, whose value cannot be changed, and the value of these constants is determined by the parameters setting of the simulation.

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