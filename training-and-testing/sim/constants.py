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
        Default value: {2, 3, 4}
    D2D_range: dict
        Limitation on the number of D2D pairs.
        Default value: {2, 3, 4}
    noise: float
        Noise power (Watt).
        Default value: 7.161e-16
    Pmax: float
        Maximum transmit power of all CUEs and all D2D pairs (Watt).
        Default value: 0.2
    rate_proportion: float
        The ratio of CUE's minimum rate requirement to CUE's maximum data rate.
        Default value: 0.2
    QoS_of_D2D: int
        Minimum rate requirement of all D2D pairs (bps/Hz).
        Default value: 3
    circuit_power: float
        Circuit power (Watt).
        Default value: 0.1
    PA_inefficiency_factor: float
        Power amplifier inefficiency factor.
        Default value: 0.35
"""
 
cell_range = {2, 3}
CUE_range = {2, 3, 4}
D2D_range = {2, 3, 4}
noise = 7.161e-16 
Pmax = 0.2 
rate_proportion = 0.2
QoS_of_D2D = 3
circuit_power = 0.1 
PA_inefficiency_factor = 0.35 