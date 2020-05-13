function summary(num_of_cells, num_of_CUEs, num_of_D2Ds, system_EE, total_data_rate, total_power_consumption, CUE_rate, D2D_rate, CUE_power, D2D_power, feasible, Pmax, QoS_of_CUE, QoS_of_D2D, proportion, max_dinkelbach_iterations, max_condensation_iterations, iteration, dinkelbach_iterations, condensation_iterations)
% Input:
% num_of_cells: Number of the cells in the system
% num_of_CUEs: Number of the CUEs in each cell
% num_of_D2Ds: Number of the D2D pairs in each cell
% system_EE: The energy efficiency of the multi-cell system
% total_data_rate: Total data rate of all users
% total_power_consumption: Total power consumption of all users
% CUE_rate: The data rate achieved by CUEs
% D2D_rate: The data rate achieved by D2D pairs
% CUE_power: The transmit power of all CUEs
% D2D_power: The transmit power of all D2D pairs
% feasible: A binary indicator which indicates whether the transmit power of all devices is feasible or not
% Pmax: Maximum transmit power of all devices (Watt)
% QoS_of_CUE: Minimum data rate requirement of all CUEs (bps/Hz)
% QoS_of_D2D: Minimum data rate requirement of all D2D pairs (bps/Hz)
% proportion: The proportion of CUE's minimum rate requirement to CUE's maximum data rate
% max_dinkelbach_iterations: Maximum iterations of dinkelbach method 
% max_condensation_iterations: Maximum iterations of condensation method
% iteration (optional): The current iteration  
% dinkelbach_iterations(optional): The current iteration of dinkelbach method
% condensation_iterations (optional): The current iteration of condensation method

if nargin == 17
    cprintf('Blue', '========================= CVX summary: Initialization =========================\n');
    fprintf('Initial system EE: %f\n', system_EE);
    fprintf('Initial system sum rate: %f\n', total_data_rate);
    fprintf('Initial totsl power comsunption: %f\n', total_power_consumption);
    fprintf('Maximum iterations of dinkelbach method: %d\n', max_dinkelbach_iterations);
    fprintf('Maximum iterations of condensation method: %d\n', max_condensation_iterations);
    
    str = string(feasible);
    str = upper(extractBefore(str,2)) + extractAfter(str,1);
    fprintf('Feasible: %s\n\n', str);
    
    print_transmit_power(num_of_cells, num_of_CUEs, num_of_D2Ds, Pmax, CUE_power, D2D_power, 2);
    print_data_rate(num_of_cells, num_of_CUEs, num_of_D2Ds, QoS_of_CUE, QoS_of_D2D, CUE_rate, D2D_rate, proportion, 2);
else
    cprintf('Blue', '========================== CVX summary: Iteration %d ==========================\n', iteration);
    fprintf('System EE: %f\n', system_EE);
    fprintf('System sum rate: %f\n', total_data_rate);
    fprintf('Totsl power comsunption: %f\n', total_power_consumption);
    fprintf('Iteration of dinkelbach method: %d\n', dinkelbach_iterations);
    fprintf('Iteration of condensation method: %d\n', condensation_iterations);
    
    str = string(feasible);
    str = upper(extractBefore(str,2)) + extractAfter(str,1);
    fprintf('Feasible: %s\n\n', str);
    
    print_transmit_power(num_of_cells, num_of_CUEs, num_of_D2Ds, Pmax, CUE_power, D2D_power, 2);
    print_data_rate(num_of_cells, num_of_CUEs, num_of_D2Ds, QoS_of_CUE, QoS_of_D2D, CUE_rate, D2D_rate, proportion, 2);
end
