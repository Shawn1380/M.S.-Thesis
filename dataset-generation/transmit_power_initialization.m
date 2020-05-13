function [initial_CUE_power, initial_D2D_power, isfeasible] = transmit_power_initialization(num_of_cells, num_of_CUEs, num_of_D2Ds, channel_gain_matrix, Pmax, QoS_of_CUE, QoS_of_D2D)
% Input:
% num_of_cells: Number of the cells in the system
% num_of_CUEs: Number of the CUEs in each cell
% num_of_D2Ds: Number of the D2D pairs in each cell
% channel_gain_matrix: A matrix represents the channel gains of all links in the entire network
% Pmax: Maximum transmit power of all devices (Watt)
% QoS_of_CUE: Minimum data rate requirement of all CUEs (bps/Hz)
% QoS_of_D2D: Minimum data rate requirement of all D2D pairs (bps/Hz)
%
% Output: 
% initial_CUE_power: The initial transmit power of all CUEs 
% initial_D2D_power: The initial transmit power of all D2D pairs
% isfeasible: A binary indicator which indicates whether the feasibility test is passed or not

cprintf('Red', 'Initializing transmit power of CUEs and D2D pairs...\n');

matrix = zeros(num_of_D2Ds + 1, num_of_D2Ds + 1);
initial_CUE_power = zeros(num_of_CUEs, 1, num_of_cells);
initial_D2D_power = zeros(num_of_D2Ds, num_of_CUEs, num_of_cells);
    
for k = 1 : num_of_cells
    for i = 1 : num_of_CUEs
        % Compute each entry of the matrix
        matrix(1, 1) = channel_gain_matrix(i, 1, k);
        matrix(1, 2 : num_of_D2Ds + 1) = channel_gain_matrix(i, 2 : num_of_D2Ds + 1, k);
        matrix(2 : num_of_D2Ds + 1, 1) = channel_gain_matrix(num_of_CUEs + 1 : num_of_CUEs + num_of_D2Ds, 1, k);
        matrix(2 : num_of_D2Ds + 1, 2 : num_of_D2Ds + 1) = channel_gain_matrix(num_of_CUEs + 1 : num_of_CUEs + num_of_D2Ds, 2 : num_of_D2Ds + 1, k);
        
        % Calculate initial transit power
        [feasible, particular_sol] = feasibility_test(num_of_CUEs, num_of_D2Ds, matrix, Pmax, QoS_of_CUE(i, 1, k), QoS_of_D2D);
        if feasible ~= 1
            cprintf('Red', 'Feasibility test: Fail\n');
            initial_CUE_power = NaN;
            initial_D2D_power = NaN;
            isfeasible = 0;
            return
        else
            initial_CUE_power(i, 1, k) = particular_sol(1);
            initial_D2D_power(:, i, k) = particular_sol(2 : num_of_D2Ds + 1);
        end
    end
end

cprintf('Red', 'Feasibility test: Pass\n\n');
isfeasible = 1;