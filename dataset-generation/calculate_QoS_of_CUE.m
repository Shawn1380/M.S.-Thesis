function [QoS_of_CUE] = calculate_QoS_of_CUE(num_of_cells, num_of_CUEs, channel_gain_matrix, Pmax, proportion)
% Input:
% num_of_cells: Number of the cells in the system
% num_of_CUEs: Number of the CUEs in each cell
% channel_gain_matrix: A matrix represents the channel gains of all links in the entire network
% Pmax: Maximum transmit power of all devices (Watt)
% proportion: The proportion of minimum rate requirement to maximum data rate 
%
% Output:
% QoS_of_CUE: Minimum data rate requirement of all CUEs (bps/Hz)

noise = 7.161e-16; % Noise power (Watt)

QoS_of_CUE = zeros(num_of_CUEs, 1, num_of_cells);

for k = 1 : num_of_cells
    for i = 1 : num_of_CUEs
        % Calculate the power of desired signal
        desired_siganl = Pmax * channel_gain_matrix(i, 1, k);
        
        % Calculate CUE's maximum data rate
        SINR = desired_siganl / noise;
        max_rate_of_CUE = log2(1 + SINR);
        
        QoS_of_CUE(i, 1, k) = max_rate_of_CUE * proportion;
    end
end