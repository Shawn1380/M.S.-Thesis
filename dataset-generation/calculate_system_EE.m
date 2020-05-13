function [system_EE, total_data_rate, total_power_consumption, CUE_rate, D2D_rate, feasible] = calculate_system_EE(num_of_cells, num_of_CUEs, num_of_D2Ds, channel_gain_matrix, CUE_power, D2D_power, Pmax, QoS_of_CUE, QoS_of_D2D)
% Input:
% num_of_cells: Number of the cells in the system
% num_of_CUEs: Number of the CUEs in each cell
% num_of_D2Ds: Number of the D2D pairs in each cell
% channel_gain_matrix: A matrix which represents the channel gains of all links in the entire network
% CUE_power: The transmit power of all CUEs
% D2D_power: The transmit power of all D2D pairs
% Pmax: Maximum transmit power of all devices (Watt)
% QoS_of_CUE: Minimum data rate requirement of all CUEs (bps/Hz)
% QoS_of_D2D: Minimum data rate requirement of all D2D pairs (bps/Hz)
%
% Output:
% system_EE: The energy efficiency of the multi-cell system
% total_data_rate: Total data rate of all users
% total_power_consumption: Total power consumption of all users
% CUE_rate: The data rate achieved by CUEs
% D2D_rate: The data rate achieved by D2D pairs
% feasible: A binary indicator which indicates whether the transmit power of all devices is feasible or not

noise = 7.161e-16; % Noise power (Watt)
circuit_power = 0.1; % Circuit power (Watt)
PA_inefficiency_factor = 0.35; % Power amplifier inefficiency factor

total_power_consumption = circuit_power * num_of_cells * (num_of_CUEs + 2 * num_of_D2Ds) + sum(CUE_power(:)) / PA_inefficiency_factor + sum(D2D_power(:)) / PA_inefficiency_factor;
total_data_rate = 0;

CUE_rate = zeros(num_of_CUEs, 1, num_of_cells);
D2D_rate = zeros(num_of_D2Ds, num_of_CUEs, num_of_cells);

feasible = true;

for k = 1 : num_of_cells
    
    % Calculate EE of each CUE
    for i = 1 : num_of_CUEs
        
        % Feasibility check
        if CUE_power(i, 1, k) > Pmax || CUE_power(i, 1, k) < 0
            feasible = false;
        end        
        
        % Calculate the power of desired signal 
        desired_signal = CUE_power(i, 1, k) * channel_gain_matrix(i, 1, k);
        
        % Calculate the power of intra-cell interference from D2D pairs 
        intra_cell_interference = sum(D2D_power(:, i, k) .* channel_gain_matrix(num_of_CUEs + 1 : num_of_CUEs + num_of_D2Ds, 1, k));
        
        % Calculate the power of inter-cell interference
        inter_cell_interference_from_CUE = 0;
        inter_cell_interference_from_D2D = 0;
        for j = 1 : num_of_cells
            if j == k
                continue
            else
                if j < k
                    % CUE part
                    interference_from_CUE = CUE_power(i, 1, j) * channel_gain_matrix(num_of_CUEs + num_of_D2Ds + (j - 1) * num_of_CUEs + i, 1, k);
                    % D2D part
                    interference_from_D2D = sum(D2D_power(:, i, j) .* channel_gain_matrix(num_of_cells * num_of_CUEs + num_of_D2Ds + (j - 1) * num_of_D2Ds + 1 : num_of_cells * num_of_CUEs + num_of_D2Ds + (j - 1) * num_of_D2Ds + num_of_D2Ds, 1, k));
                else
                    % CUE part
                    interference_from_CUE = CUE_power(i, 1, j) * channel_gain_matrix(num_of_CUEs + num_of_D2Ds + (j - 2) * num_of_CUEs + i, 1, k);
                    % D2D part
                    interference_from_D2D = sum(D2D_power(:, i, j) .* channel_gain_matrix(num_of_cells * num_of_CUEs + num_of_D2Ds + (j - 2) * num_of_D2Ds + 1 : num_of_cells * num_of_CUEs + num_of_D2Ds + (j - 2) * num_of_D2Ds + num_of_D2Ds, 1, k));
                end    
                inter_cell_interference_from_CUE = inter_cell_interference_from_CUE + interference_from_CUE;
                inter_cell_interference_from_D2D = inter_cell_interference_from_D2D + interference_from_D2D;
            end
        end
        
        SINR = desired_signal / (noise + intra_cell_interference + inter_cell_interference_from_CUE + inter_cell_interference_from_D2D);
        rate_of_CUE = log2(1 + SINR);
        CUE_rate(i, 1, k) = rate_of_CUE;
        total_data_rate = total_data_rate + rate_of_CUE;
        
        % Feasibility check
        if rate_of_CUE < QoS_of_CUE(i, 1, k) - 1e-2
            feasible = false;
        end        
    end
    
    % Calculate EE of each D2D pair
    for i = 1 : num_of_D2Ds
        
        rate_of_D2D = 0;
        
        % Feasibility check
        if sum(D2D_power(i, :, k)) > Pmax || sum(D2D_power(i, :, k)) < 0
            feasible = false;
        end        
        
        % Summation over all resource blocks 
        for j = 1 : num_of_CUEs
            
            % Feasibility check
            if D2D_power(i, j, k) > Pmax || D2D_power(i, j, k) < 0
                feasible = false;
            end            
            
            % Calculate the power of desired signal
            desired_signal = D2D_power(i, j, k) * channel_gain_matrix(num_of_CUEs + i, 1 + i, k);
            
            % Calculate the power of intra-cell inteference from CUE and other D2D pairs
            interference_from_CUE = CUE_power(j, 1, k) * channel_gain_matrix(j, 1 + i, k);
            interference_from_D2D = sum(D2D_power(:, j, k) .* channel_gain_matrix(num_of_CUEs + 1 : num_of_CUEs + num_of_D2Ds, 1 + i, k)) - desired_signal; 
            intra_cell_interference = interference_from_CUE + interference_from_D2D;
            
            % Calculate the power of inter-cell interference
            inter_cell_interference_from_CUE = 0;
            inter_cell_interference_from_D2D = 0;
            for l = 1 : num_of_cells
                if l == k
                    continue
                else
                    if l < k
                        % CUE part
                        interference_from_CUE = CUE_power(j, 1, l) * channel_gain_matrix(num_of_CUEs + num_of_D2Ds + (l - 1) * num_of_CUEs + j, 1 + i, k);
                        % D2D part
                        interference_from_D2D = sum(D2D_power(:, j, l) .* channel_gain_matrix(num_of_cells * num_of_CUEs + num_of_D2Ds + (l - 1) * num_of_D2Ds + 1 : num_of_cells * num_of_CUEs + num_of_D2Ds + (l - 1) * num_of_D2Ds + num_of_D2Ds, 1 + i, k));
                    else
                        % CUE part
                        interference_from_CUE = CUE_power(j, 1, l) * channel_gain_matrix(num_of_CUEs + num_of_D2Ds + (l - 2) * num_of_CUEs + j, 1 + i, k);
                        % D2D part
                        interference_from_D2D = sum(D2D_power(:, j, l) .* channel_gain_matrix(num_of_cells * num_of_CUEs + num_of_D2Ds + (l - 2) * num_of_D2Ds + 1 : num_of_cells * num_of_CUEs + num_of_D2Ds + (l - 2) * num_of_D2Ds + num_of_D2Ds, 1 + i, k));
                    end
                    inter_cell_interference_from_CUE = inter_cell_interference_from_CUE + interference_from_CUE;
                    inter_cell_interference_from_D2D = inter_cell_interference_from_D2D + interference_from_D2D;
                end
            end
            
            SINR = desired_signal / (noise + intra_cell_interference + inter_cell_interference_from_CUE + inter_cell_interference_from_D2D);
            D2D_rate(i, j, k) = log2(1 + SINR);
            rate_of_D2D = rate_of_D2D + log2(1 + SINR);
        end
        
        total_data_rate = total_data_rate + rate_of_D2D;
        
        % Feasibility check
        if rate_of_D2D < QoS_of_D2D - 1e-2
            feasible = false;
        end
    end
end

system_EE = total_data_rate / total_power_consumption;