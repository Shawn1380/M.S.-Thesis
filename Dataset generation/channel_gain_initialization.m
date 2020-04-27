function [channel_gain_matrix] = channel_gain_initialization(num_of_cells, num_of_CUEs, num_of_D2Ds, cenX, cenY, CUE_position, D2D_position)
% Input:
% num_of_cells: Number of the cells in the system
% num_of_CUEs: Number of the CUEs in each cell
% num_of_D2Ds: Number of the D2D pairs in each cell
% cenX: The X-coordinate of the center of the cells
% cenY: The Y-coordinate of the center of the cells
% CUE_position: The position of the CUEs
% D2D_position: The position of the D2D pairs
%
% Output:
% channel_gain_matrix: A matrix represents the channel gains of all links in the entire network, which consists of the following components:  
%                      
% Intra cell part:
% intra_CUE_to_BS_gain: Intra-cell channel gain between CUE and BS
% intra_CUE_to_D2D_gain: Intra-cell channel gain between CUE and D2D pair
% intra_D2D_to_BS_gain: Intra-cell channel gain between D2D pair and BS
% intra_D2D_to_D2D_gain: Intra-cell channel gain between D2D pairs
%
% Inter cell part:
% inter_CUE_to_BS_gain: Inter-cell channel gain between CUE and BS
% inter_CUE_to_D2D_gain: Inter-cell channel gain between CUE and D2D pair
% inter_D2D_to_BS_gain: Inter-cell channel gain between D2D pair and BS
% inter_D2D_to_D2D_gain: Inter-cell channel gain between D2D pairs

channel_gain_matrix = zeros((num_of_CUEs + num_of_D2Ds) * num_of_cells, 1 + num_of_D2Ds, num_of_cells);

intra_CUE_to_BS_gain = zeros(num_of_CUEs, 1);
intra_CUE_to_D2D_gain = zeros(num_of_CUEs, num_of_D2Ds);
intra_D2D_to_BS_gain = zeros(num_of_D2Ds, 1);
intra_D2D_to_D2D_gain = zeros(num_of_D2Ds, num_of_D2Ds);

inter_CUE_to_BS_gain = zeros(num_of_CUEs * (num_of_cells - 1), 1);
inter_CUE_to_D2D_gain = zeros(num_of_CUEs * (num_of_cells - 1), num_of_D2Ds);
inter_D2D_to_BS_gain = zeros(num_of_D2Ds * (num_of_cells - 1), 1);
inter_D2D_to_D2D_gain = zeros(num_of_D2Ds * (num_of_cells - 1), num_of_D2Ds);

for k = 1 : num_of_cells
    
    for i = 1 : num_of_CUEs
        % Calculate intra channel gain between CUE and BS 
        distance = sqrt((CUE_position(i, 2 * k - 1) - cenX(k)) ^ 2 + (CUE_position(i, 2 * k) - cenY(k)) ^ 2) / 1000; % Distance between CUE and BS (in kilometer)
        pathloss = 128.1 + 37.6 * log10(distance);
        intra_CUE_to_BS_gain(i, 1) = 10 ^ (-pathloss / 10);
   
        for j = 1 : num_of_D2Ds
            % Calculate intra channel gain between CUE and D2D 
            distance = sqrt((CUE_position(i, 2 * k - 1) - D2D_position(j, 2 * k - 1)) ^ 2 + (CUE_position(i, 2 * k) - D2D_position(j, 2 * k)) ^ 2) / 1000; % Distance between CUE and D2D (in kilometer)
            pathloss = 128.1 + 37.6 * log10(distance);
            intra_CUE_to_D2D_gain(i, j) = 10 ^ (-pathloss / 10);
        end
    end
    
    channel_gain_matrix(1 : num_of_CUEs, 1, k) = intra_CUE_to_BS_gain(:, 1);
    channel_gain_matrix(1 : num_of_CUEs, 2 : num_of_D2Ds + 1, k) = intra_CUE_to_D2D_gain(:, :);
    
    for i = 1 : num_of_D2Ds
        % Calculate intra channel gain between D2D and BS 
        distance = sqrt((D2D_position(i, 2 * k - 1) - cenX(k)) ^ 2 + (D2D_position(i, 2 * k) - cenY(k)) ^ 2) / 1000; % Distance between D2D and BS (in kilometer)
        pathloss = 128.1 + 37.6 * log10(distance);
        intra_D2D_to_BS_gain(i, 1) = 10 ^ (-pathloss / 10);
        
        for j = 1 : num_of_D2Ds
            % Calculate intra channel gain between D2D Tx and D2D Rx
            if i == j
                distance = 15 / 1000; % Distance between D2D Tx and D2D Rx (in kilometer)
                pathloss = 148 + 40 * log10(distance); 
            else
                distance = sqrt((D2D_position(i, 2 * k - 1) - D2D_position(j, 2 * k - 1)) ^ 2 + (D2D_position(i, 2 * k) - D2D_position(j, 2 * k)) ^ 2) / 1000; % Distance between D2D Tx and D2D Rx (in kilometer)
                pathloss = 128.1 + 37.6 * log10(distance);
            end
            intra_D2D_to_D2D_gain(i, j) = 10 ^ (-pathloss / 10);
        end
    end
    
    channel_gain_matrix(num_of_CUEs + 1 : num_of_CUEs + num_of_D2Ds, 1, k) = intra_D2D_to_BS_gain(:, 1);
    channel_gain_matrix(num_of_CUEs + 1 : num_of_CUEs + num_of_D2Ds, 2 : num_of_D2Ds + 1, k) = intra_D2D_to_D2D_gain(:, :);
    
    for i = 1 : num_of_cells
        for j = 1 : num_of_CUEs
            if i == k
                break
            else
                % Calculate inter channel gain between CUE and BS
                distance = sqrt((CUE_position(j, 2 * i - 1) - cenX(k)) ^ 2 + (CUE_position(j, 2 * i) - cenY(k)) ^ 2) / 1000; % Distance between CUE and BS (in kilometer)
                pathloss = 128.1 + 37.6 * log10(distance);
                if i < k
                    inter_CUE_to_BS_gain(j + (i - 1) * num_of_CUEs, 1) = 10 ^ (-pathloss / 10);
                else
                    inter_CUE_to_BS_gain(j + (i - 2) * num_of_CUEs, 1) = 10 ^ (-pathloss / 10);
                end
                
                % Calculate inter channel gain between CUE and D2D
                for l = 1 : num_of_D2Ds
                    distance = sqrt((CUE_position(j, 2 * i - 1) - D2D_position(l, 2 * k - 1)) ^ 2 + (CUE_position(j, 2 * i) - D2D_position(l, 2 * k)) ^ 2) / 1000; % Distance between CUE and D2D (in kilometer)
                    pathloss = 128.1 + 37.6 * log10(distance);
                    if i < k
                        inter_CUE_to_D2D_gain(j + (i - 1) * num_of_CUEs, l) = 10 ^ (-pathloss / 10);
                    else
                        inter_CUE_to_D2D_gain(j + (i - 2) * num_of_CUEs, l) = 10 ^ (-pathloss / 10);
                    end
                end
            end
        end
        
        for j = 1 : num_of_D2Ds
            if i == k
                break
            else
                % Calculate inter channel gain between D2D and BS
                distance = sqrt((D2D_position(j, 2 * i - 1) - cenX(k)) ^ 2 + (D2D_position(j, 2 * i) - cenY(k)) ^ 2) / 1000; % Distance between D2D and BS (in kilometer)
                pathloss = 128.1 + 37.6 * log10(distance);
                if i < k
                    inter_D2D_to_BS_gain(j + (i - 1) * num_of_D2Ds, 1) = 10 ^ (-pathloss / 10);
                else    
                    inter_D2D_to_BS_gain(j + (i - 2) * num_of_D2Ds, 1) = 10 ^ (-pathloss / 10);
                end
                
                % Calculate inter channel gain between D2D Tx and D2D Rx
                for l = 1 : num_of_D2Ds
                    distance = sqrt((D2D_position(j, 2 * i - 1) - D2D_position(l, 2 * k - 1)) ^ 2 + (D2D_position(j, 2 * i) - D2D_position(l, 2 * k)) ^ 2) / 1000; % Distance between D2D Tx and D2D Rx (in kilometer)
                    pathloss = 128.1 + 37.6 * log10(distance);
                    if i < k
                        inter_D2D_to_D2D_gain(j + (i - 1) * num_of_D2Ds, l) = 10 ^ (-pathloss / 10);
                    else
                        inter_D2D_to_D2D_gain(j + (i - 2) * num_of_D2Ds, l) = 10 ^ (-pathloss / 10);
                    end
                end
            end    
        end
    end
    
    channel_gain_matrix(num_of_CUEs + num_of_D2Ds + 1 : num_of_cells * num_of_CUEs + num_of_D2Ds, 1, k) = inter_CUE_to_BS_gain(:, 1);
    channel_gain_matrix(num_of_CUEs + num_of_D2Ds + 1 : num_of_cells * num_of_CUEs + num_of_D2Ds, 2 : num_of_D2Ds + 1, k) = inter_CUE_to_D2D_gain(:, :);
    channel_gain_matrix(num_of_cells * num_of_CUEs + num_of_D2Ds + 1 : num_of_cells * (num_of_CUEs + num_of_D2Ds), 1, k) = inter_D2D_to_BS_gain(:, 1);
    channel_gain_matrix(num_of_cells * num_of_CUEs + num_of_D2Ds + 1 : num_of_cells * (num_of_CUEs + num_of_D2Ds), 2 : num_of_D2Ds + 1, k) = inter_D2D_to_D2D_gain(:, :);
    
end