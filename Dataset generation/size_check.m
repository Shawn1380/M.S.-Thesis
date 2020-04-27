function [pass] = size_check(num_of_cells)
% Input:
% num_of_cells: Number of the cells in the system
%
% Output:
% pass: A binary indicator which indicates whether the size check is passed or not

for num_of_CUEs = 2 : 5
    for num_of_D2Ds = 2 : 5
        
        % Declaration of binary indicator
        pass = 1;
        
        % Loop over all training set (.mat file)
        filename = sprintf('data_Cell_%d_CUE_%d_D2D_%d_2000.mat', num_of_cells, num_of_CUEs, num_of_D2Ds);
        matobj = matfile(filename);
        [channel_gain_matrix_first, channel_gain_matrix_second, channel_gain_matrix_third] = cellfun(@size, matobj.input_data);
        [optimal_CUE_power_first, optimal_CUE_power_second, optimal_CUE_power_third] = cellfun(@size, matobj.target_data(1, :));
        [optimal_D2D_power_first, optimal_D2D_power_second, optimal_D2D_power_third] = cellfun(@size, matobj.target_data(2, :));
        
        % Check the size of channel_gain_matrix 
        if sum(channel_gain_matrix_first ~= (num_of_CUEs + num_of_D2Ds) * num_of_cells) ~= 0 || sum(channel_gain_matrix_second ~= 1 + num_of_D2Ds) ~= 0 || sum(channel_gain_matrix_third ~= num_of_cells) ~= 0
            pass = 0;
            cprintf('Red', 'Size check failed: channel_gain_matrix in %s\n', filename);
        end
        
        % Check the size of optimal_CUE_power
        if sum(optimal_CUE_power_first ~= num_of_CUEs) ~= 0 || sum(optimal_CUE_power_second ~= 1) ~= 0 || sum(optimal_CUE_power_third ~= num_of_cells) ~= 0
            pass = 0;
            cprintf('Red', 'Size check failed: optimal_CUE_power in %s\n', filename);
        end
        
        % Check the size of optimal_D2D_power
        if sum(optimal_D2D_power_first ~= num_of_D2Ds) ~= 0 || sum(optimal_D2D_power_second ~= num_of_CUEs) ~= 0 || sum(optimal_D2D_power_third ~= num_of_cells) ~= 0
            pass = 0;
            cprintf('Red', 'Size check failed: optimal_D2D_power in %s\n', filename);
        end
        
        % The size check is passed 
        if pass == 1
            cprintf('Red', 'Size check passed: %s\n', filename);
        end
    end
end
