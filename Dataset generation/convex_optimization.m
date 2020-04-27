function [optimal_CUE_power, optimal_D2D_power, success] = convex_optimization(num_of_cells, num_of_CUEs, num_of_D2Ds, channel_gain_matrix, initial_CUE_power, initial_D2D_power, Pmax, QoS_of_CUE, QoS_of_D2D, proportion, max_dinkelbach_iterations, max_condensation_iterations)
% Input:
% num_of_cells: Number of the cells in the system
% num_of_CUEs: Number of the CUEs in each cell
% num_of_D2Ds: Number of the D2D pairs in each cell
% channel_gain_matrix: A matrix represents the channel gains of all links in the entire network
% initial_CUE_power: The initial transmit power of all CUEs 
% initial_D2D_power: The initial transmit power of all D2D pairs
% Pmax: Maximum transmit power of all devices (Watt)
% QoS_of_CUE: Minimum data rate requirement of all CUEs (bps/Hz)
% QoS_of_D2D: Minimum data rate requirement of all D2D pairs (bps/Hz)
% proportion: The proportion of CUE's minimum rate requirement to CUE's maximum data rate
% max_dinkelbach_iterations: Maximum iterations of dinkelbach method 
% max_condensation_iterations: Maximum iterations of condensation method
%
% Output:
% optimal_system_EE: The optimal energy efficiency of the multi-cell system
% optimal_CUE_power: The optimal transmit power of all CUEs
% optimal_D2D_power: The optimal transmit power of all D2Ds
% success: Summarize the result of CVX's effort in the form of integer

cprintf('Red', 'Start solving objective function by CVX...\n\n');

noise = 7.161e-16; % Noise power (Watt)
PA_inefficiency_factor = 0.35; % Power amplifier inefficiency factor

% Used in the dinkelbach method (determine whether system EE is converged)
[system_EE, total_data_rate, total_power_consumption, initial_CUE_rate, initial_D2D_rate, feasible] = calculate_system_EE(num_of_cells, num_of_CUEs, num_of_D2Ds, channel_gain_matrix, initial_CUE_power, initial_D2D_power, Pmax, QoS_of_CUE, QoS_of_D2D);

% Summary
summary(num_of_cells, num_of_CUEs, num_of_D2Ds, system_EE, total_data_rate, total_power_consumption, initial_CUE_rate, initial_D2D_rate, initial_CUE_power, initial_D2D_power, feasible, Pmax, QoS_of_CUE, QoS_of_D2D, proportion, max_dinkelbach_iterations, max_condensation_iterations);

% Used in the condensation method (approximate the posynomial in denominator with a monomial)
CUE_power = initial_CUE_power;
D2D_power = initial_D2D_power;

iteration = 1;
dinkelbach_iterations = 1;

% Maximize system EE by dinkelbach method
for dinkelbach_iteration = 1 : max_dinkelbach_iterations
    
    condensation_iterations = 1;
    
    % Approximate objective function to GP problem by condensation method
    for condensation_iteration = 1 : max_condensation_iterations
        
        cvx_begin gp quiet
        
            % Define the variables (transmit power of CUEs and D2D pairs)
            variable variable_CUE_power(num_of_CUEs, 1, num_of_cells) nonnegative
            variable variable_D2D_power(num_of_D2Ds, num_of_CUEs, num_of_cells) nonnegative
            
            % Initialize some matrices which are used in the objective function
            intra_D2D_to_BS_interference = cvx(zeros(num_of_D2Ds, num_of_CUEs, num_of_cells));
            inter_D2D_to_BS_interference = cvx(zeros(num_of_D2Ds * (num_of_cells - 1), num_of_CUEs, num_of_cells));
            inter_CUE_to_BS_interference = cvx(zeros(num_of_cells - 1, num_of_CUEs, num_of_cells));
            CUE_desired_signal = cvx(zeros(1, num_of_CUEs, num_of_cells));
            
            alpha_intra_D2D_to_BS_interference = zeros(num_of_D2Ds, num_of_CUEs, num_of_cells);
            alpha_inter_D2D_to_BS_interference = zeros(num_of_D2Ds * (num_of_cells - 1), num_of_CUEs, num_of_cells);
            alpha_inter_CUE_to_BS_interference = zeros(num_of_cells - 1, num_of_CUEs, num_of_cells);
            alpha_CUE_desired_signal = zeros(1, num_of_CUEs, num_of_cells);
            
            intra_CUE_to_D2D_interference = cvx(zeros(num_of_CUEs, num_of_D2Ds, num_of_cells));
            intra_D2D_to_D2D_interference = cvx(zeros(num_of_CUEs * (num_of_D2Ds - 1), num_of_D2Ds, num_of_cells));
            inter_D2D_to_D2D_interference = cvx(zeros(num_of_CUEs * num_of_D2Ds * (num_of_cells - 1), num_of_D2Ds, num_of_cells));
            inter_CUE_to_D2D_interference = cvx(zeros(num_of_CUEs * (num_of_cells - 1), num_of_D2Ds, num_of_cells));
            D2D_desired_signal = cvx(zeros(num_of_CUEs, num_of_D2Ds, num_of_cells));
            
            beta_intra_CUE_to_D2D_interference = zeros(num_of_CUEs, num_of_D2Ds, num_of_cells);
            beta_intra_D2D_to_D2D_interference = zeros(num_of_CUEs * (num_of_D2Ds - 1), num_of_D2Ds, num_of_cells);
            beta_inter_D2D_to_D2D_interference = zeros(num_of_CUEs * num_of_D2Ds * (num_of_cells - 1), num_of_D2Ds, num_of_cells);
            beta_inter_CUE_to_D2D_interference = zeros(num_of_CUEs * (num_of_cells - 1), num_of_D2Ds, num_of_cells);
            beta_D2D_desired_signal = zeros(num_of_CUEs, num_of_D2Ds, num_of_cells);
            
            % Calculate the values of these matrices
            for k = 1 : num_of_cells
                
                % CUE
                for i = 1 : num_of_CUEs
                    % Calculate the power of desired signal
                    CUE_desired_signal(1, i, k) = variable_CUE_power(i, 1, k) * channel_gain_matrix(i, 1, k);
                    alpha_CUE_desired_signal(1, i, k) = CUE_power(i, 1, k) * channel_gain_matrix(i, 1, k);
                    
                    % Calculate the power of intra-cell interference 
                    intra_D2D_to_BS_interference(:, i, k) = variable_D2D_power(:, i, k) .* channel_gain_matrix(num_of_CUEs + 1 : num_of_CUEs + num_of_D2Ds, 1, k);
                    alpha_intra_D2D_to_BS_interference(:, i, k) = D2D_power(:, i, k) .* channel_gain_matrix(num_of_CUEs + 1 : num_of_CUEs + num_of_D2Ds, 1, k);
                    
                    % Calculate the power of inter-cell interference 
                    for j = 1 : num_of_cells
                        if j == k
                            continue
                        else
                            if j < k
                                % D2D part
                                inter_D2D_to_BS_interference((j - 1) * num_of_D2Ds + 1 : (j - 1) * num_of_D2Ds + num_of_D2Ds, i, k) = variable_D2D_power(:, i, j) .* channel_gain_matrix(num_of_cells * num_of_CUEs + num_of_D2Ds + (j - 1) * num_of_D2Ds + 1 : num_of_cells * num_of_CUEs + num_of_D2Ds + (j - 1) * num_of_D2Ds + num_of_D2Ds, 1, k);
                                alpha_inter_D2D_to_BS_interference((j - 1) * num_of_D2Ds + 1 : (j - 1) * num_of_D2Ds + num_of_D2Ds, i, k) = D2D_power(:, i, j) .* channel_gain_matrix(num_of_cells * num_of_CUEs + num_of_D2Ds + (j - 1) * num_of_D2Ds + 1 : num_of_cells * num_of_CUEs + num_of_D2Ds + (j - 1) * num_of_D2Ds + num_of_D2Ds, 1, k);
                                % CUE part
                                inter_CUE_to_BS_interference(j, i, k) = variable_CUE_power(i, 1, j) * channel_gain_matrix(num_of_CUEs + num_of_D2Ds + (j - 1) * num_of_CUEs + i, 1, k);
                                alpha_inter_CUE_to_BS_interference(j, i, k) = CUE_power(i, 1, j) * channel_gain_matrix(num_of_CUEs + num_of_D2Ds + (j - 1) * num_of_CUEs + i, 1, k);
                            else
                                % D2D part
                                inter_D2D_to_BS_interference((j - 2) * num_of_D2Ds + 1 : (j - 2) * num_of_D2Ds + num_of_D2Ds, i, k) = variable_D2D_power(:, i, j) .* channel_gain_matrix(num_of_cells * num_of_CUEs + num_of_D2Ds + (j - 2) * num_of_D2Ds + 1 : num_of_cells * num_of_CUEs + num_of_D2Ds + (j - 2) * num_of_D2Ds + num_of_D2Ds, 1, k);
                                alpha_inter_D2D_to_BS_interference((j - 2) * num_of_D2Ds + 1 : (j - 2) * num_of_D2Ds + num_of_D2Ds, i, k) = D2D_power(:, i, j) .* channel_gain_matrix(num_of_cells * num_of_CUEs + num_of_D2Ds + (j - 2) * num_of_D2Ds + 1 : num_of_cells * num_of_CUEs + num_of_D2Ds + (j - 2) * num_of_D2Ds + num_of_D2Ds, 1, k);
                                % CUE part
                                inter_CUE_to_BS_interference(j - 1, i, k) = variable_CUE_power(i, 1, j) * channel_gain_matrix(num_of_CUEs + num_of_D2Ds + (j - 2) * num_of_CUEs + i, 1, k);
                                alpha_inter_CUE_to_BS_interference(j - 1, i, k) = CUE_power(i, 1, j) * channel_gain_matrix(num_of_CUEs + num_of_D2Ds + (j - 2) * num_of_CUEs + i, 1, k);
                            end
                        end
                    end
                end
                
                % D2D pair
                for i = 1 : num_of_D2Ds
                    % Calculate the power of desired signal
                    D2D_desired_signal(:, i, k) = variable_D2D_power(i, :, k) * channel_gain_matrix(num_of_CUEs + i, 1 + i, k);
                    beta_D2D_desired_signal(:, i, k) = D2D_power(i, :, k) * channel_gain_matrix(num_of_CUEs + i, 1 + i, k);
                    
                    % Calculate the power of intra-cell interference
                    % CUE part
                    intra_CUE_to_D2D_interference(:, i, k) = variable_CUE_power(:, 1, k) .* channel_gain_matrix(1 : num_of_CUEs, 1 + i, k);
                    beta_intra_CUE_to_D2D_interference(:, i, k) = CUE_power(:, 1, k) .* channel_gain_matrix(1 : num_of_CUEs, 1 + i, k);
                    % D2D part
                    for j = 1 : num_of_D2Ds
                        if j == i
                            continue
                        else
                            if j < i
                                intra_D2D_to_D2D_interference((j - 1) * num_of_CUEs + 1 : (j - 1) * num_of_CUEs + num_of_CUEs, i, k) = variable_D2D_power(j, :, k) * channel_gain_matrix(num_of_CUEs + j, 1 + i, k);
                                beta_intra_D2D_to_D2D_interference((j - 1) * num_of_CUEs + 1 : (j - 1) * num_of_CUEs + num_of_CUEs, i, k) = D2D_power(j, :, k) * channel_gain_matrix(num_of_CUEs + j, 1 + i, k);
                            else
                                intra_D2D_to_D2D_interference((j - 2) * num_of_CUEs + 1 : (j - 2) * num_of_CUEs + num_of_CUEs, i, k) = variable_D2D_power(j, :, k) * channel_gain_matrix(num_of_CUEs + j, 1 + i, k);
                                beta_intra_D2D_to_D2D_interference((j - 2) * num_of_CUEs + 1 : (j - 2) * num_of_CUEs + num_of_CUEs, i, k) = D2D_power(j, :, k) * channel_gain_matrix(num_of_CUEs + j, 1 + i, k);
                            end
                        end
                    end
                    
                    % Calculate the power of inter-cell interference
                    for j = 1 : num_of_cells
                        if j == k
                            continue
                        else
                            if j < k
                                % D2D part
                                cellj_D2D_power = transpose(variable_D2D_power(:, :, j));
                                beta_cellj_D2D_power = transpose(D2D_power(:, :, j));
                                cellj_channel_gain = channel_gain_matrix(num_of_cells * num_of_CUEs + num_of_D2Ds + (j - 1) * num_of_D2Ds + 1 : num_of_cells * num_of_CUEs + num_of_D2Ds + (j - 1) * num_of_D2Ds + num_of_D2Ds, 1 + i, k);
                                cellj_channel_gain = transpose(repmat(cellj_channel_gain, 1, num_of_CUEs));
                                inter_D2D_to_D2D_interference((j - 1) * num_of_CUEs * num_of_D2Ds + 1 : (j - 1) * num_of_CUEs * num_of_D2Ds + num_of_CUEs * num_of_D2Ds, i, k) = cellj_D2D_power(:) .* cellj_channel_gain(:);
                                beta_inter_D2D_to_D2D_interference((j - 1) * num_of_CUEs * num_of_D2Ds + 1 : (j - 1) * num_of_CUEs * num_of_D2Ds + num_of_CUEs * num_of_D2Ds, i, k) = beta_cellj_D2D_power(:) .* cellj_channel_gain(:);
                                % CUE part
                                inter_CUE_to_D2D_interference((j - 1) * num_of_CUEs + 1 : (j - 1) * num_of_CUEs + num_of_CUEs, i, k) = variable_CUE_power(:, 1, j) .* channel_gain_matrix(num_of_CUEs + num_of_D2Ds + (j - 1) * num_of_CUEs + 1 : num_of_CUEs + num_of_D2Ds + (j - 1) * num_of_CUEs + num_of_CUEs, 1 + i, k);
                                beta_inter_CUE_to_D2D_interference((j - 1) * num_of_CUEs + 1 : (j - 1) * num_of_CUEs + num_of_CUEs, i, k) = CUE_power(:, 1, j) .* channel_gain_matrix(num_of_CUEs + num_of_D2Ds + (j - 1) * num_of_CUEs + 1 : num_of_CUEs + num_of_D2Ds + (j - 1) * num_of_CUEs + num_of_CUEs, 1 + i, k);
                            else
                                % D2D part
                                cellj_D2D_power = transpose(variable_D2D_power(:, :, j));
                                beta_cellj_D2D_power = transpose(D2D_power(:, :, j));
                                cellj_channel_gain = channel_gain_matrix(num_of_cells * num_of_CUEs + num_of_D2Ds + (j - 2) * num_of_D2Ds + 1 : num_of_cells * num_of_CUEs + num_of_D2Ds + (j - 2) * num_of_D2Ds + num_of_D2Ds, 1 + i, k);
                                cellj_channel_gain = transpose(repmat(cellj_channel_gain, 1, num_of_CUEs));
                                inter_D2D_to_D2D_interference((j - 2) * num_of_CUEs * num_of_D2Ds + 1 : (j - 2) * num_of_CUEs * num_of_D2Ds + num_of_CUEs * num_of_D2Ds, i, k) = cellj_D2D_power(:) .* cellj_channel_gain(:);
                                beta_inter_D2D_to_D2D_interference((j - 2) * num_of_CUEs * num_of_D2Ds + 1 : (j - 2) * num_of_CUEs * num_of_D2Ds + num_of_CUEs * num_of_D2Ds, i, k) = beta_cellj_D2D_power(:) .* cellj_channel_gain(:);
                                % CUE part
                                inter_CUE_to_D2D_interference((j - 2) * num_of_CUEs + 1 : (j - 2) * num_of_CUEs + num_of_CUEs, i, k) = variable_CUE_power(:, 1, j) .* channel_gain_matrix(num_of_CUEs + num_of_D2Ds + (j - 2) * num_of_CUEs + 1 : num_of_CUEs + num_of_D2Ds + (j - 2) * num_of_CUEs + num_of_CUEs, 1 + i, k);
                                beta_inter_CUE_to_D2D_interference((j - 2) * num_of_CUEs + 1 : (j - 2) * num_of_CUEs + num_of_CUEs, i, k) = CUE_power(:, 1, j) .* channel_gain_matrix(num_of_CUEs + num_of_D2Ds + (j - 2) * num_of_CUEs + 1 : num_of_CUEs + num_of_D2Ds + (j - 2) * num_of_CUEs + num_of_CUEs, 1 + i, k);
                            end
                        end
                    end
                end
            end
            
            % CUE_rate_related_expression: CUE's data rate related expression
            numerator_CUE = sum(intra_D2D_to_BS_interference, 1) + sum(inter_D2D_to_BS_interference, 1) + sum(inter_CUE_to_BS_interference, 1) + noise;
            
            alpha_total_signal = sum(alpha_intra_D2D_to_BS_interference, 1) + sum(alpha_inter_D2D_to_BS_interference, 1) + sum(alpha_inter_CUE_to_BS_interference, 1) + noise + alpha_CUE_desired_signal;
            alpha_1 = alpha_intra_D2D_to_BS_interference ./ alpha_total_signal + realmin;
            alpha_2 = alpha_inter_D2D_to_BS_interference ./ alpha_total_signal + realmin;
            alpha_3 = alpha_inter_CUE_to_BS_interference ./ alpha_total_signal + realmin;
            alpha_4 = noise ./ alpha_total_signal + realmin;
            alpha_5 = alpha_CUE_desired_signal ./ alpha_total_signal + realmin;
            
            denominator_CUE = prod((intra_D2D_to_BS_interference ./ alpha_1) .^ alpha_1, 1) .* prod((inter_D2D_to_BS_interference ./ alpha_2) .^ alpha_2, 1) .* prod((inter_CUE_to_BS_interference ./ alpha_3) .^ alpha_3, 1) .* ((noise ./ alpha_4) .^ alpha_4) .* ((CUE_desired_signal ./ alpha_5) .^ alpha_5);
            
            fraction_CUE = numerator_CUE ./ denominator_CUE + realmin; % Avoid dividing by zero
            CUE_rate_related_expression = prod(fraction_CUE(:));
            
            % D2D_rate_related_expression: D2D pair's data rate related expression
            sum_intra_D2D_to_D2D_interference = cvx(zeros(num_of_CUEs, num_of_D2Ds, num_of_cells));
            sum_inter_D2D_to_D2D_interference = cvx(zeros(num_of_CUEs, num_of_D2Ds, num_of_cells));
            sum_inter_CUE_to_D2D_interference = cvx(zeros(num_of_CUEs, num_of_D2Ds, num_of_cells));
            sum_beta_intra_D2D_to_D2D_interference = zeros(num_of_CUEs, num_of_D2Ds, num_of_cells);
            sum_beta_inter_D2D_to_D2D_interference = zeros(num_of_CUEs, num_of_D2Ds, num_of_cells);
            sum_beta_inter_CUE_to_D2D_interference = zeros(num_of_CUEs, num_of_D2Ds, num_of_cells);
            for i = 1 : num_of_CUEs
                sum_intra_D2D_to_D2D_interference(i, :, :) = sum(intra_D2D_to_D2D_interference(i : num_of_CUEs : end, :, :), 1);
                sum_inter_D2D_to_D2D_interference(i, :, :) = sum(inter_D2D_to_D2D_interference(i : num_of_CUEs : end, :, :), 1);
                sum_inter_CUE_to_D2D_interference(i, :, :) = sum(inter_CUE_to_D2D_interference(i : num_of_CUEs : end, :, :), 1);
                sum_beta_intra_D2D_to_D2D_interference(i, :, :) = sum(beta_intra_D2D_to_D2D_interference(i : num_of_CUEs : end, :, :), 1);
                sum_beta_inter_D2D_to_D2D_interference(i, :, :) = sum(beta_inter_D2D_to_D2D_interference(i : num_of_CUEs : end, :, :), 1);
                sum_beta_inter_CUE_to_D2D_interference(i, :, :) = sum(beta_inter_CUE_to_D2D_interference(i : num_of_CUEs : end, :, :), 1);
            end
            
            numerator_D2D = intra_CUE_to_D2D_interference + sum_intra_D2D_to_D2D_interference + sum_inter_D2D_to_D2D_interference + sum_inter_CUE_to_D2D_interference + noise;
            
            beta_total_signal = beta_intra_CUE_to_D2D_interference + sum_beta_intra_D2D_to_D2D_interference + sum_beta_inter_D2D_to_D2D_interference + sum_beta_inter_CUE_to_D2D_interference + noise + beta_D2D_desired_signal;
            beta_1 = beta_intra_CUE_to_D2D_interference ./ beta_total_signal + realmin;
            beta_2 = beta_intra_D2D_to_D2D_interference ./ repmat(beta_total_signal, num_of_D2Ds - 1, 1, 1) + realmin;
            beta_3 = beta_inter_D2D_to_D2D_interference ./ repmat(beta_total_signal, num_of_D2Ds * (num_of_cells - 1), 1, 1) + realmin;
            beta_4 = beta_inter_CUE_to_D2D_interference ./ repmat(beta_total_signal, num_of_cells - 1, 1, 1) + realmin;
            beta_5 = noise ./ beta_total_signal + realmin;
            beta_6 = beta_D2D_desired_signal ./ beta_total_signal + realmin;
            
            prod_beta_2 = cvx(zeros(num_of_CUEs, num_of_D2Ds, num_of_cells));
            prod_beta_3 = cvx(zeros(num_of_CUEs, num_of_D2Ds, num_of_cells));
            prod_beta_4 = cvx(zeros(num_of_CUEs, num_of_D2Ds, num_of_cells));
            transformed_intra_D2D_to_D2D_interference = (intra_D2D_to_D2D_interference ./ beta_2) .^ beta_2;
            transformed_inter_D2D_to_D2D_interference = (inter_D2D_to_D2D_interference ./ beta_3) .^ beta_3;
            transformed_inter_CUE_to_D2D_interference = (inter_CUE_to_D2D_interference ./ beta_4) .^ beta_4;
            for i = 1 : num_of_CUEs
               prod_beta_2(i, :, :) = prod(transformed_intra_D2D_to_D2D_interference(i : num_of_CUEs : end, :, :), 1);
               prod_beta_3(i, :, :) = prod(transformed_inter_D2D_to_D2D_interference(i : num_of_CUEs : end, :, :), 1);
               prod_beta_4(i, :, :) = prod(transformed_inter_CUE_to_D2D_interference(i : num_of_CUEs : end, :, :), 1);
            end
            
            denominator_D2D = ((intra_CUE_to_D2D_interference ./ beta_1) .^ beta_1) .* prod_beta_2 .* prod_beta_3 .* prod_beta_4 .* ((noise ./ beta_5) .^ beta_5) .* ((D2D_desired_signal ./ beta_6) .^ beta_6);
            
            fraction_D2D = numerator_D2D ./ denominator_D2D + realmin; % Avoid dividing by zero
            D2D_rate_related_expression = prod(fraction_D2D(:));
            
            % CUE_power_related_expression: CUE's power consumption related expression
            Maclaurin_CUE = 1 + (system_EE / PA_inefficiency_factor) * log(2) * variable_CUE_power;
            CUE_power_related_expression = prod(Maclaurin_CUE(:));
            
            % D2D_power_related_expression: D2D pair's power consumption related expression
            Maclaurin_D2D = 1 + (system_EE / PA_inefficiency_factor) * log(2) * variable_D2D_power;
            D2D_power_related_expression = prod(Maclaurin_D2D(:));
            
            % Define the objective function
            minimize(CUE_rate_related_expression * D2D_rate_related_expression * CUE_power_related_expression * D2D_power_related_expression)
            
            subject to
            
                % CUE's rate constraint related expression
                CUE_rate_constraint_expression = numerator_CUE ./ CUE_desired_signal;
                
                % D2D pair's rate constraint related expression
                D2D_rate_constraint_expression = prod(fraction_D2D, 1);
                
                % Define the constraints
                for k = 1 : num_of_cells 
                    for i = 1 : num_of_CUEs
                        % CUE's power budget limitation
                        variable_CUE_power(i, 1, k) <= Pmax;
                        % CUE's minimum rate requirement
                        CUE_rate_constraint_expression(1, i, k) <= 1 / (2 ^ QoS_of_CUE(i, 1, k) - 1);
                        
                        % D2D pair's power budget limitation on each resource block
                        for j = 1 : num_of_D2Ds 
                            variable_D2D_power(j, i, k) <= Pmax;
                        end    
                    end
                    
                    for i = 1 : num_of_D2Ds
                        % D2D pair's total power budget limitation
                        sum(variable_D2D_power(i, :, k)) <= Pmax;
                        % D2D pair's minimum rate requirement
                        D2D_rate_constraint_expression(1, i, k) <= 1 / 2 ^ QoS_of_D2D;
                    end
                end
                
        cvx_end
        
        if strcmp(cvx_status, 'Solved')
            % Update transmit power of all devices iteratively
            CUE_power = variable_CUE_power;
            D2D_power = variable_D2D_power;
        
            % Calculate updated system EE
            [updated_system_EE, total_data_rate, total_power_consumption, CUE_rate, D2D_rate, feasible] = calculate_system_EE(num_of_cells, num_of_CUEs, num_of_D2Ds, channel_gain_matrix, CUE_power, D2D_power, Pmax, QoS_of_CUE, QoS_of_D2D);
        
            % Summary
            summary(num_of_cells, num_of_CUEs, num_of_D2Ds, updated_system_EE, total_data_rate, total_power_consumption, CUE_rate, D2D_rate, CUE_power, D2D_power, feasible, Pmax, QoS_of_CUE, QoS_of_D2D, proportion, max_dinkelbach_iterations, max_condensation_iterations, iteration, dinkelbach_iterations, condensation_iterations);
            
            % Increase the index of the iteration
            iteration = iteration + 1;
            condensation_iterations = condensation_iterations + 1;
        elseif strcmp(cvx_status, 'Infeasible')
            cprintf('Red', 'The GP problem has been proven to be infeasible.\n\n');
            %error("Error. The GP problem has been proven to be infeasible, please increase the 'Pmax' or decrease the 'proportion' and the 'QoS_of_D2D'.")
            optimal_CUE_power = NaN;
            optimal_D2D_power = NaN;
            success = 0;
            return
        else
            %error('Error. The solver failed to make sufficient progress towards a solution.')
            cprintf('Red', 'The solver failed to make sufficient progress towards a solution.\n\n');
            optimal_CUE_power = NaN;
            optimal_D2D_power = NaN;
            success = 0;
            return
        end
    end
    
    % The system EE is converged
    if abs(total_data_rate - system_EE * total_power_consumption) < 1e-4 && feasible
        cprintf('Red', 'The objective function is solved (system EE is converged).\n');
        optimal_CUE_power = CUE_power;
        optimal_D2D_power = D2D_power;
        success = 1;
        fprintf(1, 'Optimal system enenrgy efficiency: %f\n', updated_system_EE);
        fprintf(1, 'System sum rate: %f\n', total_data_rate);
        fprintf(1, 'Total power consumption: %f\n\n', total_power_consumption);
        print_transmit_power(num_of_cells, num_of_CUEs, num_of_D2Ds, Pmax, optimal_CUE_power, optimal_D2D_power, 1);
        print_data_rate(num_of_cells, num_of_CUEs, num_of_D2Ds, QoS_of_CUE, QoS_of_D2D, CUE_rate, D2D_rate, proportion, 1)
        return
    % The system EE is not converged
    else
        system_EE = updated_system_EE;
    end
    
    dinkelbach_iterations = dinkelbach_iterations + 1;
end

% Reach the maximum iterations of dinkelbach method
if feasible
    cprintf('Red', 'The objective function is solved (reach the maximum iterations of dinkelbach method).\n');
    optimal_CUE_power = CUE_power;
    optimal_D2D_power = D2D_power;
    success = 1;
    % Show the details
    fprintf(1, 'Optimal system enenrgy efficiency: %f\n', system_EE);
    fprintf(1, 'System sum rate: %f\n', total_data_rate);
    fprintf(1, 'Total power consumption: %f\n\n', total_power_consumption);
    print_transmit_power(num_of_cells, num_of_CUEs, num_of_D2Ds, Pmax, optimal_CUE_power, optimal_D2D_power, 1);
    print_data_rate(num_of_cells, num_of_CUEs, num_of_D2Ds, QoS_of_CUE, QoS_of_D2D, CUE_rate, D2D_rate, proportion, 1)
else 
    cprintf('Red', 'The objective function is solved, but optimal solution is infeasible.\n');
    optimal_CUE_power = NaN;
    optimal_D2D_power = NaN;
    success = 0;
end