function main(num_of_cells, num_of_CUEs, num_of_D2Ds, num_of_training_data)

%%%%%%%%%%%%%%%%%%%% Parameters setting %%%%%%%%%%%%%%%%%%%%
%num_of_training_data = 3000; % Number of the training data
%num_of_cells = 2; % Number of the cells in the system
%num_of_CUEs = 2; % Number of the CUEs in each cell
%num_of_D2Ds = 2; % Number of the D2D pairs in each cell

radius = 500; % The radius of the cell (meter)
Pmax = 0.2; % Maximun transimit power of all devices (Watt)
QoS_of_D2D = 3; % Minimum data rate requirement of all D2D pairs (bps/Hz)
proportion = 0.2; % The proportion of CUE's minimum rate requirement to CUE's maximum data rate

max_dinkelbach_iterations = 3; % Maximum iterations of dinkelbach method 
max_condensation_iterations = 2; % Maximum iterations of condensation method
%%%%%%%%%%%%%%%%%%%% Parameters setting %%%%%%%%%%%%%%%%%%%%

input_data = cell(1, num_of_training_data);
target_data = cell(2, num_of_training_data);

index = 1;

% Calculate the coordinate of each base station
[cenX, cenY] = cell_deployment(num_of_cells, radius);

while index <= num_of_training_data
    
    while 1
        % Generate positions of CUEs and D2D pairs
        CUE_position = randomize_device_position(num_of_cells, radius, cenX, cenY, num_of_CUEs);
        D2D_position = randomize_device_position(num_of_cells, radius, cenX, cenY, num_of_D2Ds);

        % Calculate the channel gain between all devices
        channel_gain_matrix = channel_gain_initialization(num_of_cells, num_of_CUEs, num_of_D2Ds, cenX, cenY, CUE_position, D2D_position);
        
        % Calculate the minimum rate requiement of CUEs
        QoS_of_CUE = calculate_QoS_of_CUE(num_of_cells, num_of_CUEs, channel_gain_matrix, Pmax, proportion);
    
        % Initialize the transmit power of CUEs and D2D pairs
        [initial_CUE_power, initial_D2D_power, isfeasible] = transmit_power_initialization(num_of_cells, num_of_CUEs, num_of_D2Ds, channel_gain_matrix, Pmax, QoS_of_CUE, QoS_of_D2D);
    
        if isfeasible == true
            break
        end
    end

    % Solve the objective function by CVX 
    [optimal_CUE_power, optimal_D2D_power, success] = convex_optimization(num_of_cells, num_of_CUEs, num_of_D2Ds, channel_gain_matrix, initial_CUE_power, initial_D2D_power, Pmax, QoS_of_CUE, QoS_of_D2D, proportion, max_dinkelbach_iterations, max_condensation_iterations);

    if success == 1
        cprintf('Red', 'Training data #%d: Saved\n\n', index);
        input_data(1, index) = {channel_gain_matrix};
        target_data(1, index) = {optimal_CUE_power};
        target_data(2, index) = {optimal_D2D_power};
        index = index + 1;
    end
end

filename = sprintf('data_Cell_%d_CUE_%d_D2D_%d_%d', num_of_cells, num_of_CUEs, num_of_D2Ds, num_of_training_data);
save(filename, 'input_data', 'target_data');
