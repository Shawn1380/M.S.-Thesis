function print_data_rate(num_of_cells, num_of_CUEs, num_of_D2Ds, QoS_of_CUE, QoS_of_D2D, CUE_rate, D2D_rate, proportion, mode)
% Input:
% num_of_cells: Number of the cells in the system
% num_of_CUEs: Number of the CUEs in each cell
% num_of_D2Ds: Number of the D2D pairs in each cell
% QoS_of_CUE: Minimum data rate requirement of all CUEs (bps/Hz)
% QoS_of_D2D: Minimum data rate requirement of all D2D pairs (bps/Hz)
% CUE_rate: The data rate achieved by CUEs
% D2D_rate: The data rate achieved by D2D pairs
% proportion: The proportion of CUE's minimum rate requirement to CUE's maximum data rate
% mode: Determine what kind of message should be print

num_of_RBs = num_of_CUEs;

if mode == 0
    str = 'Initialization';   
elseif mode == 1
    str = 'Optimal solution';
elseif mode == 2
    str = 'CVX summary';
else
    % Define your mode right here
end

for cell_index = 1 : num_of_cells
    
    cprintf('Blue', '%s: Cell %d (data rate)\n', str, cell_index);
    fprintf("CUE's minimum data rate requirement: %2.1f times the speed of the maximum data rate (bps/Hz)\n", proportion);
    fprintf("D2D's minimum data rate requirement: %2.1f (bps/Hz)\n\n", QoS_of_D2D);
    
    % Print the CUE's data rate
    fprintf('\t\tRB');
    fprintf('\t\t\t\t\tRequirement\n');
    for CUE_index = 1 : num_of_CUEs
        fprintf('CUE %d\t%12.10f\t\t', CUE_index, CUE_rate(CUE_index, 1, cell_index));
        fprintf('%12.10f\n', QoS_of_CUE(CUE_index, 1, cell_index));
    end
    fprintf('\n');
    
    % Print the D2D pair's data rate
    fprintf('\t\t');
    for RB_index = 1 : num_of_RBs
        fprintf('RB %d\t\t\t\t', RB_index);    
    end
    fprintf('Total\t\t\t\t');
    fprintf('Requirement\n');
    
    for D2D_index = 1 : num_of_D2Ds
        fprintf('D2D %d\t', D2D_index);
        for RB_index = 1 : num_of_RBs
            fprintf('%12.10f\t\t', D2D_rate(D2D_index, RB_index, cell_index)) 
        end
        fprintf('%12.10f\t\t', sum(D2D_rate(D2D_index, :, cell_index)));
        fprintf('%12.10f\n', QoS_of_D2D);
    end
    
    fprintf('\n');
end


